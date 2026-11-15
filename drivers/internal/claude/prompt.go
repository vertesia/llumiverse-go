package claude

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/vertesia/llumiverse-go/core"
)

const (
	// claudeJSONSchemaInstructionPrefix starts the prompt text that asks for JSON matching the result schema.
	claudeJSONSchemaInstructionPrefix = "The answer must be a JSON object using the following JSON Schema:"
	// claudeToolAwareJSONSchemaInstructionPrefix is the prefix used when the request declares tools.
	claudeToolAwareJSONSchemaInstructionPrefix = "When not calling tools, the answer must be a JSON object using the following JSON Schema:"
	// claudeAgentPromptCacheKeyPrefix marks agent prompt cache keys, which select the agent cache layout.
	claudeAgentPromptCacheKeyPrefix = "agent-"
)

// claudeCompactJSON encodes a value as compact JSON without HTML escaping, like JSON.stringify.
func claudeCompactJSON(value any) string {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return ""
	}
	return strings.TrimSuffix(buffer.String(), "\n")
}

// claudeSchemaInstructions is the prompt text that asks Claude for JSON matching the result schema.
func claudeSchemaInstructions(options ExecutionOptions) string {
	prefix := claudeJSONSchemaInstructionPrefix
	if len(options.Tools) > 0 {
		prefix = claudeToolAwareJSONSchemaInstructionPrefix
	}
	return prefix + "\n" + claudeCompactJSON(options.ResultSchema)
}

// hasClaudeSchemaInstructions reports whether a system block or a turn's text already carries the instructions for
// this request's schema. An earlier turn may carry another schema's instructions, which do not count.
func hasClaudeSchemaInstructions(options ExecutionOptions, system []claudeBlock, messages []claudeMessage) bool {
	schema := claudeCompactJSON(options.ResultSchema)
	instructions := []string{
		claudeJSONSchemaInstructionPrefix + "\n" + schema,
		claudeToolAwareJSONSchemaInstructionPrefix + "\n" + schema,
	}
	carries := func(text string) bool {
		for _, instruction := range instructions {
			if strings.Contains(text, instruction) {
				return true
			}
		}
		return false
	}
	for _, block := range system {
		if carries(block.Text) {
			return true
		}
	}
	for _, message := range messages {
		for _, block := range message.Content {
			if block.Type == "text" && carries(block.Text) {
				return true
			}
		}
	}
	return false
}

// isClaudeResultSchemaSystemBlock reports whether a system block is generated result-schema instructions.
func isClaudeResultSchemaSystemBlock(block claudeBlock) bool {
	return block.Type == "text" &&
		(strings.HasPrefix(block.Text, claudeToolAwareJSONSchemaInstructionPrefix) ||
			strings.HasPrefix(block.Text, claudeJSONSchemaInstructionPrefix))
}

// mergeClaudeSystemBlocks appends system blocks. Agent tool continuations carry the result schema on every request;
// only the latest generated schema block is kept, so the system prefix stays byte-stable, and conversations that
// accumulated one schema block per iteration are repaired.
func mergeClaudeSystemBlocks(base, additions []claudeBlock) []claudeBlock {
	combined := append(append([]claudeBlock(nil), base...), additions...)
	hasSchema := false
	for _, block := range additions {
		if isClaudeResultSchemaSystemBlock(block) {
			hasSchema = true
			break
		}
	}
	if !hasSchema {
		return combined
	}
	latest := -1
	for i, block := range combined {
		if isClaudeResultSchemaSystemBlock(block) {
			latest = i
		}
	}
	out := make([]claudeBlock, 0, len(combined))
	for i, block := range combined {
		if !isClaudeResultSchemaSystemBlock(block) || i == latest {
			out = append(out, block)
		}
	}
	return out
}

// isClaudeAgentPromptCacheKey reports whether a cache key selects the agent cache layout.
func isClaudeAgentPromptCacheKey(key string) bool {
	return strings.HasPrefix(key, claudeAgentPromptCacheKeyPrefix)
}

// formatClaudePrompt maps provider-neutral segments onto Claude Messages blocks: system segments become system
// text, tool segments become user turns holding a tool_result, and other segments become user or assistant text
// and file blocks. Safety segments go last.
//
// The result schema becomes prompt instructions unless nativeResultSchema is set (the provider enforces it as
// output_config.format). Without a prompt cache key the instructions are a system block. With an agent cache key
// they stay one system block for the whole tool loop; with another key they are appended to the last task text
// block, after the cached prefix, falling back to a system block.
func formatClaudePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions, nativeResultSchema bool) (claudePrompt, error) {
	prompt := claudePrompt{}
	for _, segment := range segments {
		if segment.Role == PromptRoleSystem && segment.Content != "" {
			prompt.System = append(prompt.System, claudeBlock{Type: "text", Text: segment.Content})
		}
	}
	schemaText := ""
	if options.ResultSchema != nil && !nativeResultSchema {
		schemaText = claudeSchemaInstructions(options)
		if options.PromptCacheKey == "" {
			prompt.System = append(prompt.System, claudeBlock{Type: "text", Text: schemaText})
		}
	}

	var safety []claudeMessage
	for _, segment := range segments {
		switch segment.Role {
		case PromptRoleSystem, PromptRoleNegative, PromptRoleMask:
			continue
		case PromptRoleTool:
			if segment.ToolUseID == "" {
				return claudePrompt{}, errors.New("tool prompt segment requires ToolUseID")
			}
			content := []claudeBlock{}
			if segment.Content != "" {
				content = append(content, claudeBlock{Type: "text", Text: segment.Content})
			}
			files, err := claudeFileBlocks(ctx, segment)
			if err != nil {
				return claudePrompt{}, err
			}
			content = append(content, files...)
			prompt.Messages = append(prompt.Messages, claudeMessage{
				Role:    "user",
				Content: []claudeBlock{{Type: "tool_result", ToolUseID: segment.ToolUseID, Content: content}},
			})
		default:
			content := []claudeBlock{}
			if segment.Content != "" {
				content = append(content, claudeBlock{Type: "text", Text: segment.Content})
			}
			files, err := claudeFileBlocks(ctx, segment)
			if err != nil {
				return claudePrompt{}, err
			}
			content = append(content, files...)
			if len(content) == 0 {
				continue
			}
			message := claudeMessage{Role: "user", Content: content}
			if segment.Role == PromptRoleAssistant {
				message.Role = "assistant"
			}
			if segment.Role == PromptRoleSafety {
				safety = append(safety, message)
			} else {
				prompt.Messages = append(prompt.Messages, message)
			}
		}
	}

	if schemaText != "" && options.PromptCacheKey != "" {
		placed := false
		if !isClaudeAgentPromptCacheKey(options.PromptCacheKey) && len(prompt.Messages) > 0 {
			task := &prompt.Messages[len(prompt.Messages)-1]
			if last := len(task.Content) - 1; last >= 0 && task.Content[last].Type == "text" {
				task.Content[last].Text += "\n\n" + schemaText
				placed = true
			}
		}
		if !placed {
			prompt.System = append(prompt.System, claudeBlock{Type: "text", Text: schemaText})
		}
	}
	prompt.Messages = append(prompt.Messages, safety...)
	return prompt, nil
}

// claudeTextApplicationTypes are application types sent as plain-text documents.
var claudeTextApplicationTypes = map[string]bool{
	"application/json":   true,
	"application/xml":    true,
	"application/yaml":   true,
	"application/x-yaml": true,
}

// isClaudeTextMIME reports text formats sent as plain-text documents: text/*, JSON, XML and YAML, including +json
// and +xml types.
func isClaudeTextMIME(mimeType string) bool {
	kind := strings.ToLower(strings.TrimSpace(strings.Split(mimeType, ";")[0]))
	return strings.HasPrefix(kind, "text/") || claudeTextApplicationTypes[kind] ||
		strings.HasSuffix(kind, "+json") || strings.HasSuffix(kind, "+xml")
}

// claudeFileBlocks converts attached files into Claude blocks, for user turns and tool results alike: PNG, JPEG,
// GIF and WebP images (image/jpg is read as JPEG), PDFs as base64 documents, and text formats as text documents.
// Other image types and audio are errors; video and other types are skipped.
func claudeFileBlocks(ctx context.Context, segment PromptSegment) ([]claudeBlock, error) {
	blocks := make([]claudeBlock, 0, len(segment.Files))
	for _, file := range segment.Files {
		mimeType := file.MIMEType()
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			normalized := mimeType
			if normalized == "image/jpg" {
				normalized = "image/jpeg"
			}
			switch normalized {
			case "image/png", "image/jpeg", "image/gif", "image/webp":
			default:
				return nil, fmt.Errorf("unsupported image type: %s", mimeType)
			}
			encoded, err := dataSourceToBase64(ctx, file)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, claudeBlock{Type: "image", Source: &claudeSource{Type: "base64", MediaType: normalized, Data: encoded}})
		case strings.HasPrefix(mimeType, "audio/"):
			return nil, errors.New("audio input is not supported by Claude; supply a transcript instead")
		case strings.HasPrefix(mimeType, "video/"):
			// Claude takes no video input; the attachment is skipped.
		case mimeType == "application/pdf":
			encoded, err := dataSourceToBase64(ctx, file)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, claudeBlock{Type: "document", Title: file.Name(), Source: &claudeSource{Type: "base64", MediaType: "application/pdf", Data: encoded}})
		case isClaudeTextMIME(mimeType):
			text, err := readAllStringFromDataSource(ctx, file)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, claudeBlock{Type: "document", Title: file.Name(), Source: &claudeSource{Type: "text", MediaType: "text/plain", Data: text}})
		}
	}
	return blocks, nil
}

// formatClaudeDebugPrompt returns a copy of the prompt whose base64 image and document payloads are truncated. The
// input is not modified.
func formatClaudeDebugPrompt(prompt claudePrompt) claudePrompt {
	truncate := func(blocks []claudeBlock) []claudeBlock {
		out := make([]claudeBlock, len(blocks))
		for i, block := range blocks {
			if (block.Type == "image" || block.Type == "document") && block.Source != nil && block.Source.Type == "base64" {
				source := *block.Source
				source.Data = core.TruncateBinaryForDebug(source.Data)
				block.Source = &source
			}
			out[i] = block
		}
		return out
	}
	messages := make([]claudeMessage, len(prompt.Messages))
	for i, message := range prompt.Messages {
		messages[i] = claudeMessage{Role: message.Role, Content: truncate(message.Content)}
	}
	prompt.Messages = messages
	return prompt
}
