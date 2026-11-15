package bedrock

import (
	"encoding/base64"
	"strings"

	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/vertesia/llumiverse-go/core"
)

// FormatDebugPrompt returns a copy of a Bedrock prompt that is safe to return
// in ExecutionResponse.Prompt: inline image, document, video and audio bytes,
// including those nested in tool results, become base64 truncated by
// core.TruncateBinaryForDebug, and image-generation inputs are truncated the
// same way. Every other block keeps its encoding/json shape, so the copy reads
// like the conversation Bedrock stores. The prompt being sent is not modified,
// and prompts of any other type are returned unchanged.
func (d *BedrockDriver) FormatDebugPrompt(prompt any) any {
	switch value := prompt.(type) {
	case bedrockPrompt:
		return bedrockDebugPrompt(value)
	case *bedrockPrompt:
		if value == nil {
			return prompt
		}
		return bedrockDebugPrompt(*value)
	case bedrockImagePrompt:
		return bedrockDebugImagePrompt(value)
	case *bedrockImagePrompt:
		if value == nil {
			return prompt
		}
		return bedrockDebugImagePrompt(*value)
	}
	return prompt
}

// bedrockDebugPrompt is the debug copy of a bedrockPrompt.
func bedrockDebugPrompt(prompt bedrockPrompt) map[string]any {
	messages := make([]map[string]any, 0, len(prompt.Messages))
	for _, message := range prompt.Messages {
		content := make([]any, 0, len(message.Content))
		for _, block := range message.Content {
			content = append(content, bedrockDebugContentBlock(block))
		}
		messages = append(messages, map[string]any{"Role": message.Role, "Content": content})
	}
	return map[string]any{"Messages": messages, "System": prompt.System, "Turn": prompt.Turn}
}

// bedrockDebugImagePrompt is the debug copy of an image-generation prompt.
func bedrockDebugImagePrompt(prompt bedrockImagePrompt) bedrockImagePrompt {
	prompt.Images = bedrockDebugBase64Strings(prompt.Images)
	prompt.Masks = bedrockDebugBase64Strings(prompt.Masks)
	return prompt
}

// bedrockDebugBase64Strings truncates base64 strings and data URLs into a new slice.
func bedrockDebugBase64Strings(values []string) []string {
	if values == nil {
		return nil
	}
	out := make([]string, len(values))
	for i, value := range values {
		if strings.HasPrefix(value, "data:") {
			out[i] = core.TruncateDataURLForDebug(value)
		} else {
			out[i] = core.TruncateBinaryForDebug(value)
		}
	}
	return out
}

// bedrockDebugBytes renders inline bytes as truncated base64.
func bedrockDebugBytes(data []byte) string {
	return core.TruncateBinaryForDebug(base64.StdEncoding.EncodeToString(data))
}

// bedrockDebugMedia renders a media block with inline bytes in its
// encoding/json shape, with the bytes truncated.
func bedrockDebugMedia(fields map[string]any, data []byte) map[string]any {
	fields["Source"] = map[string]any{"Value": bedrockDebugBytes(data)}
	return map[string]any{"Value": fields}
}

// bedrockDebugContentBlock returns the debug form of one Converse content block.
func bedrockDebugContentBlock(block brtypes.ContentBlock) any {
	switch value := block.(type) {
	case *brtypes.ContentBlockMemberImage:
		if source, ok := value.Value.Source.(*brtypes.ImageSourceMemberBytes); ok {
			return bedrockDebugMedia(map[string]any{"Format": value.Value.Format}, source.Value)
		}
	case *brtypes.ContentBlockMemberDocument:
		if source, ok := value.Value.Source.(*brtypes.DocumentSourceMemberBytes); ok {
			return bedrockDebugMedia(map[string]any{"Format": value.Value.Format, "Name": value.Value.Name}, source.Value)
		}
	case *brtypes.ContentBlockMemberVideo:
		if source, ok := value.Value.Source.(*brtypes.VideoSourceMemberBytes); ok {
			return bedrockDebugMedia(map[string]any{"Format": value.Value.Format}, source.Value)
		}
	case *brtypes.ContentBlockMemberAudio:
		if source, ok := value.Value.Source.(*brtypes.AudioSourceMemberBytes); ok {
			return bedrockDebugMedia(map[string]any{"Format": value.Value.Format}, source.Value)
		}
	case *brtypes.ContentBlockMemberToolResult:
		if !bedrockToolResultHasInlineBytes(value.Value.Content) {
			return block
		}
		content := make([]any, 0, len(value.Value.Content))
		for _, part := range value.Value.Content {
			content = append(content, bedrockDebugToolResultBlock(part))
		}
		return map[string]any{"Value": map[string]any{
			"ToolUseId": value.Value.ToolUseId,
			"Status":    value.Value.Status,
			"Type":      value.Value.Type,
			"Content":   content,
		}}
	}
	return block
}

// bedrockDebugToolResultBlock returns the debug form of one tool-result block.
func bedrockDebugToolResultBlock(block brtypes.ToolResultContentBlock) any {
	switch value := block.(type) {
	case *brtypes.ToolResultContentBlockMemberImage:
		if source, ok := value.Value.Source.(*brtypes.ImageSourceMemberBytes); ok {
			return bedrockDebugMedia(map[string]any{"Format": value.Value.Format}, source.Value)
		}
	case *brtypes.ToolResultContentBlockMemberDocument:
		if source, ok := value.Value.Source.(*brtypes.DocumentSourceMemberBytes); ok {
			return bedrockDebugMedia(map[string]any{"Format": value.Value.Format, "Name": value.Value.Name}, source.Value)
		}
	case *brtypes.ToolResultContentBlockMemberVideo:
		if source, ok := value.Value.Source.(*brtypes.VideoSourceMemberBytes); ok {
			return bedrockDebugMedia(map[string]any{"Format": value.Value.Format}, source.Value)
		}
	}
	return block
}

// bedrockToolResultHasInlineBytes reports whether tool-result content carries
// an image, document or video with inline bytes.
func bedrockToolResultHasInlineBytes(content []brtypes.ToolResultContentBlock) bool {
	for _, part := range content {
		if _, ok := bedrockDebugToolResultBlock(part).(map[string]any); ok {
			return true
		}
	}
	return false
}
