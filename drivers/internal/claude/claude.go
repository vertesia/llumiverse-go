package claude

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const defaultAnthropicBaseURL = "https://api.anthropic.com"
const anthropicVersion = "2023-06-01"

// AnthropicOptions configures the native Anthropic Claude Messages API driver.
type AnthropicOptions struct {
	DriverOptions
	// APIKey is sent as the x-api-key header.
	APIKey string
	// BaseURL defaults to the official Anthropic API.
	BaseURL string
	// HTTPClient overrides the default HTTP client.
	HTTPClient *http.Client
}

// AnthropicDriver implements Driver for Anthropic Claude models.
type AnthropicDriver struct {
	options AnthropicOptions
	client  *http.Client
}

// NewAnthropicDriver creates a Claude driver backed by the Anthropic Messages API.
func NewAnthropicDriver(options AnthropicOptions) (*AnthropicDriver, error) {
	if options.APIKey == "" {
		return nil, errors.New("api key is required")
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultAnthropicBaseURL
	}
	client := options.HTTPClient
	if client == nil {
		client = newHTTPClient(nil, options.HTTPTimeout)
	} else if hasHTTPTimeout(options.HTTPTimeout) {
		client = newHTTPClient(client, options.HTTPTimeout)
	}
	return &AnthropicDriver{options: options, client: client}, nil
}

// Provider returns ProviderAnthropic.
func (d *AnthropicDriver) Provider() Provider {
	return ProviderAnthropic
}

// claudePrompt is the Messages API request body shape: an alternating
// user/assistant message list plus an optional system block list. Turn is
// internal bookkeeping and is never serialized.
type claudePrompt struct {
	Messages []claudeMessage `json:"messages"`
	System   []claudeBlock   `json:"system,omitempty"`
	Turn     int             `json:"-"`
}

// claudeMessage is a single Messages API turn ("user" or "assistant") carrying
// one or more content blocks.
type claudeMessage struct {
	Role    string        `json:"role"`
	Content []claudeBlock `json:"content"`
}

// claudeBlock is a polymorphic Messages API content block. Type selects which
// other fields apply: Text for "text", Source for "image"/"document", ID/Name/
// Input for "tool_use", ToolUseID/Content for "tool_result". CacheControl marks
// a prefix boundary for prompt caching.
type claudeBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text,omitempty"`
	Title        string        `json:"title,omitempty"`
	Source       *claudeSource `json:"source,omitempty"`
	ToolUseID    string        `json:"tool_use_id,omitempty"`
	Content      []claudeBlock `json:"content,omitempty"`
	ID           string        `json:"id,omitempty"`
	Name         string        `json:"name,omitempty"`
	Input        any           `json:"input,omitempty"`
	CacheControl any           `json:"cache_control,omitempty"`
}

// claudeSource carries inline base64 (or text) data for image and document
// blocks, along with its media type.
type claudeSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
}

// formatClaudePrompt maps provider-neutral PromptSegments onto Claude Messages
// API blocks: system segments become system text blocks, tool segments become
// user messages wrapping a tool_result block, and everything else becomes
// user/assistant text+file blocks. A ResultSchema is injected as a system-text
// instruction, and safety segments are deferred to the end of the message list.
func formatClaudePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (claudePrompt, error) {
	// Shared Claude prompt formatting is used by both the native Anthropic driver
	// and Claude-on-Gemini Enterprise Agent Platform. They share the Messages API
	// content shape; only the client, endpoint, and auth headers differ.
	prompt := claudePrompt{}
	var safety []claudeMessage
	for _, segment := range segments {
		switch segment.Role {
		case PromptRoleNegative, PromptRoleMask:
			continue
		case PromptRoleSystem:
			if segment.Content != "" {
				prompt.System = append(prompt.System, claudeBlock{Type: "text", Text: segment.Content})
			}
		case PromptRoleTool:
			if segment.ToolUseID == "" {
				return claudePrompt{}, errors.New("tool prompt segment requires ToolUseID")
			}
			content := []claudeBlock{}
			if segment.Content != "" {
				content = append(content, claudeBlock{Type: "text", Text: segment.Content})
			}
			fileBlocks, err := claudeFileBlocks(ctx, segment, true)
			if err != nil {
				return claudePrompt{}, err
			}
			content = append(content, fileBlocks...)
			prompt.Messages = append(prompt.Messages, claudeMessage{
				Role: "user",
				Content: []claudeBlock{{
					Type:      "tool_result",
					ToolUseID: segment.ToolUseID,
					Content:   content,
				}},
			})
		default:
			blocks := []claudeBlock{}
			if segment.Content != "" {
				blocks = append(blocks, claudeBlock{Type: "text", Text: segment.Content})
			}
			fileBlocks, err := claudeFileBlocks(ctx, segment, false)
			if err != nil {
				return claudePrompt{}, err
			}
			blocks = append(blocks, fileBlocks...)
			if len(blocks) == 0 {
				continue
			}
			msg := claudeMessage{Role: "user", Content: blocks}
			if segment.Role == PromptRoleAssistant {
				msg.Role = "assistant"
			}
			if segment.Role == PromptRoleSafety {
				safety = append(safety, msg)
			} else {
				prompt.Messages = append(prompt.Messages, msg)
			}
		}
	}
	if options.ResultSchema != nil {
		schema, _ := json.Marshal(options.ResultSchema)
		notice := "The answer must be a JSON object using the following JSON Schema:\n" + string(schema)
		if len(options.Tools) > 0 {
			notice = "When not calling tools, the answer must be a JSON object using the following JSON Schema:\n" + string(schema)
		}
		prompt.System = append(prompt.System, claudeBlock{Type: "text", Text: notice})
	}
	// Safety messages are appended at the end, matching the TS client, so they
	// remain close to the final user request without becoming system text.
	prompt.Messages = append(prompt.Messages, safety...)
	return prompt, nil
}

// claudeFileBlocks converts attached files into Claude image/document blocks.
// Images become base64 "image" blocks; PDFs and text/* files become "document"
// blocks. When restrictedTypes is true (tool_result context) only images are
// allowed, since tool_result blocks reject document types.
func claudeFileBlocks(ctx context.Context, segment PromptSegment, restrictedTypes bool) ([]claudeBlock, error) {
	blocks := make([]claudeBlock, 0, len(segment.Files))
	for _, file := range segment.Files {
		mimeType := file.MIMEType()
		if mimeType == "" {
			mimeType = "application/octet-stream"
		}
		if !isClaudeImageMIME(mimeType) {
			if restrictedTypes {
				// Claude tool_result blocks do not accept all document types.
				continue
			}
			switch {
			case mimeType == "application/pdf":
				encoded, err := dataSourceToBase64(ctx, file)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, claudeBlock{
					Type:  "document",
					Title: file.Name(),
					Source: &claudeSource{
						Type:      "base64",
						MediaType: "application/pdf",
						Data:      encoded,
					},
				})
			case strings.HasPrefix(mimeType, "text/"):
				text, err := readAllStringFromDataSource(ctx, file)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, claudeBlock{
					Type:  "document",
					Title: file.Name(),
					Source: &claudeSource{
						Type:      "text",
						MediaType: "text/plain",
						Data:      text,
					},
				})
			}
			continue
		}
		encoded, err := dataSourceToBase64(ctx, file)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, claudeBlock{
			Type: "image",
			Source: &claudeSource{
				Type:      "base64",
				MediaType: mimeType,
				Data:      encoded,
			},
		})
	}
	return blocks, nil
}

// isClaudeImageMIME reports whether the MIME type is one the Messages API
// accepts as an inline image block.
func isClaudeImageMIME(mimeType string) bool {
	switch mimeType {
	case "image/png", "image/jpeg", "image/gif", "image/webp":
		return true
	default:
		return false
	}
}

// CreatePrompt builds the Claude Messages API request body (a claudePrompt)
// from the given prompt segments.
func (d *AnthropicDriver) CreatePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (any, error) {
	return formatClaudePrompt(ctx, segments, options)
}

// Execute runs a single non-streaming completion against POST /v1/messages and
// returns the assembled response with updated conversation state.
func (d *AnthropicDriver) Execute(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (*ExecutionResponse, error) {
	return executeWithPrompt(ctx, d, segments, options, d.requestTextCompletion)
}

// Stream runs a streaming completion against POST /v1/messages (stream=true),
// parsing the SSE event stream into CompletionChunks.
func (d *AnthropicDriver) Stream(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (CompletionStream, error) {
	return streamWithPrompt(ctx, d, segments, options, d.requestTextCompletionStream)
}

// requestTextCompletion posts a non-streaming Messages API request and returns
// the parsed completion plus the conversation extended with the response.
func (d *AnthropicDriver) requestTextCompletion(ctx context.Context, prompt any, options ExecutionOptions) (*Completion, error) {
	endpoint, err := joinEndpoint(d.options.BaseURL, "v1", "messages")
	if err != nil {
		return nil, err
	}
	conversation := claudeConversationInput(options.Conversation, prompt.(claudePrompt))
	payload, headers, err := claudePayload(conversation, options, false, d.headers())
	if err != nil {
		return nil, err
	}
	var response claudeResponse
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, payload, &response); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	completion := extractClaudeCompletion(response, options)
	completion.Conversation = finalizeClaudeConversation(appendClaudeResponseToConversation(conversation, response), options)
	return completion, nil
}

// requestTextCompletionStream posts a streaming Messages API request and spawns
// a goroutine that scans the SSE body, converts each event to a CompletionChunk
// (skipping empty ones), and reconstructs the final conversation on close.
func (d *AnthropicDriver) requestTextCompletionStream(ctx context.Context, prompt any, options ExecutionOptions) (CompletionStream, error) {
	endpoint, err := joinEndpoint(d.options.BaseURL, "v1", "messages")
	if err != nil {
		return nil, err
	}
	conversation := claudeConversationInput(options.Conversation, prompt.(claudePrompt))
	payload, headers, err := claudePayload(conversation, options, true, d.headers())
	if err != nil {
		return nil, err
	}
	body, err := postSSE(ctx, d.httpClient(options.HTTPTimeout), endpoint, headers, payload)
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	ch := make(chan streamItem)
	completion := &ExecutionResponse{Prompt: prompt}
	go func() {
		defer close(ch)
		err := scanSSE(body, func(event sseEvent) error {
			if event.Data == "" {
				return nil
			}
			var envelope map[string]json.RawMessage
			if err := json.Unmarshal([]byte(event.Data), &envelope); err != nil {
				return err
			}
			chunk := claudeSSEToChunk(envelope, options)
			if len(chunk.Result) > 0 || len(chunk.ToolUse) > 0 || chunk.FinishReason != "" || chunk.TokenUsage != nil {
				ch <- streamItem{Chunk: chunk}
			}
			return nil
		})
		if err != nil && !errors.Is(err, io.EOF) {
			ch <- streamItem{Err: d.formatError(err, options.Model, "stream")}
		}
	}()
	return newChannelStreamWithFinalizer(ch, completion, body.Close, func(resp *ExecutionResponse) {
		resp.Conversation = finalizeClaudeConversation(buildClaudeStreamingConversation(conversation, &resp.Completion), options)
	}), nil
}

// claudePayload assembles the Messages API JSON body and request headers. It
// repairs and normalizes the message list, validates tool schemas, computes
// max_tokens, wires up extended/adaptive thinking, applies sampling parameters
// only when thinking is inactive, attaches prompt-cache control to tools and the
// final system block, and sets the output-128k beta header when required.
func claudePayload(prompt claudePrompt, options ExecutionOptions, stream bool, headers map[string]string) (map[string]any, map[string]string, error) {
	prompt.Messages = fixClaudeOrphanedToolUse(sanitizeClaudeMessages(mergeConsecutiveClaudeUserMessages(prompt.Messages)))
	if len(options.Tools) == 0 && claudeMessagesContainToolBlocks(prompt.Messages) {
		// Conversations restored from checkpoints may contain previous tool calls
		// even when the next request has no tools. Convert those blocks to text so
		// the API receives a valid non-tool conversation.
		prompt.Messages = convertClaudeToolBlocksToText(prompt.Messages)
	}
	for _, tool := range options.Tools {
		if tool.InputSchema != nil && tool.InputSchema["type"] != "object" {
			return nil, nil, errors.New(`claude tool input_schema.type must be "object"`)
		}
	}
	maxTokens := claudeMaxTokens(options)
	payload := map[string]any{
		"model":      options.Model,
		"messages":   prompt.Messages,
		"max_tokens": maxTokens,
	}
	if stream {
		payload["stream"] = true
	}
	thinkingActive := addClaudeThinkingPayload(payload, options)
	if !thinkingActive {
		if v := optionFloat(options.ModelOptions, "temperature"); v != nil {
			payload["temperature"] = *v
		}
		if v := optionFloat(options.ModelOptions, "top_p"); v != nil && optionFloat(options.ModelOptions, "temperature") == nil {
			payload["top_p"] = *v
		}
		if v := optionInt(options.ModelOptions, "top_k"); v != nil {
			payload["top_k"] = *v
		}
	}
	if stops := optionStringSlice(options.ModelOptions, "stop_sequence"); len(stops) > 0 {
		payload["stop_sequences"] = stops
	}
	if len(options.Tools) > 0 {
		payload["tools"] = claudeTools(options.Tools, claudeCacheEnabled(options))
	}
	system := stripClaudeCacheControlFromBlocks(prompt.System)
	if len(system) > 0 {
		if claudeCacheEnabled(options) {
			system[len(system)-1].CacheControl = claudeCacheControl(options)
		}
		payload["system"] = system
	}
	if claudeNeedsOutput128KBeta(options, maxTokens) {
		headers["anthropic-beta"] = "output-128k-2025-02-19"
	}
	return payload, headers, nil
}

// claudeTools converts tool definitions to the Messages API tools shape,
// attaching ephemeral cache_control to the last tool when caching is enabled so
// the tool definitions form a cacheable prefix.
func claudeTools(tools []ToolDefinition, cacheEnabled bool) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for i, tool := range tools {
		item := map[string]any{
			"name":         tool.Name,
			"description":  tool.Description,
			"input_schema": tool.InputSchema,
		}
		if cacheEnabled && i == len(tools)-1 {
			item["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		out = append(out, item)
	}
	return out
}

// claudeConversationInput merges any stored conversation state (a claudePrompt
// from a prior turn) with the new prompt, producing the full request prompt.
//
// Claude requires strict user/assistant alternation; sanitize and merge
// after appending stored conversation state.
func claudeConversationInput(conversation any, prompt claudePrompt) claudePrompt {
	out := claudePrompt{}
	switch value := conversation.(type) {
	case claudePrompt:
		out = value
	case *claudePrompt:
		if value != nil {
			out = *value
		}
	}
	out.System = append(out.System, prompt.System...)
	out.Messages = append(out.Messages, prompt.Messages...)
	out.Messages = mergeConsecutiveClaudeUserMessages(sanitizeClaudeMessages(out.Messages))
	return out
}

// appendClaudeResponseToConversation appends the assistant turn from a
// non-streaming response to the conversation, then re-normalizes it.
func appendClaudeResponseToConversation(conversation claudePrompt, response claudeResponse) claudePrompt {
	blocks := claudeBlocksFromResponse(response.Content)
	if len(blocks) == 0 {
		blocks = []claudeBlock{{Type: "text", Text: ""}}
	}
	role := response.Role
	if role == "" {
		role = "assistant"
	}
	conversation.Messages = append(conversation.Messages, claudeMessage{Role: role, Content: blocks})
	conversation.Messages = mergeConsecutiveClaudeUserMessages(sanitizeClaudeMessages(conversation.Messages))
	return conversation
}

// buildClaudeStreamingConversation appends the assistant turn for a streamed
// completion, since no single final response object is available.
//
// Streaming does not return a single final response object, so reconstruct the
// assistant message from accumulated text and tool-use chunks before applying
// the normal conversation cleanup policy.
func buildClaudeStreamingConversation(conversation claudePrompt, completion *Completion) claudePrompt {
	content := []claudeBlock{}
	if text := completionResultsToText(completion.Result); text != "" {
		content = append(content, claudeBlock{Type: "text", Text: text})
	}
	for _, tool := range completion.ToolUse {
		input := tool.ToolInput
		if input == nil {
			input = map[string]any{}
		}
		content = append(content, claudeBlock{Type: "tool_use", ID: tool.ID, Name: tool.ToolName, Input: input})
	}
	if len(content) > 0 {
		conversation.Messages = append(conversation.Messages, claudeMessage{Role: "assistant", Content: content})
	}
	conversation.Messages = mergeConsecutiveClaudeUserMessages(sanitizeClaudeMessages(conversation.Messages))
	return conversation
}

// claudeBlocksFromResponse converts response content blocks back into request
// blocks (text and tool_use only) for storage in conversation state.
func claudeBlocksFromResponse(blocks []claudeResponseBlock) []claudeBlock {
	out := make([]claudeBlock, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case "text":
			out = append(out, claudeBlock{Type: "text", Text: block.Text})
		case "tool_use":
			out = append(out, claudeBlock{Type: "tool_use", ID: block.ID, Name: block.Name, Input: safeJSONParse(string(block.Input))})
		}
	}
	return out
}

// sanitizeClaudeMessages drops blank text blocks and any message left with no
// content, since the Messages API rejects empty content arrays.
func sanitizeClaudeMessages(messages []claudeMessage) []claudeMessage {
	out := make([]claudeMessage, 0, len(messages))
	for _, msg := range messages {
		content := make([]claudeBlock, 0, len(msg.Content))
		for _, block := range msg.Content {
			if block.Type == "text" && strings.TrimSpace(block.Text) == "" {
				continue
			}
			content = append(content, block)
		}
		if len(content) > 0 {
			msg.Content = content
			out = append(out, msg)
		}
	}
	return out
}

// mergeConsecutiveClaudeUserMessages collapses adjacent user messages into one,
// helping enforce the strict user/assistant alternation the API requires.
func mergeConsecutiveClaudeUserMessages(messages []claudeMessage) []claudeMessage {
	if len(messages) < 2 {
		return messages
	}
	out := []claudeMessage{messages[0]}
	for _, msg := range messages[1:] {
		last := &out[len(out)-1]
		if last.Role == "user" && msg.Role == "user" {
			last.Content = append(last.Content, msg.Content...)
			continue
		}
		out = append(out, msg)
	}
	return out
}

// fixClaudeOrphanedToolUse repairs assistant tool_use blocks that lack a
// matching tool_result in the following user message.
//
// If an agent was stopped after Claude emitted a tool_use but before the tool
// ran, the next request would otherwise contain an orphaned call. Insert a
// synthetic interrupted tool_result to keep the Messages conversation valid.
func fixClaudeOrphanedToolUse(messages []claudeMessage) []claudeMessage {
	if len(messages) < 2 {
		return messages
	}
	out := make([]claudeMessage, len(messages))
	copy(out, messages)
	for i := 0; i < len(out)-1; i++ {
		current := out[i]
		if current.Role != "assistant" {
			continue
		}
		var toolBlocks []claudeBlock
		for _, block := range current.Content {
			if block.Type == "tool_use" {
				toolBlocks = append(toolBlocks, block)
			}
		}
		if len(toolBlocks) == 0 || out[i+1].Role != "user" {
			continue
		}
		existing := map[string]bool{}
		for _, block := range out[i+1].Content {
			if block.Type == "tool_result" {
				existing[block.ToolUseID] = true
			}
		}
		var synthetic []claudeBlock
		for _, block := range toolBlocks {
			if existing[block.ID] {
				continue
			}
			name := block.Name
			if name == "" {
				name = "unknown"
			}
			synthetic = append(synthetic, claudeBlock{
				Type:      "tool_result",
				ToolUseID: block.ID,
				Content:   []claudeBlock{{Type: "text", Text: fmt.Sprintf(`[Tool interrupted: The user stopped the operation before "%s" could execute.]`, name)}},
			})
		}
		if len(synthetic) > 0 {
			out[i+1].Content = append(synthetic, out[i+1].Content...)
		}
	}
	return out
}

// claudeMessagesContainToolBlocks reports whether any message carries a
// tool_use or tool_result block.
func claudeMessagesContainToolBlocks(messages []claudeMessage) bool {
	for _, msg := range messages {
		for _, block := range msg.Content {
			if block.Type == "tool_use" || block.Type == "tool_result" {
				return true
			}
		}
	}
	return false
}

// convertClaudeToolBlocksToText rewrites tool_use/tool_result blocks as plain
// text summaries so a tool-bearing history can be replayed in a request that
// declares no tools (the API otherwise rejects such blocks).
func convertClaudeToolBlocksToText(messages []claudeMessage) []claudeMessage {
	out := make([]claudeMessage, 0, len(messages))
	for _, msg := range messages {
		next := msg
		next.Content = make([]claudeBlock, 0, len(msg.Content))
		for _, block := range msg.Content {
			switch block.Type {
			case "tool_use":
				next.Content = append(next.Content, claudeBlock{
					Type: "text",
					Text: fmt.Sprintf("[Tool call: %s(%s)]", block.Name, truncateForConversation(toolInputString(block.Input), 500)),
				})
			case "tool_result":
				next.Content = append(next.Content, claudeBlock{
					Type: "text",
					Text: "[Tool result: " + truncateForConversation(claudeToolResultText(block), 500) + "]",
				})
			default:
				next.Content = append(next.Content, block)
			}
		}
		out = append(out, next)
	}
	return out
}

// toolInputString renders a tool input value as a string, marshaling non-string
// values to JSON.
func toolInputString(input any) string {
	if input == nil {
		return ""
	}
	if s, ok := input.(string); ok {
		return s
	}
	data, _ := json.Marshal(input)
	return string(data)
}

// claudeToolResultText extracts the textual payload of a tool_result block,
// joining nested text blocks and falling back to "No content".
func claudeToolResultText(block claudeBlock) string {
	if block.Text != "" {
		return block.Text
	}
	var parts []string
	for _, content := range block.Content {
		if content.Type == "text" && content.Text != "" {
			parts = append(parts, content.Text)
		}
	}
	if len(parts) == 0 {
		return "No content"
	}
	return strings.Join(parts, "\n")
}

// stripClaudeCacheControlFromBlocks returns a copy of the blocks with any
// cache_control cleared, so caching markers can be re-applied freshly per request.
func stripClaudeCacheControlFromBlocks(blocks []claudeBlock) []claudeBlock {
	out := make([]claudeBlock, 0, len(blocks))
	for _, block := range blocks {
		block.CacheControl = nil
		out = append(out, block)
	}
	return out
}

// claudeMaxTokens resolves the max_tokens value: an explicit option wins,
// otherwise Claude 3.7 Sonnet defaults to 64k (unless a large thinking budget
// pushes it onto the 128k beta path), with a 4096 fallback for other models.
func claudeMaxTokens(options ExecutionOptions) int {
	if v := optionInt(options.ModelOptions, "max_tokens"); v != nil {
		return *v
	}
	if strings.Contains(options.Model, "claude-3-7-sonnet") {
		// Claude 3.7 defaults to 64k output in the TS client unless an explicit
		// thinking budget already requires the 128k beta path.
		if budget := optionInt(options.ModelOptions, "thinking_budget_tokens"); budget == nil || *budget < 48000 {
			return 64000
		}
	}
	return 4096
}

// addClaudeThinkingPayload wires the "thinking" (and adaptive "output_config")
// fields into the payload and reports whether thinking is active (which causes
// callers to drop sampling parameters).
func addClaudeThinkingPayload(payload map[string]any, options ExecutionOptions) bool {
	// Claude 3.7+ accepts explicit thinking budgets; newer variant-first models
	// may also support adaptive effort-based thinking. Explicit budgets take
	// priority and enable extended thinking regardless of adaptive support.
	if !isClaudeVersionGTE(options.Model, 3, 7) {
		return false
	}
	if budget := optionInt(options.ModelOptions, "thinking_budget_tokens"); budget != nil {
		payload["thinking"] = map[string]any{"type": "enabled", "budget_tokens": *budget}
		return true
	}
	if effort := optionString(options.ModelOptions, "effort"); effort != "" && claudeSupportsAdaptiveThinking(options.Model) {
		// include_thoughts controls whether summaries are displayed; it does not
		// enable thinking by itself. Adaptive thinking remains off unless effort is set.
		display := "omitted"
		if optionBool(options.ModelOptions, "include_thoughts") {
			display = "summarized"
		}
		payload["thinking"] = map[string]any{"type": "adaptive", "display": display}
		payload["output_config"] = map[string]any{"effort": effort}
		return true
	}
	// Older thinking-capable models are disabled by default unless the caller
	// provides an explicit budget, matching the TS driver's behavior.
	payload["thinking"] = map[string]any{"type": "disabled"}
	return false
}

// claudeCacheEnabled reports whether prompt caching is requested via the
// cache_enabled model option.
func claudeCacheEnabled(options ExecutionOptions) bool {
	return optionBool(options.ModelOptions, "cache_enabled")
}

// claudeCacheControl builds the ephemeral cache_control object, honoring an
// optional cache_ttl model option.
func claudeCacheControl(options ExecutionOptions) map[string]any {
	out := map[string]any{"type": "ephemeral"}
	if ttl := optionString(options.ModelOptions, "cache_ttl"); ttl != "" {
		out["ttl"] = ttl
	}
	return out
}

// claudeNeedsOutput128KBeta reports whether the request must opt into the
// output-128k beta header: only Claude 3.7 Sonnet, and only when max_tokens or
// the thinking budget exceeds 64k.
func claudeNeedsOutput128KBeta(options ExecutionOptions, maxTokens int) bool {
	if !strings.Contains(options.Model, "claude-3-7-sonnet") {
		return false
	}
	if maxTokens > 64000 {
		return true
	}
	if budget := optionInt(options.ModelOptions, "thinking_budget_tokens"); budget != nil && *budget > 64000 {
		return true
	}
	return false
}

// headers returns the base auth headers (x-api-key plus the pinned
// anthropic-version) for every request.
func (d *AnthropicDriver) headers() map[string]string {
	return map[string]string{
		"x-api-key":         d.options.APIKey,
		"anthropic-version": anthropicVersion,
	}
}

// claudeResponse is the non-streaming Messages API response body, including the
// usage block whose cache read/write counts feed token accounting.
type claudeResponse struct {
	ID         string                `json:"id"`
	Role       string                `json:"role"`
	Content    []claudeResponseBlock `json:"content"`
	StopReason string                `json:"stop_reason"`
	Usage      struct {
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	} `json:"usage"`
}

// claudeResponseBlock is a single content block in a response, covering text,
// thinking/redacted_thinking, and tool_use variants.
type claudeResponseBlock struct {
	Type     string          `json:"type"`
	Text     string          `json:"text"`
	Thinking string          `json:"thinking"`
	Data     string          `json:"data"`
	ID       string          `json:"id"`
	Name     string          `json:"name"`
	Input    json.RawMessage `json:"input"`
}

// extractClaudeCompletion flattens a response into a Completion: text and
// (when include_thoughts is set) thinking blocks are joined into the result,
// tool_use blocks become ToolUse entries, and usage/finish reason are normalized.
func extractClaudeCompletion(response claudeResponse, options ExecutionOptions) *Completion {
	var parts []string
	var tools []ToolUse
	includeThoughts := optionBool(options.ModelOptions, "include_thoughts")
	for _, block := range response.Content {
		switch block.Type {
		case "text":
			parts = append(parts, block.Text)
		case "thinking":
			if includeThoughts && block.Thinking != "" {
				parts = append(parts, block.Thinking)
			}
		case "redacted_thinking":
			if includeThoughts && block.Data != "" {
				parts = append(parts, "[Redacted thinking: "+block.Data+"]")
			}
		case "tool_use":
			tools = append(tools, ToolUse{
				ID:        block.ID,
				ToolName:  block.Name,
				ToolInput: safeJSONParse(string(block.Input)),
			})
		}
	}
	text := strings.Join(parts, "\n")
	result := []CompletionResult{{Type: ResultTypeText, Value: text}}
	completion := &Completion{
		Result:       result,
		ToolUse:      tools,
		TokenUsage:   claudeUsage(response.Usage.InputTokens, response.Usage.OutputTokens, response.Usage.CacheReadInputTokens, response.Usage.CacheCreationInputTokens),
		FinishReason: claudeFinishReason(response.StopReason, len(tools) > 0),
	}
	if options.IncludeOriginalResponse {
		completion.OriginalResponse = response
	}
	return completion
}

// claudeUsage maps Claude token counts to ExecutionTokenUsage. Prompt and Total
// include the cache read/write tokens, while PromptNew tracks only the
// uncached input and PromptCached/PromptCacheWrite expose the cache split.
// Returns nil when all counts are zero.
func claudeUsage(input, output, cacheRead, cacheWrite int) *ExecutionTokenUsage {
	if input == 0 && output == 0 && cacheRead == 0 && cacheWrite == 0 {
		return nil
	}
	return &ExecutionTokenUsage{
		Prompt:           input + cacheRead + cacheWrite,
		PromptNew:        input,
		Result:           output,
		Total:            input + output + cacheRead + cacheWrite,
		PromptCached:     cacheRead,
		PromptCacheWrite: cacheWrite,
	}
}

// claudeFinishReason normalizes Claude stop_reason values to the neutral set
// (end_turn->stop, max_tokens->length, tool_use->tool_use), passing through
// anything unrecognized.
func claudeFinishReason(reason string, toolUse bool) string {
	if toolUse || reason == "tool_use" {
		return "tool_use"
	}
	switch reason {
	case "end_turn":
		return "stop"
	case "max_tokens":
		return "length"
	default:
		return reason
	}
}

// claudeSSEToChunk translates one decoded Messages API streaming event into a
// CompletionChunk: message_start/message_delta carry usage and finish reason,
// content_block_start opens tool_use (or redacted thinking) blocks, and
// content_block_delta streams text, partial tool-input JSON, and thinking deltas.
// Events that produce no output return an empty chunk.
func claudeSSEToChunk(envelope map[string]json.RawMessage, options ExecutionOptions) CompletionChunk {
	var typ string
	_ = json.Unmarshal(envelope["type"], &typ)
	includeThoughts := optionBool(options.ModelOptions, "include_thoughts")
	switch typ {
	case "message_start":
		var msg struct {
			Message claudeResponse `json:"message"`
		}
		_ = json.Unmarshal(mustRaw(envelope, "message"), &msg.Message)
		return CompletionChunk{TokenUsage: claudeUsage(msg.Message.Usage.InputTokens, msg.Message.Usage.OutputTokens, msg.Message.Usage.CacheReadInputTokens, msg.Message.Usage.CacheCreationInputTokens)}
	case "message_delta":
		var msg struct {
			Delta struct {
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage struct {
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		}
		_ = json.Unmarshal(mustMarshal(envelope), &msg)
		return CompletionChunk{FinishReason: claudeFinishReason(msg.Delta.StopReason, false), TokenUsage: claudeUsage(0, msg.Usage.OutputTokens, 0, 0)}
	case "content_block_start":
		var msg struct {
			ContentBlock struct {
				Type  string          `json:"type"`
				ID    string          `json:"id"`
				Name  string          `json:"name"`
				Input json.RawMessage `json:"input"`
				Data  string          `json:"data"`
			} `json:"content_block"`
		}
		_ = json.Unmarshal(mustMarshal(envelope), &msg)
		if msg.ContentBlock.Type == "tool_use" {
			return CompletionChunk{ToolUse: []ToolUse{{ID: msg.ContentBlock.ID, ToolName: msg.ContentBlock.Name, ToolInput: safeJSONParse(string(msg.ContentBlock.Input))}}}
		}
		if msg.ContentBlock.Type == "redacted_thinking" && includeThoughts && msg.ContentBlock.Data != "" {
			return CompletionChunk{Result: []CompletionResult{{Type: ResultTypeText, Value: "[Redacted thinking: " + msg.ContentBlock.Data + "]"}}}
		}
	case "content_block_delta":
		var msg struct {
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
				Thinking    string `json:"thinking"`
			} `json:"delta"`
		}
		_ = json.Unmarshal(mustMarshal(envelope), &msg)
		if msg.Delta.Type == "text_delta" && msg.Delta.Text != "" {
			return CompletionChunk{Result: []CompletionResult{{Type: ResultTypeText, Value: msg.Delta.Text}}}
		}
		if msg.Delta.Type == "input_json_delta" && msg.Delta.PartialJSON != "" {
			return CompletionChunk{ToolUse: []ToolUse{{ToolInput: msg.Delta.PartialJSON}}}
		}
		if msg.Delta.Type == "thinking_delta" && includeThoughts && msg.Delta.Thinking != "" {
			return CompletionChunk{Result: []CompletionResult{{Type: ResultTypeText, Value: msg.Delta.Thinking}}}
		}
		if msg.Delta.Type == "signature_delta" && includeThoughts {
			return CompletionChunk{Result: []CompletionResult{{Type: ResultTypeText, Value: "\n\n"}}}
		}
	}
	return CompletionChunk{}
}

// mustRaw returns the raw JSON for key, or the literal null if absent, so
// downstream Unmarshal calls never see a nil slice.
func mustRaw(m map[string]json.RawMessage, key string) json.RawMessage {
	if raw := m[key]; raw != nil {
		return raw
	}
	return []byte("null")
}

// mustMarshal JSON-encodes v, discarding the error (used to re-decode an SSE
// envelope into a typed struct).
func mustMarshal(v any) []byte {
	data, _ := json.Marshal(v)
	return data
}

// ListModels fetches the available Claude models from GET /v1/models, marking
// them all as text models with streaming and tool support.
func (d *AnthropicDriver) ListModels(ctx context.Context, _ *ModelSearchPayload) ([]AIModel, error) {
	endpoint, err := joinEndpoint(d.options.BaseURL, "v1", "models")
	if err != nil {
		return nil, err
	}
	var response struct {
		Data []struct {
			ID          string `json:"id"`
			DisplayName string `json:"display_name"`
		} `json:"data"`
	}
	if err := doJSON(ctx, d.client, http.MethodGet, endpoint, d.headers(), nil, &response); err != nil {
		return nil, d.formatError(err, "", "listModels")
	}
	models := make([]AIModel, 0, len(response.Data))
	for _, item := range response.Data {
		name := item.DisplayName
		if name == "" {
			name = item.ID
		}
		models = append(models, AIModel{ID: item.ID, Name: name, Provider: ProviderAnthropic, Type: "text", CanStream: true, ToolSupport: true})
	}
	return models, nil
}

// ValidateConnection verifies the Anthropic API key by listing models.
func (d *AnthropicDriver) ValidateConnection(ctx context.Context) error {
	_, err := d.ListModels(ctx, nil)
	return err
}

func (d *AnthropicDriver) httpClient(timeout HTTPTimeoutOptions) *http.Client {
	if !hasHTTPTimeout(timeout) {
		return d.client
	}
	return newHTTPClient(d.client, timeout)
}

// GenerateEmbeddings returns a non-retryable error because Anthropic Claude does not expose embeddings.
func (d *AnthropicDriver) GenerateEmbeddings(_ context.Context, options EmbeddingsOptions) (*EmbeddingsResult, error) {
	return nil, newLlumiverseError("Anthropic does not support embeddings", boolPtr(false), LlumiverseErrorContext{
		Provider:  ProviderAnthropic,
		Model:     options.Model,
		Operation: "embeddings",
	}, nil, 0, "UnsupportedOperation")
}

// formatError wraps an HTTP status error in a LlumiverseError with provider
// context and retryability derived from the status code and body; non-HTTP
// errors pass through unchanged.
func (d *AnthropicDriver) formatError(err error, model string, operation string) error {
	code, _ := errorStatusAndName(err)
	if code != 0 {
		return newLlumiverseError(err.Error(), retryableFromStatusAndMessage(code, err.Error()), LlumiverseErrorContext{
			Provider:  d.Provider(),
			Model:     model,
			Operation: operation,
		}, err, code, "AnthropicError")
	}
	return err
}
