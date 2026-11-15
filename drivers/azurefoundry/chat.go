package azurefoundry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/vertesia/llumiverse-go/drivers/openai"
)

// chatPrompt is the Chat Completions prompt and retained conversation of the
// Foundry models that are neither OpenAI nor Claude. Turn counts the completed
// turns for the retention rules.
type chatPrompt struct {
	Messages []chatMessage `json:"messages"`
	Turn     int           `json:"turn,omitempty"`
}

// chatMessage is one Chat Completions message. Content is Text, or Parts when
// the message carries images; an assistant message with tool calls and no text
// sends a null content.
type chatMessage struct {
	Role             string
	Text             string
	Parts            []chatPart
	ToolCalls        []chatToolCall
	ToolCallID       string
	ReasoningContent *string
	Reasoning        *string
}

// chatPart is a text or image_url content part.
type chatPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *chatImageURL `json:"image_url,omitempty"`
}

// chatImageURL is the image of an image_url part.
type chatImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// chatToolCall is an assistant function call.
type chatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatToolFunction `json:"function"`
}

// chatToolFunction names the function of a tool call and its JSON arguments.
type chatToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatMessageJSON struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content"`
	ToolCalls        []chatToolCall  `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
	ReasoningContent *string         `json:"reasoning_content,omitempty"`
	Reasoning        *string         `json:"reasoning,omitempty"`
}

// MarshalJSON encodes the message in the Chat Completions shape.
func (m chatMessage) MarshalJSON() ([]byte, error) {
	var content any = m.Text
	switch {
	case m.Parts != nil:
		content = m.Parts
	case m.Role == "assistant" && m.Text == "" && len(m.ToolCalls) > 0:
		content = nil
	}
	encoded, err := json.Marshal(content)
	if err != nil {
		return nil, err
	}
	return json.Marshal(chatMessageJSON{
		Role:             m.Role,
		Content:          encoded,
		ToolCalls:        m.ToolCalls,
		ToolCallID:       m.ToolCallID,
		ReasoningContent: m.ReasoningContent,
		Reasoning:        m.Reasoning,
	})
}

// UnmarshalJSON decodes a stored message whose content is a string, null, or parts.
func (m *chatMessage) UnmarshalJSON(data []byte) error {
	var raw chatMessageJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*m = chatMessage{
		Role:             raw.Role,
		ToolCalls:        raw.ToolCalls,
		ToolCallID:       raw.ToolCallID,
		ReasoningContent: raw.ReasoningContent,
		Reasoning:        raw.Reasoning,
	}
	trimmed := bytes.TrimSpace(raw.Content)
	switch {
	case len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")):
	case trimmed[0] == '[':
		return json.Unmarshal(trimmed, &m.Parts)
	default:
		return json.Unmarshal(trimmed, &m.Text)
	}
	return nil
}

// text returns the message's text: its string content or its text parts.
func (m chatMessage) text() string {
	if m.Parts == nil {
		return m.Text
	}
	texts := make([]string, 0, len(m.Parts))
	for _, part := range m.Parts {
		if part.Type == "text" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// resultSchemaInstruction is the system instruction that accompanies the
// response_format of a result schema, as in the TS client's OpenAI-like prompt.
func resultSchemaInstruction(schema map[string]any) string {
	encoded, _ := json.Marshal(schema)
	return "IMPORTANT: only answer using JSON, and respecting the schema included below, between the <response_schema> tags. <response_schema>" + string(encoded) + "</response_schema>"
}

// createChatPrompt formats prompt segments as Chat Completions messages: system
// and safety text first (safety emphasized), then user, assistant and tool
// messages in order. Image files become image_url parts of user messages;
// other attachments are not supported by the Foundry chat models and are left out.
func createChatPrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (chatPrompt, error) {
	var system, others []chatMessage
	for _, segment := range segments {
		switch segment.Role {
		case PromptRoleNegative, PromptRoleMask:
			continue
		case PromptRoleSystem:
			if segment.Content != "" {
				system = append(system, chatMessage{Role: "system", Text: segment.Content})
			}
		case PromptRoleSafety:
			if segment.Content != "" {
				system = append(system, chatMessage{Role: "system", Text: "DO NOT IGNORE - IMPORTANT: " + segment.Content})
			}
		case PromptRoleTool:
			if segment.ToolUseID == "" {
				return chatPrompt{}, errors.New("tool prompt segment requires ToolUseID")
			}
			others = append(others, chatMessage{Role: "tool", Text: segment.Content, ToolCallID: segment.ToolUseID})
		case PromptRoleAssistant:
			others = append(others, chatMessage{Role: "assistant", Text: segment.Content})
		default:
			images, err := chatImageParts(ctx, segment, options)
			if err != nil {
				return chatPrompt{}, err
			}
			if len(images) == 0 {
				others = append(others, chatMessage{Role: "user", Text: segment.Content})
				continue
			}
			parts := images
			if segment.Content != "" {
				parts = append(parts, chatPart{Type: "text", Text: segment.Content})
			}
			others = append(others, chatMessage{Role: "user", Parts: parts})
		}
	}
	if options.ResultSchema != nil {
		system = append(system, chatMessage{Role: "system", Text: resultSchemaInstruction(options.ResultSchema)})
	}
	return chatPrompt{Messages: append(system, others...)}, nil
}

// chatImageParts returns the image_url parts of a segment's image files.
func chatImageParts(ctx context.Context, segment PromptSegment, options ExecutionOptions) ([]chatPart, error) {
	var parts []chatPart
	for _, file := range segment.Files {
		if !strings.HasPrefix(strings.ToLower(file.MIMEType()), "image/") {
			continue
		}
		dataURL, err := dataSourceToDataURL(ctx, file)
		if err != nil {
			return nil, err
		}
		image := &chatImageURL{URL: dataURL}
		switch detail := optionString(options.ModelOptions, "image_detail"); detail {
		case "low", "high", "auto":
			image.Detail = detail
		}
		parts = append(parts, chatPart{Type: "image_url", ImageURL: image})
	}
	return parts, nil
}

// storedChatConversation decodes a retained conversation: the driver's own
// value, or its JSON form after a round trip through storage.
func storedChatConversation(conversation any) chatPrompt {
	switch value := conversation.(type) {
	case nil:
		return chatPrompt{}
	case chatPrompt:
		return value
	case *chatPrompt:
		if value != nil {
			return *value
		}
		return chatPrompt{}
	}
	data, err := json.Marshal(conversation)
	if err != nil {
		return chatPrompt{}
	}
	var decoded chatPrompt
	if json.Unmarshal(data, &decoded) == nil {
		return decoded
	}
	// The TS client's legacy conversations are a bare message array.
	if json.Unmarshal(data, &decoded.Messages) == nil {
		return decoded
	}
	return chatPrompt{}
}

// chatConversationInput appends the prompt to the retained conversation and
// repairs it for the request: orphaned tool calls get an interruption result,
// orphaned tool results are dropped, reasoning is replayed as the model
// requires, and tool messages become text when the request has no tools.
func chatConversationInput(conversation any, prompt chatPrompt, source string, hasTools bool) chatPrompt {
	stored := storedChatConversation(conversation)
	messages := make([]chatMessage, 0, len(stored.Messages)+len(prompt.Messages))
	messages = append(messages, stored.Messages...)
	messages = append(messages, prompt.Messages...)
	messages = fixOrphanedChatToolResults(fixOrphanedChatToolUse(messages))
	messages = prepareChatReasoning(messages, source)
	if !hasTools {
		messages = convertChatToolMessagesToText(messages)
	}
	return chatPrompt{Messages: messages, Turn: stored.Turn}
}

// fixOrphanedChatToolUse inserts an interruption result after any tool call
// that has none, so a run stopped mid-tool can continue: Chat Completions
// requires every tool call to be answered.
func fixOrphanedChatToolUse(messages []chatMessage) []chatMessage {
	if len(messages) < 2 {
		return messages
	}
	answered := map[string]bool{}
	for _, message := range messages {
		if message.Role == "tool" && message.ToolCallID != "" {
			answered[message.ToolCallID] = true
		}
	}
	out := make([]chatMessage, 0, len(messages))
	type pendingCall struct{ id, name string }
	var pending []pendingCall
	flush := func() {
		for _, call := range pending {
			out = append(out, chatMessage{
				Role:       "tool",
				ToolCallID: call.id,
				Text:       fmt.Sprintf(`[Tool interrupted: The user stopped the operation before "%s" could execute.]`, call.name),
			})
		}
		pending = nil
	}
	for _, message := range messages {
		if len(message.ToolCalls) > 0 {
			for _, call := range message.ToolCalls {
				if call.ID != "" && !answered[call.ID] {
					name := call.Function.Name
					if name == "" {
						name = "unknown"
					}
					pending = append(pending, pendingCall{id: call.ID, name: name})
				}
			}
			out = append(out, message)
			continue
		}
		if message.Role == "tool" {
			out = append(out, message)
			continue
		}
		flush()
		out = append(out, message)
	}
	flush()
	return out
}

// fixOrphanedChatToolResults drops tool results whose call is no longer in the
// conversation, usually after compaction removed the tool-call turn.
func fixOrphanedChatToolResults(messages []chatMessage) []chatMessage {
	calls := map[string]bool{}
	for _, message := range messages {
		for _, call := range message.ToolCalls {
			if call.ID != "" {
				calls[call.ID] = true
			}
		}
	}
	out := messages[:0:0]
	for _, message := range messages {
		if message.Role == "tool" && (message.ToolCallID == "" || !calls[message.ToolCallID]) {
			continue
		}
		out = append(out, message)
	}
	return out
}

// convertChatToolMessagesToText rewrites tool calls and results as text for a
// request without tools, which would otherwise be rejected.
func convertChatToolMessagesToText(messages []chatMessage) []chatMessage {
	hasTools := false
	for _, message := range messages {
		if message.Role == "tool" || len(message.ToolCalls) > 0 {
			hasTools = true
			break
		}
	}
	if !hasTools {
		return messages
	}
	out := make([]chatMessage, 0, len(messages))
	for _, message := range messages {
		switch {
		case len(message.ToolCalls) > 0:
			var lines []string
			if text := message.text(); strings.TrimSpace(text) != "" {
				lines = append(lines, text)
			}
			for _, call := range message.ToolCalls {
				name := call.Function.Name
				if name == "" {
					name = "unknown"
				}
				lines = append(lines, fmt.Sprintf("[Tool call: %s(%s)]", name, truncateToolText(call.Function.Arguments)))
			}
			out = append(out, chatMessage{Role: message.Role, Text: strings.Join(lines, "\n")})
		case message.Role == "tool":
			if hasImagePart(message.Parts) {
				parts := append([]chatPart{{Type: "text", Text: "Tool result " + message.ToolCallID + ":"}}, message.Parts...)
				out = append(out, chatMessage{Role: "user", Parts: parts})
				continue
			}
			output := message.text()
			if output == "" {
				output = "No output"
			}
			out = append(out, chatMessage{Role: "user", Text: "[Tool result: " + truncateToolText(output) + "]"})
		default:
			out = append(out, message)
		}
	}
	return out
}

func hasImagePart(parts []chatPart) bool {
	for _, part := range parts {
		if part.Type == "image_url" {
			return true
		}
	}
	return false
}

func truncateToolText(value string) string {
	if len(value) > 500 {
		return value[:500] + "..."
	}
	return value
}

// chatReasoningPolicy is how a model wants its reasoning replayed in later turns.
type chatReasoningPolicy int

const (
	reasoningKeep chatReasoningPolicy = iota
	reasoningOmit
	reasoningActiveToolTurn
	reasoningFullToolHistory
)

// reasoningPolicy follows the model family and version: DeepSeek R1 rejects
// replayed reasoning, V3.2 needs it within the active tool turn, and V4 and
// later across the tool history.
func reasoningPolicy(source string) chatReasoningPolicy {
	model := strings.ToLower(source)
	switch {
	case strings.Contains(model, "deepseek-r1") || strings.Contains(model, "deepseek-reasoner") || strings.Contains(model, "deepseek.r1"):
		return reasoningOmit
	case deepSeekVersionGTE(model, 4, 0):
		return reasoningFullToolHistory
	case strings.Contains(model, "deepseek-v3.2-") || strings.HasSuffix(model, "deepseek-v3.2"):
		return reasoningActiveToolTurn
	}
	return reasoningKeep
}

func withoutReasoning(message chatMessage) chatMessage {
	message.ReasoningContent = nil
	message.Reasoning = nil
	return message
}

func hasReasoning(message chatMessage) bool {
	return message.ReasoningContent != nil || message.Reasoning != nil
}

// prepareChatReasoning removes the reasoning a model does not accept back.
func prepareChatReasoning(messages []chatMessage, source string) []chatMessage {
	switch reasoningPolicy(source) {
	case reasoningOmit:
		out := make([]chatMessage, len(messages))
		for i, message := range messages {
			out[i] = withoutReasoning(message)
		}
		return out
	case reasoningActiveToolTurn:
		start := -1
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i].Role == "user" {
				start = i
				break
			}
		}
		if start <= 0 {
			return messages
		}
		out := make([]chatMessage, len(messages))
		for i, message := range messages {
			if i < start {
				message = withoutReasoning(message)
			}
			out[i] = message
		}
		return out
	}
	return messages
}

// finalizeChatConversation counts the turn and applies the retention rules.
// Reasoning a model must see again in a tool loop keeps the conversation as is.
func finalizeChatConversation(conversation chatPrompt, source string, options ExecutionOptions) chatPrompt {
	conversation.Turn++
	policy := reasoningPolicy(source)
	if policy == reasoningOmit {
		messages := make([]chatMessage, len(conversation.Messages))
		for i, message := range conversation.Messages {
			messages[i] = withoutReasoning(message)
		}
		conversation.Messages = messages
	}
	var latestAssistant *chatMessage
	for i := len(conversation.Messages) - 1; i >= 0; i-- {
		if conversation.Messages[i].Role == "assistant" {
			latestAssistant = &conversation.Messages[i]
			break
		}
	}
	if policy == reasoningActiveToolTurn && latestAssistant != nil && len(latestAssistant.ToolCalls) > 0 && hasReasoning(*latestAssistant) {
		return conversation
	}
	if policy == reasoningFullToolHistory {
		for _, message := range conversation.Messages {
			if len(message.ToolCalls) > 0 && hasReasoning(message) {
				return conversation
			}
		}
	}
	stripMedia := shouldStripConversationMedia(conversation.Turn, options.StripImagesAfterTurns)
	stripHeartbeats := shouldStripConversationHeartbeats(conversation.Turn, options.StripHeartbeatsAfterTurns)
	messages := make([]chatMessage, len(conversation.Messages))
	for i, message := range conversation.Messages {
		if policy == reasoningActiveToolTurn {
			message = withoutReasoning(message)
		}
		messages[i] = retainChatMessage(message, options, stripMedia, stripHeartbeats)
	}
	conversation.Messages = messages
	return conversation
}

// retainChatMessage replaces expired inline images with a placeholder and
// applies the heartbeat and text-truncation rules to the message's text.
func retainChatMessage(message chatMessage, options ExecutionOptions, stripMedia bool, stripHeartbeats bool) chatMessage {
	message.Text = processConversationText(message.Text, options, stripHeartbeats)
	if message.Parts == nil {
		return message
	}
	parts := make([]chatPart, 0, len(message.Parts))
	for _, part := range message.Parts {
		switch {
		case part.Type == "image_url" && stripMedia && part.ImageURL != nil && strings.HasPrefix(part.ImageURL.URL, "data:") && strings.Contains(part.ImageURL.URL, ";base64,"):
			parts = append(parts, chatPart{Type: "text", Text: conversationImagePlaceholder})
		case part.Type == "text":
			part.Text = processConversationText(part.Text, options, stripHeartbeats)
			parts = append(parts, part)
		default:
			parts = append(parts, part)
		}
	}
	message.Parts = parts
	return message
}

// chatPayload builds the Chat Completions request. Only the options the caller
// set are sent, except n=1: the completion interface reads one choice. Tool
// schemas use the limited (non-strict) form the Foundry models accept, and a
// result schema becomes a json_schema response format.
func (d *AzureFoundryDriver) chatPayload(conversation chatPrompt, r route, options ExecutionOptions, stream bool) (map[string]any, error) {
	modelOptions := options.ModelOptions
	payload := map[string]any{
		"model":    r.deployment,
		"messages": conversation.Messages,
		"n":        1,
		"stream":   stream,
	}
	if v := optionInt(modelOptions, "max_tokens"); v != nil {
		payload["max_tokens"] = *v
	}
	for _, key := range []string{"temperature", "top_p", "presence_penalty", "frequency_penalty"} {
		if v := optionFloat(modelOptions, key); v != nil {
			payload[key] = *v
		}
	}
	if stops := optionStringSlice(modelOptions, "stop_sequence"); len(stops) > 0 {
		payload["stop"] = stops
	}
	if v := optionInt(modelOptions, "seed"); v != nil {
		payload["seed"] = *v
	}
	if effort := optionString(modelOptions, "effort", "reasoning_effort"); effort != "" {
		payload["reasoning_effort"] = effort
	}
	if tier := optionString(modelOptions, "service_tier"); tier != "" {
		payload["service_tier"] = tier
	}
	requiredTool := optionString(modelOptions, "required_tool_name")
	toolChoice := optionString(modelOptions, "tool_choice")
	if len(options.Tools) == 0 && (requiredTool != "" || toolChoice == "required" || toolChoice == "any") {
		operation := "execute"
		if stream {
			operation = "stream"
		}
		return nil, newLlumiverseError("a required tool choice was requested, but no tools are available", boolPtr(false), LlumiverseErrorContext{
			Provider:  ProviderAzureFoundry,
			Model:     options.Model,
			Operation: operation,
		}, nil, 0, "ToolChoiceConfigurationError")
	}
	if len(options.Tools) > 0 {
		payload["tools"] = chatTools(options.Tools)
		switch {
		case requiredTool != "":
			payload["tool_choice"] = map[string]any{"type": "function", "function": map[string]any{"name": requiredTool}}
		case toolChoice == "any":
			payload["tool_choice"] = "required"
		case toolChoice != "":
			payload["tool_choice"] = toolChoice
		}
		if parallel, ok := modelOptions["parallel_tool_calls"].(bool); ok {
			payload["parallel_tool_calls"] = parallel
		}
	}
	if options.ResultSchema != nil {
		schema, strict := openai.StructuredOutputSchema(options.ResultSchema)
		payload["response_format"] = map[string]any{
			"type":        "json_schema",
			"json_schema": map[string]any{"name": "output", "strict": strict, "schema": schema},
		}
	}
	return payload, nil
}

// chatTools converts tool definitions to Chat Completions function tools.
func chatTools(tools []ToolDefinition) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		parameters := map[string]any{}
		if tool.InputSchema != nil {
			parameters = openai.LimitedSchema(tool.InputSchema)
		}
		function := map[string]any{"name": tool.Name, "parameters": parameters}
		if tool.Description != nil && *tool.Description != "" {
			function["description"] = *tool.Description
		}
		out = append(out, map[string]any{"type": "function", "function": function})
	}
	return out
}

// chatEndpoint is the Chat Completions URL of the inference endpoint.
func (d *AzureFoundryDriver) chatEndpoint() (string, error) {
	return d.v1Endpoint(d.inferenceV1, "chat", "completions")
}

// includeThoughts reports whether reasoning is returned as thoughts results (the default).
func includeThoughts(options ExecutionOptions) bool {
	value, ok := options.ModelOptions["include_thoughts"].(bool)
	return !ok || value
}

// chatResponse is a non-streaming Chat Completions response.
type chatResponse struct {
	ID          string       `json:"id"`
	Choices     []chatChoice `json:"choices"`
	Usage       *chatUsage   `json:"usage"`
	ServiceTier string       `json:"service_tier"`
}

type chatChoice struct {
	FinishReason string               `json:"finish_reason"`
	Message      *chatResponseMessage `json:"message"`
	Delta        *chatResponseMessage `json:"delta"`
}

// chatResponseMessage is a response message or a stream delta.
type chatResponseMessage struct {
	Content          json.RawMessage     `json:"content"`
	ReasoningContent *string             `json:"reasoning_content"`
	Reasoning        *string             `json:"reasoning"`
	ToolCalls        []chatToolCallDelta `json:"tool_calls"`
	Refusal          *string             `json:"refusal"`
}

// chatToolCallDelta is a tool call of a response, or a fragment of one in a stream.
type chatToolCallDelta struct {
	Index    *int   `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type chatUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens     int `json:"cached_tokens"`
		CacheWriteTokens int `json:"cache_write_tokens"`
	} `json:"prompt_tokens_details"`
}

// executionUsage maps Chat Completions usage; cache reads and writes are part
// of the prompt tokens, so the new prompt tokens are what remains.
func (u *chatUsage) executionUsage() *ExecutionTokenUsage {
	if u == nil {
		return nil
	}
	usage := &ExecutionTokenUsage{
		Prompt: tokens(u.PromptTokens),
		Result: tokens(u.CompletionTokens),
		Total:  tokens(u.TotalTokens),
	}
	newTokens := u.PromptTokens
	if details := u.PromptTokensDetails; details != nil {
		if details.CachedTokens > 0 {
			usage.PromptCached = tokens(details.CachedTokens)
		}
		if details.CacheWriteTokens > 0 {
			usage.PromptCacheWrite = tokens(details.CacheWriteTokens)
		}
		newTokens -= details.CachedTokens + details.CacheWriteTokens
	}
	usage.PromptNew = tokens(max(newTokens, 0))
	return usage
}

// contentText returns the text of a message content: a string, or its text parts.
func contentText(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}
	var text string
	if json.Unmarshal(trimmed, &text) == nil {
		return text
	}
	var parts []chatPart
	if json.Unmarshal(trimmed, &parts) != nil {
		return ""
	}
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.Type == "text" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

func reasoningText(message *chatResponseMessage) string {
	switch {
	case message.ReasoningContent != nil:
		return *message.ReasoningContent
	case message.Reasoning != nil:
		return *message.Reasoning
	}
	return ""
}

// chatFinishReason normalizes a finish reason. Length and content-filter stops
// are kept even with a partial tool call, so the incomplete call is dropped
// rather than reported as malformed.
func chatFinishReason(reason string, toolUse bool) string {
	reason = strings.ToLower(reason)
	switch {
	case reason == "length" || reason == "content_filter":
		return reason
	case toolUse || reason == "tool_calls" || reason == "function_call":
		return "tool_use"
	}
	return reason
}

func isCutOff(reason string) bool {
	reason = strings.ToLower(reason)
	return reason == "length" || reason == "content_filter"
}

// isJSONObjectText reports whether value is a complete JSON object.
func isJSONObjectText(value string) bool {
	var object map[string]any
	return json.Unmarshal([]byte(value), &object) == nil
}

// toolInputFromArguments parses tool arguments, keeping only a JSON object.
func toolInputFromArguments(arguments string) any {
	var object map[string]any
	if json.Unmarshal([]byte(arguments), &object) == nil {
		return object
	}
	return map[string]any{}
}

// requestChat runs a non-streaming Chat Completions request.
func (d *AzureFoundryDriver) requestChat(ctx context.Context, prompt chatPrompt, r route, options ExecutionOptions) (*Completion, error) {
	conversation := chatConversationInput(options.Conversation, prompt, r.source, len(options.Tools) > 0)
	payload, err := d.chatPayload(conversation, r, options, false)
	if err != nil {
		return nil, err
	}
	endpoint, err := d.chatEndpoint()
	if err != nil {
		return nil, err
	}
	headers, err := d.creds.keyOrBearer(ctx, headerAPIKey)
	if err != nil {
		return nil, err
	}
	var raw json.RawMessage
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, payload, &raw); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	var response chatResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	var choice chatChoice
	if len(response.Choices) > 0 {
		choice = response.Choices[0]
	}
	message := choice.Message
	if message == nil {
		return nil, d.formatError(errors.New("chat completions response is not valid: no data"), options.Model, "execute")
	}
	results := chatResults(contentText(message.Content), reasoningText(message), includeThoughts(options))
	calls := functionCalls(message.ToolCalls)
	// A cut-off stop may have ended inside the last tool call: leave it out when
	// its arguments are not complete JSON rather than invent empty input.
	cutOff := isCutOff(choice.FinishReason)
	if cutOff && len(calls) > 0 && !isJSONObjectText(calls[len(calls)-1].Function.Arguments) {
		calls = calls[:len(calls)-1]
	}
	toolUses := make([]ToolUse, 0, len(calls))
	for _, call := range calls {
		toolUses = append(toolUses, ToolUse{ID: call.ID, ToolName: call.Function.Name, ToolInput: toolInputObject(toolInputFromArguments(call.Function.Arguments))})
	}
	if len(results) == 0 && reasoningText(message) == "" && len(toolUses) == 0 && !cutOff {
		return nil, d.formatError(errors.New("chat completions response is not valid: no data"), options.Model, "execute")
	}
	assistant := chatMessage{
		Role:             "assistant",
		Text:             contentText(message.Content),
		ReasoningContent: message.ReasoningContent,
		Reasoning:        message.Reasoning,
	}
	if len(toolUses) > 0 {
		assistant.ToolCalls = calls
	}
	conversation.Messages = append(conversation.Messages, assistant)
	completion := &Completion{
		Result:       results,
		ToolUse:      toolUses,
		TokenUsage:   response.Usage.executionUsage(),
		FinishReason: chatFinishReason(choice.FinishReason, len(toolUses) > 0),
		ServiceTier:  response.ServiceTier,
		Conversation: finalizeChatConversation(conversation, r.source, options),
	}
	if len(completion.ToolUse) == 0 {
		completion.ToolUse = nil
	}
	if options.IncludeOriginalResponse {
		var original any
		if json.Unmarshal(raw, &original) == nil {
			completion.OriginalResponse = original
		}
	}
	return completion, nil
}

// functionCalls returns the function tool calls of a response message.
func functionCalls(deltas []chatToolCallDelta) []chatToolCall {
	calls := make([]chatToolCall, 0, len(deltas))
	for _, delta := range deltas {
		if delta.Type != "" && delta.Type != "function" {
			continue
		}
		calls = append(calls, chatToolCall{ID: delta.ID, Type: "function", Function: chatToolFunction{Name: delta.Function.Name, Arguments: delta.Function.Arguments}})
	}
	return calls
}

// chatResults returns the text and thoughts of a response: native reasoning
// first, then the content split around <think> blocks.
func chatResults(content string, reasoning string, thoughts bool) []CompletionResult {
	results := splitThinkBlocks(content, thoughts)
	if thoughts && reasoning != "" {
		results = append([]CompletionResult{newThoughtsResult(reasoning)}, results...)
	}
	return results
}

// splitThinkBlocks splits text around <think>...</think> blocks, returning the
// blocks as thoughts when thoughts is set. An unclosed block runs to the end.
func splitThinkBlocks(value string, thoughts bool) []CompletionResult {
	var results []CompletionResult
	lower := strings.ToLower(value)
	offset := 0
	for offset < len(value) {
		open := strings.Index(lower[offset:], "<think")
		if open < 0 {
			break
		}
		open += offset
		tagEnd := strings.Index(lower[open:], ">")
		if tagEnd < 0 {
			break
		}
		bodyStart := open + tagEnd + 1
		if text := strings.TrimSpace(value[offset:open]); text != "" {
			results = append(results, newTextResult(text))
		}
		closeIndex := strings.Index(lower[bodyStart:], "</think>")
		bodyEnd, next := len(value), len(value)
		if closeIndex >= 0 {
			bodyEnd = bodyStart + closeIndex
			next = bodyEnd + len("</think>")
		}
		if thoughts && bodyEnd > bodyStart {
			results = append(results, newThoughtsResult(value[bodyStart:bodyEnd]))
		}
		offset = next
	}
	if offset < len(value) {
		if text := strings.TrimSpace(value[offset:]); text != "" {
			results = append(results, newTextResult(text))
		}
	}
	return results
}

// thinkStreamProjector splits streamed text around <think> blocks, holding back
// a partial tag at a chunk boundary until the next chunk decides it.
type thinkStreamProjector struct {
	buffer            string
	inThoughts        bool
	trimNextTextStart bool
}

const (
	thinkOpenTag  = "<think"
	thinkCloseTag = "</think>"
)

// push adds streamed text and returns the results it completes; final flushes
// whatever is held back.
func (p *thinkStreamProjector) push(value string, final bool) []CompletionResult {
	p.buffer += value
	var results []CompletionResult
	for p.buffer != "" {
		lower := strings.ToLower(p.buffer)
		if p.inThoughts {
			if closeIndex := strings.Index(lower, thinkCloseTag); closeIndex >= 0 {
				results = p.append(results, ResultTypeThoughts, p.buffer[:closeIndex])
				p.buffer = p.buffer[closeIndex+len(thinkCloseTag):]
				p.inThoughts = false
				p.trimNextTextStart = true
				continue
			}
			if final {
				results = p.append(results, ResultTypeThoughts, p.buffer)
				p.buffer = ""
				break
			}
			retained := partialSuffixLength(lower, thinkCloseTag)
			results = p.append(results, ResultTypeThoughts, p.buffer[:len(p.buffer)-retained])
			p.buffer = p.buffer[len(p.buffer)-retained:]
			break
		}
		if open := strings.Index(lower, thinkOpenTag); open >= 0 {
			if tagEnd := strings.Index(lower[open:], ">"); tagEnd >= 0 {
				results = p.append(results, ResultTypeText, strings.TrimRight(p.buffer[:open], " \t\r\n"))
				p.buffer = p.buffer[open+tagEnd+1:]
				p.inThoughts = true
				continue
			}
		}
		if final {
			results = p.append(results, ResultTypeText, p.buffer)
			p.buffer = ""
			break
		}
		retained := 0
		if last := strings.LastIndex(lower, "<"); last >= 0 {
			candidate := lower[last:]
			if strings.HasPrefix(thinkOpenTag, candidate) || (strings.HasPrefix(candidate, thinkOpenTag) && !strings.Contains(candidate, ">")) {
				retained = len(candidate)
			}
		}
		results = p.append(results, ResultTypeText, p.buffer[:len(p.buffer)-retained])
		p.buffer = p.buffer[len(p.buffer)-retained:]
		break
	}
	return results
}

// partialSuffixLength returns the length of the longest suffix of value that is
// a proper prefix of tag.
func partialSuffixLength(value string, tag string) int {
	for length := min(len(value), len(tag)-1); length > 0; length-- {
		if strings.HasPrefix(tag, value[len(value)-length:]) {
			return length
		}
	}
	return 0
}

func (p *thinkStreamProjector) append(results []CompletionResult, kind string, value string) []CompletionResult {
	if kind == ResultTypeText && p.trimNextTextStart {
		value = strings.TrimLeft(value, " \t\r\n")
		if value != "" {
			p.trimNextTextStart = false
		}
	}
	if kind == ResultTypeText && strings.TrimSpace(value) == "" {
		return results
	}
	if value == "" {
		return results
	}
	if kind == ResultTypeThoughts {
		return append(results, newThoughtsResult(value))
	}
	return append(results, newTextResult(value))
}

// chatStreamState accumulates the native assistant message of a stream, for
// the retained conversation.
type chatStreamState struct {
	projector        thinkStreamProjector
	content          strings.Builder
	reasoningContent *string
	reasoning        *string
	calls            map[int]*chatToolCall
	keys             map[int]string
}

// chunk converts one stream event into a completion chunk.
func (s *chatStreamState) chunk(event chatResponse, thoughts bool) CompletionChunk {
	var choice chatChoice
	if len(event.Choices) > 0 {
		choice = event.Choices[0]
	}
	var results []CompletionResult
	var toolChunks []ToolUseChunk
	if delta := choice.Delta; delta != nil {
		if content := contentText(delta.Content); content != "" {
			s.content.WriteString(content)
			results = append(results, s.projector.push(content, choice.FinishReason != "")...)
		} else if choice.FinishReason != "" {
			results = append(results, s.projector.push("", true)...)
		}
		var reasoning string
		switch {
		case delta.ReasoningContent != nil:
			s.reasoningContent = appendString(s.reasoningContent, *delta.ReasoningContent)
			reasoning = *delta.ReasoningContent
		case delta.Reasoning != nil:
			s.reasoning = appendString(s.reasoning, *delta.Reasoning)
			reasoning = *delta.Reasoning
		}
		if thoughts && reasoning != "" {
			results = append([]CompletionResult{newThoughtsResult(reasoning)}, results...)
		}
		for i, delta := range delta.ToolCalls {
			index := i
			if delta.Index != nil {
				index = *delta.Index
			}
			call := s.calls[index]
			if call == nil {
				call = &chatToolCall{Type: "function"}
				s.calls[index] = call
			}
			if delta.ID != "" {
				call.ID = delta.ID
			}
			call.Function.Name += delta.Function.Name
			call.Function.Arguments += delta.Function.Arguments
			key, ok := s.keys[index]
			if !ok {
				key = delta.ID
				if key == "" {
					key = "tool_" + strconv.Itoa(index)
				}
				s.keys[index] = key
			}
			toolChunks = append(toolChunks, ToolUseChunk{ID: key, ToolName: delta.Function.Name, ToolInput: delta.Function.Arguments})
		}
	} else if choice.FinishReason != "" {
		results = append(results, s.projector.push("", true)...)
	}
	if !thoughts {
		filtered := results[:0]
		for _, result := range results {
			if result.Type() != ResultTypeThoughts {
				filtered = append(filtered, result)
			}
		}
		results = filtered
	}
	return CompletionChunk{
		Result:       results,
		ToolUse:      toolChunks,
		FinishReason: chatFinishReason(choice.FinishReason, len(toolChunks) > 0),
		TokenUsage:   event.Usage.executionUsage(),
		ServiceTier:  event.ServiceTier,
	}
}

func appendString(current *string, value string) *string {
	if current == nil {
		return &value
	}
	next := *current + value
	return &next
}

// assistantMessage is the native assistant message of the stream.
func (s *chatStreamState) assistantMessage() chatMessage {
	message := chatMessage{Role: "assistant", Text: s.content.String(), ReasoningContent: s.reasoningContent, Reasoning: s.reasoning}
	indexes := make([]int, 0, len(s.calls))
	for index := range s.calls {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	for _, index := range indexes {
		message.ToolCalls = append(message.ToolCalls, *s.calls[index])
	}
	return message
}

// streamChat runs a streaming Chat Completions request. Foundry models reject
// the OpenAI stream_options, so none are sent.
func (d *AzureFoundryDriver) streamChat(ctx context.Context, prompt chatPrompt, r route, options ExecutionOptions) (CompletionStream, error) {
	conversation := chatConversationInput(options.Conversation, prompt, r.source, len(options.Tools) > 0)
	payload, err := d.chatPayload(conversation, r, options, true)
	if err != nil {
		return nil, err
	}
	endpoint, err := d.chatEndpoint()
	if err != nil {
		return nil, err
	}
	headers, err := d.creds.keyOrBearer(ctx, headerAPIKey)
	if err != nil {
		return nil, err
	}
	body, err := postSSE(ctx, d.httpClient(options.HTTPTimeout), endpoint, headers, payload)
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	thoughts := includeThoughts(options)
	state := &chatStreamState{calls: map[int]*chatToolCall{}, keys: map[int]string{}}
	ch, send, closeFn := newStreamChannel(body)
	go func() {
		defer close(ch)
		err := scanSSE(body, func(event sseEvent) error {
			data := strings.TrimSpace(event.Data)
			if data == "" {
				return nil
			}
			if data == "[DONE]" {
				return io.EOF
			}
			var decoded chatResponse
			if err := json.Unmarshal([]byte(data), &decoded); err != nil {
				return err
			}
			if chunk := state.chunk(decoded, thoughts); hasChunkContent(chunk) && !send(streamItem{Chunk: chunk}) {
				return errStreamClosed
			}
			return nil
		})
		if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, errStreamClosed) {
			send(streamItem{Err: d.formatError(err, options.Model, "stream")})
		}
	}()
	completion := &ExecutionResponse{Prompt: prompt}
	return newChannelStreamWithFinalizer(ch, completion, closeFn, func(resp *ExecutionResponse) {
		assistant := state.assistantMessage()
		// Tool calls whose ID arrived after their first fragment were keyed by index.
		for i, tool := range resp.ToolUse {
			for index, key := range state.keys {
				if key == tool.ID && state.calls[index] != nil && state.calls[index].ID != "" {
					resp.ToolUse[i].ID = state.calls[index].ID
				}
			}
		}
		conversation.Messages = append(conversation.Messages, assistant)
		resp.Conversation = finalizeChatConversation(conversation, r.source, options)
	}), nil
}

// chatDebugPrompt copies a chat prompt with its inline images truncated.
func chatDebugPrompt(prompt chatPrompt) chatPrompt {
	out := prompt
	out.Messages = make([]chatMessage, len(prompt.Messages))
	for i, message := range prompt.Messages {
		if message.Parts != nil {
			parts := make([]chatPart, len(message.Parts))
			for j, part := range message.Parts {
				if part.ImageURL != nil {
					image := *part.ImageURL
					image.URL = truncateDataURLForDebug(image.URL)
					part.ImageURL = &image
				}
				parts[j] = part
			}
			message.Parts = parts
		}
		out.Messages[i] = message
	}
	return out
}
