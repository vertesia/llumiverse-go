package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/vertesia/llumiverse-go/core"
)

// ChatCompletionsProtocolOptions configures the shared OpenAI Chat Completions
// protocol used by the generic Chat Completions driver and by other providers
// that expose a Chat Completions endpoint (for example Vertex AI MaaS models).
type ChatCompletionsProtocolOptions struct {
	// Provider is reported in errors; it defaults to ProviderOpenAICompatible.
	Provider Provider
	// ModelName is the model identifier sent in the request body. It defaults to
	// ExecutionOptions.Model.
	ModelName string
	// DefaultMaxTokens is sent as max_tokens when the caller sets none. Some
	// compatible providers return empty or truncated completions without one.
	DefaultMaxTokens int
	// ExtraBody holds provider-specific request fields, sent under extra_body
	// below the caller's model_options.extra_body.
	ExtraBody map[string]any
	// ResultSchemaInPrompt requests result_schema through a prompt instruction
	// instead of response_format, for providers whose response_format support
	// is unreliable.
	ResultSchemaInPrompt bool
	// IncludeResultSchemaInPrompt adds the prompt instruction while still sending
	// response_format, for providers that enforce structured output loosely.
	IncludeResultSchemaInPrompt bool
	// IncludeResultSchemaInPromptForModel is the per-model form of
	// IncludeResultSchemaInPrompt for providers that host mixed model families.
	IncludeResultSchemaInPromptForModel func(model string) bool
	// CompatibleToolSchemas sends tool parameters as plain JSON Schema instead of
	// OpenAI strict schemas, for providers that reject the strict fields.
	CompatibleToolSchemas bool
}

// ChatCompletionsTransport sends Chat Completions requests built by
// ChatCompletionsProtocol. Payloads carry provider extensions under
// "extra_body"; a transport that talks to an endpoint expecting them at the
// body root flattens them with MergeChatCompletionsExtraBody.
type ChatCompletionsTransport interface {
	// PostChatCompletion sends a non-streaming request and returns the raw JSON response.
	PostChatCompletion(ctx context.Context, payload map[string]any, options ExecutionOptions) (json.RawMessage, error)
	// PostChatCompletionStream sends a streaming request and returns the SSE body.
	PostChatCompletionStream(ctx context.Context, payload map[string]any, options ExecutionOptions) (io.ReadCloser, error)
	// FormatChatCompletionsError wraps an error raised while reading a stream.
	FormatChatCompletionsError(err error, model string, operation string) error
}

// ChatCompletionsPrompt is the prompt built by ChatCompletionsProtocol: the new
// messages of this turn, as Chat Completions message objects.
type ChatCompletionsPrompt struct {
	Messages []map[string]any
}

// chatCompletionsConversation is the conversation state the protocol returns.
// The _is_openai_chat_completions marker matches the TS client's shape.
type chatCompletionsConversation struct {
	IsOpenAIChatCompletions bool             `json:"_is_openai_chat_completions"`
	Messages                []map[string]any `json:"messages"`
	Turn                    int              `json:"turn,omitempty"`
}

// ChatCompletionsProtocol implements prompt formatting, payload construction
// and response parsing for the OpenAI Chat Completions API over a transport.
type ChatCompletionsProtocol struct {
	Options ChatCompletionsProtocolOptions
}

var (
	thinkBlockPattern     = regexp.MustCompile(`(?is)<think\b[^>]*>(.*?)(?:</think>|$)`)
	thinkOpenPattern      = regexp.MustCompile(`(?i)<think\b[^>]*>`)
	deepSeekOmitReasoning = regexp.MustCompile(`deepseek-(?:r1|reasoner)|deepseek\.r1`)
	deepSeekV32Pattern    = regexp.MustCompile(`deepseek-v3\.2(?:-|$)`)
)

const chatCompletionsSchemaInstruction = "IMPORTANT: only answer using JSON, and respecting the schema included below, between the <response_schema> tags. <response_schema>%s</response_schema>"

// CreatePrompt converts prompt segments into Chat Completions messages: system
// segments are joined into one system message, tool segments become tool
// messages, and files become image, audio or text parts.
func (p ChatCompletionsProtocol) CreatePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (ChatCompletionsPrompt, error) {
	var messages []map[string]any
	var system strings.Builder
	for _, segment := range segments {
		if segment.Role == PromptRoleSystem && segment.Content != "" {
			system.WriteString(segment.Content)
			system.WriteString("\n")
		}
	}
	if text := strings.TrimSpace(system.String()); text != "" {
		messages = append(messages, map[string]any{"role": "system", "content": text})
	}
	if options.ResultSchema != nil && p.schemaInPrompt(options.Model) {
		schema, _ := json.Marshal(options.ResultSchema)
		messages = append(messages, map[string]any{"role": "system", "content": fmt.Sprintf(chatCompletionsSchemaInstruction, schema)})
	}
	for _, segment := range segments {
		switch segment.Role {
		case PromptRoleSystem:
			continue
		case PromptRoleTool:
			if segment.ToolUseID == "" {
				return ChatCompletionsPrompt{}, errors.New("tool prompt segment must have a tool_use_id to reference the original tool call")
			}
			parts := []any{map[string]any{"type": "text", "text": segment.Content}}
			for _, file := range segment.Files {
				part, err := chatCompletionsFilePart(ctx, file, false)
				if err != nil {
					return ChatCompletionsPrompt{}, err
				}
				if part != nil {
					parts = append(parts, part)
				}
			}
			var content any = parts
			if len(parts) == 1 {
				content = segment.Content
			}
			messages = append(messages, map[string]any{"role": "tool", "tool_call_id": segment.ToolUseID, "content": content})
		default:
			var content any = segment.Content
			if len(segment.Files) > 0 {
				var parts []any
				if strings.TrimSpace(segment.Content) != "" {
					parts = append(parts, map[string]any{"type": "text", "text": segment.Content})
				}
				for _, file := range segment.Files {
					part, err := chatCompletionsFilePart(ctx, file, true)
					if err != nil {
						return ChatCompletionsPrompt{}, err
					}
					if part != nil {
						parts = append(parts, part)
					}
				}
				if len(parts) > 0 {
					content = parts
				}
			}
			role := "user"
			if segment.Role == PromptRoleAssistant {
				role = "assistant"
			}
			messages = append(messages, map[string]any{"role": role, "content": content})
		}
	}
	return ChatCompletionsPrompt{Messages: messages}, nil
}

// schemaInPrompt reports whether result_schema goes into a prompt instruction.
func (p ChatCompletionsProtocol) schemaInPrompt(model string) bool {
	return p.Options.ResultSchemaInPrompt || p.Options.IncludeResultSchemaInPrompt ||
		(p.Options.IncludeResultSchemaInPromptForModel != nil && p.Options.IncludeResultSchemaInPromptForModel(model))
}

// chatCompletionsFilePart converts a file to a content part: images become
// image_url parts, text files their content, and (outside tool results) MP3 and
// WAV audio input_audio parts. Other files are skipped.
func chatCompletionsFilePart(ctx context.Context, file DataSource, allowAudio bool) (any, error) {
	mime := file.MIMEType()
	switch {
	case strings.HasPrefix(mime, "image/"):
		url, err := dataSourceToDataURL(ctx, file)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "image_url", "image_url": map[string]any{"url": url, "detail": "auto"}}, nil
	case allowAudio && strings.HasPrefix(mime, "audio/"):
		return openAIInputAudioPart(ctx, file)
	case strings.HasPrefix(mime, "text/"):
		text, err := dataSourceText(ctx, file)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "text", "text": "\n\nFile content:\n" + text}, nil
	}
	return nil, nil
}

// openAIInputAudioPart builds a Chat Completions input_audio part. The API
// accepts only MP3 and WAV.
func openAIInputAudioPart(ctx context.Context, file DataSource) (map[string]any, error) {
	var format string
	switch file.MIMEType() {
	case "audio/wav", "audio/x-wav":
		format = "wav"
	case "audio/mpeg", "audio/mp3":
		format = "mp3"
	default:
		return nil, errors.New("chat audio input requires MP3 or WAV")
	}
	data, err := core.DataSourceToBase64(ctx, file)
	if err != nil {
		return nil, err
	}
	return map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": data, "format": format}}, nil
}

// dataSourceText reads a text file attachment.
func dataSourceText(ctx context.Context, file DataSource) (string, error) {
	reader, err := file.Open(ctx)
	if err != nil {
		return "", err
	}
	defer func() { _ = reader.Close() }()
	data, err := io.ReadAll(reader)
	return string(data), err
}

// Execute runs a non-streaming Chat Completions request through transport.
func (p ChatCompletionsProtocol) Execute(ctx context.Context, transport ChatCompletionsTransport, prompt any, options ExecutionOptions) (*Completion, error) {
	conversation := p.requestConversation(prompt, options)
	payload, err := p.BuildPayload(conversation.Messages, options, false)
	if err != nil {
		return nil, err
	}
	raw, err := transport.PostChatCompletion(ctx, payload, options)
	if err != nil {
		return nil, err
	}
	var response chatCompletionsResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, err
	}
	includeThoughts := chatIncludeThoughts(options)
	if len(response.Choices) == 0 {
		return nil, errors.New("chat completions response is not valid: no data")
	}
	choice := response.Choices[0]
	message := choice.Message
	results := chatCompletionsResults(message.contentText(), message.reasoningText(), includeThoughts)
	cutOff := isCutOffFinishReason(choice.FinishReason)
	calls := message.functionCalls()
	if cutOff && len(calls) > 0 && !isJSONObjectText(calls[len(calls)-1].Function.Arguments) {
		// A cut-off stop may have ended inside the last tool call; earlier calls had finished.
		calls = calls[:len(calls)-1]
	}
	toolUse := chatCompletionsToolUses(calls)
	finishReason := normalizeChatCompletionsFinishReason(choice.FinishReason, len(toolUse) > 0)
	if len(results) == 0 && message.reasoningText() == "" && len(toolUse) == 0 && !cutOff {
		// A cut-off stop with nothing returned is still valid: the stop is what the caller needs.
		return nil, errors.New("chat completions response is not valid: no data")
	}
	assistant := map[string]any{"role": "assistant", "content": message.contentValue()}
	if message.ReasoningContent != nil {
		assistant["reasoning_content"] = *message.ReasoningContent
	}
	if message.Reasoning != nil {
		assistant["reasoning"] = *message.Reasoning
	}
	if len(toolUse) > 0 {
		assistant["tool_calls"] = chatToolCallMaps(message.functionCalls())
	}
	completion := &Completion{
		Result:       results,
		ToolUse:      toolUse,
		TokenUsage:   openAIChatCompletionsUsage(response.Usage),
		ServiceTier:  response.ServiceTier,
		FinishReason: finishReason,
	}
	if options.IncludeOriginalResponse {
		var original any
		_ = json.Unmarshal(raw, &original)
		completion.OriginalResponse = original
	}
	conversation.Messages = append(conversation.Messages, assistant)
	completion.Conversation = finalizeChatCompletionsConversation(conversation, options)
	return completion, nil
}

// Stream runs a streaming Chat Completions request through transport.
func (p ChatCompletionsProtocol) Stream(ctx context.Context, transport ChatCompletionsTransport, prompt any, options ExecutionOptions) (CompletionStream, error) {
	conversation := p.requestConversation(prompt, options)
	payload, err := p.BuildPayload(conversation.Messages, options, true)
	if err != nil {
		return nil, err
	}
	body, err := transport.PostChatCompletionStream(ctx, payload, options)
	if err != nil {
		return nil, err
	}
	state := &chatCompletionsStreamState{includeThoughts: chatIncludeThoughts(options), calls: map[int]*chatToolCall{}}
	ch := make(chan streamItem)
	completion := &ExecutionResponse{Prompt: prompt}
	go func() {
		defer close(ch)
		err := core.ScanSSE(body, func(event core.SSEEvent) error {
			if event.Data == "" || event.Data == "[DONE]" {
				return nil
			}
			var decoded chatCompletionsResponse
			if err := json.Unmarshal([]byte(event.Data), &decoded); err != nil {
				return err
			}
			out := state.chunk(decoded)
			if len(out.Result) == 0 && len(out.ToolUse) == 0 && out.FinishReason == "" && out.TokenUsage == nil {
				return nil
			}
			select {
			case ch <- streamItem{Chunk: out}:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		if err != nil {
			select {
			case ch <- streamItem{Err: transport.FormatChatCompletionsError(err, options.Model, "stream")}:
			case <-ctx.Done():
			}
		}
	}()
	return newChannelStreamWithFinalizer(ch, completion, body.Close, func(resp *ExecutionResponse) {
		state.finalize(resp)
		conversation.Messages = append(conversation.Messages, state.assistantMessage())
		resp.Conversation = finalizeChatCompletionsConversation(conversation, options)
	}), nil
}

// requestConversation prepends the stored conversation to the prompt and
// repairs it for sending.
func (p ChatCompletionsProtocol) requestConversation(prompt any, options ExecutionOptions) chatCompletionsConversation {
	stored := chatCompletionsConversationFrom(options.Conversation)
	stored.IsOpenAIChatCompletions = true
	switch value := prompt.(type) {
	case ChatCompletionsPrompt:
		stored.Messages = append(stored.Messages, cloneMessages(value.Messages)...)
	case *ChatCompletionsPrompt:
		if value != nil {
			stored.Messages = append(stored.Messages, cloneMessages(value.Messages)...)
		}
	}
	stored.Messages = prepareChatCompletionsMessages(stored.Messages, options)
	return stored
}

// BuildPayload assembles the Chat Completions request body for messages.
// Unset options are omitted; provider extensions are carried under extra_body.
func (p ChatCompletionsProtocol) BuildPayload(messages []map[string]any, options ExecutionOptions, stream bool) (map[string]any, error) {
	mo := options.ModelOptions
	model := p.Options.ModelName
	if model == "" {
		model = options.Model
	}
	payload := map[string]any{
		"model":    model,
		"messages": convertToChatCompletionsMessages(messages),
		// The completion interface consumes one generated choice.
		"n":      1,
		"stream": stream,
	}
	if v := optionInt(mo, "max_tokens"); v != nil {
		payload["max_tokens"] = *v
	} else if p.Options.DefaultMaxTokens > 0 {
		payload["max_tokens"] = p.Options.DefaultMaxTokens
	}
	for _, key := range []string{"temperature", "top_p", "presence_penalty", "frequency_penalty", "seed"} {
		if v := optionNumber(mo, key); v != nil {
			payload[key] = v
		}
	}
	if stop := optionStrings(mo, "stop_sequence"); len(stop) > 0 {
		payload["stop"] = stop
	}
	if effort := optionString(mo, "effort", "reasoning_effort"); effort != "" {
		payload["reasoning_effort"] = effort
	}
	if tier := optionString(mo, "service_tier"); tier != "" {
		payload["service_tier"] = tier
	}
	if extra := mergeMaps(p.Options.ExtraBody, optionMap(mo, "extra_body")); len(extra) > 0 {
		payload["extra_body"] = extra
	}
	tools := chatCompletionsTools(options.Tools, p.Options.CompatibleToolSchemas)
	requiredTool := optionString(mo, "required_tool_name")
	toolChoice := optionString(mo, "tool_choice")
	if (requiredTool != "" || toolChoice == "required" || toolChoice == "any") && len(tools) == 0 {
		return nil, toolChoiceConfigurationError("[OpenAI Chat Completions API] A required tool choice was requested, but no tools are available.", p.provider(), options.Model, streamOperation(stream))
	}
	if len(tools) > 0 {
		payload["tools"] = tools
		switch {
		case requiredTool != "":
			payload["tool_choice"] = map[string]any{"type": "function", "function": map[string]any{"name": requiredTool}}
		case toolChoice == "any":
			payload["tool_choice"] = "required"
		case toolChoice != "":
			payload["tool_choice"] = toolChoice
		}
		if v := optionBoolPtr(mo, "parallel_tool_calls"); v != nil {
			payload["parallel_tool_calls"] = *v
		}
	}
	if options.ResultSchema != nil && !p.Options.ResultSchemaInPrompt {
		schema, strict := openAISchemaForResponses(options.ResultSchema)
		payload["response_format"] = map[string]any{
			"type":        "json_schema",
			"json_schema": map[string]any{"name": "output", "strict": strict, "schema": schema},
		}
	}
	return payload, nil
}

func (p ChatCompletionsProtocol) provider() Provider {
	if p.Options.Provider != "" {
		return p.Options.Provider
	}
	return ProviderOpenAICompatible
}

func streamOperation(stream bool) string {
	if stream {
		return "stream"
	}
	return "execute"
}

// toolChoiceConfigurationError reports a forced tool choice without tools as a
// non-retryable 400, as the TS client's ToolChoiceConfigurationError does.
func toolChoiceConfigurationError(message string, provider Provider, model, operation string) error {
	cause := errors.New(message)
	return newLlumiverseError(message, boolPtr(false), LlumiverseErrorContext{Provider: provider, Model: model, Operation: operation}, cause, 400, "ToolChoiceConfigurationError")
}

// MergeChatCompletionsExtraBody moves the payload's extra_body fields to the
// body root, below the fields llumiverse sets so they cannot replace them.
func MergeChatCompletionsExtraBody(payload map[string]any) map[string]any {
	extra, _ := payload["extra_body"].(map[string]any)
	out := make(map[string]any, len(payload)+len(extra))
	for key, value := range extra {
		out[key] = value
	}
	for key, value := range payload {
		if key != "extra_body" {
			out[key] = value
		}
	}
	return out
}

// mergeMaps merges maps left to right, returning nil when all are empty.
func mergeMaps(maps ...map[string]any) map[string]any {
	var out map[string]any
	for _, m := range maps {
		for key, value := range m {
			if out == nil {
				out = map[string]any{}
			}
			out[key] = value
		}
	}
	return out
}

// chatCompletionsTools converts tool definitions to Chat Completions function
// tools: strict OpenAI schemas when possible, or plain schemas for compatible
// providers.
func chatCompletionsTools(tools []ToolDefinition, compatible bool) []any {
	if len(tools) == 0 {
		return nil
	}
	out := make([]any, 0, len(tools))
	for _, tool := range tools {
		fn := map[string]any{"name": tool.Name, "parameters": map[string]any{}}
		if tool.Description != nil {
			fn["description"] = *tool.Description
		}
		if tool.InputSchema != nil {
			if compatible {
				fn["parameters"] = openAILimitedSchema(tool.InputSchema)
			} else {
				schema, strict := openAISchemaForResponses(tool.InputSchema)
				fn["parameters"] = schema
				fn["strict"] = strict
			}
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out
}

// convertToChatCompletionsMessages turns stored messages into request
// messages. Empty string content becomes null, all-text part lists are joined,
// and images in tool results (which the API accepts only as text) are replaced
// by an indexed reference and carried in a following user message.
func convertToChatCompletionsMessages(messages []map[string]any) []any {
	converted := make([]any, 0, len(messages))
	var attachments []any
	flush := func() {
		if len(attachments) > 0 {
			converted = append(converted, map[string]any{"role": "user", "content": attachments})
			attachments = nil
		}
	}
	for _, msg := range messages {
		role := stringValue(msg["role"])
		if role != "tool" {
			// Complete all parallel tool results before ordinary user content.
			flush()
		}
		result := map[string]any{"role": role}
		if calls, ok := msg["tool_calls"].([]any); ok && len(calls) > 0 {
			result["tool_calls"] = calls
		}
		for _, key := range []string{"reasoning_content", "reasoning"} {
			if value, ok := msg[key]; ok {
				result[key] = value
			}
		}
		switch content := msg["content"].(type) {
		case string:
			if content == "" {
				result["content"] = nil
			} else {
				result["content"] = content
			}
		case []any:
			imageIndex := 0
			parts := make([]any, 0, len(content))
			allText := true
			for _, raw := range content {
				part, _ := raw.(map[string]any)
				if role == "tool" && part != nil && part["type"] == "image_url" {
					imageIndex++
					label := fmt.Sprintf("Image %d from tool result %s:", imageIndex, stringValue(msg["tool_call_id"]))
					attachments = append(attachments, map[string]any{"type": "text", "text": label}, part)
					raw = map[string]any{"type": "text", "text": fmt.Sprintf("[Image %d attached below]", imageIndex)}
					part = raw.(map[string]any)
				}
				if part == nil || part["type"] != "text" {
					allText = false
				}
				parts = append(parts, raw)
			}
			switch {
			case len(parts) > 0 && allText:
				texts := make([]string, len(parts))
				for i, raw := range parts {
					texts[i] = stringValue(raw.(map[string]any)["text"])
				}
				result["content"] = strings.Join(texts, "\n")
			case len(parts) > 0:
				result["content"] = parts
			}
		default:
			result["content"] = nil
		}
		if id := stringValue(msg["tool_call_id"]); id != "" {
			result["tool_call_id"] = id
		}
		converted = append(converted, result)
	}
	flush()
	return converted
}

// chatCompletionsConversationFrom reads a stored conversation: the protocol's
// state, a {messages} object, or a legacy message array.
func chatCompletionsConversationFrom(conversation any) chatCompletionsConversation {
	switch value := conversation.(type) {
	case nil:
		return chatCompletionsConversation{}
	case chatCompletionsConversation:
		value.Messages = cloneMessages(value.Messages)
		return value
	case *chatCompletionsConversation:
		if value == nil {
			return chatCompletionsConversation{}
		}
		copied := *value
		copied.Messages = cloneMessages(copied.Messages)
		return copied
	}
	data, err := json.Marshal(conversation)
	if err != nil {
		return chatCompletionsConversation{}
	}
	var messages []map[string]any
	if json.Unmarshal(data, &messages) == nil {
		return chatCompletionsConversation{Messages: messages}
	}
	var state chatCompletionsConversation
	_ = json.Unmarshal(data, &state)
	return state
}

func cloneMessages(messages []map[string]any) []map[string]any {
	out := make([]map[string]any, len(messages))
	for i, message := range messages {
		out[i] = cloneJSONValue(message).(map[string]any)
	}
	return out
}

// prepareChatCompletionsMessages repairs orphaned tool calls and results,
// applies the model's reasoning replay policy, and turns tool messages into
// text when the request has no tools.
func prepareChatCompletionsMessages(messages []map[string]any, options ExecutionOptions) []map[string]any {
	messages = fixOrphanedChatCompletionsToolResults(fixOrphanedChatCompletionsToolUse(messages))
	messages = prepareChatCompletionsReasoning(messages, options.Model)
	if len(options.Tools) == 0 {
		messages = convertChatCompletionsToolMessagesToText(messages)
	}
	return messages
}

type chatReasoningReplayPolicy int

const (
	chatReasoningKeep chatReasoningReplayPolicy = iota
	chatReasoningOmit
	chatReasoningActiveToolTurn
	chatReasoningFullToolHistory
)

// chatReasoningPolicy decides which reasoning a model needs replayed: DeepSeek
// R1 rejects it, DeepSeek V4+ wants it on every tool turn, and V3.2 only on the
// active tool turn.
func chatReasoningPolicy(model string) chatReasoningReplayPolicy {
	id := strings.ToLower(model)
	switch {
	case deepSeekOmitReasoning.MatchString(id):
		return chatReasoningOmit
	case isDeepSeekVersionGTE(id, 4, 0):
		return chatReasoningFullToolHistory
	case deepSeekV32Pattern.MatchString(id):
		return chatReasoningActiveToolTurn
	}
	return chatReasoningKeep
}

func withoutChatReasoning(message map[string]any) map[string]any {
	_, hasContent := message["reasoning_content"]
	_, hasReasoning := message["reasoning"]
	if !hasContent && !hasReasoning {
		return message
	}
	out := make(map[string]any, len(message))
	for key, value := range message {
		if key != "reasoning_content" && key != "reasoning" {
			out[key] = value
		}
	}
	return out
}

func prepareChatCompletionsReasoning(messages []map[string]any, model string) []map[string]any {
	switch chatReasoningPolicy(model) {
	case chatReasoningOmit:
		out := make([]map[string]any, len(messages))
		for i, message := range messages {
			out[i] = withoutChatReasoning(message)
		}
		return out
	case chatReasoningActiveToolTurn:
		start := -1
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i]["role"] == "user" {
				start = i
				break
			}
		}
		if start <= 0 {
			return messages
		}
		out := make([]map[string]any, len(messages))
		for i, message := range messages {
			if i < start {
				message = withoutChatReasoning(message)
			}
			out[i] = message
		}
		return out
	}
	return messages
}

// chatMessageToolCalls returns the tool_calls of a stored message.
func chatMessageToolCalls(message map[string]any) []map[string]any {
	raw, _ := message["tool_calls"].([]any)
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		if call, ok := item.(map[string]any); ok {
			out = append(out, call)
		}
	}
	return out
}

func chatToolCallName(call map[string]any) string {
	if fn, ok := call["function"].(map[string]any); ok {
		return stringValue(fn["name"])
	}
	return ""
}

func chatToolCallArguments(call map[string]any) string {
	if fn, ok := call["function"].(map[string]any); ok {
		return stringValue(fn["arguments"])
	}
	return ""
}

// fixOrphanedChatCompletionsToolUse inserts a tool message for every assistant
// tool call left without a result (a run stopped mid tool execution), because
// Chat Completions requires each call to be answered.
func fixOrphanedChatCompletionsToolUse(messages []map[string]any) []map[string]any {
	if len(messages) < 2 {
		return messages
	}
	answered := map[string]bool{}
	for _, message := range messages {
		if message["role"] == "tool" {
			if id := stringValue(message["tool_call_id"]); id != "" {
				answered[id] = true
			}
		}
	}
	out := make([]map[string]any, 0, len(messages))
	var pendingOrder []string
	pending := map[string]string{}
	flush := func() {
		for _, id := range pendingOrder {
			out = append(out, map[string]any{
				"role":         "tool",
				"tool_call_id": id,
				"content":      fmt.Sprintf(`[Tool interrupted: The user stopped the operation before "%s" could execute.]`, pending[id]),
			})
		}
		pendingOrder = nil
		pending = map[string]string{}
	}
	for _, message := range messages {
		if calls := chatMessageToolCalls(message); len(calls) > 0 {
			for _, call := range calls {
				id := stringValue(call["id"])
				if id == "" || answered[id] {
					continue
				}
				name := chatToolCallName(call)
				if name == "" {
					name = "unknown"
				}
				if _, seen := pending[id]; !seen {
					pendingOrder = append(pendingOrder, id)
				}
				pending[id] = name
			}
			out = append(out, message)
			continue
		}
		if message["role"] == "tool" {
			out = append(out, message)
			continue
		}
		flush()
		out = append(out, message)
	}
	flush()
	return out
}

// fixOrphanedChatCompletionsToolResults drops tool messages whose call is no
// longer in the conversation, usually after trimming removed the call.
func fixOrphanedChatCompletionsToolResults(messages []map[string]any) []map[string]any {
	calls := map[string]bool{}
	for _, message := range messages {
		for _, call := range chatMessageToolCalls(message) {
			if id := stringValue(call["id"]); id != "" {
				calls[id] = true
			}
		}
	}
	out := make([]map[string]any, 0, len(messages))
	for _, message := range messages {
		if message["role"] == "tool" {
			if id := stringValue(message["tool_call_id"]); id == "" || !calls[id] {
				continue
			}
		}
		out = append(out, message)
	}
	return out
}

// convertChatCompletionsToolMessagesToText rewrites tool calls and results as
// bracketed text for requests made without tools, keeping tool-result images.
func convertChatCompletionsToolMessagesToText(messages []map[string]any) []map[string]any {
	hasTools := false
	for _, message := range messages {
		if message["role"] == "tool" || len(chatMessageToolCalls(message)) > 0 {
			hasTools = true
			break
		}
	}
	if !hasTools {
		return messages
	}
	out := make([]map[string]any, 0, len(messages))
	for _, message := range messages {
		if calls := chatMessageToolCalls(message); len(calls) > 0 {
			var lines []string
			if text := chatContentText(message["content"]); strings.TrimSpace(text) != "" {
				lines = append(lines, text)
			}
			for _, call := range calls {
				name := chatToolCallName(call)
				if name == "" {
					name = "unknown"
				}
				lines = append(lines, fmt.Sprintf("[Tool call: %s(%s)]", name, truncateForConversation(chatToolCallArguments(call), 500)))
			}
			var content any
			if len(lines) > 0 {
				content = strings.Join(lines, "\n")
			}
			out = append(out, map[string]any{"role": message["role"], "content": content})
			continue
		}
		if message["role"] == "tool" {
			if parts, ok := message["content"].([]any); ok && chatPartsHaveImage(parts) {
				content := append([]any{map[string]any{"type": "text", "text": "Tool result " + stringValue(message["tool_call_id"]) + ":"}}, parts...)
				out = append(out, map[string]any{"role": "user", "content": content})
				continue
			}
			output := chatContentText(message["content"])
			if output == "" {
				output = "No output"
			}
			out = append(out, map[string]any{"role": "user", "content": "[Tool result: " + truncateForConversation(output, 500) + "]"})
			continue
		}
		out = append(out, message)
	}
	return out
}

func chatPartsHaveImage(parts []any) bool {
	for _, raw := range parts {
		if part, ok := raw.(map[string]any); ok && part["type"] == "image_url" {
			return true
		}
	}
	return false
}

// chatContentText returns a message content's text: the string, or its text
// parts joined by newlines.
func chatContentText(content any) string {
	switch value := content.(type) {
	case string:
		return value
	case []any:
		var texts []string
		for _, raw := range value {
			if part, ok := raw.(map[string]any); ok && part["type"] == "text" {
				if text, ok := part["text"].(string); ok {
					texts = append(texts, text)
				}
			}
		}
		return strings.Join(texts, "\n")
	}
	return ""
}

// finalizeChatCompletionsConversation increments the turn and applies the
// reasoning replay policy and the retention rules to the next conversation.
func finalizeChatCompletionsConversation(conversation chatCompletionsConversation, options ExecutionOptions) chatCompletionsConversation {
	conversation.IsOpenAIChatCompletions = true
	conversation.Turn++
	policy := chatReasoningPolicy(options.Model)
	if policy == chatReasoningOmit {
		for i, message := range conversation.Messages {
			conversation.Messages[i] = withoutChatReasoning(message)
		}
	}
	hasReasoning := func(message map[string]any) bool {
		return len(chatMessageToolCalls(message)) > 0 && (message["reasoning_content"] != nil || message["reasoning"] != nil)
	}
	switch policy {
	case chatReasoningActiveToolTurn:
		for i := len(conversation.Messages) - 1; i >= 0; i-- {
			if conversation.Messages[i]["role"] == "assistant" {
				if hasReasoning(conversation.Messages[i]) {
					// The tool turn is still active: keep its reasoning and the history verbatim.
					return conversation
				}
				break
			}
		}
		for i, message := range conversation.Messages {
			conversation.Messages[i] = withoutChatReasoning(message)
		}
	case chatReasoningFullToolHistory:
		for _, message := range conversation.Messages {
			if hasReasoning(message) {
				return conversation
			}
		}
	}
	stripMedia := shouldStripConversationMedia(conversation.Turn, options.StripImagesAfterTurns)
	stripHeartbeats := shouldStripConversationHeartbeats(conversation.Turn, options.StripHeartbeatsAfterTurns)
	for i, message := range conversation.Messages {
		switch content := message["content"].(type) {
		case string:
			message["content"] = core.ProcessConversationText(content, options, stripHeartbeats)
		case []any:
			message["content"] = processChatContentParts(content, options, stripMedia, stripHeartbeats)
		}
		conversation.Messages[i] = message
	}
	return conversation
}

// processChatContentParts applies the retention rules to content parts:
// expired inline images become a placeholder and text is truncated.
func processChatContentParts(parts []any, options ExecutionOptions, stripMedia, stripHeartbeats bool) []any {
	out := make([]any, 0, len(parts))
	for _, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok {
			out = append(out, raw)
			continue
		}
		switch part["type"] {
		case "image_url":
			image, _ := part["image_url"].(map[string]any)
			url := stringValue(image["url"])
			if stripMedia && strings.HasPrefix(url, "data:image/") && strings.Contains(url, ";base64,") {
				out = append(out, map[string]any{"type": "text", "text": conversationImagePlaceholder})
				continue
			}
		case "text":
			if text, ok := part["text"].(string); ok {
				part["text"] = core.ProcessConversationText(text, options, stripHeartbeats)
			}
		}
		out = append(out, part)
	}
	return out
}

// chatCompletionsResponse decodes both a completion and a stream chunk.
type chatCompletionsResponse struct {
	ID          string                  `json:"id"`
	ServiceTier string                  `json:"service_tier"`
	Choices     []chatCompletionsChoice `json:"choices"`
	Usage       *chatCompletionsUsage   `json:"usage"`
}

type chatCompletionsChoice struct {
	FinishReason string                 `json:"finish_reason"`
	Message      chatCompletionsMessage `json:"message"`
	Delta        chatCompletionsMessage `json:"delta"`
}

type chatCompletionsMessage struct {
	Content          json.RawMessage `json:"content"`
	Refusal          string          `json:"refusal"`
	ReasoningContent *string         `json:"reasoning_content"`
	Reasoning        *string         `json:"reasoning"`
	ToolCalls        []chatToolCall  `json:"tool_calls"`
}

type chatToolCall struct {
	Index    *int   `json:"index,omitempty"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// contentValue returns the decoded content: a string, a part list, or nil.
func (m chatCompletionsMessage) contentValue() any {
	if len(m.Content) == 0 {
		return nil
	}
	var value any
	_ = json.Unmarshal(m.Content, &value)
	return value
}

func (m chatCompletionsMessage) contentText() string {
	return chatContentText(m.contentValue())
}

// reasoningText returns the native reasoning, preferring reasoning_content.
func (m chatCompletionsMessage) reasoningText() string {
	if m.ReasoningContent != nil {
		return *m.ReasoningContent
	}
	if m.Reasoning != nil {
		return *m.Reasoning
	}
	return ""
}

// functionCalls returns the function tool calls, skipping other call types.
func (m chatCompletionsMessage) functionCalls() []chatToolCall {
	out := make([]chatToolCall, 0, len(m.ToolCalls))
	for _, call := range m.ToolCalls {
		if call.Type == "" || call.Type == "function" {
			out = append(out, call)
		}
	}
	return out
}

func chatToolCallMaps(calls []chatToolCall) []any {
	out := make([]any, len(calls))
	for i, call := range calls {
		out[i] = map[string]any{
			"id":       call.ID,
			"type":     "function",
			"function": map[string]any{"name": call.Function.Name, "arguments": call.Function.Arguments},
		}
	}
	return out
}

func chatCompletionsToolUses(calls []chatToolCall) []ToolUse {
	if len(calls) == 0 {
		return nil
	}
	out := make([]ToolUse, len(calls))
	for i, call := range calls {
		out[i] = ToolUse{ID: call.ID, ToolName: call.Function.Name, ToolInput: toolInputObject(safeJSONParse(call.Function.Arguments))}
	}
	return out
}

// normalizeChatCompletionsFinishReason keeps length and content_filter so the
// caller sees the provider's stop, and reports tool calls as tool_use.
func normalizeChatCompletionsFinishReason(reason string, hasToolUse bool) string {
	if reason == "length" || reason == "content_filter" {
		return reason
	}
	if hasToolUse || reason == "tool_calls" || reason == "function_call" {
		return "tool_use"
	}
	return reason
}

// isCutOffFinishReason reports whether a finish reason means the response may
// have stopped inside its last tool call.
func isCutOffFinishReason(reason string) bool {
	switch reason {
	case "length", "refusal", "content_filter", "content_filtered", "guardrail_intervened":
		return true
	}
	return false
}

// isJSONObjectText reports whether text parses as a JSON object.
func isJSONObjectText(text string) bool {
	var value any
	if json.Unmarshal([]byte(text), &value) != nil {
		return false
	}
	_, ok := value.(map[string]any)
	return ok
}

func chatIncludeThoughts(options ExecutionOptions) bool {
	v := optionBoolPtr(options.ModelOptions, "include_thoughts")
	return v == nil || *v
}

// chatCompletionsResults splits content into text and <think> blocks, putting
// the native reasoning first as thoughts.
func chatCompletionsResults(content, reasoning string, includeThoughts bool) []CompletionResult {
	results := splitChatCompletionsThinkBlocks(content, includeThoughts)
	if includeThoughts && reasoning != "" {
		results = append([]CompletionResult{newThoughtsResult(reasoning)}, results...)
	}
	return results
}

// splitChatCompletionsThinkBlocks splits <think> blocks some open models write
// inline out of the text, as thoughts.
func splitChatCompletionsThinkBlocks(value string, includeThoughts bool) []CompletionResult {
	var results []CompletionResult
	offset := 0
	for _, match := range thinkBlockPattern.FindAllStringSubmatchIndex(value, -1) {
		if match[0] > offset {
			if text := strings.TrimSpace(value[offset:match[0]]); text != "" {
				results = append(results, newTextResult(text))
			}
		}
		if includeThoughts && match[3] > match[2] {
			results = append(results, newThoughtsResult(value[match[2]:match[3]]))
		}
		offset = match[1]
	}
	if offset < len(value) {
		if text := strings.TrimSpace(value[offset:]); text != "" {
			results = append(results, newTextResult(text))
		}
	}
	return results
}

// openAIChatCompletionsUsage maps Chat Completions usage, including the cache
// writes and provider cost OpenAI-compatible gateways report (cost is not
// charged on bring-your-own-key requests).
func openAIChatCompletionsUsage(usage *chatCompletionsUsage) *ExecutionTokenUsage {
	if usage == nil {
		return nil
	}
	var cached, cacheWrite *int64
	if usage.PromptTokensDetails != nil {
		cached = usage.PromptTokensDetails.CachedTokens
		cacheWrite = usage.PromptTokensDetails.CacheWriteTokens
	}
	out := openAIPromptUsage(usage.PromptTokens, cached, cacheWrite)
	out.Result = tokens(int(usage.CompletionTokens))
	out.Total = tokens(int(usage.TotalTokens))
	if usage.Cost != nil && (usage.IsBYOK == nil || !*usage.IsBYOK) {
		cost := *usage.Cost
		out.ProviderCostUsd = &cost
	}
	return out
}

type chatCompletionsUsage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens     *int64 `json:"cached_tokens"`
		CacheWriteTokens *int64 `json:"cache_write_tokens"`
	} `json:"prompt_tokens_details"`
	Cost   *float64 `json:"cost"`
	IsBYOK *bool    `json:"is_byok"`
}

// openAIPromptUsage splits prompt tokens by cache use. OpenAI-compatible APIs
// count cache reads and writes in the prompt tokens, so the new prompt tokens
// are what remains.
func openAIPromptUsage(prompt int64, cached, cacheWrite *int64) *ExecutionTokenUsage {
	out := &ExecutionTokenUsage{Prompt: tokens(int(prompt))}
	remaining := prompt
	if cached != nil {
		remaining -= *cached
		if *cached > 0 {
			out.PromptCached = tokens(int(*cached))
		}
	}
	if cacheWrite != nil {
		remaining -= *cacheWrite
		if *cacheWrite > 0 {
			out.PromptCacheWrite = tokens(int(*cacheWrite))
		}
	}
	out.PromptNew = tokens(int(max(0, remaining)))
	return out
}

// chatCompletionsStreamState projects stream chunks and accumulates the native
// assistant message for the conversation.
type chatCompletionsStreamState struct {
	includeThoughts  bool
	projector        thinkStreamProjector
	content          strings.Builder
	reasoningContent *string
	reasoning        *string
	calls            map[int]*chatToolCall
	finishReason     string
	serviceTier      string
}

func (s *chatCompletionsStreamState) chunk(response chatCompletionsResponse) CompletionChunk {
	out := CompletionChunk{TokenUsage: openAIChatCompletionsUsage(response.Usage), ServiceTier: response.ServiceTier}
	if response.ServiceTier != "" {
		s.serviceTier = response.ServiceTier
	}
	if len(response.Choices) == 0 {
		return out
	}
	choice := response.Choices[0]
	delta := choice.Delta
	var results []CompletionResult
	if content := delta.contentText(); content != "" {
		s.content.WriteString(content)
		results = append(results, s.projector.push(content, choice.FinishReason != "")...)
	} else if choice.FinishReason != "" {
		results = append(results, s.projector.push("", true)...)
	}
	appendReasoning := func(target **string, value string) {
		if *target == nil {
			*target = new(string)
		}
		**target += value
		if s.includeThoughts && value != "" {
			results = append([]CompletionResult{newThoughtsResult(value)}, results...)
		}
	}
	if delta.ReasoningContent != nil {
		appendReasoning(&s.reasoningContent, *delta.ReasoningContent)
	} else if delta.Reasoning != nil {
		appendReasoning(&s.reasoning, *delta.Reasoning)
	}
	for _, call := range delta.ToolCalls {
		index := 0
		if call.Index != nil {
			index = *call.Index
		}
		native := s.calls[index]
		if native == nil {
			native = &chatToolCall{Type: "function"}
			s.calls[index] = native
		}
		if call.ID != "" {
			native.ID = call.ID
		}
		native.Function.Name += call.Function.Name
		native.Function.Arguments += call.Function.Arguments
		// Arguments arrive as string fragments; keeping them as strings (even
		// empty ones) lets core stitch them without a placeholder replacing bytes.
		out.ToolUse = append(out.ToolUse, ToolUseChunk{ID: "tool_" + strconv.Itoa(index), ToolName: call.Function.Name, ToolInput: call.Function.Arguments})
	}
	if !s.includeThoughts {
		results = withoutThoughts(results)
	}
	out.Result = results
	out.FinishReason = normalizeChatCompletionsFinishReason(choice.FinishReason, len(out.ToolUse) > 0)
	if choice.FinishReason != "" {
		s.finishReason = choice.FinishReason
	}
	return out
}

func withoutThoughts(results []CompletionResult) []CompletionResult {
	out := results[:0]
	for _, result := range results {
		if result.Type() != ResultTypeThoughts {
			out = append(out, result)
		}
	}
	return out
}

// finalize replaces the stream's positional tool IDs with the provider's IDs,
// drops a last call cut off by a content filter (core drops those cut off by
// the token limit), and records the service tier.
func (s *chatCompletionsStreamState) finalize(resp *ExecutionResponse) {
	if s.serviceTier != "" {
		resp.ServiceTier = s.serviceTier
	}
	last := -1
	for index := range s.calls {
		last = max(last, index)
	}
	dropLast := last >= 0 && isCutOffFinishReason(s.finishReason) && !isJSONObjectText(s.calls[last].Function.Arguments)
	tools := resp.ToolUse[:0]
	for _, tool := range resp.ToolUse {
		if index, err := strconv.Atoi(strings.TrimPrefix(tool.ID, "tool_")); err == nil && strings.HasPrefix(tool.ID, "tool_") {
			if dropLast && index == last {
				continue
			}
			if native := s.calls[index]; native != nil && native.ID != "" {
				tool.ID = native.ID
			}
		}
		tools = append(tools, tool)
	}
	resp.ToolUse = tools
}

// assistantMessage returns the streamed assistant message in native form.
func (s *chatCompletionsStreamState) assistantMessage() map[string]any {
	message := map[string]any{"role": "assistant", "content": nil}
	if content := s.content.String(); content != "" {
		message["content"] = content
	}
	if s.reasoningContent != nil {
		message["reasoning_content"] = *s.reasoningContent
	}
	if s.reasoning != nil {
		message["reasoning"] = *s.reasoning
	}
	if len(s.calls) > 0 {
		indexes := make([]int, 0, len(s.calls))
		for index := range s.calls {
			indexes = append(indexes, index)
		}
		sort.Ints(indexes)
		calls := make([]chatToolCall, len(indexes))
		for i, index := range indexes {
			calls[i] = *s.calls[index]
		}
		message["tool_calls"] = chatToolCallMaps(calls)
	}
	return message
}

// thinkStreamProjector splits streamed text into text and thoughts at <think>
// tags, holding back a partial tag until the next chunk shows what it is.
type thinkStreamProjector struct {
	buffer            string
	inThoughts        bool
	trimNextTextStart bool
}

func (p *thinkStreamProjector) push(value string, final bool) []CompletionResult {
	p.buffer += value
	var results []CompletionResult
	for p.buffer != "" {
		if p.inThoughts {
			if index := strings.Index(strings.ToLower(p.buffer), "</think>"); index >= 0 {
				results = p.append(results, false, p.buffer[:index])
				p.buffer = p.buffer[index+len("</think>"):]
				p.inThoughts = false
				p.trimNextTextStart = true
				continue
			}
			if final {
				results = p.append(results, false, p.buffer)
				p.buffer = ""
				break
			}
			retained := p.partialTagSuffixLength("</think>")
			results = p.append(results, false, p.buffer[:len(p.buffer)-retained])
			p.buffer = p.buffer[len(p.buffer)-retained:]
			break
		}
		if loc := thinkOpenPattern.FindStringIndex(p.buffer); loc != nil {
			results = p.append(results, true, strings.TrimRight(p.buffer[:loc[0]], " \t\r\n"))
			p.buffer = p.buffer[loc[1]:]
			p.inThoughts = true
			continue
		}
		if final {
			results = p.append(results, true, p.buffer)
			p.buffer = ""
			break
		}
		lower := strings.ToLower(p.buffer)
		retained := 0
		if lastOpen := strings.LastIndex(lower, "<"); lastOpen >= 0 {
			candidate := lower[lastOpen:]
			if strings.HasPrefix("<think", candidate) || (strings.HasPrefix(candidate, "<think") && !strings.Contains(candidate, ">")) {
				retained = len(candidate)
			}
		}
		results = p.append(results, true, p.buffer[:len(p.buffer)-retained])
		p.buffer = p.buffer[len(p.buffer)-retained:]
		break
	}
	return results
}

func (p *thinkStreamProjector) partialTagSuffixLength(tag string) int {
	lower := strings.ToLower(p.buffer)
	for length := min(len(lower), len(tag)-1); length > 0; length-- {
		if strings.HasPrefix(tag, lower[len(lower)-length:]) {
			return length
		}
	}
	return 0
}

func (p *thinkStreamProjector) append(results []CompletionResult, text bool, value string) []CompletionResult {
	if text {
		if p.trimNextTextStart {
			value = strings.TrimLeft(value, " \t\r\n")
			if value != "" {
				p.trimNextTextStart = false
			}
		}
		if strings.TrimSpace(value) == "" {
			return results
		}
		return append(results, newTextResult(value))
	}
	if value == "" {
		return results
	}
	return append(results, newThoughtsResult(value))
}

// FormatChatCompletionsDebugPrompt returns a copy of a Chat Completions prompt
// with its inline image and audio payloads truncated, for
// ExecutionResponse.Prompt.
func FormatChatCompletionsDebugPrompt(prompt any) any {
	switch value := prompt.(type) {
	case ChatCompletionsPrompt:
		value.Messages = truncateDebugItems(value.Messages)
		return value
	case *ChatCompletionsPrompt:
		if value == nil {
			return value
		}
		copied := *value
		copied.Messages = truncateDebugItems(copied.Messages)
		return &copied
	}
	return prompt
}
