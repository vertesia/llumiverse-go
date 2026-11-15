package mistral

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/vertesia/llumiverse-go/capability"
	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
	"github.com/vertesia/llumiverse-go/wire"
)

// DefaultEndpoint is the Mistral API root; the driver sends requests under /v1.
const DefaultEndpoint = "https://api.mistral.ai"

// DefaultEmbeddingModel is the embedding model used when EmbeddingsOptions.Model is empty.
const DefaultEmbeddingModel = "mistral-embed"

// MistralOptions configures a Mistral AI driver.
type MistralOptions struct {
	common.DriverOptions
	// APIKey is the Mistral API key. Required.
	APIKey string
	// Endpoint overrides the API root, as the TS client's endpoint_url does. A trailing /v1 is removed. It
	// defaults to DefaultEndpoint.
	Endpoint string
	// DefaultMaxTokens is sent as max_tokens when the request sets none.
	DefaultMaxTokens int
	// HTTPClient overrides the HTTP client.
	HTTPClient *http.Client
}

// MistralDriver implements Driver for Mistral AI on its native Chat Completions API, which returns reasoning as
// thinking content chunks.
type MistralDriver struct {
	options MistralOptions
	v1      string
	client  *http.Client
	headers map[string]string
}

var versionSuffix = regexp.MustCompile(`/v1/?$`)

// NewMistralDriver creates a Mistral AI driver.
func NewMistralDriver(options MistralOptions) (*MistralDriver, error) {
	if options.APIKey == "" {
		return nil, errors.New("mistral api key is required")
	}
	root := strings.TrimRight(versionSuffix.ReplaceAllString(options.Endpoint, ""), "/")
	if root == "" {
		root = DefaultEndpoint
	}
	client := options.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	return &MistralDriver{
		options: options,
		v1:      root + "/v1",
		client:  core.NewHTTPClient(client, options.HTTPTimeout),
		headers: map[string]string{"Authorization": "Bearer " + options.APIKey},
	}, nil
}

// Provider returns ProviderMistralAI.
func (d *MistralDriver) Provider() common.Provider {
	return common.ProviderMistralAI
}

// Prompt is the Mistral prompt: the new messages of this turn, in Mistral's wire shape.
type Prompt struct {
	Messages []map[string]any `json:"messages"`
}

// CreatePrompt converts prompt segments into Mistral messages: system segments become one system message, a
// result schema an instruction, tool segments tool messages, and files image, audio or text chunks.
func (d *MistralDriver) CreatePrompt(ctx context.Context, segments []common.PromptSegment, options common.ExecutionOptions) (any, error) {
	var messages []map[string]any
	var system []string
	for _, segment := range segments {
		if segment.Role == common.PromptRoleSystem && segment.Content != "" {
			system = append(system, segment.Content)
		}
	}
	if len(system) > 0 {
		messages = append(messages, map[string]any{"role": "system", "content": strings.Join(system, "\n")})
	}
	if options.ResultSchema != nil {
		schema, err := json.Marshal(options.ResultSchema)
		if err != nil {
			return nil, err
		}
		messages = append(messages, map[string]any{"role": "system", "content": "Only answer with JSON matching this schema: " + string(schema)})
	}
	for _, segment := range segments {
		if segment.Role == common.PromptRoleSystem {
			continue
		}
		parts := []any{}
		if segment.Content != "" {
			parts = append(parts, map[string]any{"type": "text", "text": segment.Content})
		}
		for _, file := range segment.Files {
			part, err := filePart(ctx, file)
			if err != nil {
				return nil, err
			}
			if part != nil {
				parts = append(parts, part)
			}
		}
		var content any = parts
		if len(parts) == 1 {
			if part := parts[0].(map[string]any); part["type"] == "text" {
				content = part["text"]
			}
		}
		switch segment.Role {
		case common.PromptRoleTool:
			if segment.ToolUseID == "" {
				return nil, errors.New("mistral tool response requires tool_use_id")
			}
			messages = append(messages, map[string]any{"role": "tool", "tool_call_id": segment.ToolUseID, "content": content})
		case common.PromptRoleAssistant:
			messages = append(messages, map[string]any{"role": "assistant", "content": content})
		default:
			messages = append(messages, map[string]any{"role": "user", "content": content})
		}
	}
	return Prompt{Messages: messages}, nil
}

// filePart converts an attached file to a Mistral content chunk; files other than images, audio and text are
// skipped.
func filePart(ctx context.Context, file common.DataSource) (map[string]any, error) {
	mime := file.MIMEType()
	switch {
	case strings.HasPrefix(mime, "image/"):
		url, err := core.DataSourceToDataURL(ctx, file)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "image_url", "image_url": url}, nil
	case strings.HasPrefix(mime, "audio/"):
		data, err := core.DataSourceToBase64(ctx, file)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "input_audio", "input_audio": data}, nil
	case strings.HasPrefix(mime, "text/"):
		text, err := core.ReadAllStringFromDataSource(ctx, file)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "text", "text": text}, nil
	}
	return nil, nil
}

// FormatDebugPrompt returns a copy of the prompt with inline images and audio truncated.
func (d *MistralDriver) FormatDebugPrompt(prompt any) any {
	value, ok := prompt.(Prompt)
	if !ok {
		return prompt
	}
	messages := make([]map[string]any, len(value.Messages))
	for i, message := range value.Messages {
		copied := make(map[string]any, len(message))
		for key, item := range message {
			copied[key] = item
		}
		if parts, ok := message["content"].([]any); ok {
			truncated := make([]any, len(parts))
			for j, raw := range parts {
				truncated[j] = raw
				part, ok := raw.(map[string]any)
				if !ok {
					continue
				}
				switch part["type"] {
				case "image_url":
					if url, ok := part["image_url"].(string); ok {
						truncated[j] = map[string]any{"type": "image_url", "image_url": core.TruncateDataURLForDebug(url)}
					}
				case "input_audio":
					if data, ok := part["input_audio"].(string); ok {
						truncated[j] = map[string]any{"type": "input_audio", "input_audio": core.TruncateBinaryForDebug(data)}
					}
				}
			}
			copied["content"] = truncated
		}
		messages[i] = copied
	}
	return Prompt{Messages: messages}
}

// Execute runs a non-streaming completion.
func (d *MistralDriver) Execute(ctx context.Context, segments []common.PromptSegment, options common.ExecutionOptions) (*common.ExecutionResponse, error) {
	return core.ExecuteWithPrompt(ctx, d, segments, options, d.requestCompletion)
}

// Stream runs a streaming completion.
func (d *MistralDriver) Stream(ctx context.Context, segments []common.PromptSegment, options common.ExecutionOptions) (common.CompletionStream, error) {
	stream, err := core.StreamWithPrompt(ctx, d, segments, options, d.requestCompletionStream)
	if err != nil {
		return nil, err
	}
	return core.LeaseCompletionStream(stream, d.options.StreamStartTimeout), nil
}

// buildRequest assembles the chat request body. Mistral serves models with different effort sets (GLM's
// low/high/max for example), so effort is passed through for the provider to validate.
func (d *MistralDriver) buildRequest(conversation Conversation, options common.ExecutionOptions, stream bool) map[string]any {
	mo := options.ModelOptions
	body := map[string]any{
		"model":    options.Model,
		"messages": conversation.Messages,
		"n":        1,
		"stream":   stream,
	}
	if v := core.OptionInt(mo, "max_tokens"); v != nil {
		body["max_tokens"] = *v
	} else if d.options.DefaultMaxTokens > 0 {
		body["max_tokens"] = d.options.DefaultMaxTokens
	}
	for option, field := range map[string]string{
		"temperature":       "temperature",
		"top_p":             "top_p",
		"presence_penalty":  "presence_penalty",
		"frequency_penalty": "frequency_penalty",
		"random_seed":       "random_seed",
	} {
		if v := core.OptionFloat(mo, option); v != nil {
			if float64(int64(*v)) == *v {
				body[field] = int64(*v)
			} else {
				body[field] = *v
			}
		}
	}
	if stop := core.OptionStringSlice(mo, "stop_sequence"); len(stop) > 0 {
		body["stop"] = stop
	}
	for _, key := range []string{"safe_prompt", "parallel_tool_calls"} {
		if value, ok := mo[key].(bool); ok {
			body[key] = value
		}
	}
	if mode := core.OptionString(mo, "prompt_mode"); mode != "" {
		body["prompt_mode"] = mode
	}
	if effort := core.OptionString(mo, "effort"); effort != "" {
		body["reasoning_effort"] = effort
	}
	if name := core.OptionString(mo, "required_tool_name"); name != "" {
		body["tool_choice"] = map[string]any{"type": "function", "function": map[string]any{"name": name}}
	} else if choice, ok := mo["tool_choice"]; ok && choice != nil && choice != "" {
		body["tool_choice"] = choice
	}
	if options.PromptCacheKey != "" {
		body["prompt_cache_key"] = options.PromptCacheKey
	}
	if len(options.Tools) > 0 {
		tools := make([]any, len(options.Tools))
		for i, tool := range options.Tools {
			function := map[string]any{"name": tool.Name, "parameters": map[string]any{}}
			if tool.Description != nil {
				function["description"] = *tool.Description
			}
			if tool.InputSchema != nil {
				function["parameters"] = tool.InputSchema
			}
			tools[i] = map[string]any{"type": "function", "function": function}
		}
		body["tools"] = tools
	}
	return body
}

// chatResponse decodes a chat completion and a stream chunk.
type chatResponse struct {
	Choices []struct {
		FinishReason string         `json:"finish_reason"`
		Message      map[string]any `json:"message"`
		Delta        map[string]any `json:"delta"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     *int `json:"prompt_tokens"`
		CompletionTokens *int `json:"completion_tokens"`
		TotalTokens      *int `json:"total_tokens"`
	} `json:"usage"`
}

func (r chatResponse) tokenUsage() *common.ExecutionTokenUsage {
	if r.Usage == nil {
		return nil
	}
	count := func(value *int) *float64 {
		n := 0.0
		if value != nil {
			n = float64(*value)
		}
		return &n
	}
	return &common.ExecutionTokenUsage{Prompt: count(r.Usage.PromptTokens), Result: count(r.Usage.CompletionTokens), Total: count(r.Usage.TotalTokens)}
}

func (d *MistralDriver) requestCompletion(ctx context.Context, prompt any, options common.ExecutionOptions) (*common.Completion, error) {
	conversation, err := requestConversation(options.Conversation, prompt)
	if err != nil {
		return nil, err
	}
	var raw json.RawMessage
	if err := core.DoJSON(ctx, d.requestClient(options.HTTPTimeout), http.MethodPost, d.v1+"/chat/completions", d.headers, d.buildRequest(conversation, options, false), &raw); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	var response chatResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, err
	}
	if len(response.Choices) == 0 || response.Choices[0].Message == nil {
		return nil, errors.New("mistral response is not valid: no assistant message")
	}
	choice := response.Choices[0]
	message := choice.Message
	message["role"] = "assistant"
	toolUse := toolUses(message["tool_calls"])
	finishReason := choice.FinishReason
	if len(toolUse) > 0 {
		finishReason = "tool_use"
	}
	completion := &common.Completion{
		Result:       projectContent(message["content"], includeThoughts(options)),
		ToolUse:      toolUse,
		TokenUsage:   response.tokenUsage(),
		FinishReason: finishReason,
	}
	if options.IncludeOriginalResponse {
		var original any
		_ = json.Unmarshal(raw, &original)
		completion.OriginalResponse = original
	}
	conversation.Messages = append(conversation.Messages, message)
	completion.Conversation = finalizeConversation(conversation, options)
	return completion, nil
}

func (d *MistralDriver) requestCompletionStream(ctx context.Context, prompt any, options common.ExecutionOptions) (common.CompletionStream, error) {
	conversation, err := requestConversation(options.Conversation, prompt)
	if err != nil {
		return nil, err
	}
	body, err := core.PostSSE(ctx, d.requestClient(options.HTTPTimeout), d.v1+"/chat/completions", d.headers, d.buildRequest(conversation, options, true))
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	state := &streamState{includeThoughts: includeThoughts(options), calls: map[int]map[string]any{}}
	ch := make(chan core.StreamItem)
	go func() {
		defer close(ch)
		err := core.ScanSSE(body, func(event core.SSEEvent) error {
			if event.Data == "" || event.Data == "[DONE]" {
				return nil
			}
			var chunk chatResponse
			if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
				return err
			}
			out, ok := state.chunk(chunk)
			if !ok {
				return nil
			}
			select {
			case ch <- core.StreamItem{Chunk: out}:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		if err != nil {
			select {
			case ch <- core.StreamItem{Err: d.formatError(err, options.Model, "stream")}:
			case <-ctx.Done():
			}
		}
	}()
	return core.NewChannelStreamWithFinalizer(ch, &common.ExecutionResponse{Prompt: prompt}, body.Close, func(resp *common.ExecutionResponse) {
		state.finalize(resp)
		conversation.Messages = append(conversation.Messages, state.assistantMessage())
		resp.Conversation = finalizeConversation(conversation, options)
	}), nil
}

func (d *MistralDriver) requestClient(timeout common.HTTPTimeoutOptions) *http.Client {
	if !core.HasHTTPTimeout(timeout) {
		return d.client
	}
	return core.NewHTTPClient(d.client, timeout)
}

// includeThoughts reports whether thinking chunks become thoughts results; Mistral returns them unless
// include_thoughts is false.
func includeThoughts(options common.ExecutionOptions) bool {
	value, ok := options.ModelOptions["include_thoughts"].(bool)
	return !ok || value
}

// projectContent turns message content (a string or content chunks) into results: text chunks as text, and the
// text of thinking chunks as thoughts when includeThoughts is set.
func projectContent(content any, includeThoughts bool) []common.CompletionResult {
	results := []common.CompletionResult{}
	switch value := content.(type) {
	case string:
		if value != "" {
			results = append(results, wire.NewTextResult(value))
		}
	case []any:
		for _, raw := range value {
			part, _ := raw.(map[string]any)
			switch part["type"] {
			case "text":
				if text, _ := part["text"].(string); text != "" {
					results = append(results, wire.NewTextResult(text))
				}
			case "thinking":
				if !includeThoughts {
					continue
				}
				thoughts, _ := part["thinking"].([]any)
				for _, rawThought := range thoughts {
					thought, _ := rawThought.(map[string]any)
					if text, _ := thought["text"].(string); thought["type"] == "text" && text != "" {
						results = append(results, wire.NewThoughtsResult(text))
					}
				}
			}
		}
	}
	return results
}

// toolUses converts the tool calls of a message; arguments arrive as a JSON string or an object.
func toolUses(raw any) []common.ToolUse {
	calls, _ := raw.([]any)
	var out []common.ToolUse
	for _, item := range calls {
		call, _ := item.(map[string]any)
		function, _ := call["function"].(map[string]any)
		if function == nil {
			continue
		}
		id, _ := call["id"].(string)
		name, _ := function["name"].(string)
		input := function["arguments"]
		if text, ok := input.(string); ok {
			input = core.SafeJSONParse(text)
		}
		if _, ok := input.(map[string]any); !ok {
			input = map[string]any{}
		}
		out = append(out, common.ToolUse{ID: id, ToolName: name, ToolInput: common.ToolInputObject(input)})
	}
	return out
}

// streamState projects stream chunks and accumulates the native assistant message for the conversation.
type streamState struct {
	includeThoughts bool
	content         []any
	calls           map[int]map[string]any
}

func (s *streamState) chunk(response chatResponse) (common.CompletionChunk, bool) {
	out := common.CompletionChunk{TokenUsage: response.tokenUsage()}
	if len(response.Choices) == 0 || response.Choices[0].Delta == nil {
		return out, out.TokenUsage != nil
	}
	choice := response.Choices[0]
	content := deltaContent(choice.Delta["content"])
	s.appendContent(content)
	out.Result = projectContent(content, s.includeThoughts)
	calls, _ := choice.Delta["tool_calls"].([]any)
	for offset, raw := range calls {
		call, _ := raw.(map[string]any)
		function, _ := call["function"].(map[string]any)
		index := offset
		if value, ok := call["index"].(float64); ok {
			index = int(value)
		}
		native := s.calls[index]
		if native == nil {
			native = map[string]any{"id": "", "index": index, "type": "function", "function": map[string]any{"name": "", "arguments": ""}}
			s.calls[index] = native
		}
		nativeFunction := native["function"].(map[string]any)
		if id, _ := call["id"].(string); id != "" {
			native["id"] = id
		}
		name, _ := function["name"].(string)
		nativeFunction["name"] = nativeFunction["name"].(string) + name
		var input any
		switch args := function["arguments"].(type) {
		case string:
			if previous, ok := nativeFunction["arguments"].(string); ok {
				nativeFunction["arguments"] = previous + args
			} else {
				nativeFunction["arguments"] = args
			}
			input = args
		case map[string]any:
			nativeFunction["arguments"] = args
			input = args
		}
		out.ToolUse = append(out.ToolUse, common.ToolUseChunk{ID: "tool_" + strconv.Itoa(index), ToolName: name, ToolInput: input})
	}
	out.FinishReason = choice.FinishReason
	if len(out.ToolUse) > 0 {
		out.FinishReason = "tool_use"
	}
	return out, len(out.Result) > 0 || len(out.ToolUse) > 0 || out.FinishReason != "" || out.TokenUsage != nil
}

// deltaContent returns delta content as chunks: a string becomes one text chunk.
func deltaContent(content any) []any {
	switch value := content.(type) {
	case string:
		if value == "" {
			return nil
		}
		return []any{map[string]any{"type": "text", "text": value}}
	case []any:
		return value
	}
	return nil
}

// appendContent merges streamed chunks into the native content: consecutive text chunks join, and thinking
// continues an open thinking chunk.
func (s *streamState) appendContent(incoming []any) {
	for _, raw := range incoming {
		part, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		var previous map[string]any
		if len(s.content) > 0 {
			previous, _ = s.content[len(s.content)-1].(map[string]any)
		}
		switch {
		case part["type"] == "text" && previous != nil && previous["type"] == "text":
			text, _ := part["text"].(string)
			previous["text"] = previous["text"].(string) + text
		case part["type"] == "thinking" && previous != nil && previous["type"] == "thinking" && previous["closed"] != true:
			thoughts, _ := part["thinking"].([]any)
			existing, _ := previous["thinking"].([]any)
			previous["thinking"] = append(existing, cloneValue(thoughts).([]any)...)
			if signature, ok := part["signature"]; ok {
				previous["signature"] = signature
			}
			if closed, ok := part["closed"]; ok {
				previous["closed"] = closed
			}
		default:
			copied := cloneValue(part).(map[string]any)
			if copied["type"] == "text" {
				if _, ok := copied["text"].(string); !ok {
					copied["text"] = ""
				}
			}
			s.content = append(s.content, copied)
		}
	}
}

// finalize replaces the stream's positional tool IDs with Mistral's IDs.
func (s *streamState) finalize(resp *common.ExecutionResponse) {
	for i, tool := range resp.ToolUse {
		index, err := strconv.Atoi(strings.TrimPrefix(tool.ID, "tool_"))
		if err != nil || !strings.HasPrefix(tool.ID, "tool_") {
			continue
		}
		if native := s.calls[index]; native != nil {
			if id, _ := native["id"].(string); id != "" {
				resp.ToolUse[i].ID = id
			}
		}
	}
}

// assistantMessage returns the streamed assistant message in native form.
func (s *streamState) assistantMessage() map[string]any {
	message := map[string]any{"role": "assistant", "content": s.content}
	if s.content == nil {
		message["content"] = []any{}
	}
	if len(s.calls) > 0 {
		indexes := make([]int, 0, len(s.calls))
		for index := range s.calls {
			indexes = append(indexes, index)
		}
		sort.Ints(indexes)
		calls := make([]any, len(indexes))
		for i, index := range indexes {
			calls[i] = s.calls[index]
		}
		message["tool_calls"] = calls
	}
	return message
}

type mistralModel struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	Description  string `json:"description"`
	OwnedBy      string `json:"owned_by"`
	Capabilities *struct {
		CompletionChat *bool `json:"completion_chat"`
	} `json:"capabilities"`
}

// ListModels lists the chat models. Models the API marks as unable to run chat completions are left out, and
// those without capability metadata kept.
func (d *MistralDriver) ListModels(ctx context.Context, _ *common.ModelSearchPayload) ([]common.AIModel, error) {
	var response struct {
		Data []mistralModel `json:"data"`
	}
	if err := core.DoJSON(ctx, d.client, http.MethodGet, d.v1+"/models", d.headers, nil, &response); err != nil {
		return nil, d.formatError(err, "", "listModels")
	}
	models := make([]common.AIModel, 0, len(response.Data))
	for _, model := range response.Data {
		if model.ID == "" || capability.IsEmbeddingModel(capability.Listing{ID: model.ID}, common.ProviderMistralAI) {
			continue
		}
		if model.Capabilities != nil && model.Capabilities.CompletionChat != nil && !*model.Capabilities.CompletionChat {
			continue
		}
		name := model.Name
		if name == "" {
			name = model.ID
		}
		owner := model.OwnedBy
		aiModel := common.AIModel{
			ID:        model.ID,
			Name:      name,
			Provider:  string(common.ProviderMistralAI),
			Owner:     &owner,
			Type:      ptr(wire.ModelTypeText),
			CanStream: ptr(true),
		}
		if model.Description != "" {
			aiModel.Description = ptr(model.Description)
		}
		capability.ForModel(model.ID, common.ProviderMistralAI).Apply(&aiModel)
		models = append(models, aiModel)
	}
	return models, nil
}

// ValidateConnection checks the API key against the model listing.
func (d *MistralDriver) ValidateConnection(ctx context.Context) error {
	if err := core.DoJSON(ctx, d.client, http.MethodGet, d.v1+"/models", d.headers, nil, nil); err != nil {
		return d.formatError(err, "", "validateConnection")
	}
	return nil
}

// GenerateEmbeddings embeds text inputs, with DefaultEmbeddingModel unless options name a model.
func (d *MistralDriver) GenerateEmbeddings(ctx context.Context, options common.EmbeddingsOptions) (*common.EmbeddingsResult, error) {
	options, err := core.NormalizeEmbeddingsOptions(options)
	if err != nil {
		return nil, err
	}
	model := options.Model
	if model == "" {
		model = DefaultEmbeddingModel
	}
	texts := make([]string, len(options.Inputs))
	for i, input := range options.Inputs {
		if input.Type != "" && input.Type != "text" {
			return nil, fmt.Errorf("provider 'mistralai' does not support '%s' embeddings; only 'text' is supported", input.Type)
		}
		texts[i] = input.Text
	}
	var response struct {
		Data []struct {
			Index     *int      `json:"index"`
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
		Usage *struct {
			PromptTokens *float64 `json:"prompt_tokens"`
			TotalTokens  *float64 `json:"total_tokens"`
		} `json:"usage"`
	}
	body := map[string]any{"model": model, "input": texts, "encoding_format": "float"}
	if err := core.DoJSON(ctx, d.requestClient(options.HTTPTimeout), http.MethodPost, d.v1+"/embeddings", d.headers, body, &response); err != nil {
		return nil, d.formatError(err, model, "execute")
	}
	sort.SliceStable(response.Data, func(i, j int) bool {
		return indexOf(response.Data[i].Index, i) < indexOf(response.Data[j].Index, j)
	})
	results := make([]common.EmbeddingResultItem, len(response.Data))
	text := wire.EmbeddingOutputModality("text")
	for i, entry := range response.Data {
		if len(entry.Embedding) == 0 {
			return nil, fmt.Errorf("mistral embedding empty for input index %d", indexOf(entry.Index, i))
		}
		results[i] = common.EmbeddingResultItem{Outputs: []common.EmbeddingOutput{{Values: entry.Embedding, Modality: &text}}}
	}
	result := &common.EmbeddingsResult{Model: model, Results: results}
	if response.Usage != nil {
		tokens := response.Usage.PromptTokens
		if tokens == nil {
			tokens = response.Usage.TotalTokens
		}
		if tokens != nil {
			result.Usage = &common.EmbeddingsTokenUsage{InputTokens: ptr(*tokens), InputTextTokens: ptr(*tokens)}
		}
	}
	return result, nil
}

// formatError wraps a transport or HTTP error as a LlumiverseError with its status and retryability: a 4xx
// response (an invalid request) is not retryable, rate limits and 5xx responses are.
func (d *MistralDriver) formatError(err error, model, operation string) error {
	var llumiErr *common.LlumiverseError
	if err == nil || errors.As(err, &llumiErr) {
		return err
	}
	code, name := core.ErrorStatusAndName(err)
	if code != 0 {
		name = "MistralError"
	}
	message := err.Error()
	if errors.Is(err, io.ErrUnexpectedEOF) {
		message = "mistral stream ended unexpectedly: " + message
	}
	return common.NewLlumiverseError(fmt.Sprintf("[%s] %s", common.ProviderMistralAI, message), core.RetryableFromStatusAndMessage(code, message), common.LlumiverseErrorContext{
		Provider:  common.ProviderMistralAI,
		Model:     model,
		Operation: operation,
	}, err, code, name)
}

func indexOf(index *int, fallback int) int {
	if index != nil {
		return *index
	}
	return fallback
}

func ptr[T any](v T) *T { return &v }

var _ common.Driver = (*MistralDriver)(nil)
