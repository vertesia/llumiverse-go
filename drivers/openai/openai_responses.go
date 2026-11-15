package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"github.com/vertesia/llumiverse-go/core"
)

// responseItem is one Responses API input item in its JSON shape. Items are kept
// as JSON objects, as the TS client keeps them, so that output items the request
// types do not model (reasoning, compaction, image generation calls) round-trip
// unchanged through the conversation.
type responseItem = map[string]any

// openAIResponsesPrompt is the prepared Responses API input: the ordered input
// items plus a flattened Text concatenation used for image-generation prompts.
type openAIResponsesPrompt struct {
	Items []responseItem
	Text  string
}

// createResponsesPrompt builds the ordered Responses input items from prompt
// segments. System and safety material is grouped first in the instruction role
// chosen for the model (safety text emphasized), followed by user/assistant/tool
// items in order; the concatenated Text is retained for image models. Tool
// segments require a ToolUseID.
func (d *OpenAICompatibleDriver) createResponsesPrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (openAIResponsesPrompt, error) {
	prompt := openAIResponsesPrompt{}
	var system []responseItem
	var others []responseItem
	systemRole := openAIResponsesSystemRole(options.Model)
	for _, segment := range segments {
		switch segment.Role {
		case PromptRoleNegative, PromptRoleMask:
			continue
		case PromptRoleTool:
			if segment.ToolUseID == "" {
				return openAIResponsesPrompt{}, errors.New("tool prompt segment requires ToolUseID")
			}
			files, err := openAIResponseFileParts(ctx, segment.Files, options)
			if err != nil {
				return openAIResponsesPrompt{}, err
			}
			var output any = segment.Content
			if len(files) > 0 {
				// Tool-result attachments travel in the output itself.
				var parts []any
				if segment.Content != "" {
					parts = append(parts, map[string]any{"type": "input_text", "text": segment.Content})
				}
				for _, part := range files {
					parts = append(parts, part)
				}
				output = parts
			}
			others = append(others, responseItem{"type": "function_call_output", "call_id": segment.ToolUseID, "output": output})
			continue
		}
		parts, err := openAIResponseContentParts(ctx, segment, options)
		if err != nil {
			return openAIResponsesPrompt{}, err
		}
		if segment.Content != "" {
			if prompt.Text != "" {
				prompt.Text += "\n"
			}
			prompt.Text += segment.Content
		}
		if len(parts) == 0 {
			continue
		}
		switch segment.Role {
		case PromptRoleSystem:
			texts := inputTextParts(parts)
			if len(texts) == 1 && len(segment.Files) == 0 {
				system = append(system, responseItem{"type": "message", "role": systemRole, "content": texts[0]["text"]})
			} else {
				system = append(system, responseMessage(systemRole, texts))
			}
		case PromptRoleSafety:
			safety := inputTextParts(parts)
			for _, part := range safety {
				part["text"] = "DO NOT IGNORE - IMPORTANT: " + stringValue(part["text"])
			}
			if len(safety) > 0 {
				system = append(system, responseMessage(systemRole, safety))
			}
		case PromptRoleAssistant:
			others = append(others, responseMessage("assistant", parts))
		default:
			others = append(others, responseMessage("user", parts))
		}
	}
	if options.ResultSchema != nil {
		schemaPart := []responseItem{{"type": "input_text", "text": openAISchemaInstruction(options.ResultSchema)}}
		switch {
		case options.PromptCacheSchemaSuffix:
			// After the cached prefix, so changing the schema keeps the prompt cache.
			others = append(others, responseMessage("user", schemaPart))
		case !d.supportsSchema(options.Model):
			// Without native structured output, the schema goes into the instructions.
			system = append(system, responseMessage(systemRole, schemaPart))
		}
	}
	prompt.Items = append(prompt.Items, system...)
	prompt.Items = append(prompt.Items, others...)
	return prompt, nil
}

// supportsSchema reports whether the model takes native structured output:
// realtime models and models without tool support do not.
func (d *OpenAICompatibleDriver) supportsSchema(model string) bool {
	return !strings.Contains(strings.ToLower(model), "realtime") && !isLikelyNonToolOpenAIModel(model)
}

// openAISchemaInstruction is the prompt instruction that carries a result schema.
func openAISchemaInstruction(schema map[string]any) string {
	data, _ := json.Marshal(schema)
	return fmt.Sprintf(chatCompletionsSchemaInstruction, data)
}

// responseMessage builds an input message item from content parts.
func responseMessage(role string, parts []responseItem) responseItem {
	content := make([]any, 0, len(parts))
	for _, part := range parts {
		content = append(content, part)
	}
	return responseItem{"type": "message", "role": role, "content": content}
}

// inputTextParts keeps only the input_text parts, which is all system messages accept.
func inputTextParts(parts []responseItem) []responseItem {
	out := make([]responseItem, 0, len(parts))
	for _, part := range parts {
		if part["type"] == "input_text" {
			out = append(out, part)
		}
	}
	return out
}

// openAIResponseContentParts builds the input content parts for a segment: a
// part per supported file, then an input_text part for its content.
func openAIResponseContentParts(ctx context.Context, segment PromptSegment, options ExecutionOptions) ([]responseItem, error) {
	parts, err := openAIResponseFileParts(ctx, segment.Files, options)
	if err != nil {
		return nil, err
	}
	if segment.Content != "" {
		parts = append(parts, responseItem{"type": "input_text", "text": segment.Content})
	}
	return parts, nil
}

// openAIImageDataURL inlines an image, assuming JPEG when the file has no type.
func openAIImageDataURL(ctx context.Context, file DataSource) (string, error) {
	if file.MIMEType() != "" {
		return dataSourceToDataURL(ctx, file)
	}
	data, err := core.DataSourceToBase64(ctx, file)
	if err != nil {
		return "", err
	}
	return "data:image/jpeg;base64," + data, nil
}

// openAIResponseFileParts converts files to Responses content parts: images
// (or files without a type) become input_image, PDFs input_file, and text files
// input_text with their content. Other types (audio, video) are skipped.
func openAIResponseFileParts(ctx context.Context, files []DataSource, options ExecutionOptions) ([]responseItem, error) {
	parts := make([]responseItem, 0, len(files))
	for _, file := range files {
		mime := file.MIMEType()
		switch {
		case mime == "" || strings.HasPrefix(mime, "image/"):
			dataURL, err := openAIImageDataURL(ctx, file)
			if err != nil {
				return nil, err
			}
			parts = append(parts, responseItem{"type": "input_image", "image_url": dataURL, "detail": openAIImageDetail(options)})
		case mime == "application/pdf":
			dataURL, err := dataSourceToDataURL(ctx, file)
			if err != nil {
				return nil, err
			}
			parts = append(parts, responseItem{"type": "input_file", "filename": file.Name(), "file_data": dataURL})
		case strings.HasPrefix(mime, "text/"):
			text, err := dataSourceText(ctx, file)
			if err != nil {
				return nil, err
			}
			parts = append(parts, responseItem{"type": "input_text", "text": text})
		}
	}
	return parts, nil
}

// requestResponsesCompletion runs a non-streaming Responses request, routing
// image models to the Images API, and returns the completion with the updated
// conversation state appended.
func (d *OpenAICompatibleDriver) requestResponsesCompletion(ctx context.Context, prompt any, options ExecutionOptions) (*Completion, error) {
	rp := prompt.(openAIResponsesPrompt)
	if isOpenAIImageModel(options.Model) {
		return d.requestOpenAIImageGeneration(ctx, rp, options)
	}
	input := openAIResponsesInput(options.Conversation, rp.Items, len(options.Tools) > 0)
	payload := d.responsePayload(input, options)
	var raw json.RawMessage
	if err := d.sdk.Post(ctx, "responses", payload, &raw, d.requestOptions(options.HTTPTimeout)...); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	var response responsesResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, err
	}
	completion := extractOpenAIResponseCompletion(&response, openAIIncludeThoughts(options))
	if options.IncludeOriginalResponse {
		var original any
		_ = json.Unmarshal(raw, &original)
		completion.OriginalResponse = original
	}
	completion.Conversation = buildOpenAIResponsesConversation(input, rawOutputItems(response.Output), completion, options)
	return completion, nil
}

// requestResponsesCompletionStream opens a Responses stream, driving the events
// through openAIResponsesStreamState and finalizing the conversation state once
// the stream completes.
func (d *OpenAICompatibleDriver) requestResponsesCompletionStream(ctx context.Context, prompt any, options ExecutionOptions) (CompletionStream, error) {
	rp := prompt.(openAIResponsesPrompt)
	// Image generation models use the Images API path and do not stream through
	// Responses; callers should use Execute for those models.
	input := openAIResponsesInput(options.Conversation, rp.Items, len(options.Tools) > 0)
	payload := d.responsePayload(input, options)
	payload["stream"] = true
	var res *http.Response
	if err := d.sdk.Post(ctx, "responses", payload, &res, d.requestOptions(options.HTTPTimeout)...); err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	ch := make(chan streamItem)
	completion := &ExecutionResponse{Prompt: prompt}
	state := &openAIResponsesStreamState{includeThoughts: openAIIncludeThoughts(options)}
	go func() {
		defer close(ch)
		state.tools = map[string]openAIResponsesStreamTool{}
		err := core.ScanSSE(res.Body, func(event core.SSEEvent) error {
			if event.Data == "" || event.Data == "[DONE]" {
				return nil
			}
			var decoded responsesStreamEvent
			if err := json.Unmarshal([]byte(event.Data), &decoded); err != nil {
				return err
			}
			out, err := state.chunk(decoded)
			if err != nil {
				return err
			}
			if len(out.Result) > 0 || len(out.ToolUse) > 0 || out.FinishReason != "" || out.TokenUsage != nil {
				select {
				case ch <- streamItem{Chunk: out}:
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			return nil
		})
		if err != nil {
			select {
			case ch <- streamItem{Err: d.formatError(err, options.Model, "stream")}:
			case <-ctx.Done():
			}
		}
	}()
	return newChannelStreamWithFinalizer(ch, completion, res.Body.Close, func(resp *ExecutionResponse) {
		var output []responseItem
		if state.final != nil {
			output = rawOutputItems(state.final.Output)
		}
		resp.Conversation = buildOpenAIResponsesConversation(input, output, &resp.Completion, options)
	}), nil
}

// responsePayload assembles the Responses request. Temperature and top_p are
// only sent for non-reasoning models; reasoning effort is mapped per-model;
// tools are omitted for likely non-tool models; and a ResultSchema becomes a
// json_schema text format (strict when convertible), skipped for realtime models.
func (d *OpenAICompatibleDriver) responsePayload(input []responseItem, options ExecutionOptions) map[string]any {
	payload := map[string]any{
		"model": options.Model,
		"input": input,
	}
	if key := openAIPromptCacheKey(options); key != "" {
		payload["prompt_cache_key"] = key
		marked, cacheOptions := configureOpenAIPromptCaching(input, options.Model, key)
		payload["input"] = marked
		if cacheOptions != nil {
			payload["prompt_cache_options"] = cacheOptions
		}
	}
	if v := optionInt(options.ModelOptions, "max_tokens"); v != nil {
		payload["max_output_tokens"] = *v
	}
	if !isOpenAIReasoningModel(options.Model) {
		if v := optionFloat(options.ModelOptions, "temperature"); v != nil {
			payload["temperature"] = *v
		}
		if v := optionFloat(options.ModelOptions, "top_p"); v != nil {
			payload["top_p"] = *v
		}
	}
	if reasoning := openAIReasoning(options); reasoning != nil {
		payload["reasoning"] = reasoning
		// Encrypted reasoning items let the next turn continue the same reasoning.
		payload["include"] = []string{"reasoning.encrypted_content"}
	}
	// Provider tiers are passed through so values newer than this client still work.
	if tier := optionString(options.ModelOptions, "service_tier"); tier != "" {
		payload["service_tier"] = tier
	}
	if len(options.Labels) > 0 {
		payload["metadata"] = options.Labels
	}
	if len(options.Tools) > 0 && !isLikelyNonToolOpenAIModel(options.Model) {
		payload["tools"] = openAIResponseTools(options.Tools)
	}
	if text := d.openAIResponseTextConfig(options); text != nil {
		payload["text"] = text
	}
	return payload
}

// openAIResponseTextConfig builds the text config: a json_schema format for
// native structured output (strict when convertible), json_object when the
// schema travels as a prompt suffix, and the verbosity option.
func (d *OpenAICompatibleDriver) openAIResponseTextConfig(options ExecutionOptions) map[string]any {
	text := map[string]any{}
	switch {
	case options.ResultSchema != nil && options.PromptCacheSchemaSuffix:
		text["format"] = map[string]any{"type": "json_object"}
	case options.ResultSchema != nil && d.supportsSchema(options.Model):
		schema, strict := openAISchemaForResponses(options.ResultSchema)
		text["format"] = map[string]any{
			"type":   "json_schema",
			"name":   "format_output",
			"schema": schema,
			"strict": strict,
		}
	}
	if verbosity := optionString(options.ModelOptions, "verbosity"); verbosity != "" {
		text["verbosity"] = verbosity
	}
	if len(text) == 0 {
		return nil
	}
	return text
}

// openAIResponseTools converts ToolDefinitions into Responses function tools,
// applying strict schema mode when the input schema can be converted.
func openAIResponseTools(tools []ToolDefinition) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		schema, strict := openAISchemaForResponses(tool.InputSchema)
		fn := map[string]any{
			"type":       "function",
			"name":       tool.Name,
			"parameters": schema,
			"strict":     strict,
		}
		if tool.Description != nil && *tool.Description != "" {
			fn["description"] = *tool.Description
		}
		out = append(out, fn)
	}
	return out
}

// responsesResponse is the part of a Responses API response the driver reads.
type responsesResponse struct {
	ID                string                      `json:"id"`
	Status            string                      `json:"status"`
	Error             *responsesError             `json:"error"`
	IncompleteDetails *responsesIncompleteDetails `json:"incomplete_details"`
	Output            []responsesOutputItem       `json:"output"`
	Usage             *responsesUsage             `json:"usage"`
	ServiceTier       string                      `json:"service_tier"`
}

// responsesError is the error of a failed response or of an error stream event.
type responsesError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// responsesIncompleteDetails says why a response is incomplete.
type responsesIncompleteDetails struct {
	Reason string `json:"reason"`
}

// responsesOutputItem is one output item, with its raw JSON kept for the conversation.
type responsesOutputItem struct {
	Type         string                   `json:"type"`
	ID           string                   `json:"id"`
	CallID       string                   `json:"call_id"`
	Name         string                   `json:"name"`
	Arguments    string                   `json:"arguments"`
	Status       string                   `json:"status"`
	Result       string                   `json:"result"`
	OutputFormat string                   `json:"output_format"`
	Content      []responsesOutputContent `json:"content"`
	Summary      []responsesOutputContent `json:"summary"`
	Raw          responseItem             `json:"-"`
}

// UnmarshalJSON decodes the typed fields and keeps the raw item.
func (item *responsesOutputItem) UnmarshalJSON(data []byte) error {
	type plain responsesOutputItem
	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	var raw responseItem
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*item = responsesOutputItem(decoded)
	item.Raw = raw
	return nil
}

// rawOutputItems returns the output items in their JSON shape.
func rawOutputItems(output []responsesOutputItem) []responseItem {
	items := make([]responseItem, 0, len(output))
	for _, item := range output {
		if item.Raw != nil {
			items = append(items, item.Raw)
		}
	}
	return items
}

// responsesOutputContent is one content part of an output message or reasoning summary.
type responsesOutputContent struct {
	Type    string `json:"type"`
	Text    string `json:"text"`
	Refusal string `json:"refusal"`
}

// responsesUsage is the usage block of a Responses API response, plus the fields
// some OpenAI-compatible Responses endpoints report instead (prompt_tokens_details,
// cached_tokens).
type responsesUsage struct {
	InputTokens         int64                    `json:"input_tokens"`
	OutputTokens        int64                    `json:"output_tokens"`
	TotalTokens         int64                    `json:"total_tokens"`
	InputTokensDetails  *responsesInputTokenInfo `json:"input_tokens_details"`
	PromptTokensDetails *responsesInputTokenInfo `json:"prompt_tokens_details"`
	CachedTokens        *int64                   `json:"cached_tokens"`
	CacheWriteTokens    *int64                   `json:"cache_write_tokens"`
}

// responsesInputTokenInfo details the input tokens of a response.
type responsesInputTokenInfo struct {
	CachedTokens     *int64 `json:"cached_tokens"`
	CacheWriteTokens *int64 `json:"cache_write_tokens"`
}

// extractOpenAIResponseCompletion maps a Responses output into a Completion,
// collecting text/image results, tool uses, usage, and a normalized finish reason.
func extractOpenAIResponseCompletion(response *responsesResponse, includeThoughts bool) *Completion {
	if response == nil {
		return &Completion{Result: []CompletionResult{newTextResult("")}}
	}
	result := openAIResponseResults(response.Output, includeThoughts)
	tools := openAIResponseToolUses(response.Output)
	completion := &Completion{
		Result:       result,
		TokenUsage:   openAIResponseUsage(response.Usage),
		ToolUse:      tools,
		FinishReason: openAIResponseFinishReason(response, len(tools) > 0),
		ServiceTier:  response.ServiceTier,
	}
	if len(completion.Result) == 0 && len(completion.ToolUse) == 0 {
		completion.Result = []CompletionResult{newTextResult("")}
	}
	return completion
}

// openAIResponseResults extracts reasoning summaries (as thoughts), text and
// generated-image results from Responses output items, in order, wrapping bare
// base64 image data in a PNG data URL.
func openAIResponseResults(output []responsesOutputItem, includeThoughts bool) []CompletionResult {
	results := []CompletionResult{}
	for _, item := range output {
		switch item.Type {
		case "reasoning":
			if !includeThoughts {
				continue
			}
			for _, summary := range item.Summary {
				if summary.Type == "summary_text" && summary.Text != "" {
					results = append(results, newThoughtsResult(summary.Text))
				}
			}
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" && part.Text != "" {
					results = append(results, newTextResult(part.Text))
				}
			}
		case "image_generation_call":
			if item.Result != "" {
				value := item.Result
				if !strings.HasPrefix(value, "data:") {
					value = "data:image/png;base64," + value
				}
				results = append(results, newImageResult(value))
			}
		}
	}
	return results
}

// openAIResponseToolUses extracts function_call items as ToolUses, preferring
// CallID over ID and parsing the arguments JSON.
func openAIResponseToolUses(output []responsesOutputItem) []ToolUse {
	tools := []ToolUse{}
	for _, item := range output {
		if item.Type != "function_call" {
			continue
		}
		id := item.CallID
		if id == "" {
			id = item.ID
		}
		if id == "" {
			continue
		}
		tools = append(tools, ToolUse{ID: id, ToolName: item.Name, ToolInput: toolInputObject(safeJSONParse(item.Arguments))})
	}
	return tools
}

// openAIResponseUsage normalizes Responses token usage, returning nil when
// empty. Cache reads and writes are read from input_tokens_details, then from
// the prompt_tokens_details or top-level fields some compatible endpoints use.
func openAIResponseUsage(usage *responsesUsage) *ExecutionTokenUsage {
	if usage == nil || (usage.InputTokens == 0 && usage.OutputTokens == 0 && usage.TotalTokens == 0) {
		return nil
	}
	pick := func(get func(*responsesInputTokenInfo) *int64, top *int64) *int64 {
		for _, details := range []*responsesInputTokenInfo{usage.InputTokensDetails, usage.PromptTokensDetails} {
			if details != nil {
				if v := get(details); v != nil {
					return v
				}
			}
		}
		return top
	}
	cached := pick(func(d *responsesInputTokenInfo) *int64 { return d.CachedTokens }, usage.CachedTokens)
	cacheWrite := pick(func(d *responsesInputTokenInfo) *int64 { return d.CacheWriteTokens }, usage.CacheWriteTokens)
	out := openAIPromptUsage(usage.InputTokens, cached, cacheWrite)
	out.Result = tokens(int(usage.OutputTokens))
	out.Total = tokens(int(usage.TotalTokens))
	return out
}

// openAIResponseFinishReason maps the Responses status into a canonical finish
// reason: "tool_use" when tool calls are present, "stop" on completion, and
// "length"/the incomplete reason otherwise.
func openAIResponseFinishReason(response *responsesResponse, toolUse bool) string {
	if toolUse {
		return "tool_use"
	}
	switch response.Status {
	case "completed":
		return "stop"
	case "incomplete":
		reason := ""
		if response.IncompleteDetails != nil {
			reason = response.IncompleteDetails.Reason
		}
		if reason == "max_output_tokens" {
			return "length"
		}
		if reason != "" {
			return reason
		}
		return "incomplete"
	default:
		return response.Status
	}
}

// responsesStreamEvent is one Responses API stream event, with the fields of
// every event type the driver reads.
type responsesStreamEvent struct {
	Type        string               `json:"type"`
	OutputIndex int                  `json:"output_index"`
	ItemID      string               `json:"item_id"`
	Delta       string               `json:"delta"`
	Text        string               `json:"text"`
	Arguments   string               `json:"arguments"`
	Refusal     string               `json:"refusal"`
	Code        string               `json:"code"`
	Message     string               `json:"message"`
	Item        *responsesOutputItem `json:"item"`
	Response    *responsesResponse   `json:"response"`
}

// openAIResponsesStreamTool tracks the id and name of an in-flight streaming
// tool call so argument deltas can be attributed to it.
type openAIResponsesStreamTool struct {
	id   string
	name string
}

// openAIResponsesStreamState accumulates per-stream tool metadata, keyed by both
// item ID and call ID, so argument-delta events resolve to the right tool call.
type openAIResponsesStreamState struct {
	tools           map[string]openAIResponsesStreamTool
	hasTextDeltas   bool
	refusal         string
	includeThoughts bool
	// final is the terminal response, whose output items become the conversation.
	final *responsesResponse
}

// chunk advances the streaming state machine for one Responses event, emitting a
// CompletionChunk for tool-call starts, argument deltas, text deltas, and the
// terminal completed/incomplete/failed events (which carry finish reason and usage).
// A refusal or an error event ends the stream with an error.
func (s *openAIResponsesStreamState) chunk(event responsesStreamEvent) (CompletionChunk, error) {
	switch event.Type {
	case "response.output_item.added":
		if event.Item != nil && event.Item.Type == "function_call" {
			id := event.Item.CallID
			if id == "" {
				id = event.Item.ID
			}
			if id == "" {
				id = fmt.Sprintf("tool_%d", event.OutputIndex)
			}
			meta := openAIResponsesStreamTool{id: id, name: event.Item.Name}
			if event.Item.ID != "" {
				s.tools[event.Item.ID] = meta
			}
			if event.Item.CallID != "" {
				s.tools[event.Item.CallID] = meta
			}
			return CompletionChunk{ToolUse: []ToolUseChunk{{ID: id, ToolName: event.Item.Name, ToolInput: ""}}}, nil
		}
	case "response.function_call_arguments.delta":
		meta := s.tools[event.ItemID]
		id := meta.id
		if id == "" {
			id = event.ItemID
		}
		if id == "" {
			id = fmt.Sprintf("tool_%d", event.OutputIndex)
		}
		return CompletionChunk{ToolUse: []ToolUseChunk{{ID: id, ToolName: meta.name, ToolInput: event.Delta}}}, nil
	case "response.output_text.delta":
		s.hasTextDeltas = true
		if event.Delta != "" {
			return CompletionChunk{Result: []CompletionResult{newTextResult(event.Delta)}}, nil
		}
	case "response.output_text.done":
		// Some models (for example gpt-5 with json_schema structured output) buffer
		// the whole output and send only the done event. Emit it only when no delta
		// arrived, so the text is not duplicated.
		if !s.hasTextDeltas && event.Text != "" {
			return CompletionChunk{Result: []CompletionResult{newTextResult(event.Text)}}, nil
		}
	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		if s.includeThoughts && event.Delta != "" {
			return CompletionChunk{Result: []CompletionResult{newThoughtsResult(event.Delta)}}, nil
		}
	case "response.refusal.delta":
		s.refusal += event.Delta
	case "response.refusal.done":
		refusal := event.Refusal
		if refusal == "" {
			refusal = s.refusal
		}
		return CompletionChunk{}, fmt.Errorf("[OpenAI] Model refused: %s", refusal)
	case "error", "response.error":
		// OpenAI sends "error"; some compatible endpoints still send the legacy alias.
		message := "[OpenAI Responses API] " + event.Message
		if event.Code != "" {
			message += " (" + event.Code + ")"
		}
		return CompletionChunk{}, errors.New(message)
	case "response.completed", "response.incomplete", "response.failed":
		if event.Response == nil {
			return CompletionChunk{}, nil
		}
		s.final = event.Response
		tools := openAIResponseToolUses(event.Response.Output)
		return CompletionChunk{
			FinishReason: openAIResponseFinishReason(event.Response, event.Type != "response.failed" && len(tools) > 0),
			TokenUsage:   openAIResponseUsage(event.Response.Usage),
			ServiceTier:  event.Response.ServiceTier,
		}, nil
	}
	return CompletionChunk{}, nil
}

// openAIResponsesSystemRole picks the role used for system/safety instructions:
// "user" for o1-mini/o1-preview, "developer" for other o1/o3 reasoning models,
// and "system" otherwise.
func openAIResponsesSystemRole(model string) string {
	normalized := strings.ToLower(shortResourceName(model))
	if strings.HasPrefix(normalized, "o1-mini") || strings.HasPrefix(normalized, "o1-preview") {
		return "user"
	}
	if strings.HasPrefix(normalized, "o1") || strings.HasPrefix(normalized, "o3") {
		return "developer"
	}
	return "system"
}

var openAIReasoningModelPattern = regexp.MustCompile(`(?:^|[/:.])o(?:1|3|4)(?:-|$)`)

// isOpenAIReasoningModel detects o1/o3/o4 and GPT-5+ models, which take
// reasoning and reject temperature and top_p.
func isOpenAIReasoningModel(model string) bool {
	return openAIReasoningModelPattern.MatchString(strings.ToLower(model)) || isOpenAIGptVersionGTE(model, 5, 0)
}

// openAIReasoning builds the reasoning request for reasoning models or when the
// caller sets an effort or reasoning context. Effort (or its reasoning_effort
// alias) is passed through unchanged so the provider validates it, which also
// serves OpenAI-compatible endpoints whose models are not known here. Summaries
// are always requested, for the thoughts projection.
func openAIReasoning(options ExecutionOptions) map[string]any {
	effort := optionString(options.ModelOptions, "effort", "reasoning_effort")
	context := optionString(options.ModelOptions, "reasoning_context")
	if effort == "" && context == "" && !isOpenAIReasoningModel(options.Model) {
		return nil
	}
	reasoning := map[string]any{"summary": "auto"}
	if effort != "" {
		reasoning["effort"] = effort
	}
	if context != "" {
		reasoning["context"] = context
	}
	return reasoning
}

// openAIIncludeThoughts reports whether reasoning is returned as thoughts
// (unless include_thoughts is false).
func openAIIncludeThoughts(options ExecutionOptions) bool {
	v := optionBoolPtr(options.ModelOptions, "include_thoughts")
	return v == nil || *v
}

// openAIImageDetail returns the image_detail option; OpenAI accepts only low,
// high, or auto, so other values fall back to auto.
func openAIImageDetail(options ExecutionOptions) string {
	switch detail := optionString(options.ModelOptions, "image_detail"); detail {
	case "low", "high", "auto":
		return detail
	default:
		return "auto"
	}
}

// completionResultsToText flattens text and JSON completion results into a single
// string for retention as an assistant message.
func completionResultsToText(results []CompletionResult) string {
	var b strings.Builder
	for _, result := range results {
		switch result.Type() {
		case ResultTypeText:
			b.WriteString(result.String())
		case ResultTypeJSON:
			jsonResult, err := result.AsJSONResult()
			if err != nil {
				continue
			}
			if s, ok := jsonResult.Value.(string); ok {
				b.WriteString(s)
			} else {
				data, _ := json.Marshal(jsonResult.Value)
				b.Write(data)
			}
		}
	}
	return b.String()
}

// stringValue returns v when it is a string, and "" otherwise.
func stringValue(v any) string {
	s, _ := v.(string)
	return s
}

// openAISchemaForResponses prepares a JSON schema for the Responses API,
// returning the strict-mode transform (and true) when possible, or falling back
// to the limited transform (and false).
func openAISchemaForResponses(schema map[string]any) (map[string]any, bool) {
	if strict, err := openAIStrictSchema(schema, 0); err == nil {
		return strict, true
	}
	return openAILimitedSchema(schema), false
}

// openAIStrictSchema transforms a schema to satisfy OpenAI strict structured
// output: it drops defaults, sets additionalProperties=false on objects, marks
// every property required, and recurses. It errors on missing property types,
// empty objects, or nesting deeper than five levels (cases strict mode rejects).
func openAIStrictSchema(schema map[string]any, depth int) (map[string]any, error) {
	if depth > 5 {
		return nil, errors.New("schema nesting too deep")
	}
	out := copyMapWithoutDefault(schema)
	if out["type"] == "object" {
		out["additionalProperties"] = false
	}
	if props, ok := out["properties"].(map[string]any); ok {
		required := make([]string, 0, len(props))
		nextProps := map[string]any{}
		for name, raw := range props {
			required = append(required, name)
			prop, ok := raw.(map[string]any)
			if !ok {
				nextProps[name] = raw
				continue
			}
			if _, ok := prop["type"]; !ok {
				return nil, fmt.Errorf("schema property %q is missing type", name)
			}
			next, err := openAIStrictSchema(prop, depth+1)
			if err != nil {
				return nil, err
			}
			nextProps[name] = next
		}
		out["required"] = required
		out["properties"] = nextProps
	}
	if out["type"] == "object" {
		if props, ok := out["properties"].(map[string]any); !ok || len(props) == 0 {
			return nil, errors.New("empty object schema is not supported in strict mode")
		}
	}
	return out, nil
}

// openAILimitedSchema is the non-strict fallback transform: it drops defaults,
// ensures a top-level object type, and recurses into properties without imposing
// strict-mode constraints.
func openAILimitedSchema(schema map[string]any) map[string]any {
	out := copyMapWithoutDefault(schema)
	if _, ok := out["type"]; !ok {
		out["type"] = "object"
	}
	if props, ok := out["properties"].(map[string]any); ok {
		nextProps := map[string]any{}
		for name, raw := range props {
			if prop, ok := raw.(map[string]any); ok {
				nextProps[name] = openAILimitedSchema(prop)
			} else {
				nextProps[name] = raw
			}
		}
		out["properties"] = nextProps
	}
	return out
}

// copyMapWithoutDefault shallow-copies a map omitting the "default" key, which
// OpenAI strict schema mode does not accept.
func copyMapWithoutDefault(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for key, value := range in {
		if key == "default" {
			continue
		}
		out[key] = value
	}
	return out
}
