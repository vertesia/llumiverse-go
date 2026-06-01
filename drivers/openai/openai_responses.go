package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

// openAIResponsesPrompt is the prepared Responses API input: the ordered input
// items plus a flattened Text concatenation used for image-generation prompts.
type openAIResponsesPrompt struct {
	Items responses.ResponseInputParam
	Text  string
}

// createResponsesPrompt builds the ordered Responses input items from prompt
// segments. System and safety material is grouped first in the instruction role
// chosen for the model (safety text emphasized), followed by user/assistant/tool
// items in order; the concatenated Text is retained for image models. Tool
// segments require a ToolUseID.
func (d *OpenAICompatibleDriver) createResponsesPrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (openAIResponsesPrompt, error) {
	// Responses uses ordered input items. Keep system and safety material in the
	// instruction role while preserving user/assistant/tool ordering.
	prompt := openAIResponsesPrompt{}
	var system []responses.ResponseInputItemUnionParam
	var safety []responses.ResponseInputItemUnionParam
	var others []responses.ResponseInputItemUnionParam
	systemRole := openAIResponsesSystemRole(options.Model)
	for _, segment := range segments {
		switch segment.Role {
		case PromptRoleNegative, PromptRoleMask:
			continue
		case PromptRoleTool:
			if segment.ToolUseID == "" {
				return openAIResponsesPrompt{}, errors.New("tool prompt segment requires ToolUseID")
			}
			others = append(others, responses.ResponseInputItemParamOfFunctionCallOutput(segment.ToolUseID, segment.Content))
			continue
		}
		parts, text, err := openAIResponseContentParts(ctx, segment, options)
		if err != nil {
			return openAIResponsesPrompt{}, err
		}
		if text != "" {
			if prompt.Text != "" {
				prompt.Text += "\n"
			}
			prompt.Text += text
		}
		if len(parts) == 0 {
			continue
		}
		switch segment.Role {
		case PromptRoleSystem:
			system = append(system, responses.ResponseInputItemParamOfMessage(parts, systemRole))
		case PromptRoleSafety:
			safetyParts := make(responses.ResponseInputMessageContentListParam, 0, len(parts))
			for _, part := range parts {
				if textValue := part.GetText(); textValue != nil {
					safetyParts = append(safetyParts, responses.ResponseInputContentParamOfInputText("DO NOT IGNORE - IMPORTANT: "+*textValue))
				}
			}
			if len(safetyParts) > 0 {
				system = append(system, responses.ResponseInputItemParamOfMessage(safetyParts, systemRole))
			}
		case PromptRoleAssistant:
			others = append(others, responses.ResponseInputItemParamOfMessage(parts, responses.EasyInputMessageRoleAssistant))
		default:
			others = append(others, responses.ResponseInputItemParamOfMessage(parts, responses.EasyInputMessageRoleUser))
		}
	}
	prompt.Items = append(prompt.Items, system...)
	prompt.Items = append(prompt.Items, others...)
	prompt.Items = append(prompt.Items, safety...)
	return prompt, nil
}

// openAIResponseContentParts builds the typed input-content parts for a segment
// (input_image parts for files, an input_text part for content) and also returns
// the segment's text for prompt-text accumulation.
func openAIResponseContentParts(ctx context.Context, segment PromptSegment, options ExecutionOptions) (responses.ResponseInputMessageContentListParam, string, error) {
	// openai-go models Responses content as a union; build typed parts rather
	// than marshaling maps so SDK validation catches shape drift.
	parts := make(responses.ResponseInputMessageContentListParam, 0, len(segment.Files)+1)
	for _, file := range segment.Files {
		dataURL, err := dataSourceToDataURL(ctx, file)
		if err != nil {
			return nil, "", err
		}
		parts = append(parts, responses.ResponseInputContentUnionParam{
			OfInputImage: &responses.ResponseInputImageParam{
				Detail:   responses.ResponseInputImageDetail(openAIImageDetail(options)),
				ImageURL: param.NewOpt(dataURL),
			},
		})
	}
	if segment.Content != "" {
		parts = append(parts, responses.ResponseInputContentParamOfInputText(segment.Content))
	}
	return parts, segment.Content, nil
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
	response, err := d.sdk.Responses.New(ctx, payload, d.requestOptions(options.HTTPTimeout)...)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	completion := extractOpenAIResponseCompletion(response, options.IncludeOriginalResponse)
	completion.Conversation = buildOpenAIResponsesConversation(input, completion, options)
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
	stream := d.sdk.Responses.NewStreaming(ctx, d.responsePayload(input, options), d.requestOptions(options.HTTPTimeout)...)
	ch := make(chan streamItem)
	completion := &ExecutionResponse{Prompt: prompt}
	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()
		state := openAIResponsesStreamState{tools: map[string]openAIResponsesStreamTool{}}
		for stream.Next() {
			out := state.chunk(stream.Current())
			if len(out.Result) > 0 || len(out.ToolUse) > 0 || out.FinishReason != "" || out.TokenUsage != nil {
				ch <- streamItem{Chunk: out}
			}
		}
		if err := stream.Err(); err != nil {
			ch <- streamItem{Err: d.formatError(err, options.Model, "stream")}
		}
	}()
	return newChannelStreamWithFinalizer(ch, completion, stream.Close, func(resp *ExecutionResponse) {
		resp.Conversation = buildOpenAIResponsesConversation(input, &resp.Completion, options)
	}), nil
}

// responsePayload assembles the Responses request. Temperature and top_p are
// only sent for non-reasoning models; reasoning effort is mapped per-model;
// tools are omitted for likely non-tool models; and a ResultSchema becomes a
// json_schema text format (strict when convertible), skipped for realtime models.
func (d *OpenAICompatibleDriver) responsePayload(input responses.ResponseInputParam, options ExecutionOptions) responses.ResponseNewParams {
	payload := responses.ResponseNewParams{
		Model: shared.ResponsesModel(options.Model),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
	}
	if v := optionInt(options.ModelOptions, "max_tokens"); v != nil {
		payload.MaxOutputTokens = param.NewOpt(int64(*v))
	}
	if !isOpenAIReasoningModel(options.Model) {
		if v := optionFloat(options.ModelOptions, "temperature"); v != nil {
			payload.Temperature = param.NewOpt(*v)
		}
		if v := optionFloat(options.ModelOptions, "top_p"); v != nil {
			payload.TopP = param.NewOpt(*v)
		}
	}
	if effort := openAIReasoningEffort(options.Model, optionString(options.ModelOptions, "effort", "reasoning_effort")); effort != "" {
		payload.Reasoning = shared.ReasoningParam{Effort: shared.ReasoningEffort(effort)}
	}
	if len(options.Labels) > 0 {
		payload.Metadata = shared.Metadata(options.Labels)
	}
	if len(options.Tools) > 0 && !isLikelyNonToolOpenAIModel(options.Model) {
		payload.Tools = openAIResponseTools(options.Tools)
	}
	if options.ResultSchema != nil && !strings.Contains(strings.ToLower(options.Model), "realtime") {
		schema, strict := openAISchemaForResponses(options.ResultSchema)
		payload.Text = responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:   "format_output",
					Schema: schema,
					Strict: param.NewOpt(strict),
				},
			},
		}
	}
	return payload
}

// openAIResponseTools converts ToolDefinitions into Responses function tool
// params, applying strict schema mode when the input schema can be converted.
func openAIResponseTools(tools []ToolDefinition) []responses.ToolUnionParam {
	out := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		schema, strict := openAISchemaForResponses(tool.InputSchema)
		fn := responses.FunctionToolParam{
			Name:       tool.Name,
			Parameters: schema,
			Strict:     param.NewOpt(strict),
		}
		if tool.Description != "" {
			fn.Description = param.NewOpt(tool.Description)
		}
		out = append(out, responses.ToolUnionParam{OfFunction: &fn})
	}
	return out
}

// extractOpenAIResponseCompletion maps a Responses output into a Completion,
// collecting text/image results, tool uses, usage, and a normalized finish
// reason, and optionally attaching the raw response.
func extractOpenAIResponseCompletion(response *responses.Response, includeOriginal bool) *Completion {
	if response == nil {
		return &Completion{Result: []CompletionResult{{Type: ResultTypeText, Value: ""}}}
	}
	result := openAIResponseResults(response.Output)
	tools := openAIResponseToolUses(response.Output)
	completion := &Completion{
		Result:       result,
		TokenUsage:   openAIResponseUsage(response.Usage),
		ToolUse:      tools,
		FinishReason: openAIResponseFinishReason(*response, len(tools) > 0),
	}
	if len(completion.Result) == 0 && len(completion.ToolUse) == 0 {
		completion.Result = []CompletionResult{{Type: ResultTypeText, Value: ""}}
	}
	if includeOriginal {
		completion.OriginalResponse = response
	}
	return completion
}

// openAIResponseResults extracts text and generated-image results from Responses
// output items, wrapping bare base64 image data in a PNG data URL.
func openAIResponseResults(output []responses.ResponseOutputItemUnion) []CompletionResult {
	results := []CompletionResult{}
	for _, item := range output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" && part.Text != "" {
					results = append(results, CompletionResult{Type: ResultTypeText, Value: part.Text})
				}
			}
		case "image_generation_call":
			if item.Result != "" {
				value := item.Result
				if !strings.HasPrefix(value, "data:") {
					value = "data:image/png;base64," + value
				}
				results = append(results, CompletionResult{Type: ResultTypeImage, Value: value})
			}
		}
	}
	return results
}

// openAIResponseToolUses extracts function_call items as ToolUses, preferring
// CallID over ID and parsing the arguments JSON.
func openAIResponseToolUses(output []responses.ResponseOutputItemUnion) []ToolUse {
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
		tools = append(tools, ToolUse{ID: id, ToolName: item.Name, ToolInput: safeJSONParse(item.Arguments)})
	}
	return tools
}

// openAIResponseUsage normalizes Responses token usage, returning nil when empty.
func openAIResponseUsage(usage responses.ResponseUsage) *ExecutionTokenUsage {
	if usage.InputTokens == 0 && usage.OutputTokens == 0 && usage.TotalTokens == 0 {
		return nil
	}
	return openAIUsage(usage.InputTokens, usage.OutputTokens, usage.TotalTokens, usage.InputTokensDetails.CachedTokens)
}

// openAIResponseFinishReason maps the Responses status into a canonical finish
// reason: "tool_use" when tool calls are present, "stop" on completion, and
// "length"/the incomplete reason otherwise.
func openAIResponseFinishReason(response responses.Response, toolUse bool) string {
	if toolUse {
		return "tool_use"
	}
	switch response.Status {
	case "completed":
		return "stop"
	case "incomplete":
		if response.IncompleteDetails.Reason == "max_output_tokens" {
			return "length"
		}
		if response.IncompleteDetails.Reason != "" {
			return response.IncompleteDetails.Reason
		}
		return "incomplete"
	default:
		if response.Status != "" {
			return string(response.Status)
		}
		return ""
	}
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
	tools map[string]openAIResponsesStreamTool
}

// chunk advances the streaming state machine for one Responses event, emitting a
// CompletionChunk for tool-call starts, argument deltas, text deltas, and the
// terminal completed/incomplete/failed events (which carry finish reason and usage).
func (s *openAIResponsesStreamState) chunk(event responses.ResponseStreamEventUnion) CompletionChunk {
	switch value := event.AsAny().(type) {
	case responses.ResponseOutputItemAddedEvent:
		if value.Item.Type == "function_call" {
			id := value.Item.CallID
			if id == "" {
				id = value.Item.ID
			}
			if id == "" {
				id = fmt.Sprintf("tool_%d", value.OutputIndex)
			}
			meta := openAIResponsesStreamTool{id: id, name: value.Item.Name}
			if value.Item.ID != "" {
				s.tools[value.Item.ID] = meta
			}
			if value.Item.CallID != "" {
				s.tools[value.Item.CallID] = meta
			}
			return CompletionChunk{ToolUse: []ToolUse{{ID: id, ToolName: value.Item.Name, ToolInput: ""}}}
		}
	case responses.ResponseFunctionCallArgumentsDeltaEvent:
		meta := s.tools[value.ItemID]
		id := meta.id
		if id == "" {
			id = value.ItemID
		}
		if id == "" {
			id = fmt.Sprintf("tool_%d", value.OutputIndex)
		}
		return CompletionChunk{ToolUse: []ToolUse{{ID: id, ToolName: meta.name, ToolInput: value.Delta}}}
	case responses.ResponseTextDeltaEvent:
		if value.Delta != "" {
			return CompletionChunk{Result: []CompletionResult{{Type: ResultTypeText, Value: value.Delta}}}
		}
	case responses.ResponseCompletedEvent:
		tools := openAIResponseToolUses(value.Response.Output)
		return CompletionChunk{FinishReason: openAIResponseFinishReason(value.Response, len(tools) > 0), TokenUsage: openAIResponseUsage(value.Response.Usage)}
	case responses.ResponseIncompleteEvent:
		tools := openAIResponseToolUses(value.Response.Output)
		return CompletionChunk{FinishReason: openAIResponseFinishReason(value.Response, len(tools) > 0), TokenUsage: openAIResponseUsage(value.Response.Usage)}
	case responses.ResponseFailedEvent:
		return CompletionChunk{FinishReason: openAIResponseFinishReason(value.Response, false), TokenUsage: openAIResponseUsage(value.Response.Usage)}
	}
	return CompletionChunk{}
}

// openAIResponsesInput combines retained conversation items (in any of the
// accepted forms) with the new prompt items, repairs orphaned tool calls, and
// converts tool items to text when the current request has no tools.
func openAIResponsesInput(conversation any, prompt responses.ResponseInputParam, hasTools bool) responses.ResponseInputParam {
	// Accept both raw SDK input slices and the structured state returned by this
	// driver so callers can persist conversation values without conversion.
	out := responses.ResponseInputParam{}
	switch value := conversation.(type) {
	case responses.ResponseInputParam:
		out = append(out, value...)
	case []responses.ResponseInputItemUnionParam:
		out = append(out, value...)
	case openAIResponsesConversationState:
		out = append(out, value.Items...)
	case *openAIResponsesConversationState:
		if value != nil {
			out = append(out, value.Items...)
		}
	}
	out = append(out, prompt...)
	out = fixOpenAIOrphanedToolUse(out)
	if !hasTools && openAIResponsesInputHasToolItems(out) {
		// Stored conversations may include tool call/output items from checkpoint
		// summaries. If this request has no tools, convert them to text to avoid
		// Responses API validation errors.
		out = convertOpenAIResponsesToolItemsToText(out)
	}
	return out
}

// openAIResponsesConversationState is the driver's retained conversation: the
// accumulated Responses input items plus the turn counter used by retention rules.
type openAIResponsesConversationState struct {
	Items responses.ResponseInputParam
	Turn  int
}

// buildOpenAIResponsesConversation appends the completion's assistant text and
// tool calls to the request input, increments the turn, and applies media and
// heartbeat stripping retention rules to produce the next conversation state.
func buildOpenAIResponsesConversation(input responses.ResponseInputParam, completion *Completion, options ExecutionOptions) openAIResponsesConversationState {
	// Responses conversations are retained as input items, not provider response
	// objects. Streaming and non-streaming paths both append assistant text/tool
	// calls here before applying retention rules.
	out := make(responses.ResponseInputParam, 0, len(input)+1+len(completion.ToolUse))
	out = append(out, input...)
	text := completionResultsToText(completion.Result)
	if text != "" {
		out = append(out, responses.ResponseInputItemParamOfMessage(text, responses.EasyInputMessageRoleAssistant))
	}
	for _, tool := range completion.ToolUse {
		args := ""
		if s, ok := tool.ToolInput.(string); ok {
			args = s
		} else {
			data, _ := json.Marshal(tool.ToolInput)
			args = string(data)
		}
		out = append(out, responses.ResponseInputItemParamOfFunctionCall(args, tool.ID, tool.ToolName))
	}
	turn := openAIResponsesTurn(options.Conversation) + 1
	stripMedia := shouldStripConversationMedia(turn, options.StripImagesAfterTurns)
	stripHeartbeats := shouldStripConversationHeartbeats(turn, options.StripHeartbeatsAfterTurns)
	return openAIResponsesConversationState{
		Items: processOpenAIResponsesItemsForConversation(out, options, stripMedia, stripHeartbeats),
		Turn:  turn,
	}
}

// openAIResponsesTurn reads the turn counter from a retained conversation value,
// returning 0 for unknown or absent state.
func openAIResponsesTurn(conversation any) int {
	switch value := conversation.(type) {
	case openAIResponsesConversationState:
		return value.Turn
	case *openAIResponsesConversationState:
		if value != nil {
			return value.Turn
		}
	}
	return 0
}

func openAIImageDetail(options ExecutionOptions) string {
	// OpenAI accepts only low, high, or auto; keep invalid user values harmless.
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
		switch result.Type {
		case ResultTypeText:
			b.WriteString(toString(result.Value))
		case ResultTypeJSON:
			if s, ok := result.Value.(string); ok {
				b.WriteString(s)
			} else {
				data, _ := json.Marshal(result.Value)
				b.Write(data)
			}
		}
	}
	return b.String()
}

// fixOpenAIOrphanedToolUse inserts synthetic function_call_output items for any
// function_call lacking a matching output, keeping the conversation valid.
func fixOpenAIOrphanedToolUse(items responses.ResponseInputParam) responses.ResponseInputParam {
	// If an agent was stopped after a function_call but before its output was
	// recorded, the next Responses request would contain an orphaned tool call.
	// Insert a synthetic interruption output so the conversation remains valid.
	if len(items) < 2 {
		return items
	}
	outputs := map[string]bool{}
	for _, item := range items {
		if out := item.GetCallID(); out != nil && openAIResponsesItemType(item) == "function_call_output" {
			outputs[*out] = true
		}
	}
	out := make(responses.ResponseInputParam, 0, len(items))
	pending := map[string]string{}
	flushPending := func() {
		for id, name := range pending {
			out = append(out, responses.ResponseInputItemParamOfFunctionCallOutput(id, fmt.Sprintf(`[Tool interrupted: The user stopped the operation before "%s" could execute.]`, name)))
			delete(pending, id)
		}
	}
	for _, item := range items {
		itemType := openAIResponsesItemType(item)
		if itemType == "function_call" {
			callID := ""
			if id := item.GetCallID(); id != nil {
				callID = *id
			}
			if callID != "" && !outputs[callID] {
				name := "unknown"
				if item.GetName() != nil {
					name = *item.GetName()
				}
				pending[callID] = name
			}
			out = append(out, item)
			continue
		}
		if itemType == "function_call_output" {
			out = append(out, item)
			continue
		}
		flushPending()
		out = append(out, item)
	}
	flushPending()
	return out
}

// openAIResponsesInputHasToolItems reports whether any input item is a tool call
// or tool output.
func openAIResponsesInputHasToolItems(items responses.ResponseInputParam) bool {
	for _, item := range items {
		itemType := openAIResponsesItemType(item)
		if itemType == "function_call" || itemType == "function_call_output" {
			return true
		}
	}
	return false
}

// convertOpenAIResponsesToolItemsToText rewrites tool call/output items as
// bracketed transcript messages (truncated) for requests made without tools.
func convertOpenAIResponsesToolItemsToText(items responses.ResponseInputParam) responses.ResponseInputParam {
	// When tools are unavailable, preserve prior tool context as readable
	// transcript text instead of dropping it from conversation history.
	out := make(responses.ResponseInputParam, 0, len(items))
	for _, item := range items {
		itemType := openAIResponsesItemType(item)
		switch itemType {
		case "function_call":
			name := "unknown"
			if item.GetName() != nil {
				name = *item.GetName()
			}
			args := ""
			if item.GetArguments() != nil {
				args = *item.GetArguments()
			}
			out = append(out, responses.ResponseInputItemParamOfMessage(
				fmt.Sprintf("[Tool call: %s(%s)]", name, truncateForConversation(args, 500)),
				responses.EasyInputMessageRoleAssistant,
			))
		case "function_call_output":
			output := ""
			if item.OfFunctionCallOutput != nil {
				output = item.OfFunctionCallOutput.Output
			}
			out = append(out, responses.ResponseInputItemParamOfMessage(
				"[Tool result: "+truncateForConversation(output, 500)+"]",
				responses.EasyInputMessageRoleUser,
			))
		default:
			out = append(out, item)
		}
	}
	return out
}

// openAIResponsesItemType returns a coarse type tag ("function_call",
// "function_call_output", "message") for a Responses input item union.
func openAIResponsesItemType(item responses.ResponseInputItemUnionParam) string {
	switch {
	case item.OfFunctionCall != nil:
		return "function_call"
	case item.OfFunctionCallOutput != nil:
		return "function_call_output"
	case item.OfMessage != nil:
		return "message"
	case item.OfInputMessage != nil:
		return "message"
	case item.OfOutputMessage != nil:
		return "message"
	}
	if t := item.GetType(); t != nil {
		return *t
	}
	return ""
}

// openAIResponsesSystemRole picks the role used for system/safety instructions:
// "user" for o1-mini/o1-preview, "developer" for other o1/o3 reasoning models,
// and "system" otherwise.
func openAIResponsesSystemRole(model string) responses.EasyInputMessageRole {
	normalized := strings.ToLower(shortResourceName(model))
	if strings.HasPrefix(normalized, "o1-mini") || strings.HasPrefix(normalized, "o1-preview") {
		return responses.EasyInputMessageRoleUser
	}
	if strings.HasPrefix(normalized, "o1") || strings.HasPrefix(normalized, "o3") {
		return responses.EasyInputMessageRoleDeveloper
	}
	return responses.EasyInputMessageRoleSystem
}

// isOpenAIReasoningModel detects reasoning models (o1/o3/o4/gpt-5) by name,
// which disables temperature/top_p and enables reasoning effort.
func isOpenAIReasoningModel(model string) bool {
	normalized := strings.ToLower(model)
	return strings.Contains(normalized, "o1") || strings.Contains(normalized, "o3") || strings.Contains(normalized, "o4") || strings.Contains(normalized, "gpt-5")
}

// openAIReasoningEffort validates the requested effort for reasoning models,
// forcing "high" for gpt-5-pro and returning "" when effort does not apply.
func openAIReasoningEffort(model string, effort string) string {
	if effort == "" || !isOpenAIReasoningModel(model) {
		return ""
	}
	if strings.HasPrefix(strings.ToLower(shortResourceName(model)), "gpt-5-pro") {
		return "high"
	}
	switch effort {
	case "low", "medium", "high":
		return effort
	default:
		return ""
	}
}

// isOpenAIImageModel detects image-generation models (dall-e/gpt-image/
// chatgpt-image), which route to the Images API instead of Responses.
func isOpenAIImageModel(model string) bool {
	model = strings.ToLower(model)
	return strings.Contains(model, "dall-e") || strings.Contains(model, "gpt-image") || strings.Contains(model, "chatgpt-image")
}

// requestOpenAIImageGeneration calls the Images API using the flattened prompt
// text and image ModelOptions (size, n, quality, style, background, format,
// etc.), defaulting base64 output for dall-e, and returns image results as data
// URLs or URLs.
func (d *OpenAICompatibleDriver) requestOpenAIImageGeneration(ctx context.Context, prompt openAIResponsesPrompt, options ExecutionOptions) (*Completion, error) {
	payload := openai.ImageGenerateParams{
		Model:  openai.ImageModel(options.Model),
		Prompt: strings.TrimSpace(prompt.Text),
		Size:   openai.ImageGenerateParamsSize1024x1024,
	}
	if size := optionString(options.ModelOptions, "size"); size != "" {
		payload.Size = openai.ImageGenerateParamsSize(size)
	}
	if n := optionInt(options.ModelOptions, "n"); n != nil {
		payload.N = param.NewOpt(int64(*n))
	}
	if responseFormat := optionString(options.ModelOptions, "response_format"); responseFormat != "" {
		payload.ResponseFormat = openai.ImageGenerateParamsResponseFormat(responseFormat)
	} else if strings.Contains(options.Model, "dall-e") {
		payload.ResponseFormat = openai.ImageGenerateParamsResponseFormatB64JSON
	}
	if quality := optionString(options.ModelOptions, "image_quality", "quality"); quality != "" {
		payload.Quality = openai.ImageGenerateParamsQuality(quality)
	} else if strings.Contains(options.Model, "dall-e-3") {
		payload.Quality = openai.ImageGenerateParamsQualityStandard
	}
	if style := optionString(options.ModelOptions, "style"); style != "" {
		payload.Style = openai.ImageGenerateParamsStyle(style)
	}
	if background := optionString(options.ModelOptions, "background"); background != "" {
		payload.Background = openai.ImageGenerateParamsBackground(background)
	}
	if outputFormat := optionString(options.ModelOptions, "output_format"); outputFormat != "" {
		payload.OutputFormat = openai.ImageGenerateParamsOutputFormat(outputFormat)
	}
	if outputCompression := optionInt(options.ModelOptions, "output_compression"); outputCompression != nil {
		payload.OutputCompression = param.NewOpt(int64(*outputCompression))
	}
	if moderation := optionString(options.ModelOptions, "moderation"); moderation != "" {
		payload.Moderation = openai.ImageGenerateParamsModeration(moderation)
	}
	response, err := d.sdk.Images.Generate(ctx, payload, d.requestOptions(options.HTTPTimeout)...)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	results := []CompletionResult{}
	for _, image := range response.Data {
		if image.B64JSON != "" {
			results = append(results, CompletionResult{Type: ResultTypeImage, Value: "data:image/png;base64," + image.B64JSON})
		} else if image.URL != "" {
			results = append(results, CompletionResult{Type: ResultTypeImage, Value: image.URL})
		}
	}
	return &Completion{Result: results}, nil
}

// optionString returns the first option value found under the given keys as a
// string, or "" when none is set.
func optionString(options map[string]any, keys ...string) string {
	if options == nil {
		return ""
	}
	for _, key := range keys {
		if value, ok := options[key].(string); ok {
			return value
		}
	}
	return ""
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
