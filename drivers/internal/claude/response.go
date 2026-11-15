package claude

import "encoding/json"

// claudeResponse is a Messages API response: the content blocks, the stop reason and the usage.
type claudeResponse struct {
	ID          string          `json:"id,omitempty"`
	Type        string          `json:"type,omitempty"`
	Role        string          `json:"role,omitempty"`
	Model       string          `json:"model,omitempty"`
	Content     []claudeBlock   `json:"content"`
	StopReason  string          `json:"stop_reason,omitempty"`
	StopDetails json.RawMessage `json:"stop_details,omitempty"`
	Usage       claudeUsage     `json:"usage"`
}

// claudeUsage is the usage block of a response.
type claudeUsage struct {
	InputTokens              int  `json:"input_tokens"`
	OutputTokens             int  `json:"output_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens,omitempty"`
	CacheCreation            *struct {
		Ephemeral1HInputTokens int `json:"ephemeral_1h_input_tokens"`
	} `json:"cache_creation,omitempty"`
	ServiceTier string `json:"service_tier,omitempty"`
	// Speed is "fast" when the request ran in fast mode.
	Speed string `json:"speed,omitempty"`
	// Iterations is the per-iteration usage; compaction entries are not counted in the fields above.
	Iterations []claudeIterationUsage `json:"iterations,omitempty"`
}

// intValue returns an optional count, or 0.
func intValue(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}

// claudeTokenUsage maps Claude usage to the normalized usage. A compaction runs as its own billed iteration that
// the top-level fields leave out, so it is counted with the request. prompt includes cache reads and writes;
// prompt_new is the uncached input.
func claudeTokenUsage(usage claudeUsage) *ExecutionTokenUsage {
	compaction := claudeCompactionUsage(usage.Iterations)
	input := usage.InputTokens + compaction.Input
	output := usage.OutputTokens + compaction.Output
	cacheRead := intValue(usage.CacheReadInputTokens) + compaction.CacheRead
	cacheWrite := intValue(usage.CacheCreationInputTokens) + compaction.CacheWrite
	cacheWrite1H := compaction.CacheWrite1H
	if usage.CacheCreation != nil {
		cacheWrite1H += usage.CacheCreation.Ephemeral1HInputTokens
	}
	out := &ExecutionTokenUsage{
		PromptNew: tokens(input),
		Prompt:    tokens(input + cacheRead + cacheWrite),
		Result:    tokens(output),
		Total:     tokens(input + output + cacheRead + cacheWrite),
	}
	if compaction.Count > 0 {
		out.PromptCached = tokens(cacheRead)
		out.PromptCacheWrite = tokens(cacheWrite)
	} else {
		if usage.CacheReadInputTokens != nil {
			out.PromptCached = tokens(*usage.CacheReadInputTokens)
		}
		if usage.CacheCreationInputTokens != nil {
			out.PromptCacheWrite = tokens(*usage.CacheCreationInputTokens)
		}
	}
	if cacheWrite1H > 0 {
		out.PromptCacheWrite1H = tokens(cacheWrite1H)
	}
	return out
}

// claudeServiceTier is the processing tier a response reports: "fast" for fast mode, else its service tier.
func claudeServiceTier(usage claudeUsage) string {
	if usage.Speed == "fast" {
		return "fast"
	}
	return usage.ServiceTier
}

// claudeFinishReason normalizes Claude's stop reason: end_turn is stop, max_tokens and
// model_context_window_exceeded are length, and the others (tool_use, refusal, compaction, ...) pass through.
func claudeFinishReason(reason string) string {
	switch reason {
	case "end_turn":
		return "stop"
	case "max_tokens", "model_context_window_exceeded":
		return "length"
	default:
		return reason
	}
}

// claudeCutOffStopReasons can end a response in the middle of its last content block.
var claudeCutOffStopReasons = map[string]bool{"max_tokens": true, "model_context_window_exceeded": true, "refusal": true}

// claudeResults are the text results of a response, with thinking as thoughts when include_thoughts is set.
func claudeResults(content []claudeBlock, includeThoughts bool) []CompletionResult {
	var results []CompletionResult
	for _, block := range content {
		switch {
		case block.Type == "thinking" && block.Thinking != "" && includeThoughts:
			results = append(results, newThoughtsResult(block.Thinking))
		case block.Type == "text" && block.Text != "":
			results = append(results, newTextResult(block.Text))
		}
	}
	return results
}

// claudeToolInput decodes a tool_use input.
func claudeToolInput(input any) *JSONObject {
	if raw, ok := input.(json.RawMessage); ok {
		var decoded any
		if json.Unmarshal(raw, &decoded) != nil {
			return &JSONObject{}
		}
		return toolInputObject(decoded)
	}
	return toolInputObject(input)
}

// claudeTools are the tool calls of a response.
func claudeTools(content []claudeBlock) []ToolUse {
	var tools []ToolUse
	for _, block := range content {
		if block.Type == "tool_use" {
			tools = append(tools, ToolUse{ID: block.ID, ToolName: block.Name, ToolInput: claudeToolInput(block.Input)})
		}
	}
	return tools
}

// extractClaudeCompletion builds the completion of a response. An on-demand compaction answers with its summary
// alone. A token limit or refusal can stop the response inside its last tool call, so that call is left out (the
// conversation keeps it), and Claude's own stop is reported: callers must not mistake those calls for a completed
// tool_use turn.
func extractClaudeCompletion(response claudeResponse, options ExecutionOptions) *Completion {
	var results []CompletionResult
	if options.CompactConversation != nil {
		results = claudeCompactionResults(response.Content)
	} else {
		results = claudeResults(response.Content, optionBool(options.ModelOptions, "include_thoughts"))
	}
	if len(results) == 0 {
		results = []CompletionResult{newTextResult("")}
	}
	toolBlocks := response.Content
	if n := len(toolBlocks); n > 0 && toolBlocks[n-1].Type == "tool_use" && claudeCutOffStopReasons[response.StopReason] {
		toolBlocks = toolBlocks[:n-1]
	}
	completion := &Completion{
		Result:       results,
		ToolUse:      claudeTools(toolBlocks),
		TokenUsage:   claudeTokenUsage(response.Usage),
		ServiceTier:  claudeServiceTier(response.Usage),
		FinishReason: claudeFinishReason(response.StopReason),
	}
	if options.IncludeOriginalResponse {
		completion.OriginalResponse = response
	}
	return completion
}
