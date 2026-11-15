package claude

// claudeThinkingBindingBeta enables thinking.block_binding.
const claudeThinkingBindingBeta = "thinking-binding-controls-2026-08-01"

// claudeThinkingResult is the resolved thinking and effort configuration of a request.
type claudeThinkingResult struct {
	// Thinking is the thinking field, or nil to leave it out.
	Thinking map[string]any
	// Betas are the beta values the thinking field requires.
	Betas []string
	// OutputConfig carries the effort, or is nil when no effort was set.
	OutputConfig map[string]any
	// HasSamplingRestriction drops temperature, top_p and top_k.
	HasSamplingRestriction bool
}

// claudeThinkingDisplay is the display of adaptive thinking: summaries when the caller asked for thoughts.
func claudeThinkingDisplay(modelOptions map[string]any) string {
	if optionBool(modelOptions, "include_thoughts") {
		return "summarized"
	}
	return "omitted"
}

// resolveClaudeThinking resolves the thinking and effort configuration of a Claude request, as the TS client's
// resolveClaudeThinking does:
//   - thinking_mode "between_tools" sends {type: between_tools} and "adaptive" sends adaptive thinking;
//   - models before 3.7 have no thinking;
//   - an effort on a model with adaptive thinking selects adaptive thinking, ignoring a stale budget;
//   - a thinking budget selects extended thinking, except on models that reject sampling parameters (4.7+);
//   - Claude 5+ Opus, Sonnet, Fable and Mythos and Haiku 5.5+ send adaptive thinking even when it is omitted;
//   - other adaptive models leave the provider default; older thinking models are sent thinking disabled.
//
// include_thoughts only selects the display. Adaptive thinking on a model that binds thinking blocks to their
// conversation prefix asks the API to drop a replayed block whose prefix changed instead of rejecting the request.
func resolveClaudeThinking(model string, modelOptions map[string]any) claudeThinkingResult {
	supportsAdaptive := claudeSupportsAdaptiveThinking(model)
	samplingRestriction := claudeHasSamplingRestriction(model)
	supportsThinking := isClaudeVersionGTE(model, 3, 7)
	budget := optionInt(modelOptions, "thinking_budget_tokens")
	effort := optionString(modelOptions, "effort")
	mode := optionString(modelOptions, "thinking_mode")
	display := claudeThinkingDisplay(modelOptions)

	var thinking map[string]any
	switch {
	case mode == "between_tools":
		// This mode accepts only its type: no display or budget, even if stale settings are present.
		thinking = map[string]any{"type": "between_tools"}
	case mode == "adaptive":
		thinking = map[string]any{"type": "adaptive", "display": display}
	case !supportsThinking:
	case supportsAdaptive && effort != "":
		thinking = map[string]any{"type": "adaptive", "display": display}
	case budget != nil && !samplingRestriction:
		thinking = map[string]any{"type": "enabled", "budget_tokens": *budget}
	case claudeDefaultsToAdaptiveThinking(model):
		thinking = map[string]any{"type": "adaptive", "display": display}
	case supportsAdaptive:
		// Other adaptive models keep the provider default when effort is omitted.
	default:
		thinking = map[string]any{"type": "disabled"}
	}

	result := claudeThinkingResult{Thinking: thinking, HasSamplingRestriction: samplingRestriction}
	if effort != "" {
		result.OutputConfig = map[string]any{"effort": effort}
	}
	if thinking != nil && thinking["type"] == "adaptive" && claudeEnforcesThinkingBinding(model) {
		thinking["block_binding"] = map[string]any{"prefix_mismatch_behavior": "drop_block"}
		result.Betas = append(result.Betas, claudeThinkingBindingBeta)
	}
	return result
}
