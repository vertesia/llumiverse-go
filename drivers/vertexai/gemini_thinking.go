package vertexai

import "strings"

// geminiIncludeThoughts reports whether thought summaries are returned: on unless
// the caller sets include_thoughts to false.
func geminiIncludeThoughts(modelOptions map[string]any) bool {
	value, ok := modelOptions["include_thoughts"].(bool)
	return !ok || value
}

// geminiThinkingConfig builds thinkingConfig as the TS client does. Native controls
// (thinking_budget_tokens, thinking_level) take precedence over the generic effort;
// thought summaries default on, except for a zero budget. Effort maps to a
// thinkingLevel on Gemini 3+ and Nano Banana, and to a thinkingBudget before. With
// no thinking control the provider's model-specific default is preserved: only an
// explicit include_thoughts is sent, and otherwise no thinkingConfig at all.
func geminiThinkingConfig(options ExecutionOptions) map[string]any {
	modelOptions := options.ModelOptions
	explicitInclude, hasInclude := modelOptions["include_thoughts"].(bool)
	budget := optionInt(modelOptions, "thinking_budget_tokens")
	level := optionString(modelOptions, "thinking_level")
	if budget != nil || level != "" {
		out := map[string]any{}
		switch {
		case hasInclude:
			out["includeThoughts"] = explicitInclude
		default:
			out["includeThoughts"] = budget == nil || *budget != 0
		}
		if budget != nil {
			out["thinkingBudget"] = *budget
		}
		if level != "" {
			out["thinkingLevel"] = level
		}
		return out
	}
	includeThoughts := geminiIncludeThoughts(modelOptions)
	if effort := optionString(modelOptions, "effort"); effort != "" {
		if strings.Contains(options.Model, "gemini-nano-banana") || isGeminiVersionGTE(options.Model, "3.0") {
			out := map[string]any{"includeThoughts": includeThoughts}
			if thinkingLevel := geminiThinkingLevelForEffort(effort); thinkingLevel != "" {
				out["thinkingLevel"] = thinkingLevel
			}
			return out
		}
		return map[string]any{"includeThoughts": includeThoughts, "thinkingBudget": geminiBudgetForEffort(options.Model, effort)}
	}
	if hasInclude {
		return map[string]any{"includeThoughts": includeThoughts}
	}
	return nil
}

// geminiThinkingLevelForEffort maps a neutral effort to a Gemini thinkingLevel.
// The caller's effort passes through even where a model advertises fewer levels,
// for provider validation.
func geminiThinkingLevelForEffort(effort string) string {
	switch effort {
	case "minimal":
		return "MINIMAL"
	case "low":
		return "LOW"
	case "medium":
		return "MEDIUM"
	case "high":
		return "HIGH"
	case "xhigh":
		return "XHIGH"
	default:
		return ""
	}
}

// geminiBudgetForEffort maps a neutral effort value to a numeric thinkingBudget
// for Gemini 2.5 models, scaled by the model tier (pro/flash/flash-lite).
func geminiBudgetForEffort(model string, effort string) int {
	isFlashLite := strings.Contains(model, "flash-lite")
	isFlash := strings.Contains(model, "flash") && !isFlashLite
	isPro := strings.Contains(model, "pro")
	switch effort {
	case "minimal", "low":
		if isPro {
			return 128
		}
		if isFlashLite {
			return 512
		}
		if isFlash {
			return 1
		}
		return 1024
	case "medium":
		return 8192
	default:
		if isPro {
			return 32768
		}
		if isFlash || isFlashLite {
			return 24576
		}
		return 8192
	}
}

// hasGeminiSamplingRestriction reports Gemini models that reject the
// single-candidate contract (Gemini 4+ and Flash 3.7+ chat models). Only the
// driver's own candidateCount is withheld; caller sampling options still pass
// through for provider validation.
func hasGeminiSamplingRestriction(model string) bool {
	lower := strings.ToLower(model)
	for _, marker := range []string{"image", "tts", "transcribe", "live"} {
		if strings.Contains(lower, marker) {
			return false
		}
	}
	return isGeminiVersionGTE(lower, "4.0") || (strings.Contains(lower, "flash") && isGeminiVersionGTE(lower, "3.7"))
}
