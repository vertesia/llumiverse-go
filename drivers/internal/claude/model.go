package claude

import (
	"regexp"
	"strings"

	"github.com/vertesia/llumiverse-go/core"
)

// Claude model routing. Every rule below is expressed through the family (variant) and version parsed by
// core.ParseClaudeVersion, never through a list of model IDs, so a newly released model inherits the behaviour of
// the newest generation of its family, as the TS client's version-parsing.ts does.

// claudeMythosPreviewIDPattern is the Mythos preview ID without a suffix, which predates mid-conversation tool
// changes.
var claudeMythosPreviewIDPattern = regexp.MustCompile(`(?i)claude-mythos-preview\b`)

// parseClaudeVersion extracts the family and version from a Claude model ID, including region-prefixed Bedrock IDs
// (us.anthropic.claude-…) and Vertex AI IDs (claude-…@date).
func parseClaudeVersion(model string) (core.ClaudeVersion, bool) {
	return core.ParseClaudeVersion(model)
}

// atLeast reports whether a version is major.minor or later.
func atLeast(v core.ClaudeVersion, major, minor int) bool {
	return v.Major > major || (v.Major == major && v.Minor >= minor)
}

// isClaudeVersionGTE reports whether the model is Claude major.minor or later, whatever its family.
func isClaudeVersionGTE(model string, major, minor int) bool {
	version, ok := parseClaudeVersion(model)
	return ok && atLeast(version, major, minor)
}

// isClaudeVariantVersionGTE reports whether the model is the given family at major.minor or later.
func isClaudeVariantVersionGTE(model, variant string, major, minor int) bool {
	version, ok := parseClaudeVersion(model)
	return ok && version.Variant == variant && atLeast(version, major, minor)
}

// claudeSupportsAdaptiveThinking reports whether the model accepts adaptive (effort-based) thinking:
// Opus and Sonnet 4.6 and later, and every Claude 5 model.
func claudeSupportsAdaptiveThinking(model string) bool {
	return isClaudeVersionGTE(model, 5, 0) ||
		isClaudeVariantVersionGTE(model, "opus", 4, 6) ||
		isClaudeVariantVersionGTE(model, "sonnet", 4, 6)
}

// claudeHasSamplingRestriction reports whether the model rejects temperature, top_p and top_k:
// every model from 4.7 on.
func claudeHasSamplingRestriction(model string) bool {
	return isClaudeVersionGTE(model, 4, 7)
}

// keepsClaudePriorTurnThinking reports whether the model keeps earlier turns' thinking blocks in context when a
// plain user turn follows: Opus 4.5 and later and every 4.6+ model. Earlier models strip those blocks on that
// request, so the prompt cache breaks at the first one.
func keepsClaudePriorTurnThinking(model string) bool {
	return isClaudeVersionGTE(model, 4, 6) || isClaudeVariantVersionGTE(model, "opus", 4, 5)
}

// claudeSupportsMidConversationToolChanges reports whether the model accepts tool_addition / tool_removal blocks
// in a system message: Opus 4.8+, Fable and Mythos 5+, and Sonnet and Haiku 5.5+. Mythos Preview does not.
func claudeSupportsMidConversationToolChanges(model string) bool {
	if claudeMythosPreviewIDPattern.MatchString(model) {
		return false
	}
	version, ok := parseClaudeVersion(model)
	if !ok {
		return false
	}
	switch version.Variant {
	case "opus":
		return atLeast(version, 4, 8)
	case "fable", "mythos":
		return atLeast(version, 5, 0)
	default:
		return atLeast(version, 5, 5)
	}
}

// claudeSupportsForcedToolChoice reports whether the model accepts tool_choice "any" or a named tool. Opus and
// Sonnet 5.5+ and Fable and Mythos 5.1+ reject it.
func claudeSupportsForcedToolChoice(model string) bool {
	version, ok := parseClaudeVersion(model)
	if !ok {
		return true
	}
	switch version.Variant {
	case "opus", "sonnet":
		return !atLeast(version, 5, 5)
	case "fable", "mythos":
		return !atLeast(version, 5, 1)
	default:
		return true
	}
}

// claudeFastModeStatus classifies a model's fast mode on the Claude API: "available" on Opus 5.5+ and on Fable and
// Mythos 5.2+, "research_preview" on Opus 5, and "" otherwise. Opus 4.8's fast mode is deprecated, so it is left out.
func claudeFastModeStatus(model string) string {
	version, ok := parseClaudeVersion(model)
	if !ok {
		return ""
	}
	switch version.Variant {
	case "opus":
		if atLeast(version, 5, 5) {
			return "available"
		}
		if atLeast(version, 5, 0) {
			return "research_preview"
		}
	case "fable", "mythos":
		if atLeast(version, 5, 2) {
			return "available"
		}
	}
	return ""
}

// SupportsVertexAIFastMode reports whether Claude on Vertex AI takes speed "fast": Fable and Mythos 5.2 and later,
// where fast mode is generally available. Opus fast mode is not served there.
func SupportsVertexAIFastMode(model string) bool {
	version, ok := parseClaudeVersion(model)
	return ok && (version.Variant == "fable" || version.Variant == "mythos") && atLeast(version, 5, 2)
}

// claudeMaxTokensLimit is the canonical output limit of a Claude model, used for max_tokens when the caller omits
// it (the Messages API requires the field): 128K for 3.7 and 4.7+ and for Opus and Sonnet 4.6, 64K for Opus 4.5 and
// for other 3.7+ models, 32K for Opus 4.0 and 4.1, 8K for 3.5, and 4K for older models.
func claudeMaxTokensLimit(model string) int {
	version, ok := parseClaudeVersion(model)
	if !ok {
		if strings.Contains(strings.ToLower(model), "opus-") {
			return 32000
		}
		return 4096
	}
	switch {
	case version.Major == 3 && version.Minor == 7:
		return 128000
	case atLeast(version, 4, 7), (version.Variant == "opus" || version.Variant == "sonnet") && atLeast(version, 4, 6):
		return 128000
	case version.Variant == "opus" && atLeast(version, 4, 5):
		return 64000
	case version.Variant == "opus" && atLeast(version, 4, 0):
		return 32000
	case atLeast(version, 3, 7):
		return 64000
	case atLeast(version, 3, 5):
		return 8192
	default:
		return 4096
	}
}

// BedrockMaxTokensLimit is the output limit Amazon Bedrock enforces for a Claude model. Bedrock's bound is
// exclusive for the 128K generation (it rejects max_tokens >= 128000), and Sonnet 4.6 is capped at 64K there.
// Region prefixes (us., eu., global.) do not change the parsed version. Zero means unknown.
func BedrockMaxTokensLimit(model string) int {
	version, ok := parseClaudeVersion(model)
	if ok {
		switch {
		case atLeast(version, 4, 7), version.Variant == "opus" && atLeast(version, 4, 6):
			return 127999
		case version.Variant == "sonnet" && atLeast(version, 4, 6):
			return 64000
		case atLeast(version, 4, 5):
			return 64000
		case atLeast(version, 4, 0):
			if version.Variant == "opus" {
				return 32000
			}
			return 64000
		}
	}
	lower := strings.ToLower(model)
	switch {
	case strings.Contains(lower, "claude-3-5-haiku"):
		return 8192
	case strings.Contains(lower, "claude-3-haiku"):
		return 4096
	}
	if strings.Contains(lower, "claude") {
		return claudeMaxTokensLimit(model)
	}
	return 0
}

// claudeSupportsNativeResultSchema reports whether the model accepts structured outputs (output_config.format):
// Claude 4.5 and later.
func claudeSupportsNativeResultSchema(model string) bool {
	return isClaudeVersionGTE(model, 4, 5)
}

// claudeEnforcesThinkingBinding reports whether the model binds each thinking block to the conversation before it
// and rejects a replayed block once that prefix changes: Fable and Mythos 5.1+ and every Claude 5.5+ model.
func claudeEnforcesThinkingBinding(model string) bool {
	version, ok := parseClaudeVersion(model)
	if !ok {
		return false
	}
	if version.Major != 5 {
		return version.Major > 5
	}
	return version.Minor >= 5 || ((version.Variant == "fable" || version.Variant == "mythos") && version.Minor >= 1)
}

// claudeDefaultsToAdaptiveThinking reports whether the model sends adaptive thinking even when the caller omits
// it, so the display follows include_thoughts: Claude 5+ Opus, Sonnet, Fable and Mythos, and Haiku 5.5+.
func claudeDefaultsToAdaptiveThinking(model string) bool {
	version, ok := parseClaudeVersion(model)
	if !ok {
		return false
	}
	switch version.Variant {
	case "opus", "sonnet", "fable", "mythos":
		return version.Major >= 5
	case "haiku":
		return atLeast(version, 5, 5)
	}
	return false
}
