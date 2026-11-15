package core

import (
	"regexp"
	"strconv"
	"strings"
)

// ClaudeVersion is a parsed Claude model version: the major/minor release
// numbers and the model family ("opus", "sonnet", "haiku", "fable", or "mythos").
type ClaudeVersion struct {
	Major   int
	Minor   int
	Variant string
}

var (
	// claudeMythosPreviewPattern matches the Mythos Preview model, which carries no version and is Mythos 5.0.
	claudeMythosPreviewPattern = regexp.MustCompile(`(?i)claude-mythos-preview(?:-|\b)`)
	// claudeVariantVersionPattern matches family-first names like claude-sonnet-4-6, or
	// claude-opus-4.6 as aggregators write them. The minor version has at most two digits so a
	// date snapshot is not read as one.
	claudeVariantVersionPattern = regexp.MustCompile(`(?i)claude-(opus|sonnet|haiku|fable|mythos)-?(\d+)(?:[-.](\d{1,2}))?(?:-|\b)`)
	// claudeLegacyVersionPattern matches legacy names like claude-3-7-sonnet.
	claudeLegacyVersionPattern = regexp.MustCompile(`(?i)claude-(\d+)[-.](\d+)-(opus|sonnet|haiku|fable|mythos)(?:-|\b)`)
)

// ParseClaudeVersion extracts the version and family from a Claude model ID, the TS client's
// parseClaudeVersion: family-first names (claude-sonnet-4-6, claude-opus-4.6) first, then legacy
// names (claude-3-7-sonnet-20250219). Mythos Preview is Mythos 5.0. Drivers route Claude features
// on the family and version rather than on model ID lists. The bool reports whether the ID parsed.
func ParseClaudeVersion(model string) (ClaudeVersion, bool) {
	return parseClaudeVersion(model)
}

func parseClaudeVersion(model string) (ClaudeVersion, bool) {
	if claudeMythosPreviewPattern.MatchString(model) {
		return ClaudeVersion{Major: 5, Minor: 0, Variant: "mythos"}, true
	}
	if match := claudeVariantVersionPattern.FindStringSubmatch(model); len(match) == 4 {
		major, _ := strconv.Atoi(match[2])
		minor := 0
		if match[3] != "" {
			minor, _ = strconv.Atoi(match[3])
		}
		return ClaudeVersion{Major: major, Minor: minor, Variant: strings.ToLower(match[1])}, true
	}
	if match := claudeLegacyVersionPattern.FindStringSubmatch(model); len(match) == 4 {
		major, _ := strconv.Atoi(match[1])
		minor, _ := strconv.Atoi(match[2])
		return ClaudeVersion{Major: major, Minor: minor, Variant: strings.ToLower(match[3])}, true
	}
	return ClaudeVersion{}, false
}

// isClaudeVersionGTE reports whether the model's version is at least
// major.minor, regardless of variant; unparseable names return false.
func isClaudeVersionGTE(model string, major int, minor int) bool {
	version, ok := parseClaudeVersion(model)
	if !ok {
		return false
	}
	if version.Major != major {
		return version.Major > major
	}
	return version.Minor >= minor
}

// isClaudeVariantVersionGTE reports whether the model is the given variant and
// at least major.minor; a different variant or unparseable name returns false.
func isClaudeVariantVersionGTE(model string, variant string, major int, minor int) bool {
	version, ok := parseClaudeVersion(model)
	if !ok || version.Variant != variant {
		return false
	}
	if version.Major != major {
		return version.Major > major
	}
	return version.Minor >= minor
}

// claudeSupportsAdaptiveThinking reports whether the model supports adaptive
// effort-based thinking (Opus/Sonnet 4.6+).
//
// Adaptive effort-based thinking is only supported by newer variant-first
// Claude models. Older 3.7/4.5 thinking models still require an explicit
// thinking_budget_tokens value.
func claudeSupportsAdaptiveThinking(model string) bool {
	return isClaudeVariantVersionGTE(model, "opus", 4, 6) || isClaudeVariantVersionGTE(model, "sonnet", 4, 6)
}

// claudeHasSamplingParameterRestriction reports whether the model (4.7+) rejects
// certain sampling parameters alongside reasoning fields, so drivers can drop them.
//
// Newer Claude models reject some sampling parameters when reasoning fields
// are present, so drivers suppress them before sending provider requests.
func claudeHasSamplingParameterRestriction(model string) bool {
	version, ok := parseClaudeVersion(model)
	if !ok {
		return false
	}
	if version.Major > 4 {
		return true
	}
	return version.Major == 4 && version.Minor >= 7
}
