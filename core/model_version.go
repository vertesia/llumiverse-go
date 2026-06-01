package core

import (
	"regexp"
	"strconv"
	"strings"
)

// claudeVersion is a parsed Claude model version: the major/minor release
// numbers and the family variant ("opus", "sonnet", or "haiku").
type claudeVersion struct {
	Major   int
	Minor   int
	Variant string
}

var (
	// claudeVariantVersionPattern matches variant-first names like claude-sonnet-4-6.
	claudeVariantVersionPattern = regexp.MustCompile(`(?i)claude-(opus|sonnet|haiku)-?(\d+)(?:-(\d{1,2}))?(?:-|$)`)
	// claudeLegacyVersionPattern matches legacy names like claude-3-7-sonnet.
	claudeLegacyVersionPattern = regexp.MustCompile(`(?i)claude-(\d+)-(\d+)-(\w+)`)
)

// parseClaudeVersion extracts the version from a Claude model id, trying the
// variant-first naming scheme first and then the legacy scheme. The bool
// reports whether the name matched a known pattern.
func parseClaudeVersion(model string) (claudeVersion, bool) {
	// Anthropic and Bedrock use both legacy names like claude-3-7-sonnet and
	// newer variant-first names like claude-sonnet-4-6.
	if match := claudeVariantVersionPattern.FindStringSubmatch(model); len(match) == 4 {
		major, _ := strconv.Atoi(match[2])
		minor := 0
		if match[3] != "" {
			minor, _ = strconv.Atoi(match[3])
		}
		return claudeVersion{Major: major, Minor: minor, Variant: strings.ToLower(match[1])}, true
	}
	if match := claudeLegacyVersionPattern.FindStringSubmatch(model); len(match) == 4 {
		major, _ := strconv.Atoi(match[1])
		minor, _ := strconv.Atoi(match[2])
		return claudeVersion{Major: major, Minor: minor, Variant: strings.ToLower(match[3])}, true
	}
	return claudeVersion{}, false
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
