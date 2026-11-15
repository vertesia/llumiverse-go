package openai

import (
	"regexp"
	"strconv"
	"strings"
)

var (
	familyVersionPattern = regexp.MustCompile(`^(\d+)(?:[.-](\d+))?`)
	gptVersionPattern    = regexp.MustCompile(`(?:^|[./:])(?:openai\.)?gpt-(?:daybreak-[a-z]+-)?(\d+)(?:\.(\d+))?(?:-|[.:@]|$)`)
)

// versionGTE compares a parsed major/minor pair against a target.
func versionGTE(major, minor, targetMajor, targetMinor int) bool {
	return major > targetMajor || (major == targetMajor && minor >= targetMinor)
}

// atoiOrZero parses a regexp capture, treating a missing group as zero.
func atoiOrZero(value string) int {
	n, _ := strconv.Atoi(value)
	return n
}

// modelFamilyVersionGTE checks the numeric version that follows a model-family
// prefix such as "deepseek-v", accepting later versions so new models inherit
// the newest known behaviour.
func modelFamilyVersionGTE(model, family string, targetMajor, targetMinor int) bool {
	normalized := strings.ToLower(model)
	index := strings.Index(normalized, strings.ToLower(family))
	if index < 0 {
		return false
	}
	match := familyVersionPattern.FindStringSubmatch(normalized[index+len(family):])
	if match == nil {
		return false
	}
	return versionGTE(atoiOrZero(match[1]), atoiOrZero(match[2]), targetMajor, targetMinor)
}

// isDeepSeekVersionGTE matches DeepSeek versions written as deepseek.v3.2 or
// deepseek-v3.2.
func isDeepSeekVersionGTE(model string, major, minor int) bool {
	return modelFamilyVersionGTE(model, "deepseek.v", major, minor) || modelFamilyVersionGTE(model, "deepseek-v", major, minor)
}

// parseOpenAIGptVersion parses the GPT generation from bare IDs, snapshots, or
// provider-qualified paths.
func parseOpenAIGptVersion(model string) (major, minor int, ok bool) {
	match := gptVersionPattern.FindStringSubmatch(strings.ToLower(model))
	if match == nil {
		return 0, 0, false
	}
	return atoiOrZero(match[1]), atoiOrZero(match[2]), true
}

// isOpenAIGptVersionGTE reports whether model is a GPT model at or after the
// target version.
func isOpenAIGptVersionGTE(model string, targetMajor, targetMinor int) bool {
	major, minor, ok := parseOpenAIGptVersion(model)
	return ok && versionGTE(major, minor, targetMajor, targetMinor)
}
