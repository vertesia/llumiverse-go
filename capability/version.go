package capability

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/vertesia/llumiverse-go/core"
)

// atLeast compares major.minor against a target version.
func atLeast(major, minor, targetMajor, targetMinor int) bool {
	return major > targetMajor || (major == targetMajor && minor >= targetMinor)
}

// atoi parses a regexp capture, treating an empty capture as zero.
func atoi(s string) int {
	n, _ := strconv.Atoi(s)
	return n
}

var familyVersionPattern = regexp.MustCompile(`^(\d+)(?:[.-](\d+))?`)

// familyVersionGTE checks the numeric version that follows a model family, such as `gpt-`, `grok-` or `deepseek-v`,
// in major/minor order. Later versions match, so new models inherit the newest known behaviour.
func familyVersionGTE(model, family string, targetMajor, targetMinor int) bool {
	normalized := strings.ToLower(model)
	index := strings.Index(normalized, strings.ToLower(family))
	if index == -1 {
		return false
	}
	match := familyVersionPattern.FindStringSubmatch(normalized[index+len(family):])
	if match == nil {
		return false
	}
	return atLeast(atoi(match[1]), atoi(match[2]), targetMajor, targetMinor)
}

var qwenVersionPattern = regexp.MustCompile(`qwen(\d+)(?:\.(\d+))?`)

// qwenVersionGTE reads Qwen versions, whose minor follows a dot (`qwen3.5`); a dash starts the size (`qwen3-235b`).
func qwenVersionGTE(model string, targetMajor, targetMinor int) bool {
	match := qwenVersionPattern.FindStringSubmatch(strings.ToLower(model))
	if match == nil {
		return false
	}
	return atLeast(atoi(match[1]), atoi(match[2]), targetMajor, targetMinor)
}

var singleDigitGrokPattern = regexp.MustCompile(`grok-(\d+)(?:\.(\d))?(?:[-_.]|$)`)

// singleDigitGrokVersionGTE orders single-digit Grok releases. Grok 4.20 is a separate line, not a release after 4.7.
func singleDigitGrokVersionGTE(model string, targetMajor, targetMinor int) bool {
	match := singleDigitGrokPattern.FindStringSubmatch(model)
	if match == nil {
		return false
	}
	return atLeast(atoi(match[1]), atoi(match[2]), targetMajor, targetMinor)
}

// isPegasus15OrLater reports TwelveLabs Pegasus 1.5 and later, which changed limits and request shape.
func isPegasus15OrLater(model string) bool {
	return familyVersionGTE(model, "pegasus-", 1, 5)
}

// deepSeekVersionGTE reads `deepseek.v3.2` (Bedrock) and `deepseek-v3.2` (elsewhere).
func deepSeekVersionGTE(model string, targetMajor, targetMinor int) bool {
	return familyVersionGTE(model, "deepseek.v", targetMajor, targetMinor) ||
		familyVersionGTE(model, "deepseek-v", targetMajor, targetMinor)
}

var metaMusePattern = regexp.MustCompile(`(?:^|[-_.])muse-`)

// isMetaMuseModel reports Meta Muse models (`muse-spark-1.3`, `muse-wonder`, `muse-glimmer-30b`).
func isMetaMuseModel(model string) bool {
	return metaMusePattern.MatchString(model)
}

var kimiVersionPattern = regexp.MustCompile(`kimi-k(\d+)(?:\.(\d+))?`)

// kimiVersionGTE reads Kimi versions, whose minor follows a dot (`kimi-k2.5`); a dash starts a date (`kimi-k2-0905`).
func kimiVersionGTE(model string, targetMajor, targetMinor int) bool {
	match := kimiVersionPattern.FindStringSubmatch(model)
	if match == nil {
		return false
	}
	return atLeast(atoi(match[1]), atoi(match[2]), targetMajor, targetMinor)
}

// grokReasoningEffortLevels lists Grok reasoning efforts by generation; the Grok 4.20 multi-agent line is an
// exception.
func grokReasoningEffortLevels(model string) []string {
	if index := strings.Index(model, "grok-4.20"); index != -1 &&
		strings.Contains(model[index+len("grok-4.20"):], "multi-agent") {
		return []string{"low", "medium", "high", "xhigh"}
	}
	switch {
	case singleDigitGrokVersionGTE(model, 4, 6):
		return []string{"low", "medium", "high", "xhigh"}
	case singleDigitGrokVersionGTE(model, 4, 5):
		return []string{"low", "medium", "high"}
	case singleDigitGrokVersionGTE(model, 4, 3):
		return []string{"none", "low", "medium", "high"}
	}
	return nil
}

var (
	geminiSpecialPattern    = regexp.MustCompile(`(?:image|tts|transcribe|live)`)
	geminiLongDecodeExclude = regexp.MustCompile(`(?:image|tts|transcribe|live|native-audio|robotics|embedding|omni|nano-banana)`)
	geminiXHighPattern      = regexp.MustCompile(`(?:^|[/.])gemini-\d+(?:\.\d+)?-hydrogen(?:$|[^a-z0-9])`)
	geminiOmniPattern       = regexp.MustCompile(`^gemini-omni(?:-(\d+)(?:\.(\d+))?)?-[a-z][a-z0-9]*(?:-.+)?$`)
	geminiVersionPattern    = regexp.MustCompile(`(?i)^gemini-(\d+)(?:\.(\d+))?`)
)

// geminiSamplingRestriction reports Gemini Flash 3.7+ and every Gemini 4+ chat tier, which reject candidate
// count, penalties and sampling controls.
func geminiSamplingRestriction(model string) bool {
	if geminiSpecialPattern.MatchString(model) {
		return false
	}
	return geminiVersionGTE(model, 4, 0) || (strings.Contains(model, "flash") && geminiVersionGTE(model, 3, 7))
}

// geminiOmniVersion parses Gemini Omni video model versions: `gemini-omni-flash-preview` is 1.0, later releases
// put the version before the tier (`gemini-omni-1.2-flash`).
func geminiOmniVersion(model string) (major, minor int, ok bool) {
	match := geminiOmniPattern.FindStringSubmatch(lastSegment(strings.ToLower(model)))
	if match == nil {
		return 0, 0, false
	}
	major = 1
	if match[1] != "" {
		major = atoi(match[1])
	}
	return major, atoi(match[2]), true
}

// IsGeminiOmniVideoModel reports Gemini Omni video models, which run only on the Vertex AI Interactions API.
func IsGeminiOmniVideoModel(model string) bool {
	_, _, ok := geminiOmniVersion(model)
	return ok
}

func geminiOmniVersionGTE(model string, targetMajor, targetMinor int) bool {
	major, minor, ok := geminiOmniVersion(model)
	return ok && atLeast(major, minor, targetMajor, targetMinor)
}

// supportsGeminiXHighThinking reports the Gemini Hydrogen tier, which adds the xhigh thinking level.
func supportsGeminiXHighThinking(model string) bool {
	return geminiXHighPattern.MatchString(strings.ToLower(model))
}

// isGeminiLongDecodeModel reports Gemini 4+ chat models, whose output may continue across requests up to the
// unified input-plus-output window on Vertex AI.
func isGeminiLongDecodeModel(model string) bool {
	if geminiLongDecodeExclude.MatchString(model) {
		return false
	}
	return geminiVersionGTE(model, 4, 0)
}

// geminiVersionGTE compares the version of a `gemini-X.Y` model ID (any path prefix) with a target version.
func geminiVersionGTE(model string, targetMajor, targetMinor int) bool {
	match := geminiVersionPattern.FindStringSubmatch(lastSegment(model))
	if match == nil {
		return false
	}
	return atLeast(atoi(match[1]), atoi(match[2]), targetMajor, targetMinor)
}

// claudeVersionAtLeast reports whether a parsed Claude version is at least major.minor.
func claudeVersionAtLeast(version core.ClaudeVersion, major, minor int) bool {
	return atLeast(version.Major, version.Minor, major, minor)
}

var (
	gptVersionPattern = regexp.MustCompile(`(?:^|[./:])(?:openai\.)?gpt-(?:daybreak-[a-z]+-)?(\d+)(?:\.(\d+))?(?:-|[.:@]|$)`)
	gptProPattern     = regexp.MustCompile(`(?i)(?:^|[./:])(?:openai\.)?gpt-\d+(?:\.\d+)?-pro(?:-|[.:@]|$)`)
	gptAstraPattern   = regexp.MustCompile(`(?i)(?:^|[-_.])astra(?:[-_.]|$)`)
	gptSolPattern     = regexp.MustCompile(`(?i)(?:^|[-_.])sol(?:[-_.]|$)`)
)

// parseGptVersion parses GPT versions from bare IDs, snapshots and provider-qualified paths, including the
// restricted Daybreak models (`openai.gpt-daybreak-blue-5.6-sol`).
func parseGptVersion(model string) (major, minor int, ok bool) {
	match := gptVersionPattern.FindStringSubmatch(strings.ToLower(model))
	if match == nil {
		return 0, 0, false
	}
	return atoi(match[1]), atoi(match[2]), true
}

func gptVersionGTE(model string, targetMajor, targetMinor int) bool {
	major, minor, ok := parseGptVersion(model)
	return ok && atLeast(major, minor, targetMajor, targetMinor)
}

func isGptProModel(model string) bool { return gptProPattern.MatchString(model) }

func isGptAstraModel(model string) bool {
	return gptVersionGTE(model, 6, 0) && gptAstraPattern.MatchString(model)
}

// openAIReasoningEffortLevels lists the reasoning efforts of GPT-5 and later, by version threshold.
func openAIReasoningEffortLevels(model string) []string {
	if !gptVersionGTE(model, 5, 0) {
		return nil
	}
	if isGptProModel(model) {
		if gptVersionGTE(model, 5, 2) {
			return []string{"medium", "high", "xhigh"}
		}
		return []string{"high"}
	}
	if gptVersionGTE(model, 5, 6) {
		levels := []string{"low", "medium", "high", "xhigh", "max"}
		// Astra and GPT-6.1+ Sol reject `none`; GPT-6.0 Sol and Luna accept it.
		reasoningSol := gptVersionGTE(model, 6, 1) && gptSolPattern.MatchString(model)
		if !isGptAstraModel(model) && !reasoningSol {
			return append([]string{"none"}, levels...)
		}
		return levels
	}
	if gptVersionGTE(model, 5, 2) {
		return []string{"none", "low", "medium", "high", "xhigh"}
	}
	if gptVersionGTE(model, 5, 1) {
		return []string{"none", "low", "medium", "high"}
	}
	return []string{"minimal", "low", "medium", "high"}
}

// lastSegment returns the part of a model path after the last slash.
func lastSegment(model string) string {
	if index := strings.LastIndex(model, "/"); index != -1 {
		return model[index+1:]
	}
	return model
}
