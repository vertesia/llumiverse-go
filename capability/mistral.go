package capability

import (
	"regexp"
	"strings"
	"sync"
)

var patternCache sync.Map

// compiled returns the compiled pattern, compiling it once.
func compiled(pattern string) *regexp.Regexp {
	if re, ok := patternCache.Load(pattern); ok {
		return re.(*regexp.Regexp)
	}
	re, _ := patternCache.LoadOrStore(pattern, regexp.MustCompile(pattern))
	return re.(*regexp.Regexp)
}

// MistralKnowledge is what the catalogue knows about a Mistral model.
type MistralKnowledge struct {
	Capabilities Capabilities
	// ContextWindow is 0 when unknown.
	ContextWindow         int
	ReasoningEffortLevels []string
}

// datedVersionGTE checks a four-digit YYMM release after the family (`mistral-small-2503`).
func datedVersionGTE(model, family string, target int) bool {
	match := compiled(regexp.QuoteMeta(family) + `(\d{4})(?:[-_.]|$)`).FindStringSubmatch(model)
	return match != nil && atoi(match[1]) >= target
}

// semanticVersionGTE checks a major[.-minor] version after the family (`mistral-medium-3-5`).
func semanticVersionGTE(model, family string, targetMajor, targetMinor int) bool {
	match := compiled(regexp.QuoteMeta(family) + `(\d+)(?:[-.](\d+))?(?:[-_.]|$)`).FindStringSubmatch(model)
	return match != nil && atLeast(atoi(match[1]), atoi(match[2]), targetMajor, targetMinor)
}

func isMistralSmall4OrLater(model string) bool {
	return strings.Contains(model, "mistral-small-latest") ||
		datedVersionGTE(model, "mistral-small-", 2603) ||
		semanticVersionGTE(model, "mistral-small-", 4, 0)
}

func isMistralMedium35OrLater(model string) bool {
	return strings.Contains(model, "mistral-medium-latest") ||
		datedVersionGTE(model, "mistral-medium-", 2604) ||
		semanticVersionGTE(model, "mistral-medium-", 3, 5)
}

var mistralLargeGenerationPattern = regexp.MustCompile(`mistral-large-(\d{1,2})(?:[-_.:]|$)`)

// isMistralLargeGTE checks Mistral Large semantic generations (`mistral-large-4`, `mistral.mistral-large-3-...`).
// Dated legacy IDs such as `mistral-large-2402` are not generations.
func isMistralLargeGTE(model string, target int) bool {
	match := mistralLargeGenerationPattern.FindStringSubmatch(strings.ToLower(model))
	return match != nil && atoi(match[1]) >= target
}

// IsMistralAdjustableReasoningModel reports Mistral models whose reasoning can be switched with reasoning_effort:
// Small 4+, Medium 3.5+ and Large 4+.
func IsMistralAdjustableReasoningModel(model string) bool {
	normalized := strings.ToLower(model)
	return isMistralSmall4OrLater(normalized) || isMistralMedium35OrLater(normalized) || isMistralLargeGTE(normalized, 4)
}

var ministralDatedPattern = regexp.MustCompile(`ministral-(?:3|8|14)b-(?:latest|\d{4})(?:[-_.]|$)`)

func hasMistralVision(model string) bool {
	return strings.Contains(model, "pixtral") ||
		strings.Contains(model, "mistral-small-latest") ||
		datedVersionGTE(model, "mistral-small-", 2503) ||
		semanticVersionGTE(model, "mistral-small-", 3, 1) ||
		strings.Contains(model, "mistral-medium-latest") ||
		datedVersionGTE(model, "mistral-medium-", 2508) ||
		semanticVersionGTE(model, "mistral-medium-", 3, 1) ||
		strings.Contains(model, "mistral-large-latest") ||
		datedVersionGTE(model, "mistral-large-", 2512) ||
		semanticVersionGTE(model, "mistral-large-", 3, 0) ||
		ministralDatedPattern.MatchString(model)
}

var (
	magistralVersionPattern = regexp.MustCompile(`magistral-(?:small|medium)-(\d{4})(?:[-_.]|$)`)
	mistral24Pattern        = regexp.MustCompile(`mistral-(?:small|medium|large)-24\d{2}`)
)

func mistralContextWindow(model string) int {
	// https://docs.mistral.ai/models/mistral-large-4-0
	if isMistralLargeGTE(model, 4) {
		return 1_000_000
	}
	if strings.Contains(model, "voxtral-small") || strings.Contains(model, "voxtral-mini") {
		return 32_000
	}
	if strings.Contains(model, "magistral-") {
		match := magistralVersionPattern.FindStringSubmatch(model)
		if strings.Contains(model, "latest") || (match != nil && atoi(match[1]) >= 2509) {
			return 128_000
		}
		return 40_000
	}
	if isMistralSmall4OrLater(model) ||
		isMistralMedium35OrLater(model) ||
		strings.Contains(model, "mistral-large-latest") ||
		datedVersionGTE(model, "mistral-large-", 2512) ||
		semanticVersionGTE(model, "mistral-large-", 3, 0) ||
		ministralDatedPattern.MatchString(model) ||
		strings.Contains(model, "devstral-2") ||
		strings.Contains(model, "leanstral") ||
		datedVersionGTE(model, "codestral-", 2508) {
		return 256_000
	}
	if datedVersionGTE(model, "mistral-small-", 2503) ||
		semanticVersionGTE(model, "mistral-small-", 3, 1) ||
		datedVersionGTE(model, "mistral-medium-", 2508) ||
		semanticVersionGTE(model, "mistral-medium-", 3, 1) ||
		containsAny(model, "mistral-nemo", "open-mistral-nemo", "codestral", "pixtral") {
		return 128_000
	}
	if mistral24Pattern.MatchString(model) || strings.Contains(model, "open-mistral-7b") {
		return 32_000
	}
	return 0
}

var voxtralDedicatedPattern = regexp.MustCompile(`(transcribe|realtime)`)

// MistralModelKnowledge returns the capabilities, context window and reasoning efforts of a Mistral model.
func MistralModelKnowledge(model string) MistralKnowledge {
	knowledge := mistralModelKnowledge(model)
	knowledge.Capabilities = knowledge.Capabilities.clone()
	return knowledge
}

func mistralModelKnowledge(model string) MistralKnowledge {
	normalized := strings.ToLower(model)
	switch {
	case strings.Contains(normalized, "moderation") || strings.Contains(normalized, "shieldstral"):
		return MistralKnowledge{Capabilities: tools(some("text"), some("text"), false, false)}
	case strings.Contains(normalized, "mistral-ocr"):
		return MistralKnowledge{Capabilities: tools(some("text", "image"), some("text"), false, false)}
	case strings.Contains(normalized, "tts"):
		return MistralKnowledge{Capabilities: tools(some("text"), some("audio"), false, false)}
	case strings.Contains(normalized, "voxtral-mini") && voxtralDedicatedPattern.MatchString(normalized):
		return MistralKnowledge{Capabilities: tools(some("audio"), some("text"), false, false)}
	}
	voxtralMini := strings.Contains(normalized, "voxtral-mini")
	knowledge := MistralKnowledge{
		Capabilities: Capabilities{
			Input: Modalities{
				Text:  yes,
				Image: flagOrNil(hasMistralVision(normalized)),
				Audio: flagOrNil(strings.Contains(normalized, "voxtral-small") || voxtralMini),
			},
			Output:               some("text"),
			ToolSupport:          flag(!voxtralMini),
			ToolSupportStreaming: flag(!voxtralMini),
		},
		ContextWindow: mistralContextWindow(normalized),
	}
	if IsMistralAdjustableReasoningModel(normalized) {
		knowledge.ReasoningEffortLevels = []string{"none", "high"}
	}
	return knowledge
}
