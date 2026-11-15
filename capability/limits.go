package capability

import (
	"regexp"
	"strings"

	"github.com/vertesia/llumiverse-go/core"
)

// claudeLimits returns Claude limits by parsed generation, so `claude-opus-4-6` and `claude-opus-4.6` agree.
func claudeLimits(model string) (contextWindow, maxOutput int) {
	claude, ok := core.ParseClaudeVersion(model)
	at := func(major, minor int) bool { return ok && claudeVersionAtLeast(claude, major, minor) }
	opus := ok && claude.Variant == "opus"
	// Sonnet 4.6 supports 128K output on Anthropic and Vertex AI; Bedrock Runtime lowers it to 64K.
	longContext := at(4, 7) || ((opus || (ok && claude.Variant == "sonnet")) && at(4, 6))
	maxOutput = 4_096 // older Claude 3
	switch {
	case longContext:
		maxOutput = 128_000
	case opus && at(4, 5):
		maxOutput = 64_000
	case opus && at(4, 0):
		maxOutput = 32_000 // Opus 4.0, 4.1
	case at(3, 7):
		maxOutput = 64_000 // Sonnet 4.x, Haiku 4.5, 3.7
	case at(3, 5):
		maxOutput = 8_192
	case strings.Contains(model, "opus-"):
		maxOutput = 32_000 // legacy undated Opus
	}
	if longContext {
		return 1_000_000, maxOutput
	}
	return 200_000, maxOutput
}

var (
	oSeriesPattern  = regexp.MustCompile(`(?:^|[~/.])o\d+(?:[-_.]|$)`)
	gpt5ChatPattern = regexp.MustCompile(`gpt-5(?:\.\d+)?-chat`)
	mistralFamilies = regexp.MustCompile(`(?:mistral|mixtral|ministral|magistral|voxtral|codestral|devstral|leanstral|mathstral|pixtral)`)
	kimiK2Long      = regexp.MustCompile(`kimi-k2-(?:thinking|0905)`)
)

// MaxOutputTokens returns the source model's output token limit, or 0 for dedicated APIs and models without a
// published limit. Provider limits are applied by ResolveProfile.
func MaxOutputTokens(model string) int {
	model = strings.ToLower(model)
	if strings.Contains(model, "claude") {
		_, maxOutput := claudeLimits(model)
		return maxOutput
	}
	if strings.Contains(model, "gemini") {
		_, maxOutput := GeminiModelLimits(model)
		return maxOutput
	}
	switch {
	case strings.Contains(model, "o1-mini"):
		return 65_536
	case strings.Contains(model, "o1"):
		return 100_000
	case oSeriesPattern.MatchString(model):
		return 100_000
	}
	major, minor, gpt := parseGptVersion(model)
	switch {
	// GPT-5.x chat snapshots keep the smaller chat-model limits.
	case gpt5ChatPattern.MatchString(model):
		return 16_384
	case isGptProModel(model) && gpt && major == 5 && minor == 0:
		return 272_000
	case gptVersionGTE(model, 5, 0):
		return 128_000
	case strings.Contains(model, "gpt-oss"):
		return 16_384
	case strings.Contains(model, "gpt-4o"):
		return 16_384
	case strings.Contains(model, "gpt-4"):
		return 8_192
	case strings.Contains(model, "gpt-3.5"):
		return 4_096
	case familyVersionGTE(model, "grok-", 4, 3):
		return 131_072
	case strings.Contains(model, "nova"):
		return 10_000
	case strings.Contains(model, "mistral"):
		return 8_192
	case strings.Contains(model, "deepseek-r1-0528"):
		return 32_768
	// DeepSeek V4 has a 1M context with up to 384K output. https://api-docs.deepseek.com/quick_start/pricing
	case deepSeekVersionGTE(model, 4, 0):
		return 393_216
	case deepSeekVersionGTE(model, 3, 2):
		return 65_536
	case strings.Contains(model, "deepseek"):
		return 128_000
	case strings.Contains(model, "qwen3-next"):
		return 262_144
	case strings.Contains(model, "qwen3-coder"):
		return 65_536
	case strings.Contains(model, "qwen"):
		return 32_768
	case strings.Contains(model, "kimi-k2-thinking"):
		return 262_144
	case familyVersionGTE(model, "minimax-m", 3, 0):
		return 512_000
	case strings.Contains(model, "minimax-m2"):
		return 196_608
	case familyVersionGTE(model, "glm-", 5, 3):
		return 131_072
	case strings.Contains(model, "glm-"):
		return 32_768
	// Xiaomi MiMo and StepFun Step, as published on OpenRouter.
	case familyVersionGTE(model, "mimo-v", 2, 6):
		return 131_072
	case familyVersionGTE(model, "step-", 5, 0):
		return 64_000
	// Meta publishes no Muse output limit, and hosting providers report different caps.
	case isMetaMuseModel(model):
		return 0
	case strings.Contains(model, "gemma-4"):
		return 128_000
	case strings.Contains(model, "llama"):
		return 8_192
	case strings.Contains(model, "command-a"):
		return 8_000
	case strings.Contains(model, "command"):
		return 4_096
	}
	return 8_192 // conservative default
}

// ContextWindowSize returns the source model's input-plus-output window, or 0 for an unknown model.
func ContextWindowSize(model string) int {
	model = strings.ToLower(model)
	if strings.Contains(model, "claude") {
		contextWindow, _ := claudeLimits(model)
		return contextWindow
	}
	if strings.Contains(model, "gemini") {
		contextWindow, _ := GeminiModelLimits(model)
		return contextWindow
	}
	switch {
	case oSeriesPattern.MatchString(model):
		return 200_000
	case gpt5ChatPattern.MatchString(model):
		return 128_000
	case containsAny(model, "gpt-5.4-mini", "gpt-5.4-nano"):
		return 400_000
	case gptVersionGTE(model, 5, 4):
		return 1_050_000
	case gptVersionGTE(model, 5, 0):
		return 400_000
	case strings.Contains(model, "gpt-oss"):
		return 131_072
	case containsAny(model, "gpt-4.1", "gpt-4-1"):
		return 1_000_000
	case containsAny(model, "gpt-4-turbo", "gpt-4o"):
		return 128_000
	case strings.Contains(model, "gpt-4"):
		return 8_000
	case strings.Contains(model, "gpt-3.5"):
		return 16_000
	case familyVersionGTE(model, "grok-", 4, 3):
		return 1_000_000
	case containsAny(model, "grok-4.1", "grok-4-1"):
		return 256_000
	case strings.Contains(model, "grok"):
		return 131_072
	case strings.Contains(model, "nova"):
		return 300_000
	case mistralFamilies.MatchString(model):
		return mistralModelKnowledge(model).ContextWindow
	case deepSeekVersionGTE(model, 4, 0):
		return 1_048_576
	case deepSeekVersionGTE(model, 3, 2):
		return 163_840
	case strings.Contains(model, "deepseek-r1-0528"):
		return 163_840
	case strings.Contains(model, "deepseek"):
		return 128_000
	case strings.Contains(model, "qwen"):
		return 262_144
	case kimiVersionGTE(model, 3, 0):
		return 1_048_576
	case kimiVersionGTE(model, 2, 5) || kimiK2Long.MatchString(model):
		return 262_144
	case strings.Contains(model, "kimi-k2"):
		return 131_072
	case familyVersionGTE(model, "minimax-m", 3, 0):
		return 1_048_576
	case strings.Contains(model, "minimax-m2"):
		return 196_608
	case familyVersionGTE(model, "glm-", 5, 3):
		return 1_000_000
	case strings.Contains(model, "glm-"):
		return 128_000
	// Meta Muse (Glimmer is the small 128K model), Xiaomi MiMo and StepFun Step.
	case strings.Contains(model, "muse-glimmer"):
		return 131_072
	case isMetaMuseModel(model):
		return 1_048_576
	case familyVersionGTE(model, "mimo-v", 2, 6):
		return 1_050_000
	case familyVersionGTE(model, "step-", 5, 0):
		return 1_048_576
	case strings.Contains(model, "gemma-4"):
		return 256_000
	// Gemma 3 1B is 32K; the larger Gemma 3 models are 128K.
	case strings.Contains(model, "gemma-3-1b"):
		return 32_768
	case strings.Contains(model, "gemma-3-"):
		return 131_072
	case familyVersionGTE(model, "llama-", 4, 0) || familyVersionGTE(model, "llama", 4, 0):
		if strings.Contains(model, "scout") {
			return 10_000_000
		}
		return 1_000_000
	case containsAny(model, "llama-3.1", "llama-3.2", "llama-3.3"):
		return 128_000
	case strings.Contains(model, "llama"):
		return 8_000
	case strings.Contains(model, "command-a"):
		return 256_000
	case strings.Contains(model, "command"):
		return 128_000
	}
	return 0
}

// MaxInputTokens returns the context window minus the output limit, or 0 when either is unknown.
func MaxInputTokens(model string) int {
	contextWindow, maxOutput := ContextWindowSize(model), MaxOutputTokens(model)
	if contextWindow == 0 || maxOutput == 0 {
		return 0
	}
	return contextWindow - maxOutput
}

var geminiDedicatedPattern = regexp.MustCompile(`(?:transcribe|live|robotics|tts|embedding)`)

// GeminiModelLimits returns documented Google model limits; later generations inherit the latest known family
// limits. Dedicated APIs (transcribe, live, robotics, tts, embedding) return 0, 0.
func GeminiModelLimits(model string) (contextWindow, maxOutput int) {
	name := lastSegment(strings.ToLower(model))
	switch {
	case geminiDedicatedPattern.MatchString(name):
		return 0, 0
	// https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/omni-1-2-flash
	case geminiOmniVersionGTE(name, 1, 2):
		return 131_072, 260_640
	case strings.Contains(name, "gemini-omni-"):
		return 131_072, 57_920
	case strings.Contains(name, "gemini-nano-banana"):
		return 131_072, 32_768
	// Dedicated image families have smaller windows than chat models; narrower variants first.
	case strings.Contains(name, "-flash-lite-image") && geminiVersionGTE(name, 3, 1):
		return 65_536, 4_096
	case strings.Contains(name, "-pro-image") && geminiVersionGTE(name, 3, 0):
		return 65_536, 32_768
	case strings.Contains(name, "-flash-image") && geminiVersionGTE(name, 3, 1):
		return 131_072, 32_768
	case strings.Contains(name, "-flash-image") && geminiVersionGTE(name, 2, 5):
		return 32_768, 32_768
	case geminiVersionGTE(name, 2, 5):
		return 1_048_576, 65_536
	// Older releases keep their documented limits.
	case strings.Contains(name, "thinking") && geminiVersionGTE(name, 2, 0):
		return 1_048_576, 65_535
	case geminiVersionGTE(name, 2, 0):
		return 1_048_576, 8_192
	case strings.Contains(name, "-1.0-"):
		return 32_000, 2_048
	}
	// Legacy unversioned and 1.x behaviour.
	return 1_000_000, 65_535
}

type limits struct{ contextWindow, maxOutput int }

// vertexModelLimits holds documented Vertex AI limits; newer generations inherit them in VertexAIModelLimits.
var vertexModelLimits = map[string]limits{
	"glm-5.2-maas":       {1_000_000, 64_000},
	"deepseek-v3.1-maas": {163_840, 32_768},
	// Google lists 1,048,576 for context and output; DeepSeek caps V4.1 Flash output at 384K.
	"deepseek-v4.1-flash-maas":    {1_048_576, 393_216},
	"grok-4.7":                    {524_288, 0},
	"grok-4.6":                    {524_288, 0},
	"grok-4.3":                    {200_000, 0},
	"grok-4.20-reasoning":         {2_000_000, 0},
	"grok-4.20-non-reasoning":     {2_000_000, 0},
	"grok-4.1-fast-reasoning":     {128_000, 0},
	"grok-4.1-fast-non-reasoning": {128_000, 0},
	"mistral-small-2503":          {128_000, 0},
	"mistral-medium-3":            {128_000, 0},
	"codestral-2":                 {128_000, 0},
}

// VertexAIModelLimits returns the limits Vertex AI documents for a hosted model, or 0, 0 when it documents none.
func VertexAIModelLimits(model string) (contextWindow, maxOutput int) {
	l := vertexAIModelLimits(model)
	return l.contextWindow, l.maxOutput
}

func vertexAIModelLimits(model string) limits {
	name := lastSegment(strings.ToLower(model))
	if exact, ok := vertexModelLimits[name]; ok {
		return exact
	}
	// Vertex AI long decodes continue output across requests, so output may use the whole window.
	if isGeminiLongDecodeModel(name) {
		if contextWindow, _ := GeminiModelLimits(name); contextWindow != 0 {
			return limits{contextWindow, contextWindow}
		}
	}
	switch {
	case strings.HasSuffix(name, "-maas") && familyVersionGTE(name, "glm-", 5, 2):
		return vertexModelLimits["glm-5.2-maas"]
	case strings.HasSuffix(name, "-flash-maas") && familyVersionGTE(name, "deepseek-v", 4, 1):
		return vertexModelLimits["deepseek-v4.1-flash-maas"]
	case singleDigitGrokVersionGTE(name, 4, 6):
		return vertexModelLimits["grok-4.7"]
	case singleDigitGrokVersionGTE(name, 4, 3):
		return vertexModelLimits["grok-4.3"]
	case familyVersionGTE(name, "mistral-small-", 2503, 0):
		return vertexModelLimits["mistral-small-2503"]
	case familyVersionGTE(name, "mistral-medium-", 3, 0):
		return vertexModelLimits["mistral-medium-3"]
	case familyVersionGTE(name, "codestral-", 2, 0):
		return vertexModelLimits["codestral-2"]
	}
	return limits{}
}
