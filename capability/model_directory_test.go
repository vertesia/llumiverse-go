package capability

import (
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

// Ported from common/src/model-directory.test.ts. The getOptions assertions are not ported: option metadata is not
// part of this package. An undefined limit is 0 and undefined effort levels are nil.

func TestOpenRouterKeepsSourceSemantics(t *testing.T) {
	profile := ResolveProfile("google/gemini-3.5-flash", common.ProviderOpenRouter)
	checkString(t, "family", profile.Family, "gemini")
	checkString(t, "source_provider", profile.SourceProvider, "google")
	checkLevels(t, "reasoning_effort_levels", profile.ReasoningEffortLevels, []string{"minimal", "low", "medium", "high"})
}

func TestVertexGrokOutputLimitsUnknown(t *testing.T) {
	for _, model := range []string{
		"grok-4.3",
		"grok-4.6",
		"grok-4.7",
		"grok-4.20-reasoning",
		"grok-4.20-non-reasoning",
		"grok-4.1-fast-reasoning",
		"grok-4.1-fast-non-reasoning",
		"grok-4.8",
		"grok-5",
	} {
		id := "locations/global/publishers/xai/models/" + model
		checkInt(t, id+" max_output_tokens", ResolveProfile(id, common.ProviderVertexAI).MaxOutputTokens, 0)
	}
}

func TestVertexMaaSLimitsDistinctFromDirect(t *testing.T) {
	const model = "locations/global/publishers/zai-org/models/glm-5.2-maas"
	profile := ResolveProfile(model, common.ProviderVertexAI)
	checkInt(t, "context_window", profile.ContextWindow, 1_000_000)
	checkInt(t, "max_output_tokens", profile.MaxOutputTokens, 64_000)
	checkInt(t, "vertex grok-4.7", ResolveProfile("grok-4.7", common.ProviderVertexAI).ContextWindow, 524_288)
	checkInt(t, "xai grok-4.7", ResolveProfile("grok-4.7", common.ProviderXAI).ContextWindow, 500_000)
}

func TestOpenAICompatibleGemini(t *testing.T) {
	const model = "google/gemini-3.5-flash"
	profile := ResolveProfile(model, common.ProviderOpenAICompatible)
	caps := ForModel(model, common.ProviderOpenAICompatible)
	checkString(t, "family", profile.Family, "gemini")
	checkString(t, "source_provider", profile.SourceProvider, "google")
	checkFlag(t, "input.image", caps.Input.Image, T)
	checkFlag(t, "tool_support", caps.ToolSupport, T)
}

func TestVertexGeminiFamily(t *testing.T) {
	checkString(t, "family", ResolveProfile("gemini-3.5-flash", common.ProviderVertexAI).Family, "gemini")
}

func TestBedrockMantleOverrides(t *testing.T) {
	direct := ResolveProfile("gpt-5.5", common.ProviderOpenAI)
	mantle := ResolveProfile("openai.gpt-5.5", common.ProviderBedrockMantle)
	checkInt(t, "direct context_window", direct.ContextWindow, 1_050_000)
	checkInt(t, "mantle context_window", mantle.ContextWindow, 1_050_000)
	checkInt(t, "mantle max_output_tokens", mantle.MaxOutputTokens, 128_000)
}

func TestVertexCapabilitiesInherited(t *testing.T) {
	for _, tc := range []struct{ model, latestKnown string }{
		{"gemma-5-27b-it-maas", "gemma-4-26b-a4b-it-maas"},
		{"mistral-small-2703", "mistral-small-2503"},
		{"mistral-medium-4", "mistral-medium-3"},
		{"codestral-3", "codestral-2"},
	} {
		checkCapsEqual(t, tc.model, ResolveProfile(tc.model, common.ProviderVertexAI).Capabilities,
			ResolveProfile(tc.latestKnown, common.ProviderVertexAI).Capabilities)
	}
}

func TestFutureHostedGrokLimits(t *testing.T) {
	checkInt(t, "vertex grok-4.8", ResolveProfile("grok-4.8", common.ProviderVertexAI).ContextWindow, 524_288)
	checkInt(t, "xai grok-4.8", ResolveProfile("grok-4.8", common.ProviderXAI).ContextWindow, 500_000)
}

func TestVersionRulesCarryIntoFutureReleases(t *testing.T) {
	checkLevels(t, "gpt-6.1", ResolveProfile("gpt-6.1", common.ProviderOpenAI).ReasoningEffortLevels,
		[]string{"none", "low", "medium", "high", "xhigh", "max"})
	checkString(t, "gemini-4.0-flash family",
		ResolveProfile("gemini-4.0-flash", common.ProviderOpenAICompatible).Family, "gemini")
	for _, tc := range []struct {
		model string
		want  []string
	}{
		{"google/gemini-4-argon", []string{"low", "medium", "high"}},
		{"google/gemini-4-carbon", []string{"low", "medium", "high"}},
		// OpenRouter lists xhigh, high, medium and low for Gemini 4 Hydrogen.
		{"google/gemini-4-hydrogen", []string{"low", "medium", "high", "xhigh"}},
		{"google/gemini-5-hydrogen:batch", []string{"low", "medium", "high", "xhigh"}},
	} {
		checkLevels(t, tc.model, ResolveProfile(tc.model, common.ProviderOpenAICompatible).ReasoningEffortLevels, tc.want)
	}
	checkString(t, "anthropic/claude-opus-6 family",
		ResolveProfile("anthropic/claude-opus-6", common.ProviderOpenAICompatible).Family, "claude")
}

func TestOpenAIVariantLimits(t *testing.T) {
	for _, tc := range []struct {
		model           string
		context, output int
	}{
		{"gpt-5.4-mini", 400_000, 128_000},
		{"gpt-5.4-nano", 400_000, 128_000},
		{"gpt-5-chat-latest", 128_000, 16_384},
		{"chat-latest", 400_000, 128_000},
	} {
		profile := ResolveProfile(tc.model, common.ProviderOpenAI)
		checkInt(t, tc.model+" context_window", profile.ContextWindow, tc.context)
		checkInt(t, tc.model+" max_output_tokens", profile.MaxOutputTokens, tc.output)
	}
	checkString(t, "gpt-5-chat-latest family", ResolveProfile("gpt-5-chat-latest", common.ProviderOpenAI).Family, "gpt")
	latest := ResolveProfile("chat-latest", common.ProviderOpenAI)
	checkString(t, "chat-latest family", latest.Family, "gpt")
	checkString(t, "chat-latest source_provider", latest.SourceProvider, "openai")
	checkCaps(t, "chat-latest", latest.Capabilities,
		capsMatch{in: mods{"text": true, "image": true}, out: mods{"text": true}, tool: T})
}

func TestEmbeddingsAreNonInference(t *testing.T) {
	profile := ResolveProfile("google/text-embedding-005", common.ProviderOpenAICompatible)
	checkString(t, "family", profile.Family, "embedding")
	checkFlag(t, "output.embed", profile.Capabilities.Output.Embed, T)
	checkFlag(t, "tool_support", profile.Capabilities.ToolSupport, F)
}

func TestCompatibleGptAudioOutput(t *testing.T) {
	checkFlag(t, "gpt-audio output.audio", ForModel("gpt-audio", common.ProviderOpenAICompatible).Output.Audio, T)
}

func TestListingAliasesAndQualifiedIDs(t *testing.T) {
	compat := common.ProviderOpenAICompatible
	checkCaps(t, "~openai/gpt-latest", ResolveProfile("~openai/gpt-latest", compat).Capabilities,
		capsMatch{in: mods{"text": true, "image": true}, out: mods{"text": true}, tool: T})
	checkString(t, "~google/gemini-flash-latest", ResolveProfile("~google/gemini-flash-latest", compat).Family, "gemini")
	checkString(t, "google/gemma-4-E4B-it", ResolveProfile("google/gemma-4-E4B-it", compat).Family, "gemma")
	checkString(t, "bedrock claude ARN", ResolveProfile(
		"arn:aws:bedrock:us-east-1:381492301031:inference-profile/global.anthropic.claude-sonnet-4-6",
		common.ProviderBedrock).Family, "claude")
	checkString(t, "OPENAI/GPT-5.6-SOL canonical_id", ResolveProfile("OPENAI/GPT-5.6-SOL", compat).CanonicalID, "gpt-5.6-sol")
	checkFlag(t, "openai/gpt-oss-120b input.image",
		ResolveProfile("openai/gpt-oss-120b", compat).Capabilities.Input.Image, F)
	checkString(t, "o5-preview", ResolveProfile("o5-preview", common.ProviderOpenAI).Family, "gpt")
	checkString(t, "openai/gpt-5.6-sol-batch", ResolveProfile("openai/gpt-5.6-sol-batch", compat).Family, "gpt")
}

func TestOpenModelFamiliesOnTogetherAndCompatible(t *testing.T) {
	checkCaps(t, "google/gemma-4-31B-it", ResolveProfile("google/gemma-4-31B-it", common.ProviderTogetherAI).Capabilities,
		capsMatch{in: mods{"text": true, "image": true}, out: mods{"text": true}, tool: T})
	checkFlag(t, "google/gemma-3-1B-it input.image",
		ResolveProfile("google/gemma-3-1B-it", common.ProviderTogetherAI).Capabilities.Input.Image, F)
	checkFlag(t, "llama-4-scout input.image",
		ResolveProfile("llama-4-scout", common.ProviderOpenAICompatible).Capabilities.Input.Image, T)
	checkFlag(t, "qwen3-vl-32b input.image",
		ResolveProfile("qwen3-vl-32b", common.ProviderOpenAICompatible).Capabilities.Input.Image, T)
}

func TestSourceIdentitySeparateFromTransportLimits(t *testing.T) {
	checkInt(t, "compatible", ResolveProfile("openai.gpt-5.6-sol", common.ProviderOpenAICompatible).ContextWindow, 1_050_000)
	checkInt(t, "mantle", ResolveProfile("openai.gpt-5.6-sol", common.ProviderBedrockMantle).ContextWindow, 1_000_000)
}

func TestUnknownModelsAreConservative(t *testing.T) {
	profile := ResolveProfile("future-provider/new-capability-v1", common.ProviderOpenAICompatible)
	checkCaps(t, "compatible", profile.Capabilities, capsMatch{in: mods{"text": true}, out: mods{"text": true}})
	checkFlag(t, "tool_support", profile.Capabilities.ToolSupport, nil)
	checkInt(t, "context_window", profile.ContextWindow, 0)
	checkInt(t, "max_output_tokens", profile.MaxOutputTokens, 0)
	checkCaps(t, "vertex", ResolveProfile("future-provider/new-capability-v1", common.ProviderVertexAI).Capabilities,
		capsMatch{in: mods{"text": true}, out: mods{"text": true}})
	checkString(t, "vectorized-chat-v1", ResolveProfile("vectorized-chat-v1", common.ProviderOpenAICompatible).Family, "generic")
}

func TestFutureLlamaAcrossTransports(t *testing.T) {
	checkCaps(t, "compatible", ResolveProfile("meta-llama/llama-5-scout", common.ProviderOpenAICompatible).Capabilities,
		capsMatch{in: mods{"text": true, "image": true}, out: mods{"text": true}, tool: T, streaming: T})
	bedrock := ResolveProfile("meta.llama5-scout", common.ProviderBedrock)
	checkCaps(t, "bedrock", bedrock.Capabilities,
		capsMatch{in: mods{"text": true, "image": true}, out: mods{"text": true}, tool: T, streaming: F})
	checkInt(t, "bedrock context_window", bedrock.ContextWindow, 10_000_000)
	checkCaps(t, "vertex", ResolveProfile("llama-5-scout-maas", common.ProviderVertexAI).Capabilities,
		capsMatch{in: mods{"text": true, "image": true}, out: mods{"text": true}, tool: T})
	onFoundry := ResolveProfile("llama-5-scout", common.ProviderAzureFoundry)
	checkFlag(t, "foundry input.image", onFoundry.Capabilities.Input.Image, T)
	checkInt(t, "foundry context_window", onFoundry.ContextWindow, 10_000_000)
}

func TestCurrentMistralFamilyRules(t *testing.T) {
	small := ResolveProfile("MISTRAL-SMALL-2603", common.ProviderMistralAI)
	checkString(t, "small family", small.Family, "mistral")
	checkString(t, "small source_provider", small.SourceProvider, "mistralai")
	checkInt(t, "small context_window", small.ContextWindow, 256_000)
	checkCaps(t, "small", small.Capabilities,
		capsMatch{in: mods{"text": true, "image": true}, out: mods{"text": true}, tool: T})
	checkLevels(t, "small reasoning_effort_levels", small.ReasoningEffortLevels, []string{"none", "high"})
	checkInt(t, "small max_output_tokens", small.MaxOutputTokens, 0)
	checkLevels(t, "mistral-medium-3.6",
		ResolveProfile("mistral-medium-3.6", common.ProviderMistralAI).ReasoningEffortLevels, []string{"none", "high"})
	checkFlag(t, "voxtral-small-latest input.audio",
		ResolveProfile("voxtral-small-latest", common.ProviderMistralAI).Capabilities.Input.Audio, T)
	checkCaps(t, "voxtral-mini-2507", ResolveProfile("voxtral-mini-2507", common.ProviderMistralAI).Capabilities,
		capsMatch{in: mods{"text": true, "audio": true}, out: mods{"text": true}, tool: F})
	checkCaps(t, "voxtral-mini-transcribe-2602",
		ResolveProfile("voxtral-mini-transcribe-2602", common.ProviderMistralAI).Capabilities,
		capsMatch{in: mods{"audio": true}, out: mods{"text": true}, tool: F})
	checkCaps(t, "mistral-tts-latest", ResolveProfile("mistral-tts-latest", common.ProviderMistralAI).Capabilities,
		capsMatch{in: mods{"text": true}, out: mods{"audio": true}, tool: F})
	leanstral := ResolveProfile("labs-leanstral-1-5", common.ProviderMistralAI)
	checkString(t, "leanstral family", leanstral.Family, "mistral")
	checkInt(t, "leanstral context_window", leanstral.ContextWindow, 256_000)
	checkFlag(t, "leanstral tool_support", leanstral.Capabilities.ToolSupport, T)
}

func TestMistralLarge4Reasoning(t *testing.T) {
	for _, model := range []string{"mistral-large-4", "mistral-large-4-0", "mistral-large-5"} {
		profile := ResolveProfile(model, common.ProviderMistralAI)
		checkString(t, model+" family", profile.Family, "mistral")
		checkInt(t, model+" context_window", profile.ContextWindow, 1_000_000)
		checkCaps(t, model, profile.Capabilities, capsMatch{in: mods{"text": true, "image": true}, tool: T})
		checkLevels(t, model+" reasoning_effort_levels", profile.ReasoningEffortLevels, []string{"none", "high"})
		checkInt(t, model+" max_output_tokens", profile.MaxOutputTokens, 0)
	}
}

func TestEarlierMistralLargeLimits(t *testing.T) {
	profile := ResolveProfile("mistral-large-2411", common.ProviderMistralAI)
	checkInt(t, "context_window", profile.ContextWindow, 256_000)
	checkLevels(t, "reasoning_effort_levels", profile.ReasoningEffortLevels, nil)
}

func TestGLM53(t *testing.T) {
	for _, tc := range []struct {
		model    string
		provider common.Provider
	}{
		{"z-ai/glm-5.3", common.ProviderOpenAICompatible},
		{"zai-org/GLM-5.3", common.ProviderTogetherAI},
		{"zai-glm-5-3", common.ProviderMistralAI},
	} {
		profile := ResolveProfile(tc.model, tc.provider)
		checkString(t, tc.model+" family", profile.Family, "glm")
		checkInt(t, tc.model+" context_window", profile.ContextWindow, 1_000_000)
		checkInt(t, tc.model+" max_output_tokens", profile.MaxOutputTokens, 131_072)
		checkLevels(t, tc.model+" reasoning_effort_levels", profile.ReasoningEffortLevels, []string{"low", "high", "max"})
	}
}

func TestTogetherQwenReasoningControl(t *testing.T) {
	effort := []string{"low", "medium", "xhigh"}
	for _, tc := range []struct {
		model  string
		levels []string
	}{
		// Hybrid generations: thinking can only be switched off.
		{"Qwen/Qwen3.5-9B", []string{"none"}},
		{"Qwen/Qwen3.6-Plus", []string{"none"}},
		{"Qwen/Qwen3.7-Max", []string{"none"}},
		// Effort generations, including a plausible future release.
		{"Qwen/Qwen3.8-Flash", effort},
		{"Qwen/Qwen3.8-2.4T-A95B", effort},
		{"Qwen/Qwen4-Plus", effort},
		// Earlier generations and coder models have no documented Together reasoning control.
		{"Qwen/Qwen3-235B-A22B-Instruct-2507-tput", nil},
		{"Qwen/Qwen3-Coder-Next", nil},
	} {
		checkLevels(t, tc.model, ResolveProfile(tc.model, common.ProviderTogetherAI).ReasoningEffortLevels, tc.levels)
	}
	// The rule is Together-specific: other providers keep their own Qwen behaviour.
	checkLevels(t, "compatible qwen/qwen3.5-9b",
		ResolveProfile("qwen/qwen3.5-9b", common.ProviderOpenAICompatible).ReasoningEffortLevels, nil)
}

func TestGroqQwenReasoningControl(t *testing.T) {
	for _, tc := range []struct {
		model  string
		levels []string
	}{
		{"qwen/qwen3.8-27b", []string{"none", "low", "medium", "high"}},
		{"qwen/qwen3.6-27b", []string{"none"}},
		// Qwen 3 32B, not 3.32: the dash starts the size, so it keeps the Groq Qwen 3 on/off control.
		{"qwen/qwen3-32b", []string{"none"}},
	} {
		checkLevels(t, tc.model, ResolveProfile(tc.model, common.ProviderGroq).ReasoningEffortLevels, tc.levels)
	}
}

func TestGroqQwen38OutputLimit(t *testing.T) {
	checkInt(t, "qwen/qwen3.8-27b", ResolveProfile("qwen/qwen3.8-27b", common.ProviderGroq).MaxOutputTokens, 16_384)
	if got := ResolveProfile("qwen/qwen3-27b", common.ProviderGroq).MaxOutputTokens; got == 16_384 {
		t.Errorf("qwen/qwen3-27b max_output_tokens = %d, want anything but 16384", got)
	}
}

func TestOpenRouterHostedModels(t *testing.T) {
	museLatest := []string{"minimal", "low", "medium", "high", "xhigh", "max"}
	for _, tc := range []struct {
		model, family   string
		context, output int
		effort          []string
	}{
		{"meta/muse-spark-1.3", "muse", 1_048_576, 0, museLatest},
		{"meta/muse-wonder", "muse", 1_048_576, 0, museLatest},
		{"meta/muse-ember", "muse", 1_048_576, 0, museLatest},
		{"meta/muse-spark-1.1", "muse", 1_048_576, 0, []string{"minimal", "low", "medium", "high", "xhigh"}},
		{"meta/muse-glimmer-30b", "muse", 131_072, 0, []string{"low", "medium", "high", "xhigh"}},
		{"stepfun/step-5-preview", "step", 1_048_576, 64_000, []string{"low", "medium", "high"}},
		{"xiaomi/mimo-v2.6-pro", "mimo", 1_050_000, 131_072, nil},
		{"minimax/minimax-m3", "minimax", 1_048_576, 512_000, nil},
	} {
		profile := ResolveProfile(tc.model, common.ProviderOpenAICompatible)
		checkString(t, tc.model+" family", profile.Family, tc.family)
		checkInt(t, tc.model+" context_window", profile.ContextWindow, tc.context)
		checkFlag(t, tc.model+" tool_support", profile.Capabilities.ToolSupport, T)
		checkInt(t, tc.model+" max_output_tokens", profile.MaxOutputTokens, tc.output)
		checkFlag(t, tc.model+" input.image", profile.Capabilities.Input.Image, T)
		checkLevels(t, tc.model+" reasoning_effort_levels", profile.ReasoningEffortLevels, tc.effort)
	}
}

func TestAggregatorLimits(t *testing.T) {
	// An output of 0 is not checked, as the TypeScript test skips undefined outputs.
	for _, tc := range []struct {
		model           string
		context, output int
	}{
		{"anthropic/claude-opus-4.6", 1_000_000, 128_000},
		{"anthropic/claude-opus-4.8:batch", 1_000_000, 128_000},
		{"anthropic/claude-3.7-sonnet", 200_000, 64_000},
		{"moonshotai/kimi-k3", 1_048_576, 0},
		{"moonshotai/kimi-k2.6", 262_144, 0},
		{"moonshotai/kimi-k2-0905", 262_144, 0},
		{"moonshotai/kimi-k2", 131_072, 0},
		{"nvidia/nemotron-3-super-120b-a12b", 256_000, 0},
		{"nvidia/nemotron-3.5-lightning", 256_000, 0},
		{"openai/gpt-5.2-chat", 128_000, 16_384},
		{"google/gemma-3-27b-it", 131_072, 0},
		{"mistralai/codestral-2508", 256_000, 0},
		{"mistralai/mistral-small-2603:batch", 256_000, 0},
	} {
		profile := ResolveProfile(tc.model, common.ProviderOpenAICompatible)
		checkInt(t, tc.model+" context_window", profile.ContextWindow, tc.context)
		if tc.output != 0 {
			checkInt(t, tc.model+" max_output_tokens", profile.MaxOutputTokens, tc.output)
		}
	}
}

func TestAggregatorImageInput(t *testing.T) {
	for _, tc := range []struct {
		model string
		image bool
	}{
		{"moonshotai/kimi-k2.5", true},
		{"moonshotai/kimi-k3", true},
		{"moonshotai/kimi-k2-0905", false},
		{"openai/o3-mini", false},
		{"openai/o4-mini", true},
		{"mistralai/mistral-small-2603:batch", true},
	} {
		checkBool(t, tc.model, isTrue(ResolveProfile(tc.model, common.ProviderOpenAICompatible).Capabilities.Input.Image), tc.image)
	}
}

func TestAggregatorAudioInput(t *testing.T) {
	for _, tc := range []struct {
		model string
		audio bool
	}{
		{"meta/muse-wonder", true},
		{"meta/muse-spark-1.3", false},
		{"xiaomi/mimo-v2.6-pro", true},
		{"stepfun/step-5-preview", false},
	} {
		checkBool(t, tc.model, isTrue(ResolveProfile(tc.model, common.ProviderOpenAICompatible).Capabilities.Input.Audio), tc.audio)
	}
}

func TestDeepSeekV4(t *testing.T) {
	pro := ResolveProfile("deepseek-v4-pro", common.ProviderOpenAICompatible)
	checkInt(t, "deepseek-v4-pro context_window", pro.ContextWindow, 1_048_576)
	checkInt(t, "deepseek-v4-pro max_output_tokens", pro.MaxOutputTokens, 393_216)
	checkCaps(t, "deepseek-v4-pro", pro.Capabilities, capsMatch{in: mods{"image": false}})
	checkFlag(t, "deepseek/deepseek-v4.1-flash input.image",
		ResolveProfile("deepseek/deepseek-v4.1-flash", common.ProviderOpenAICompatible).Capabilities.Input.Image, T)
	vertex := ResolveProfile("locations/global/publishers/deepseek-ai/models/deepseek-v4.1-flash-maas",
		common.ProviderVertexAI)
	checkInt(t, "vertex context_window", vertex.ContextWindow, 1_048_576)
	checkInt(t, "vertex max_output_tokens", vertex.MaxOutputTokens, 393_216)
	checkCaps(t, "vertex", vertex.Capabilities, capsMatch{in: mods{"text": true, "image": true}, tool: T})
}

func TestSourceSemanticsAfterProviderOverlays(t *testing.T) {
	for _, provider := range []common.Provider{common.ProviderOpenAI, common.ProviderAzureFoundry, common.ProviderBedrock} {
		checkCaps(t, "gpt-image-1 on "+string(provider), ResolveProfile("gpt-image-1", provider).Capabilities,
			capsMatch{in: mods{"text": true, "image": true}, out: mods{"text": false, "image": true}, tool: F, streaming: F})
	}
	checkString(t, "image-deployment::gpt-image-1",
		ResolveProfile("image-deployment::gpt-image-1", common.ProviderAzureFoundry).Family, "image")
	checkString(t, "speech-deployment::gpt-4o-mini-tts",
		ResolveProfile("speech-deployment::gpt-4o-mini-tts", common.ProviderAzureFoundry).Family, "speech")
}

func TestOpenAIAudioModels(t *testing.T) {
	audio := ResolveProfile("gpt-audio-1.5", common.ProviderOpenAI)
	checkString(t, "gpt-audio-1.5 family", audio.Family, "audio")
	checkInt(t, "gpt-audio-1.5 context_window", audio.ContextWindow, 128_000)
	checkInt(t, "gpt-audio-1.5 max_output_tokens", audio.MaxOutputTokens, 16_384)
	checkCaps(t, "gpt-audio-1.5", audio.Capabilities,
		capsMatch{in: mods{"text": true, "audio": true}, out: mods{"text": true, "audio": true}, tool: F})

	realtime := ResolveProfile("gpt-realtime-3", common.ProviderOpenAI)
	checkString(t, "gpt-realtime-3 family", realtime.Family, "realtime")
	checkInt(t, "gpt-realtime-3 context_window", realtime.ContextWindow, 128_000)
	checkInt(t, "gpt-realtime-3 max_output_tokens", realtime.MaxOutputTokens, 32_000)

	translate := ResolveProfile("gpt-realtime-translate", common.ProviderOpenAI)
	checkString(t, "gpt-realtime-translate family", translate.Family, "realtime")
	checkInt(t, "gpt-realtime-translate context_window", translate.ContextWindow, 16_000)
	checkInt(t, "gpt-realtime-translate max_output_tokens", translate.MaxOutputTokens, 2_000)
	checkCaps(t, "gpt-realtime-translate", translate.Capabilities, capsMatch{
		in:  mods{"text": false, "image": false, "audio": true},
		out: mods{"text": true, "audio": true}, tool: F,
	})
}

func TestVertexTranscriptionAndSpeechModalities(t *testing.T) {
	checkCaps(t, "gemini-3.5-transcribe-preview", ForModel("gemini-3.5-transcribe-preview", common.ProviderVertexAI),
		capsMatch{
			in:  mods{"audio": true, "text": false, "image": false, "video": false},
			out: mods{"text": true, "audio": false}, tool: F,
		})
	checkCaps(t, "gemini-3.1-flash-tts-preview", ForModel("gemini-3.1-flash-tts-preview", common.ProviderVertexAI),
		capsMatch{
			in:  mods{"text": true, "audio": false, "image": false, "video": false},
			out: mods{"audio": true, "text": false},
		})
}

func TestVertexImageGenerationCapabilities(t *testing.T) {
	for _, tc := range []struct {
		model string
		video bool
	}{
		{"gemini-3-pro-image", false},
		{"gemini-3-pro-image-preview", false},
		{"gemini-4-pro-image", false},
		{"gemini-3.1-flash-lite-image", true},
		{"gemini-3.1-flash-lite-image-preview", true},
		{"gemini-4-flash-lite-image", true},
	} {
		id := "locations/global/publishers/google/models/" + tc.model
		checkCaps(t, tc.model, ForModel(id, common.ProviderVertexAI), capsMatch{
			in:   mods{"text": true, "image": true, "video": tc.video, "audio": false},
			out:  mods{"text": true, "image": true, "video": false, "audio": false},
			tool: F,
		})
	}
}

func TestVertexTranscriptionCapabilities(t *testing.T) {
	for _, model := range []string{"gemini-3.5-transcribe", "gemini-3.5-transcribe-preview", "gemini-4-transcribe"} {
		id := "locations/global/publishers/google/models/" + model
		checkCaps(t, model, ForModel(id, common.ProviderVertexAI), capsMatch{
			in:   mods{"text": false, "image": false, "video": false, "audio": true},
			out:  mods{"text": true, "image": false, "video": false, "audio": false},
			tool: F,
		})
	}
}

func checkImageGenerationProfile(t *testing.T, model string) {
	t.Helper()
	profile := ResolveProfile(model, common.ProviderVertexAI)
	checkInt(t, model+" context_window", profile.ContextWindow, 131_072)
	checkInt(t, model+" max_output_tokens", profile.MaxOutputTokens, 32_768)
	checkCaps(t, model, profile.Capabilities, capsMatch{
		in:   mods{"text": true, "image": true, "video": true, "audio": false},
		out:  mods{"text": true, "image": true},
		tool: F,
	})
}

func TestGemini31FlashImageMetadata(t *testing.T) {
	checkImageGenerationProfile(t, "gemini-3.1-flash-image-preview")
}

func TestNanoBananaMetadata(t *testing.T) {
	for _, model := range []string{"gemini-nano-banana-2.1", "locations/global/publishers/google/models/gemini-nano-banana-3.0"} {
		checkImageGenerationProfile(t, model)
	}
}

func TestNemotronFamily(t *testing.T) {
	nano := ResolveProfile("nvidia.nemotron-nano-12b-v2-vl-bf16", common.ProviderBedrock)
	checkString(t, "nano family", nano.Family, "nemotron")
	checkString(t, "nano source_provider", nano.SourceProvider, "nvidia")
	checkInt(t, "nano context_window", nano.ContextWindow, 128_000)
	checkInt(t, "nano max_output_tokens", nano.MaxOutputTokens, 8_192)
	checkCaps(t, "nano", nano.Capabilities, capsMatch{in: mods{"text": true, "image": true}, out: mods{"text": true}})
	super := ResolveProfile("nvidia.nemotron-super-4-180b", common.ProviderBedrock)
	checkString(t, "super family", super.Family, "nemotron")
	checkInt(t, "super context_window", super.ContextWindow, 256_000)
	checkInt(t, "super max_output_tokens", super.MaxOutputTokens, 32_768)
}

func TestModerationModelsWithoutTools(t *testing.T) {
	profile := ResolveProfile("meta-llama/llama-prompt-guard-2-86m", common.ProviderGroq)
	checkString(t, "family", profile.Family, "moderation")
	checkCaps(t, "capabilities", profile.Capabilities,
		capsMatch{in: mods{"text": true}, out: mods{"text": true}, tool: F, streaming: F})
}

func TestReasoningEffortOnlyWhenTransportCompatible(t *testing.T) {
	for _, tc := range []struct {
		model    string
		provider common.Provider
		levels   []string
	}{
		{"openai/gpt-oss-120b", common.ProviderTogetherAI, []string{"low", "medium", "high"}},
		{"gpt-5.6-sol", common.ProviderTogetherAI, nil},
		{"mistral-small-latest", common.ProviderMistralAI, []string{"none", "high"}},
		{"grok-4.3", common.ProviderXAI, []string{"none", "low", "medium", "high"}},
		{"grok-4.5", common.ProviderXAI, []string{"low", "medium", "high"}},
		{"grok-4.20-multi-agent", common.ProviderXAI, []string{"low", "medium", "high", "xhigh"}},
	} {
		checkLevels(t, tc.model+" on "+string(tc.provider), ResolveProfile(tc.model, tc.provider).ReasoningEffortLevels, tc.levels)
	}
}

func TestFoundryClaudeSourceCapabilities(t *testing.T) {
	// The getOptions comparison (Anthropic options without `speed`) is not ported.
	for _, model := range []string{"claude-opus-5-5", "claude-sonnet-5-5-20260901", "claude-opus-6", "claude-fable-5-2"} {
		onFoundry := ResolveProfile("opaque::"+model, common.ProviderAzureFoundry)
		checkCapsEqual(t, model, onFoundry.Capabilities, ResolveProfile(model, common.ProviderAnthropic).Capabilities)
		checkFlag(t, model+" input.image", onFoundry.Capabilities.Input.Image, T)
		checkFlag(t, model+" tool_support", onFoundry.Capabilities.ToolSupport, T)
	}
}
