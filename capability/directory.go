package capability

import (
	"regexp"
	"slices"
	"strings"

	"github.com/vertesia/llumiverse-go/common"
)

// profileOverride is an exact source-model rule that wins over family inference.
type profileOverride struct {
	family, sourceProvider string
	contextWindow          int
	maxOutput              int
	capabilities           *Capabilities
}

var exactModelOverrides = map[string]profileOverride{
	"chat-latest": {family: "gpt", sourceProvider: "openai", contextWindow: 400_000, maxOutput: 128_000},
	// OpenAI image endpoints are not text inference models, although their IDs contain `gpt`.
	"gpt-image-1": {family: "image", capabilities: &Capabilities{
		Input:                Modalities{Text: yes, Image: yes, Audio: no, Video: no},
		Output:               Modalities{Text: no, Image: yes, Audio: no, Video: no},
		ToolSupport:          no,
		ToolSupportStreaming: no,
	}},
	"gpt-realtime-translate": {family: "realtime", contextWindow: 16_000, maxOutput: 2_000, capabilities: &Capabilities{
		Input:                Modalities{Text: no, Image: no, Audio: yes, Video: no},
		Output:               Modalities{Text: yes, Image: no, Audio: yes, Video: no},
		ToolSupport:          no,
		ToolSupportStreaming: no,
	}},
	"gpt-5-chat-latest": {family: "gpt", sourceProvider: "openai", contextWindow: 128_000, maxOutput: 16_384},
}

var (
	regionPrefix    = regexp.MustCompile(`^(?:global|us|eu|apac)\.`)
	publisherPrefix = regexp.MustCompile(`^(?:openai|anthropic|google|xai)\.`)
	// OpenRouter routing variants select a serving mode, not a different model.
	routingSuffix = regexp.MustCompile(`:(?:batch|free|nitro|floor|extended|exacto|online|thinking)$`)
)

// canonicalModelID returns the source model an ID names: lower case, without a leading `~`, the deployment half of
// an Azure Foundry `deployment::model` ID, the path, the region and publisher prefixes and an OpenRouter routing
// suffix.
func canonicalModelID(model string) string {
	normalized := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(model)), "~")
	// Azure Foundry IDs are deployment::source-model; the family follows the source model, not the deployment name.
	if strings.Contains(normalized, "::") {
		parts := strings.Split(normalized, "::")
		normalized = parts[len(parts)-1]
	}
	leaf := lastSegment(normalized)
	leaf = regionPrefix.ReplaceAllString(leaf, "")
	leaf = publisherPrefix.ReplaceAllString(leaf, "")
	return routingSuffix.ReplaceAllString(leaf, "")
}

var (
	embeddingPattern     = regexp.MustCompile(`(^|[-_.:])(?:embed|embedding|vector)(?:[-_.:]|$)`)
	moderationPattern    = regexp.MustCompile(`(?:prompt-guard|moderation|safeguard)`)
	realtimePattern      = regexp.MustCompile(`(?:realtime|live|native-audio)`)
	transcriptionPattern = regexp.MustCompile(`(?:whisper|transcribe)`)
	audioPattern         = regexp.MustCompile(`(?:^|[-_.])audio(?:[-_.]|$)`)
	oModelPattern        = regexp.MustCompile(`^o\d+(?:[-_.]|$)`)
	mimoPattern          = regexp.MustCompile(`(?:^|[-_.])mimo-v\d`)
	stepPattern          = regexp.MustCompile(`(?:^|[-_.])step-\d`)
)

// inferFamily returns the family and maker of a model from its source ID.
func inferFamily(model string) (family, sourceProvider string) {
	normalized := canonicalModelID(model)
	switch {
	case embeddingPattern.MatchString(normalized):
		return "embedding", ""
	case moderationPattern.MatchString(normalized):
		return "moderation", ""
	case strings.Contains(normalized, "grok") && strings.Contains(normalized, "image"):
		return "image", "xai"
	case containsAny(normalized, "gpt-image", "dall-e", "imagen-"):
		if strings.Contains(normalized, "imagen") {
			return "image", "google"
		}
		return "image", "openai"
	case realtimePattern.MatchString(normalized):
		if strings.Contains(normalized, "gemini") {
			return "realtime", "google"
		}
		return "realtime", "openai"
	case transcriptionPattern.MatchString(normalized):
		return "transcription", "openai"
	case ttsPattern.MatchString(normalized):
		return "speech", "openai"
	case audioPattern.MatchString(normalized):
		return "audio", "openai"
	case strings.Contains(normalized, "sora"):
		return "video", "openai"
	case strings.Contains(normalized, "gemini"):
		return "gemini", "google"
	case strings.Contains(normalized, "claude"):
		return "claude", "anthropic"
	case strings.Contains(normalized, "gpt") || oModelPattern.MatchString(normalized):
		return "gpt", "openai"
	case strings.Contains(normalized, "grok"):
		return "grok", "xai"
	case strings.Contains(normalized, "nemotron"):
		return "nemotron", "nvidia"
	case mistralFamilies.MatchString(normalized):
		return "mistral", "mistralai"
	// Models offered through aggregators such as OpenRouter, with no first-party driver.
	case isMetaMuseModel(normalized):
		return "muse", "meta"
	case mimoPattern.MatchString(normalized):
		return "mimo", "xiaomi"
	case stepPattern.MatchString(normalized):
		return "step", "stepfun"
	}
	for _, family := range []string{"llama", "qwen", "deepseek", "gemma", "mistral", "kimi", "minimax", "glm"} {
		if strings.Contains(normalized, family) {
			return family, ""
		}
	}
	return "generic", ""
}

func llamaVersionGTE(model string, targetMajor int) bool {
	return familyVersionGTE(model, "llama-", targetMajor, 0) || familyVersionGTE(model, "llama", targetMajor, 0)
}

// openWeights is an open-weight chat family with tools while streaming.
func openWeights(input Modalities) Capabilities {
	return Capabilities{Input: input, Output: some("text"), ToolSupport: yes, ToolSupportStreaming: yes}
}

// familyCapabilities returns the capabilities of the source model by family.
func familyCapabilities(model, family string) Capabilities {
	switch family {
	case "gemini":
		return vertexAICapabilities(model)
	case "claude":
		return anthropicCapabilities(model)
	case "gpt":
		caps := openAICapabilities(model)
		caps.Input.Text = orDefault(caps.Input.Text, yes)
		caps.Input.Image = orDefault(caps.Input.Image, yes)
		caps.Output.Text = orDefault(caps.Output.Text, yes)
		caps.ToolSupportStreaming = orDefault(caps.ToolSupportStreaming, orDefault(caps.ToolSupport, yes))
		caps.ToolSupport = orDefault(caps.ToolSupport, yes)
		return caps
	case "embedding":
		return tools(some("text"), some("embed"), false, false)
	case "moderation":
		return tools(some("text"), some("text"), false, false)
	case "transcription":
		return tools(some("audio"), some("text"), false, false)
	case "speech":
		return tools(some("text"), some("audio"), false, false)
	case "realtime":
		return tools(some("text", "image", "audio"), some("text", "audio"), true, true)
	case "audio":
		return tools(some("text", "audio"), some("text", "audio"), false, false)
	case "video":
		return tools(some("text", "image", "video"), some("video", "audio"), false, false)
	case "image":
		return Capabilities{Input: some("text", "image"), Output: some("image"), ToolSupport: no}
	case "grok":
		return Capabilities{
			Input:       Modalities{Text: yes, Image: flag(strings.Contains(model, "vision") || familyVersionGTE(model, "grok-", 4, 3))},
			Output:      some("text"),
			ToolSupport: yes,
		}
	case "llama":
		// Future Llama generations inherit the newest known family behaviour.
		return openWeights(Modalities{Text: yes, Image: flag(strings.Contains(model, "vision") || llamaVersionGTE(model, 4))})
	case "gemma":
		image := (strings.Contains(model, "gemma-3") && !containsAny(model, "gemma-3-1b", "gemma-3-270m")) ||
			strings.Contains(model, "gemma-4")
		return openWeights(Modalities{Text: yes, Image: flag(image)})
	case "qwen":
		return openWeights(Modalities{Text: yes, Image: flag(qwenVision(model))})
	case "deepseek":
		// DeepSeek V4.1 Flash adds image input; earlier and non-Flash models are text-only.
		return openWeights(Modalities{Text: yes, Image: flag(familyVersionGTE(model, "deepseek-v", 4, 1) && strings.Contains(model, "flash"))})
	case "kimi":
		// Kimi K2.5 added image input; later generations keep it.
		return openWeights(Modalities{Text: yes, Image: flag(kimiVersionGTE(model, 2, 5))})
	case "glm":
		return openWeights(some("text"))
	case "minimax":
		// MiniMax M3 adds image and video input. https://openrouter.ai/minimax/minimax-m3
		m3 := familyVersionGTE(model, "minimax-m", 3, 0)
		return openWeights(Modalities{Text: yes, Image: flag(m3), Video: flag(m3)})
	// Source modalities as published on OpenRouter: Muse Wonder and MiMo also take audio.
	case "muse":
		return openWeights(Modalities{
			Text:  yes,
			Image: yes,
			Video: flag(!strings.Contains(model, "glimmer")),
			Audio: flag(strings.Contains(model, "wonder")),
		})
	case "mimo":
		return openWeights(some("text", "image", "video", "audio"))
	case "step":
		return openWeights(some("text", "image", "video"))
	case "nemotron":
		return openWeights(Modalities{Text: yes, Image: flag(strings.Contains(model, "nemotron-nano-12b"))})
	case "mistral":
		return mistralModelKnowledge(model).Capabilities
	}
	return Capabilities{Input: some("text"), Output: some("text")}
}

// qwenVision reports Qwen 3.5 and the Qwen 2.5 and 3 vision-language (`-vl`) models.
func qwenVision(model string) bool {
	if strings.Contains(model, "qwen3.5") {
		return true
	}
	for _, prefix := range []string{"qwen2.5", "qwen3"} {
		if index := strings.Index(model, prefix); index != -1 {
			suffix := model[index+len(prefix):]
			if strings.Contains(suffix, "-vl") || strings.Contains(suffix, ".vl") {
				return true
			}
		}
	}
	return false
}

// mergeCapabilities applies an exact override over the provider capabilities.
func mergeCapabilities(base Capabilities, override *Capabilities) Capabilities {
	if override == nil {
		return base
	}
	merged := base
	merged.Input = base.Input.merge(override.Input)
	merged.Output = base.Output.merge(override.Output)
	if override.ToolSupport != nil {
		merged.ToolSupport = override.ToolSupport
	}
	if override.ToolSupportStreaming != nil {
		merged.ToolSupportStreaming = override.ToolSupportStreaming
	}
	return merged
}

var (
	nemotronNanoGeneration = regexp.MustCompile(`nemotron-nano-(\d+)-`)
	nemotronGeneration     = regexp.MustCompile(`nemotron-(\d+)[.-]`)
)

// familyLimits returns the source model's limits, only where family data backs them.
func familyLimits(model, family string) limits {
	switch {
	case family == "audio":
		return limits{128_000, 16_384}
	case family == "realtime" && familyVersionGTE(model, "gpt-realtime-", 2, 0):
		// Later Realtime generations inherit the latest documented family limits.
		return limits{128_000, 32_000}
	case strings.Contains(model, "gemini"):
		contextWindow, maxOutput := GeminiModelLimits(model)
		return limits{contextWindow, maxOutput}
	case family == "nemotron":
		if strings.Contains(model, "nemotron-super-") {
			return limits{256_000, 32_768}
		}
		// NVIDIA also names models generation-first (`nemotron-3-super-120b`, `nemotron-3.5-lightning`).
		match := nemotronNanoGeneration.FindStringSubmatch(model)
		if match == nil {
			match = nemotronGeneration.FindStringSubmatch(model)
		}
		if match != nil && atoi(match[1]) >= 3 {
			return limits{256_000, 8_192}
		}
		return limits{128_000, 8_192}
	}
	switch family {
	case "generic", "embedding", "moderation", "image", "transcription", "speech", "realtime", "video":
		return limits{}
	case "mistral":
		return limits{contextWindow: mistralModelKnowledge(model).ContextWindow}
	}
	return limits{ContextWindowSize(model), MaxOutputTokens(model)}
}

var groqGptOssPattern = regexp.MustCompile(`gpt-oss-(?:20b|120b)(?:-|$)`)

// isOpenAICompatibleTransport reports transports that carry a source model's semantics without its
// provider-native fields.
func isOpenAICompatibleTransport(provider common.Provider) bool {
	return provider == common.ProviderOpenAICompatible || provider == common.ProviderOpenRouter
}

// providerOverlay applies what a provider changes about the source model.
func providerOverlay(model, sourceModel, family string, provider common.Provider, caps Capabilities) (Capabilities, limits) {
	switch {
	case provider == common.ProviderBedrock || provider == common.ProviderBedrockMantle:
		endpoint := BedrockRuntime
		if provider == common.ProviderBedrockMantle {
			endpoint = BedrockMantle
		}
		id := NormalizeBedrockModelID(model)
		contextWindow, maxOutput := bedrockLimits(id)
		return bedrockCapabilities(model, endpoint), limits{contextWindow, maxOutput}
	case provider == common.ProviderXAI && singleDigitGrokVersionGTE(sourceModel, 4, 7):
		return caps, limits{contextWindow: 500_000}
	case provider == common.ProviderGroq && groqGptOssPattern.MatchString(sourceModel):
		// Groq publishes different completion limits from the open-weight source models.
		return caps, limits{131_072, 65_536}
	case provider == common.ProviderGroq && qwenVersionGTE(sourceModel, 3, 8) && strings.Contains(sourceModel, "-27b"):
		return caps, limits{131_072, 16_384}
	case provider == common.ProviderVertexAI:
		// A documented hosted limit is authoritative, including an unknown output limit.
		if hosted := vertexAIModelLimits(model); hosted.contextWindow != 0 {
			return caps, hosted
		}
		return caps, familyLimits(sourceModel, family)
	case isOpenAICompatibleTransport(provider):
		// OpenRouter and other compatible transports keep the source model's capabilities but not provider-native
		// fields such as Vertex AI Flex or thinking_level.
		caps.ToolSupportStreaming = orDefault(caps.ToolSupportStreaming, caps.ToolSupport)
		return caps, familyLimits(sourceModel, family)
	case provider == common.ProviderAzureFoundry:
		// Deployment metadata describes the transport. Dedicated endpoints and Claude keep their source
		// capabilities, so a deployment named like a GPT model cannot become text inference by accident.
		if !slices.Contains([]string{"claude", "embedding", "moderation", "image", "transcription", "speech",
			"realtime", "audio", "video"}, family) {
			caps = azureFoundryCapabilities(model)
		}
		l := familyLimits(sourceModel, family)
		if contextWindow, maxOutput := AzureFoundryModelLimits(sourceModel); contextWindow != 0 || maxOutput != 0 {
			if contextWindow != 0 {
				l.contextWindow = contextWindow
			}
			if maxOutput != 0 {
				l.maxOutput = maxOutput
			}
		}
		return caps, l
	}
	return caps, familyLimits(sourceModel, family)
}

var (
	groqQwenNoEffort = regexp.MustCompile(`qwen3(?:\.6-27b|-32b)(?:-|$)`)
	museMaxPattern   = regexp.MustCompile(`muse-(?:wonder|ember)`)
)

// reasoningEffortLevels lists the reasoning efforts a model takes on a provider.
func reasoningEffortLevels(model, family string, provider common.Provider) []string {
	compatible := isOpenAICompatibleTransport(provider)
	switch {
	case provider == common.ProviderGroq && !strings.Contains(model, "coder") && qwenVersionGTE(model, 3, 8):
		return []string{"none", "low", "medium", "high"}
	// The 0813 release changed DeepSeek V4 Pro effort levels.
	case provider == common.ProviderTogetherAI && strings.Contains(model, "deepseek-v4-pro-0813"):
		return []string{"high", "max"}
	case provider == common.ProviderGroq && groqQwenNoEffort.MatchString(model):
		return []string{"none"}
	// Together serves Qwen 3.5-3.7 as hybrid reasoning models whose thinking can only be switched off; Qwen 3.8
	// takes low/medium/xhigh. https://docs.together.ai/docs/inference/chat/reasoning
	case provider == common.ProviderTogetherAI && family == "qwen" && !strings.Contains(model, "coder") && qwenVersionGTE(model, 3, 8):
		return []string{"low", "medium", "xhigh"}
	case provider == common.ProviderTogetherAI && family == "qwen" && !strings.Contains(model, "coder") && qwenVersionGTE(model, 3, 5):
		return []string{"none"}
	}
	if family == "gpt" {
		if strings.Contains(model, "gpt-oss") {
			if provider == common.ProviderGroq && strings.Contains(model, "safeguard") {
				return nil
			}
			switch {
			case provider == common.ProviderGroq, provider == common.ProviderTogetherAI, compatible,
				provider == common.ProviderVertexAI, provider == common.ProviderBedrock,
				provider == common.ProviderBedrockMantle, provider == common.ProviderAzureFoundry:
				return []string{"low", "medium", "high"}
			}
			return nil
		}
		switch {
		case provider == common.ProviderOpenAI, provider == common.ProviderAzureOpenAI, compatible,
			provider == common.ProviderAzureFoundry, provider == common.ProviderBedrockMantle,
			provider == common.ProviderBedrock:
			if oModelPattern.MatchString(model) {
				return []string{"low", "medium", "high"}
			}
			return openAIReasoningEffortLevels(model)
		}
		return nil
	}
	switch {
	case family == "gemini" && compatible && geminiVersionGTE(model, 3, 5):
		// Sampling-restricted generations (Flash 3.7+, Gemini 4+) reject `minimal`, as on Vertex AI.
		if supportsGeminiXHighThinking(model) {
			return []string{"low", "medium", "high", "xhigh"}
		}
		if geminiSamplingRestriction(model) {
			return []string{"low", "medium", "high"}
		}
		return []string{"minimal", "low", "medium", "high"}
	// GLM 5.3 replaces thinking on/off with low/high/max effort. https://docs.z.ai/guides/llm/glm-5.3
	case family == "glm" && familyVersionGTE(model, "glm-", 5, 3) &&
		(compatible || provider == common.ProviderTogetherAI || provider == common.ProviderMistralAI):
		return []string{"low", "high", "max"}
	case compatible && family == "muse":
		// Muse models always reason; Spark 1.3+, Wonder and Ember add max. https://openrouter.ai/meta
		if strings.Contains(model, "glimmer") {
			return []string{"low", "medium", "high", "xhigh"}
		}
		if museMaxPattern.MatchString(model) || familyVersionGTE(model, "muse-spark-", 1, 3) {
			return []string{"minimal", "low", "medium", "high", "xhigh", "max"}
		}
		return []string{"minimal", "low", "medium", "high", "xhigh"}
	case compatible && family == "step":
		return []string{"low", "medium", "high"}
	case provider == common.ProviderMistralAI && family == "mistral":
		return mistralModelKnowledge(model).ReasoningEffortLevels
	// Bedrock Runtime hosts Grok with the same reasoning controls as xAI.
	case (provider == common.ProviderXAI || provider == common.ProviderBedrock) && family == "grok":
		return grokReasoningEffortLevels(model)
	}
	return nil
}

// ResolveProfile returns what the catalogue knows about a model on a provider: its family, capabilities, limits
// and reasoning efforts. Family rules follow the source model, so transport paths, region prefixes, deployment
// names and routing suffixes do not change them; the provider then overlays what it serves differently. Unknown
// models resolve to the generic family with text in and out.
func ResolveProfile(model string, provider common.Provider) Profile {
	canonical := canonicalModelID(model)
	override, hasOverride := exactModelOverrides[canonical]
	family, sourceProvider := inferFamily(canonical)
	if hasOverride && override.family != "" {
		family = override.family
	}
	if hasOverride && override.sourceProvider != "" {
		sourceProvider = override.sourceProvider
	}
	base := familyCapabilities(canonical, family)
	if provider == common.ProviderVertexAI {
		if vertex := vertexAICapabilities(model); vertex.Input.anyTrue() || vertex.Output.anyTrue() || vertex.ToolSupport != nil {
			base = vertex
		}
	}
	caps, l := providerOverlay(model, canonical, family, provider, base)
	if hasOverride {
		caps = mergeCapabilities(caps, override.capabilities)
		if override.contextWindow != 0 {
			l.contextWindow = override.contextWindow
		}
		if override.maxOutput != 0 {
			l.maxOutput = override.maxOutput
		}
	}
	profile := Profile{
		ModelID:         model,
		CanonicalID:     canonical,
		Family:          family,
		SourceProvider:  sourceProvider,
		Capabilities:    caps.clone(),
		ContextWindow:   l.contextWindow,
		MaxOutputTokens: l.maxOutput,
	}
	if levels := reasoningEffortLevels(canonical, family, provider); len(levels) > 0 {
		profile.ReasoningEffortLevels = slices.Clone(levels)
	}
	return profile
}

// IsDirectoryEmbedding reports whether a model ID names an embedding model (an `embed`, `embedding` or `vector`
// segment), as given or as its source model.
func IsDirectoryEmbedding(model string) bool {
	normalized := strings.ToLower(strings.TrimSpace(model))
	return embeddingPattern.MatchString(normalized) || embeddingPattern.MatchString(canonicalModelID(normalized))
}
