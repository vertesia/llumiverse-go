package capability

import (
	"regexp"
	"strings"

	"github.com/vertesia/llumiverse-go/core"
)

// BedrockEndpoint names the Bedrock API a model is reached through.
type BedrockEndpoint string

const (
	// BedrockRuntime is Bedrock Runtime (Converse and InvokeModel).
	BedrockRuntime BedrockEndpoint = "runtime"
	// BedrockMantle is Bedrock Mantle, the OpenAI-compatible endpoint.
	BedrockMantle BedrockEndpoint = "mantle"
)

// BedrockKnowledge is what the catalogue knows about a Bedrock model.
type BedrockKnowledge struct {
	Input  Modalities
	Output Modalities
	// ContextWindow and MaxOutputTokens are 0 when unknown.
	ContextWindow   int
	MaxOutputTokens int
}

var daybreakPrefix = regexp.MustCompile(`^daybreak-[a-z]+-`)

// bedrockFamilyVersionGTE checks the version after a Bedrock family prefix; Daybreak GPT models carry a
// `daybreak-<color>-` segment before the GPT version.
func bedrockFamilyVersionGTE(model, family string, targetMajor, targetMinor int) bool {
	index := strings.Index(model, family)
	if index == -1 {
		return false
	}
	rest := model[index+len(family):]
	if family == "openai.gpt-" {
		rest = daybreakPrefix.ReplaceAllString(rest, "")
	}
	match := familyVersionPattern.FindStringSubmatch(rest)
	return match != nil && atLeast(atoi(match[1]), atoi(match[2]), targetMajor, targetMinor)
}

// geoPrefix is the cross-region inference prefix (`us.`, `eu.`, `apac.`, `us-gov.`, `global.`).
var geoPrefix = regexp.MustCompile(`^(?:[a-z]{2}(?:-gov)?|apac|global)\.`)

// NormalizeBedrockModelID lower-cases a Bedrock model ID or ARN and strips the ARN path and the cross-region
// prefix, leaving `<publisher>.<model>`.
func NormalizeBedrockModelID(model string) string {
	return geoPrefix.ReplaceAllString(lastSegment(strings.ToLower(model)), "")
}

func bedrockModalities(model string) (input, output Modalities) {
	if containsAny(model, "nova-2-multimodal-embeddings", "titan-embed", "cohere.embed", "marengo-embed") {
		return Modalities{
			Text:  yes,
			Image: flag(containsAny(model, "multimodal", "embed-image", "embed-v4", "marengo")),
			Video: flag(containsAny(model, "multimodal", "marengo")),
			Audio: flag(strings.Contains(model, "multimodal")),
			Embed: no,
		}, only("embed")
	}
	if containsAny(model, "nova-canvas", "titan-image-generator") {
		return only("text", "image"), only("image")
	}
	if strings.Contains(model, "nova-reel") {
		return only("text", "image"), only("video")
	}
	image := (strings.Contains(model, "anthropic.claude") && !strings.Contains(model, "claude-3-5-haiku")) ||
		(strings.Contains(model, "amazon.nova") && !strings.Contains(model, "nova-micro")) ||
		strings.Contains(model, "google.gemma") ||
		containsAny(model, "meta.llama3-2-11b", "meta.llama3-2-90b") ||
		bedrockFamilyVersionGTE(model, "meta.llama", 4, 0) ||
		containsAny(model, "mistral.magistral-", "mistral.ministral-3-", "mistral.pixtral-") ||
		isMistralLargeGTE(model, 3) ||
		kimiVersionGTE(model, 2, 5) ||
		strings.Contains(model, "nemotron-nano-12b") ||
		strings.Contains(model, "qwen3-vl") ||
		strings.Contains(model, "palmyra-vision") ||
		(strings.HasPrefix(model, "openai.gpt-") && !strings.Contains(model, "gpt-oss")) ||
		strings.HasPrefix(model, "xai.grok-")
	video := (strings.Contains(model, "amazon.nova") && !strings.Contains(model, "nova-micro")) ||
		strings.Contains(model, "twelvelabs.pegasus") ||
		bedrockFamilyVersionGTE(model, "google.gemma-", 4, 0)
	// AWS lists Voxtral with text input only, so it is described as a text model.
	audio := strings.Contains(model, "nova-sonic") || strings.Contains(model, "google.gemma-4-e2b")
	input = Modalities{
		Text:  flag(!strings.Contains(model, "nova-sonic-v1")),
		Image: flag(image),
		Video: flag(video),
		Audio: flag(audio),
		Embed: no,
	}
	output = Modalities{
		Text:  yes,
		Image: no,
		Video: no,
		Audio: flag(strings.Contains(model, "nova-sonic")),
		Embed: no,
	}
	return input, output
}

var (
	bedrockSolPattern     = regexp.MustCompile(`^openai\.gpt-[\d.]+-sol(?:[-:]|$)`)
	nemotronNanoGenerated = regexp.MustCompile(`nvidia\.nemotron-nano-(\d+)-`)
)

func bedrockLimits(model string) (contextWindow, maxOutput int) {
	switch {
	case strings.Contains(model, "ai21.jamba"):
		return 256_000, 4_096
	case strings.Contains(model, "amazon.nova-2"):
		return 1_000_000, 65_535
	case strings.Contains(model, "amazon.nova-premier"):
		return 1_000_000, 25_600
	case strings.Contains(model, "amazon.nova-micro"):
		return 128_000, 5_120
	case containsAny(model, "amazon.nova-lite", "amazon.nova-pro"):
		return 300_000, 5_120
	}
	if strings.Contains(model, "anthropic.claude") {
		// Version rules, so a new variant inherits the newest understood generation's limits.
		if claude, ok := core.ParseClaudeVersion(model); ok {
			switch {
			case claudeVersionAtLeast(claude, 4, 7) || (claude.Variant == "opus" && claudeVersionAtLeast(claude, 4, 6)):
				// Bedrock rejects max_tokens >= 128000 for this family.
				return 1_000_000, 127_999
			case claude.Variant == "sonnet" && claudeVersionAtLeast(claude, 4, 6):
				return 1_000_000, 64_000
			case claudeVersionAtLeast(claude, 4, 5):
				return 200_000, 64_000
			case claudeVersionAtLeast(claude, 4, 0):
				if claude.Variant == "opus" {
					return 200_000, 32_000
				}
				return 200_000, 64_000
			}
		}
		if strings.Contains(model, "claude-3-5-haiku") {
			return 200_000, 8_192
		}
		if strings.Contains(model, "claude-3-haiku") {
			return 200_000, 4_096
		}
	}
	switch {
	case strings.Contains(model, "cohere.command-r"):
		return 128_000, 4_096
	// https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus-1-5.html
	case isPegasus15OrLater(model):
		return 0, 98_304
	case strings.Contains(model, "twelvelabs.pegasus-"):
		return 0, 4_096
	case bedrockFamilyVersionGTE(model, "deepseek.v", 3, 2):
		return 163_840, 8_192
	case containsAny(model, "deepseek.v3", "deepseek.r1"):
		return 128_000, 8_192
	case containsAny(model, "google.gemma-4-31b", "google.gemma-4-26b"):
		return 256_000, 0
	case strings.Contains(model, "google.gemma-4-e2b"):
		return 128_000, 0
	case strings.Contains(model, "google.gemma-3-"):
		return 128_000, 8_192
	case bedrockFamilyVersionGTE(model, "meta.llama", 4, 0):
		if strings.Contains(model, "scout") {
			return 10_000_000, 8_192
		}
		return 1_000_000, 8_192
	case strings.Contains(model, "meta.llama3-") && !containsAny(model, "llama3-1", "llama3-2", "llama3-3"):
		return 8_192, 8_192
	case containsAny(model, "meta.llama3-1", "meta.llama3-2", "meta.llama3-3"):
		return 128_000, 4_096
	case bedrockFamilyVersionGTE(model, "minimax.minimax-m", 2, 1):
		return 196_000, 8_192
	case strings.Contains(model, "minimax.minimax-m2"):
		return 1_000_000, 8_192
	// Mistral Large 4 publishes a 1M context but no output limit. https://docs.mistral.ai/models/mistral-large-4-0
	case isMistralLargeGTE(model, 4):
		return 1_000_000, 0
	case containsAny(model, "mistral.devstral-2", "mistral.mistral-large-3"):
		return 256_000, 32_768
	case strings.Contains(model, "mistral.magistral-"):
		return 128_000, 40_960
	case strings.Contains(model, "mistral.ministral-3-"):
		return 128_000, 8_192
	case strings.Contains(model, "mistral.pixtral-"):
		return 128_000, 16_384
	case containsAny(model, "mistral.mistral-7b", "mistral.mistral-large-2402", "mistral.mistral-small-2402",
		"mistral.mixtral-", "mistral.voxtral-"):
		if strings.Contains(model, "voxtral") {
			return 32_000, 0
		}
		return 32_000, 4_096
	// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3.html
	case kimiVersionGTE(model, 3, 0):
		return 1_000_000, 0
	case containsAny(model, "moonshot.kimi-k2", "moonshotai.kimi-k2"):
		return 256_000, 16_384
	case strings.Contains(model, "nvidia.nemotron-super-"):
		return 256_000, 32_768
	}
	if match := nemotronNanoGenerated.FindStringSubmatch(model); match != nil && atoi(match[1]) >= 3 {
		return 256_000, 8_192
	}
	switch {
	case strings.Contains(model, "nvidia.nemotron-nano-"):
		return 128_000, 8_192
	case bedrockFamilyVersionGTE(model, "openai.gpt-", 5, 0) && !strings.Contains(model, "gpt-oss"):
		// 272K is the long-context pricing threshold on older cards. GPT-5.6 Sol and GPT-6.1+ Sol specify 1M
		// (6.1+ with 131,072 output); other variants, GPT-6 Sol and Daybreak Blue 5.6 Sol included, 1.05M.
		// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-1-terra.html
		sol := bedrockSolPattern.MatchString(model)
		if sol && bedrockFamilyVersionGTE(model, "openai.gpt-", 6, 1) {
			return 1_000_000, 131_072
		}
		if bedrockFamilyVersionGTE(model, "openai.gpt-", 5, 5) {
			if sol && strings.HasPrefix(model, "openai.gpt-5.6-") {
				return 1_000_000, 128_000
			}
			return 1_050_000, 128_000
		}
		return 272_000, 128_000
	case strings.Contains(model, "openai.gpt-oss"):
		return 128_000, 16_384
	case strings.Contains(model, "qwen3-235b"):
		return 256_000, 8_192
	case strings.Contains(model, "qwen3-32b"):
		return 32_000, 8_192
	case strings.Contains(model, "qwen3-coder-30b"):
		return 256_000, 16_384
	case strings.Contains(model, "qwen3-coder-480b"):
		return 128_000, 16_384
	case strings.Contains(model, "qwen3-coder-next"):
		return 256_000, 16_384
	case containsAny(model, "qwen3-next-", "qwen3-vl-"):
		return 256_000, 8_192
	case strings.Contains(model, "writer.palmyra-vision"):
		return 4_096, 4_096
	case containsAny(model, "writer.palmyra-x4", "writer.palmyra-x5"):
		return 128_000, 8_192
	// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-xai-grok-4-7.html
	case strings.HasPrefix(model, "xai.grok-"):
		if singleDigitGrokVersionGTE(model, 4, 7) {
			return 500_000, 0
		}
		return 1_000_000, 0
	// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-zai-glm-5-3.html
	case bedrockFamilyVersionGTE(model, "zai.glm-", 5, 3):
		return 1_000_000, 131_072
	case strings.Contains(model, "zai.glm-5"):
		return 200_000, 131_072
	// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-zai-glm-4-7.html
	case containsAny(model, "zai.glm-4.7", "zai.glm-4.6"):
		return 203_000, 131_072
	}
	return 0, 0
}

// bedrockRuntimeToolSupport returns tool use on Bedrock Runtime, or nil flags when unknown.
func bedrockRuntimeToolSupport(model string) (tool, streaming *bool) {
	var known, supported, stream bool
	switch {
	case containsAny(model, "gpt-oss-safeguard", "deepseek.r1"):
		known = true
	case strings.Contains(model, "anthropic.claude"):
		known, supported, stream = true, true, !strings.Contains(model, "claude-3-5-haiku")
	case strings.Contains(model, "ai21.jamba") && !strings.Contains(model, "jamba-instruct"):
		known, supported, stream = true, true, true
	case strings.Contains(model, "amazon.nova") && !containsAny(model, "nova-canvas", "nova-reel"):
		known, supported, stream = true, true, true
	case strings.Contains(model, "cohere.command-r"):
		known, supported, stream = true, true, true
	case strings.Contains(model, "meta.llama3-1"):
		known, supported = true, true
	case containsAny(model, "meta.llama3-2-11b", "meta.llama3-2-90b"):
		known, supported = true, true
	case bedrockFamilyVersionGTE(model, "meta.llama", 4, 0):
		// Converse behaviour is family-stable; later Llama generations inherit Llama 4 tool support.
		known, supported = true, true
	case containsAny(model, "qwen.qwen3-235b", "qwen.qwen3-32b"):
		known, supported = true, true
	case strings.HasPrefix(model, "deepseek.") && !strings.Contains(model, "deepseek.r1"):
		known, supported = true, true
	case strings.HasPrefix(model, "openai.gpt-") || strings.HasPrefix(model, "xai.grok-"):
		known, supported, stream = true, true, true
	case bedrockFamilyVersionGTE(model, "google.gemma-", 4, 0):
		known, supported, stream = true, true, true
	case containsAny(model, "mistral.mistral-large", "mistral.pixtral"):
		known, supported = true, true
	case strings.Contains(model, "twelvelabs.pegasus"):
		// Pegasus runs through InvokeModel with one prompt and a video; its request has no tools.
		known = true
	case strings.Contains(model, "writer.palmyra"):
		known, supported = true, true
	case containsAny(model, "moonshot.kimi-", "moonshotai.kimi-", "zai.glm-"):
		// AWS documents client-side tool calling for Kimi and GLM; streamed Converse tool use is unverified.
		known, supported = true, true
	}
	if !known {
		return nil, nil
	}
	return flag(supported), flag(supported && stream)
}

// BedrockModelKnowledge returns the modalities and limits of a Bedrock model ID or ARN.
func BedrockModelKnowledge(model string) BedrockKnowledge {
	id := NormalizeBedrockModelID(model)
	input, output := bedrockModalities(id)
	contextWindow, maxOutput := bedrockLimits(id)
	knowledge := BedrockKnowledge{Input: input, Output: output, ContextWindow: contextWindow, MaxOutputTokens: maxOutput}
	caps := Capabilities{Input: knowledge.Input, Output: knowledge.Output}.clone()
	knowledge.Input, knowledge.Output = caps.Input, caps.Output
	return knowledge
}

// BedrockCapabilities returns the capabilities of a Bedrock model on Bedrock Runtime or Bedrock Mantle.
func BedrockCapabilities(model string, endpoint BedrockEndpoint) Capabilities {
	return bedrockCapabilities(model, endpoint).clone()
}

func bedrockCapabilities(model string, endpoint BedrockEndpoint) Capabilities {
	id := NormalizeBedrockModelID(model)
	input, output := bedrockModalities(id)
	caps := Capabilities{Input: input, Output: output}
	if endpoint == BedrockMantle && strings.HasPrefix(id, "nvidia.nemotron-") {
		// AWS documents client-side tool calling for Nemotron on Mantle, not Bedrock Runtime tool use.
		caps.ToolSupport = yes
	} else {
		caps.ToolSupport, caps.ToolSupportStreaming = bedrockRuntimeToolSupport(id)
	}
	// Mantle streams tool calls, but only for source models known to call tools.
	if endpoint == BedrockMantle && isTrue(caps.ToolSupport) {
		caps.ToolSupportStreaming = yes
	}
	return caps
}
