package capability

import (
	"regexp"
	"strings"
)

// azureFoundryModelCapabilities holds exact Microsoft Foundry exceptions to the family patterns.
var azureFoundryModelCapabilities = map[string]Capabilities{
	// o1-mini has no tool support, unlike other o1 models.
	"o1-mini": tools(only("text", "image"), only("text"), false, false),
	// o3 is text-only, unlike other o-series models.
	"o3":                             tools(only("text"), only("text"), true, false),
	"deepseek-r1-0528":               textOnly(false, false),
	"deepseek-v3-0324":               textOnly(false, false),
	"mistral-medium-2505":            textOnly(false, false),
	"mistral-nemo":                   textOnly(false, false),
	"llama-4-scout-17b-16e-instruct": tools(only("text", "image"), only("text"), false, false),
}

// azureFoundryFamilyCapabilities is matched by longest prefix: the lowest common denominator of each family.
var azureFoundryFamilyCapabilities = map[string]Capabilities{
	"gpt-3.5-turbo": textOnly(false, false),
	"gpt-35-turbo":  textOnly(false, false),
	"gpt-35":        textOnly(false, false),
	"gpt-4":         tools(only("text", "image"), only("text"), true, true),
	"gpt-4.1":       tools(only("text", "image"), only("text"), true, true),
	"gpt-4.5":       tools(only("text", "image"), only("text"), true, true),
	"gpt-4o":        tools(only("text", "image"), only("text"), true, true),
	"gpt":           tools(only("text", "image"), only("text"), true, true),

	"o1":         tools(only("text", "image"), only("text"), true, false),
	"o1-preview": tools(only("text", "image"), only("text"), true, false),
	"o1-pro":     tools(only("text", "image"), only("text"), true, false),
	"o3-mini":    tools(only("text", "image"), only("text"), true, false),
	"o4-mini":    tools(only("text", "image"), only("text"), true, false),
	"o4":         tools(only("text", "image"), only("text"), true, false),
	"o":          tools(only("text", "image"), only("text"), true, false),

	"llama-3.1": textOnly(true, false),
	"llama-3.2": tools(only("text", "image"), only("text"), false, false),
	"llama-3.3": textOnly(true, false),
	"llama-3":   textOnly(true, false),
	"llama-4":   tools(only("text", "image"), only("text"), false, false),
	"llama":     textOnly(false, false),

	"mistral-large": textOnly(true, false),
	"mistral-small": textOnly(true, false),
	"mistral":       textOnly(false, false),

	"phi-4": textOnly(false, false),
	"phi":   textOnly(false, false),

	"deepseek-r1": textOnly(false, false),
	"deepseek-v3": textOnly(true, false),
	"deepseek":    textOnly(false, false),

	"ai21-jamba": textOnly(true, true),
	"ai21":       textOnly(true, true),
	"jamba":      textOnly(true, true),

	"cohere-command": textOnly(true, true),
	"cohere":         textOnly(true, true),
	"command":        textOnly(true, true),

	"grok-3": textOnly(true, true),
	"grok":   textOnly(true, true),
}

// foundryPartnerRule is what Microsoft Foundry documents for a model it serves through Chat Completions: whether it
// calls tools and whether it takes images. A family without a rule keeps tools off.
//
// https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure
// https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-from-partners
type foundryPartnerRule struct {
	matches func(model string) bool
	tools   bool
	// image is nil when the rule leaves image input as the family says.
	image func(model string) *bool
}

func always(v bool) func(string) *bool { return func(string) *bool { return flag(v) } }

var (
	commandAPattern     = regexp.MustCompile(`^(?:cohere-)?command-a(?:[-_.]|$)`)
	foundryDeepSeekVer  = regexp.MustCompile(`^deepseek-v(\d+)(?:\.(\d+))?(?:[-_]|$)`)
	foundryOpenAIPrefix = regexp.MustCompile(`^(?:gpt-|o1|o3|o4)`)
)

var foundryPartnerRules = []foundryPartnerRule{
	// DeepSeek V3.2 and later call tools and are text-only on Foundry. R1 and V3-0324 are not documented with tools.
	{matches: isFoundryDeepSeekV32OrLater, tools: true},
	// Kimi K2.5 and later take text and images and call tools.
	{matches: func(m string) bool { return kimiVersionGTE(m, 2, 5) }, tools: true, image: always(true)},
	// Every Grok model calls tools. Grok 4.1 Fast and Grok 4.6 and later also take images.
	{
		matches: func(m string) bool { return strings.HasPrefix(m, "grok-") },
		tools:   true,
		image: func(m string) *bool {
			return flag(strings.HasPrefix(m, "grok-4.1-fast") || singleDigitGrokVersionGTE(m, 4, 6))
		},
	},
	// Mistral Large calls tools; Large 3 and later also take images.
	{
		matches: func(m string) bool { return strings.HasPrefix(m, "mistral-large") },
		tools:   true,
		image:   func(m string) *bool { return flag(isMistralLargeGTE(m, 3)) },
	},
	{matches: func(m string) bool {
		return strings.HasPrefix(m, "mistral-small") || strings.HasPrefix(m, "ministral")
	}, tools: true},
	// Mistral Medium takes images but is documented without tool calling.
	{matches: func(m string) bool { return strings.HasPrefix(m, "mistral-medium") }, image: always(true)},
	// Cohere Command A (and Command A Plus) call tools.
	{matches: commandAPattern.MatchString, tools: true},
	// MAI-Thinking is a Chat Completions reasoning model with tool calling.
	{matches: func(m string) bool { return strings.HasPrefix(m, "mai-thinking") }, tools: true},
	// Phi-4 multimodal also takes audio, which the Chat Completions prompt does not send, so only images are listed.
	{matches: func(m string) bool { return strings.HasPrefix(m, "phi-4-multimodal") }, image: always(true)},
}

// isFoundryDeepSeekV32OrLater reads DeepSeek versions with a dotted minor (`deepseek-v3.2`); a dash starts a date.
func isFoundryDeepSeekV32OrLater(model string) bool {
	match := foundryDeepSeekVer.FindStringSubmatch(model)
	return match != nil && atLeast(atoi(match[1]), atoi(match[2]), 3, 2)
}

// applyFoundryPartnerRules applies the documented tool calling and image input of non-OpenAI Foundry models.
func applyFoundryPartnerRules(caps Capabilities, model string) Capabilities {
	if foundryOpenAIPrefix.MatchString(model) {
		return caps
	}
	for _, rule := range foundryPartnerRules {
		if !rule.matches(model) {
			continue
		}
		if rule.image != nil {
			caps.Input = caps.Input.merge(Modalities{Image: rule.image(model)})
		}
		caps.ToolSupport, caps.ToolSupportStreaming = flag(rule.tools), flag(rule.tools)
		return caps
	}
	caps.ToolSupport, caps.ToolSupportStreaming = no, no
	return caps
}

// azureFoundryBaseModel returns the source model of a `deployment::model` ID, or the ID itself.
func azureFoundryBaseModel(model string) string {
	if parts := strings.Split(model, "::"); len(parts) == 2 {
		return parts[1]
	}
	return model
}

// AzureFoundryModelLimits returns the limits Microsoft Foundry documents differently from the source provider; 0
// means the source model's limit applies.
func AzureFoundryModelLimits(model string) (contextWindow, maxOutput int) {
	normalized := strings.ToLower(azureFoundryBaseModel(model))
	switch {
	case kimiVersionGTE(normalized, 3, 0):
		return 1_000_000, 262_144
	case kimiVersionGTE(normalized, 2, 5):
		return 262_144, 262_144
	// Grok 4.7 and later share a 500,000-token window between input and output.
	case singleDigitGrokVersionGTE(normalized, 4, 7):
		return 500_000, 0
	case strings.HasPrefix(normalized, "mai-thinking"):
		return 256_000, 64_000
	}
	return 0, 0
}

// AzureFoundryCapabilities returns the capabilities of a Microsoft Foundry model, given as `deployment::model` or
// as the model name.
func AzureFoundryCapabilities(model string) Capabilities {
	return azureFoundryCapabilities(model).clone()
}

func azureFoundryCapabilities(model string) Capabilities {
	normalized := strings.ToLower(azureFoundryBaseModel(model))
	if record, ok := azureFoundryModelCapabilities[normalized]; ok {
		return applyFoundryPartnerRules(record, normalized)
	}
	if familyVersionGTE(normalized, "llama-", 4, 0) || familyVersionGTE(normalized, "llama", 4, 0) {
		return applyFoundryPartnerRules(azureFoundryFamilyCapabilities["llama-4"], normalized)
	}
	if family, ok := longestPrefix(azureFoundryFamilyCapabilities, normalized); ok {
		return applyFoundryPartnerRules(family, normalized)
	}
	input, output := inferModalities(normalized, "image", "vision")
	// Unknown models default to text in and out.
	input.Text, output.Text = yes, yes
	tool := flagOrNil(containsAny(normalized, "tool", "gpt-4", "gpt-4o", "o1", "o3", "o4", "llama-3",
		"mistral-large", "mistral-small", "jamba", "cohere", "command", "grok"))
	return applyFoundryPartnerRules(Capabilities{Input: input, Output: output, ToolSupport: tool, ToolSupportStreaming: tool}, normalized)
}
