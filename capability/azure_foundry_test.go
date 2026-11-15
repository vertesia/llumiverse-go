package capability

import (
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

// Ported from common/src/capability/azure_foundry.test.ts.

func foundry(model string) Profile {
	return ResolveProfile("deployment::"+model, common.ProviderAzureFoundry)
}

func TestFoundryPartnerToolsOn(t *testing.T) {
	// Families whose Foundry model cards say "Tool calling: Yes", including later generations of each family.
	for _, model := range []string{
		"DeepSeek-V3.2",
		"DeepSeek-V3.2-Speciale",
		"DeepSeek-V4-Pro",
		"DeepSeek-V4.1-Flash",
		"DeepSeek-V5",
		"Kimi-K2.5",
		"Kimi-K3.1-Pro",
		"Kimi-K4",
		"grok-4",
		"grok-4.7",
		"grok-4-20-reasoning",
		"grok-code-fast-1",
		"grok-5",
		"Mistral-Large-3",
		"mistral-large-4-latest",
		"mistral-small-2503",
		"Ministral-3B",
		"Cohere-command-a",
		"Cohere-command-a-plus-05-2026",
		"MAI-Thinking-1",
		"MAI-Thinking-2",
	} {
		caps := foundry(model).Capabilities
		checkFlag(t, model+" tool_support", caps.ToolSupport, T)
		checkFlag(t, model+" tool_support_streaming", caps.ToolSupportStreaming, T)
	}
}

func TestFoundryPartnerToolsOff(t *testing.T) {
	// Documented without tool calling, or not documented at all.
	for _, model := range []string{
		"DeepSeek-R1-0528",
		"DeepSeek-V3-0324",
		"Kimi-K2-Thinking",
		"Llama-4-Maverick-17B-128E-Instruct-FP8",
		"Llama-4-Scout-17B-16E-Instruct",
		"Llama-3.3-70B-Instruct",
		"Llama-5-Scout",
		"mistral-medium-3-5",
		"mistral-medium-2505",
		"Codestral-2501",
		"Phi-4",
		"Phi-4-multimodal-instruct",
		"tsuzumi-7b",
		"AI21-Jamba-1.5-Large",
		"some-future-partner-model",
	} {
		caps := foundry(model).Capabilities
		checkFlag(t, model+" tool_support", caps.ToolSupport, F)
		checkFlag(t, model+" tool_support_streaming", caps.ToolSupportStreaming, F)
	}
}

func TestFoundryPartnerImageInput(t *testing.T) {
	for _, tc := range []struct {
		model string
		image *bool
	}{
		{"Kimi-K2.6", T},
		{"Kimi-K3.1-Pro", T},
		{"grok-4.1-fast-non-reasoning", T},
		{"grok-4.6", T},
		{"grok-4.7", T},
		{"grok-4", F},
		{"grok-4.3", F},
		{"grok-4-20-non-reasoning", F},
		{"Mistral-Large-3", T},
		{"Mistral-large", F},
		{"mistral-medium-3-5", T},
		{"DeepSeek-V4.1-Flash", F},
		{"Phi-4-multimodal-instruct", T},
		{"Phi-4", F},
		{"MAI-Thinking-1", nil},
	} {
		checkFlag(t, tc.model+" input.image", foundry(tc.model).Capabilities.Input.Image, tc.image)
	}
}

func TestFoundryNoAudioInputTheFormatterCannotSend(t *testing.T) {
	if isTrue(foundry("Phi-4-multimodal-instruct").Capabilities.Input.Audio) {
		t.Error("Phi-4-multimodal-instruct advertises audio input")
	}
}

func TestFoundryOpenAIFamiliesKeepTheirRecords(t *testing.T) {
	for _, model := range []string{"gpt-6.1-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-7"} {
		caps := foundry(model).Capabilities
		checkFlag(t, model+" tool_support", caps.ToolSupport, T)
		checkFlag(t, model+" input.image", caps.Input.Image, T)
	}
	checkFlag(t, "o1-mini tool_support", foundry("o1-mini").Capabilities.ToolSupport, F)
}

func TestFoundryLimits(t *testing.T) {
	for _, tc := range []struct {
		model           string
		context, output int
	}{
		{"Kimi-K2.6", 262_144, 262_144},
		{"Kimi-K3.1-Pro", 1_000_000, 262_144},
		{"MAI-Thinking-1", 256_000, 64_000},
	} {
		profile := foundry(tc.model)
		checkInt(t, tc.model+" context_window", profile.ContextWindow, tc.context)
		checkInt(t, tc.model+" max_output_tokens", profile.MaxOutputTokens, tc.output)
	}
}

func TestFoundryGrok47Window(t *testing.T) {
	checkInt(t, "grok-4.7", foundry("grok-4.7").ContextWindow, 500_000)
	checkInt(t, "grok-4.8", foundry("grok-4.8").ContextWindow, 500_000)
}

func TestFoundrySourceCapabilities(t *testing.T) {
	checkFlag(t, "claude-opus-5-5 tool_support", foundry("claude-opus-5-5").Capabilities.ToolSupport, T)
	checkFlag(t, "text-embedding-3-large output.embed", foundry("text-embedding-3-large").Capabilities.Output.Embed, T)
	checkFlag(t, "gpt-image-2.5 output.image", foundry("gpt-image-2.5").Capabilities.Output.Image, T)
}
