package capability

import (
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

func TestBedrockModelKnowledgeLimits(t *testing.T) {
	cases := []struct {
		model                  string
		contextWindow, maxOuts int
	}{
		// Bedrock rejects max_tokens >= 128000 for the 1M Claude family.
		{"arn:aws:bedrock:us-east-1:123:inference-profile/us.anthropic.claude-opus-4-8-v1:0", 1_000_000, 127_999},
		{"anthropic.claude-sonnet-4-6", 1_000_000, 64_000},
		{"anthropic.claude-opus-4-1-20250805-v1:0", 200_000, 32_000},
		{"anthropic.claude-3-5-haiku-20241022-v1:0", 200_000, 8_192},
		{"openai.gpt-daybreak-blue-5.6-sol", 1_050_000, 128_000},
		{"openai.gpt-6.1-sol", 1_000_000, 131_072},
		{"openai.gpt-5.6-sol", 1_000_000, 128_000},
		{"xai.grok-4.7", 500_000, 0},
		{"mistral.voxtral-small-24b-2507", 32_000, 0},
		{"twelvelabs.pegasus-1-5", 0, 98_304},
		{"unknown.model", 0, 0},
	}
	for _, c := range cases {
		knowledge := BedrockModelKnowledge(c.model)
		if knowledge.ContextWindow != c.contextWindow || knowledge.MaxOutputTokens != c.maxOuts {
			t.Errorf("%s: limits = %d/%d, want %d/%d", c.model, knowledge.ContextWindow, knowledge.MaxOutputTokens, c.contextWindow, c.maxOuts)
		}
	}
}

func TestBedrockToolSupportByEndpoint(t *testing.T) {
	cases := []struct {
		model                 string
		endpoint              BedrockEndpoint
		tool, streaming, none bool
	}{
		{"meta.llama4-maverick-17b-instruct-v1:0", BedrockRuntime, true, false, false},
		{"meta.llama4-maverick-17b-instruct-v1:0", BedrockMantle, true, true, false},
		{"anthropic.claude-3-5-haiku-20241022-v1:0", BedrockRuntime, true, false, false},
		{"deepseek.r1-v1:0", BedrockMantle, false, false, false},
		{"nvidia.nemotron-super-3-120b", BedrockMantle, true, true, false},
		{"nvidia.nemotron-super-3-120b", BedrockRuntime, false, false, true},
	}
	for _, c := range cases {
		caps := BedrockCapabilities(c.model, c.endpoint)
		if c.none {
			if caps.ToolSupport != nil || caps.ToolSupportStreaming != nil {
				t.Errorf("%s on %s: tools = %v/%v, want unknown", c.model, c.endpoint, caps.ToolSupport, caps.ToolSupportStreaming)
			}
			continue
		}
		if !isTrue(caps.ToolSupport) == c.tool || !isTrue(caps.ToolSupportStreaming) == c.streaming {
			t.Errorf("%s on %s: tools = %v/%v, want %v/%v", c.model, c.endpoint, isTrue(caps.ToolSupport), isTrue(caps.ToolSupportStreaming), c.tool, c.streaming)
		}
	}
	// The profile carries the Bedrock overlay, and Mantle relaxes streamed tool use.
	if !SupportsToolUse("us.meta.llama4-maverick-17b-instruct-v1:0", common.ProviderBedrockMantle, true) ||
		SupportsToolUse("us.meta.llama4-maverick-17b-instruct-v1:0", common.ProviderBedrock, true) {
		t.Error("streamed Llama 4 tool use should differ between Mantle and Runtime")
	}
}

func TestCatalogueValuesAreNotShared(t *testing.T) {
	caps := ForModel("gpt-5.5", common.ProviderOpenAI)
	*caps.Input.Image = false
	*caps.ToolSupport = false
	again := ForModel("gpt-5.5", common.ProviderOpenAI)
	if !isTrue(again.Input.Image) || !isTrue(again.ToolSupport) {
		t.Fatal("changing returned capabilities changed the catalogue")
	}
	if got := strings.Join(again.Input.List(), ","); got != "text,image" {
		t.Fatalf("input = %s", got)
	}
}
