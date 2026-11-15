package capability

import "testing"

// Ported from common/src/capability/vertexai-limits.test.ts. An undefined limit is 0.

func checkVertexLimits(t *testing.T, model string, wantContext, wantOutput int) {
	t.Helper()
	contextWindow, maxOutput := VertexAIModelLimits(model)
	checkInt(t, model+" context_window", contextWindow, wantContext)
	checkInt(t, model+" max_output_tokens", maxOutput, wantOutput)
}

func TestVertexAIDocumentedLimits(t *testing.T) {
	checkVertexLimits(t, "glm-5.2-maas", 1_000_000, 64_000)
	checkVertexLimits(t, "deepseek-v3.1-maas", 163_840, 32_768)
}

func TestVertexAIContextOnlyOverrides(t *testing.T) {
	// The context is overridden without guessing output limits.
	for _, tc := range []struct {
		model   string
		context int
	}{
		{"grok-4.7", 524_288},
		{"grok-4.6", 524_288},
		{"grok-4.3", 200_000},
		{"grok-4.20-reasoning", 2_000_000},
		{"grok-4.20-non-reasoning", 2_000_000},
		{"grok-4.1-fast-reasoning", 128_000},
		{"grok-4.1-fast-non-reasoning", 128_000},
		{"mistral-small-2503", 128_000},
		{"mistral-medium-3", 128_000},
		{"codestral-2", 128_000},
	} {
		checkVertexLimits(t, tc.model, tc.context, 0)
	}
}

func TestVertexAILimitsInherited(t *testing.T) {
	for _, tc := range []struct{ model, latestKnown string }{
		{"grok-4.8", "grok-4.7"},
		{"grok-5", "grok-4.7"},
		{"glm-5.3-maas", "glm-5.2-maas"},
		{"mistral-small-2603", "mistral-small-2503"},
		{"mistral-medium-3.6", "mistral-medium-3"},
		{"codestral-3", "codestral-2"},
	} {
		wantContext, wantOutput := VertexAIModelLimits(tc.latestKnown)
		checkVertexLimits(t, tc.model, wantContext, wantOutput)
	}
}

func TestVertexAILimitsResourceName(t *testing.T) {
	checkVertexLimits(t, "projects/example/locations/global/publishers/zai-org/models/GLM-5.2-MAAS", 1_000_000, 64_000)
}

func TestVertexAILimitsUnrelatedModels(t *testing.T) {
	for _, model := range []string{"deepseek-v3.1", "unknown-model", ""} {
		checkVertexLimits(t, model, 0, 0)
	}
}
