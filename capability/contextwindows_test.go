package capability

import "testing"

// Ported from common/src/options/context-windows.test.ts. An undefined limit is 0.

func checkSourceLimits(t *testing.T, model string, wantContext, wantOutput int) {
	t.Helper()
	checkInt(t, model+" max output", MaxOutputTokens(model), wantOutput)
	checkInt(t, model+" context window", ContextWindowSize(model), wantContext)
}

func TestClaudeFutureCompatibleLimits(t *testing.T) {
	for _, model := range []string{
		"claude-sonnet-5",
		"claude-fable-5",
		"claude-mythos-5",
		"claude-fable-5-2",
		"claude-mythos-5-2",
		"global.anthropic.claude-fable-5-2",
		"claude-mythos-preview",
		"claude-sonnet-5-20260701",
		"publishers/anthropic/models/claude-sonnet-5@20260701",
		"us.anthropic.claude-sonnet-5-v1:0",
		"arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-5-v1:0",
		"claude-sonnet-5-1",
	} {
		checkSourceLimits(t, model, 1_000_000, 128_000)
	}
}

func TestClaude46Limits(t *testing.T) {
	// 1M context and 128K output for Claude 4.6 outside Bedrock.
	for _, model := range []string{"claude-sonnet-4-6", "claude-opus-4-6"} {
		checkSourceLimits(t, model, 1_000_000, 128_000)
	}
}

func TestOlderClaudeLimits(t *testing.T) {
	checkSourceLimits(t, "claude-3-5-sonnet-20241022", 200_000, 8192)
}

func TestGPT5Through53Limits(t *testing.T) {
	for _, model := range []string{"gpt-5", "gpt-5.2", "models/gpt-5.3-20260101"} {
		checkSourceLimits(t, model, 400_000, 128_000)
	}
}

func TestCurrentAndFutureGPT5Limits(t *testing.T) {
	for _, model := range []string{"gpt-5.4", "gpt-5.6-sol", "models/gpt-5.7-20270101"} {
		checkSourceLimits(t, model, 1_050_000, 128_000)
	}
}

func TestUppercaseQualifiedOpenAILimits(t *testing.T) {
	checkSourceLimits(t, "~OPENAI/GPT-5.6-SOL", 1_050_000, 128_000)
}

func TestSourceQualifiedOpenAILimits(t *testing.T) {
	// Transport limits are not inferred from a source-qualified model ID.
	for _, model := range []string{"openai.gpt-5.4", "us.openai.gpt-5.6-sol-v1:0"} {
		checkSourceLimits(t, model, 1_050_000, 128_000)
	}
}

func TestGPTProAndOSeriesLimits(t *testing.T) {
	checkInt(t, "gpt-5-pro", MaxOutputTokens("gpt-5-pro"), 272_000)
	checkInt(t, "gpt-5.4-pro", MaxOutputTokens("gpt-5.4-pro"), 128_000)
	checkInt(t, "gpt-6-pro", MaxOutputTokens("gpt-6-pro"), 128_000)
	checkInt(t, "o4-mini", MaxOutputTokens("o4-mini"), 100_000)
	checkSourceLimits(t, "~O5-PREVIEW", 200_000, 100_000)
}

func TestFutureLlamaContext(t *testing.T) {
	checkInt(t, "meta-llama/LLAMA-5-SCOUT", ContextWindowSize("meta-llama/LLAMA-5-SCOUT"), 10_000_000)
	checkInt(t, "meta.llama6-maverick-v1:0", ContextWindowSize("meta.llama6-maverick-v1:0"), 1_000_000)
}

func TestMistralContextAndUnknownLimits(t *testing.T) {
	for _, tc := range []struct {
		model   string
		context int
	}{
		{"mistral-small-2603", 256_000},
		{"mistral-medium-3.6", 256_000},
		{"magistral-medium-2509", 128_000},
		{"voxtral-small-2507", 32_000},
		{"labs-leanstral-1-5", 256_000},
		{"future-provider/new-capability-v1", 0},
	} {
		checkInt(t, tc.model, ContextWindowSize(tc.model), tc.context)
	}
}
