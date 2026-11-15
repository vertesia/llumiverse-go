package capability

import (
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

// Ported from common/src/capability.test.ts.

func TestDedicatedInferenceModelClassification(t *testing.T) {
	for _, model := range []string{"gpt-image-1", "text-embedding-3-small", "whisper-1", "sora-2"} {
		checkBool(t, model, IsDedicatedInferenceModel(model, common.ProviderOpenAI), true)
	}
	// Unknown future models stay in normal inference listings.
	checkBool(t, "future-inference-model-v1",
		IsDedicatedInferenceModel("future-inference-model-v1", common.ProviderOpenAI), false)
}

func TestEmbeddingModelClassification(t *testing.T) {
	embeddings := []struct {
		model    Listing
		provider common.Provider
	}{
		{Listing{ID: "text-embedding-3-small"}, common.ProviderOpenAI},
		{Listing{ID: "text-embedding-ada-002"}, common.ProviderAzureOpenAI},
		{Listing{ID: "mistral-embed"}, common.ProviderMistralAI},
		{Listing{ID: "embedding-model", OutputModalities: []string{"vectors"}}, common.ProviderXAI},
		{Listing{ID: "chat-model", Type: "embedding"}, common.ProviderTogetherAI},
	}
	for _, tc := range embeddings {
		checkBool(t, tc.model.ID, IsEmbeddingModel(tc.model, tc.provider), true)
	}

	// Normal inference models are not embeddings.
	notEmbeddings := []struct {
		model    Listing
		provider common.Provider
	}{
		{Listing{ID: "gpt-5.6-sol"}, common.ProviderOpenAI},
		{Listing{ID: "grok-4.3", OutputModalities: []string{"text"}}, common.ProviderXAI},
		// Non-embedding models are not hidden when capability metadata is incomplete.
		{Listing{ID: "whisper-large-v3"}, common.ProviderGroq},
		{Listing{ID: "canopylabs/orpheus-v1-english"}, common.ProviderGroq},
		{Listing{ID: "meta-llama/llama-prompt-guard-2-86m"}, common.ProviderGroq},
		{Listing{ID: "llama-3.3-70b-versatile"}, common.ProviderGroq},
		{Listing{ID: "gpt-image-1"}, common.ProviderOpenAI},
		{Listing{ID: "amazon.titan-image-generator-v3"}, common.ProviderBedrock},
		{Listing{ID: "amazon.nova-reel-v1"}, common.ProviderBedrock},
	}
	for _, tc := range notEmbeddings {
		checkBool(t, tc.model.ID, IsEmbeddingModel(tc.model, tc.provider), false)
	}
}

func TestXAIGrokToolCapabilities(t *testing.T) {
	// Tool use is on without setting tool_support_streaming; streaming agents must still attach tools.
	for _, model := range []string{"grok-2", "grok-3", "grok-4", "grok-4-fast-reasoning"} {
		caps := ForModel(model, common.ProviderXAI)
		checkFlag(t, model+" tool_support", caps.ToolSupport, T)
		checkFlag(t, model+" tool_support_streaming", caps.ToolSupportStreaming, nil)
		checkBool(t, model+" streaming tools", SupportsToolUse(model, common.ProviderXAI, true), true)
		checkBool(t, model+" tools", SupportsToolUse(model, common.ProviderXAI, false), true)
	}
}

func TestGrokImageInput(t *testing.T) {
	checkFlag(t, "grok-4.20", ForModel("grok-4.20", common.ProviderXAI).Input.Image, T)
	checkFlag(t, "grok-4.5", ForModel("grok-4.5", common.ProviderXAI).Input.Image, T)
}

func TestMediaInputsKeptWhileUnsupportedOutputsMasked(t *testing.T) {
	caps := ForModel("gemini-4.0-flash", common.ProviderVertexAI)
	checkFlag(t, "input.audio", caps.Input.Audio, T)
	checkFlag(t, "input.video", caps.Input.Video, T)
	checkFlag(t, "output.audio", caps.Output.Audio, F)
	checkFlag(t, "output.video", caps.Output.Video, F)
}

func TestGeminiOmniVideoCapabilities(t *testing.T) {
	want := Capabilities{
		Input:                Modalities{Text: T, Image: T, Video: T, Audio: F, Embed: F},
		Output:               Modalities{Text: T, Image: F, Video: T, Audio: F, Embed: F},
		ToolSupport:          F,
		ToolSupportStreaming: F,
	}
	for _, model := range []string{
		"gemini-omni-flash-preview",
		"gemini-omni-1.1-flash-preview",
		"gemini-omni-1.2-flash",
		"publishers/google/models/gemini-omni-1.3-flash",
	} {
		checkCapsEqual(t, model, ForModel(model, common.ProviderVertexAI), want)
	}
}

func TestSupportsToolUseStreamingDefault(t *testing.T) {
	t.Run("falls back to tool_support when tool_support_streaming is unset", func(t *testing.T) {
		// Vertex Grok records tool_support but not tool_support_streaming.
		caps := ForModel("grok-3", common.ProviderVertexAI)
		checkFlag(t, "tool_support", caps.ToolSupport, T)
		checkFlag(t, "tool_support_streaming", caps.ToolSupportStreaming, nil)
		checkBool(t, "streaming tools", SupportsToolUse("grok-3", common.ProviderVertexAI, true), true)
	})

	t.Run("honors explicit tool_support_streaming false", func(t *testing.T) {
		// Bedrock runtime Llama has tools but not streaming tools.
		const model = "meta.llama3-1-70b-instruct-v1:0"
		caps := ForModel(model, common.ProviderBedrock)
		checkFlag(t, "tool_support", caps.ToolSupport, T)
		checkFlag(t, "tool_support_streaming", caps.ToolSupportStreaming, F)
		checkBool(t, "streaming tools", SupportsToolUse(model, common.ProviderBedrock, true), false)
		checkBool(t, "tools", SupportsToolUse(model, common.ProviderBedrock, false), true)
	})

	t.Run("leaves genuinely unknown Bedrock tool support unset", func(t *testing.T) {
		caps := ForModel("future-provider.unknown-chat-v1", common.ProviderBedrock)
		checkFlag(t, "tool_support", caps.ToolSupport, nil)
		checkFlag(t, "tool_support_streaming", caps.ToolSupportStreaming, nil)
	})
}
