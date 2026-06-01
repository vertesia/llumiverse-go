package llumiverse

import (
	"context"
	"errors"
	"testing"
)

func TestFacadeHelpers(t *testing.T) {
	t.Parallel()

	if *BoolPtr(false) {
		t.Fatal("BoolPtr(false) returned true")
	}
	original := errors.New("boom")
	err := NewLlumiverseError("boom", BoolPtr(false), LlumiverseErrorContext{
		Provider:  ProviderOpenAI,
		Model:     "gpt-test",
		Operation: "execute",
	}, original, 400, "ProviderError")
	if err.Error() != "[openai] boom" || err.Name != "ProviderError" || !errors.Is(err, original) {
		t.Fatalf("error = %#v", err)
	}
}

func TestFacadeConstructorsValidateRequiredOptions(t *testing.T) {
	t.Parallel()

	if _, err := NewOpenAIDriver(OpenAICompatibleOptions{}); err == nil {
		t.Fatal("NewOpenAIDriver without API key returned nil error")
	}
	if _, err := NewOpenAICompatibleDriver(OpenAICompatibleOptions{APIKey: "key"}); err == nil {
		t.Fatal("NewOpenAICompatibleDriver without endpoint returned nil error")
	}
	if _, err := NewAnthropicDriver(AnthropicOptions{}); err == nil {
		t.Fatal("NewAnthropicDriver without API key returned nil error")
	}
	if _, err := NewVertexAIDriver(VertexAIOptions{}); err == nil {
		t.Fatal("NewVertexAIDriver without project/region returned nil error")
	}
	if _, err := NewBedrockDriver(context.Background(), BedrockOptions{}); err == nil {
		t.Fatal("NewBedrockDriver without region returned nil error")
	}
	if driver := NewBedrockDriverWithClient(BedrockOptions{}, nil, nil); driver == nil {
		t.Fatal("NewBedrockDriverWithClient returned nil")
	}
}
