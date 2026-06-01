package anthropic

import "testing"

func TestNewAnthropicDriverRequiresAPIKey(t *testing.T) {
	t.Parallel()

	if _, err := NewAnthropicDriver(AnthropicOptions{}); err == nil {
		t.Fatal("expected missing API key error")
	}
}
