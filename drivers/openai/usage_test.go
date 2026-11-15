package openai

import (
	"encoding/json"
	"testing"
)

// Mirrors the TS test for OpenAI-compatible Responses endpoints that report cached
// tokens under prompt_tokens_details or as a top-level cached_tokens field.
func TestOpenAIResponsesUsageReadsCompatibleCachedTokens(t *testing.T) {
	t.Parallel()

	for name, raw := range map[string]string{
		"input_tokens_details":  `{"input_tokens":10,"output_tokens":2,"total_tokens":12,"input_tokens_details":{"cached_tokens":4}}`,
		"prompt_tokens_details": `{"input_tokens":10,"output_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":4}}`,
		"cached_tokens":         `{"input_tokens":10,"output_tokens":2,"total_tokens":12,"cached_tokens":4}`,
	} {
		var usage responsesUsage
		if err := json.Unmarshal([]byte(raw), &usage); err != nil {
			t.Fatal(err)
		}
		got := openAIResponseUsage(&usage)
		if got == nil || count(got.PromptCached) != 4 || count(got.PromptNew) != 6 || count(got.Prompt) != 10 {
			t.Fatalf("%s: usage = %#v", name, got)
		}
	}
}
