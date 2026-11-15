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
		got := openAIResponseUsage(&usage, nil)
		if got == nil || count(got.PromptCached) != 4 || count(got.PromptNew) != 6 || count(got.Prompt) != 10 {
			t.Fatalf("%s: usage = %#v", name, got)
		}
	}
}

// Mirrors the TS Responses usage tests: 12-hour cache writes are split out,
// hosted tools are reported apart from the model, and unused tools are absent.
func TestOpenAIResponsesUsageToolsAndRetention(t *testing.T) {
	t.Parallel()

	_, resp, err := executeResponses(t, ExecutionOptions{}, `{"id":"r","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],
		"usage":{"input_tokens":100,"output_tokens":5,"total_tokens":105,"input_tokens_details":{"cached_tokens":10,"cache_write_tokens":40,"cache_write_tokens_details":{"ttl_12h_tokens":30}}},
		"tool_usage":{"image_gen":{"input_tokens":20,"output_tokens":1000,"total_tokens":1020,"output_tokens_details":{"image_tokens":990}},"web_search":{"num_requests":2}}}`)
	if err != nil {
		t.Fatal(err)
	}
	if got := marshalNoEscape(resp.TokenUsage); got != `{"prompt":100,"prompt_cache_write":40,"prompt_cache_write_12h":30,"prompt_cached":10,"prompt_new":50,"result":5,"tool_usage":{"image_generation":{"prompt":20,"result":1000,"result_image":990,"total":1020},"web_search_requests":2},"total":105}` {
		t.Fatalf("usage = %s", got)
	}

	var unused responsesToolUsage
	if err := json.Unmarshal([]byte(`{"image_gen":{"input_tokens":0,"output_tokens":0,"total_tokens":0},"web_search":{"num_requests":0}}`), &unused); err != nil {
		t.Fatal(err)
	}
	if got := openAIResponseUsage(nil, &unused); got != nil {
		t.Fatalf("unused tools = %#v", got)
	}
	unused.WebSearch.NumRequests = 1
	if got := openAIResponseUsage(nil, &unused); got == nil || count(got.ToolUsage.WebSearchRequests) != 1 || got.Prompt != nil {
		t.Fatalf("tools without model usage = %#v", got)
	}
}
