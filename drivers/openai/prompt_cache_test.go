package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// executeResponses runs a non-streaming Responses request against a server
// that answers with response, and returns the request body and the result.
func executeResponses(t *testing.T, options ExecutionOptions, response string, segments ...PromptSegment) (map[string]any, *ExecutionResponse, error) {
	t.Helper()
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()
	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "k", Endpoint: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	if options.Model == "" {
		options.Model = "gpt-4.1"
	}
	if len(segments) == 0 {
		segments = []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}
	}
	resp, err := driver.Execute(context.Background(), segments, options)
	return body, resp, err
}

const okResponse = `{"id":"r","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`

// Mirrors the TS explicit cache boundary tests: on GPT-5.6+, with a cache key and
// at least two user turns, the latest user message with an image becomes the
// breakpoint; the stored conversation is not marked.
func TestOpenAIResponsesExplicitPromptCacheBoundary(t *testing.T) {
	t.Parallel()

	conversation := []responseItem{
		{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_image", "image_url": "https://x/a.png"}, map[string]any{"type": "input_text", "text": "look"}}},
		{"type": "message", "role": "assistant", "content": "seen"},
	}
	body, resp, err := executeResponses(t, ExecutionOptions{Model: "gpt-5.6", PromptCacheKey: "tenant-1", Conversation: conversation}, okResponse)
	if err != nil {
		t.Fatal(err)
	}
	if body["prompt_cache_key"] != "tenant-1" {
		t.Fatalf("prompt_cache_key = %#v", body["prompt_cache_key"])
	}
	if options, _ := body["prompt_cache_options"].(map[string]any); options["mode"] != "explicit" {
		t.Fatalf("prompt_cache_options = %#v", body["prompt_cache_options"])
	}
	input, _ := json.Marshal(body["input"])
	if !strings.Contains(string(input), `{"prompt_cache_breakpoint":{"mode":"explicit"},"text":"look","type":"input_text"}`) {
		t.Fatalf("input = %s", input)
	}
	stored, _ := json.Marshal(resp.Conversation)
	if strings.Contains(string(stored), "prompt_cache_breakpoint") {
		t.Fatalf("conversation was marked: %s", stored)
	}

	body, _, err = executeResponses(t, ExecutionOptions{Model: "gpt-5.4", ModelOptions: map[string]any{"prompt_cache_key": "k2"}, Conversation: conversation}, okResponse)
	if err != nil {
		t.Fatal(err)
	}
	if body["prompt_cache_key"] != "k2" || body["prompt_cache_options"] != nil {
		t.Fatalf("gpt-5.4 body = %#v", body)
	}
}

func TestOpenAIResponsesUsageReadsCacheWrites(t *testing.T) {
	t.Parallel()

	var usage responsesUsage
	if err := json.Unmarshal([]byte(`{"input_tokens":10,"output_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":4}}`), &usage); err != nil {
		t.Fatal(err)
	}
	got := openAIResponseUsage(&usage)
	if count(got.PromptCached) != 3 || count(got.PromptCacheWrite) != 4 || count(got.PromptNew) != 3 {
		t.Fatalf("usage = %#v", got)
	}
}

func TestOpenAIResponsesSchemaInPromptWithoutNativeSupport(t *testing.T) {
	t.Parallel()

	schema := map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}}
	body, _, err := executeResponses(t, ExecutionOptions{Model: "gpt-realtime", ResultSchema: schema}, okResponse)
	if err != nil {
		t.Fatal(err)
	}
	input := marshalNoEscape(body["input"])
	if !strings.Contains(input, "<response_schema>") || body["text"] != nil {
		t.Fatalf("body = %#v", body)
	}
}
