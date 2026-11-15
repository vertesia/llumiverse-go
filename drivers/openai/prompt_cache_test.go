package openai

import (
	"context"
	"encoding/json"
	"errors"
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

// Mirrors the TS cache_ttl tests: the lifetime goes in the parameter the model
// takes, and one the model does not accept is dropped.
func TestOpenAIResponsesPromptCacheTTL(t *testing.T) {
	t.Parallel()

	body, _, err := executeResponses(t, ExecutionOptions{Model: "gpt-5.6", PromptCacheKey: "k", ModelOptions: map[string]any{"cache_ttl": "12h"}}, okResponse)
	if err != nil {
		t.Fatal(err)
	}
	if got := marshalNoEscape(body["prompt_cache_options"]); got != `{"ttl":"12h"}` || body["prompt_cache_retention"] != nil {
		t.Fatalf("gpt-5.6 body = %#v", body)
	}
	body, _, err = executeResponses(t, ExecutionOptions{Model: "gpt-4.1-2025-04-14", ModelOptions: map[string]any{"cache_ttl": "24h"}}, okResponse)
	if err != nil || body["prompt_cache_retention"] != "24h" || body["prompt_cache_options"] != nil {
		t.Fatalf("gpt-4.1: %v, %#v", err, body)
	}
	for model, ttl := range map[string]string{"gpt-5.6": "24h", "gpt-4.1": "12h", "gpt-4o": "24h"} {
		body, _, err = executeResponses(t, ExecutionOptions{Model: model, ModelOptions: map[string]any{"cache_ttl": ttl}}, okResponse)
		if err != nil || body["prompt_cache_retention"] != nil || body["prompt_cache_options"] != nil {
			t.Errorf("%s %s: %v, %#v", model, ttl, err, body)
		}
	}
	if got := openAIPromptCacheTTLs("gpt-5.6", ProviderOpenAICompatible); got != nil {
		t.Fatalf("compatible lifetimes = %v", got)
	}
}

// Mirrors the TS extra_body and seed tests: extensions reach the body root
// without replacing the fields llumiverse sets, and a seed is passed through.
func TestOpenAIResponsesExtraBodyAndSeed(t *testing.T) {
	t.Parallel()

	body, _, err := executeResponses(t, ExecutionOptions{ModelOptions: map[string]any{
		"seed":       7,
		"max_tokens": 50,
		"extra_body": map[string]any{"provider_flag": true, "max_output_tokens": 999, "model": "other"},
	}}, okResponse)
	if err != nil {
		t.Fatal(err)
	}
	if body["seed"] != float64(7) || body["provider_flag"] != true || body["max_output_tokens"] != float64(50) || body["model"] != "gpt-4.1" || body["extra_body"] != nil {
		t.Fatalf("body = %#v", body)
	}
	body, _, err = executeResponses(t, ExecutionOptions{}, okResponse)
	if _, ok := body["seed"]; err != nil || ok {
		t.Fatalf("no seed: %v, %#v", err, body)
	}
}

func TestOpenAIResponsesUsageReadsCacheWrites(t *testing.T) {
	t.Parallel()

	var usage responsesUsage
	if err := json.Unmarshal([]byte(`{"input_tokens":10,"output_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":4}}`), &usage); err != nil {
		t.Fatal(err)
	}
	got := openAIResponseUsage(&usage, nil)
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

// Mirrors the TS prompt_cache_schema_suffix test: the schema goes in a final user
// message and the native format becomes json_object, so the cached prefix does
// not depend on the schema.
func TestOpenAIResponsesSchemaSuffixKeepsPromptCache(t *testing.T) {
	t.Parallel()

	schema := map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}}
	body, _, err := executeResponses(t, ExecutionOptions{ResultSchema: schema, PromptCacheSchemaSuffix: true, ModelOptions: map[string]any{"verbosity": "low"}}, okResponse,
		PromptSegment{Role: PromptRoleSystem, Content: "Be brief"}, PromptSegment{Role: PromptRoleUser, Content: "Hi"})
	if err != nil {
		t.Fatal(err)
	}
	input, _ := body["input"].([]any)
	last := marshalNoEscape(input[len(input)-1])
	if !strings.Contains(last, `"role":"user"`) || !strings.Contains(last, "<response_schema>") {
		t.Fatalf("last input = %s", last)
	}
	if got := marshalNoEscape(body["text"]); got != `{"format":{"type":"json_object"},"verbosity":"low"}` {
		t.Fatalf("text = %s", got)
	}
}

// Mirrors the TS service tier tests: the tier is sent and the tier the
// provider reports is captured, streamed or not.
func TestOpenAIResponsesServiceTier(t *testing.T) {
	t.Parallel()

	body, resp, err := executeResponses(t, ExecutionOptions{ModelOptions: map[string]any{"service_tier": "flex"}},
		`{"id":"r","status":"completed","service_tier":"flex","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`)
	if err != nil {
		t.Fatal(err)
	}
	if body["service_tier"] != "flex" || resp.ServiceTier != "flex" {
		t.Fatalf("body tier %v, response tier %q", body["service_tier"], resp.ServiceTier)
	}
	stream, err := streamResponses(t, ExecutionOptions{}, `{"type":"response.completed","response":{"id":"r","status":"completed","service_tier":"priority","output":[]}}`)
	if err != nil {
		t.Fatal(err)
	}
	if stream.ServiceTier != "priority" {
		t.Fatalf("stream tier = %q", stream.ServiceTier)
	}
}

// Mirrors the TS tool-result attachment tests (and openai_format file parts):
// tool outputs carry their files, PDFs become input_file, text files their
// content, audio is skipped, and a plain system message is a string.
func TestOpenAIResponsesAttachments(t *testing.T) {
	t.Parallel()

	files := []DataSource{
		&BytesDataSource{FileName: "chart.png", MIME: "image/png", Data: []byte("png")},
		&BytesDataSource{FileName: "report.pdf", MIME: "application/pdf", Data: []byte("pdf")},
		&BytesDataSource{FileName: "notes.txt", MIME: "text/plain", Data: []byte("plain notes")},
		&BytesDataSource{FileName: "clip.mp3", MIME: "audio/mpeg", Data: []byte("mp3")},
	}
	conversation := []responseItem{{"type": "function_call", "call_id": "c1", "name": "fetch", "arguments": "{}"}}
	body, _, err := executeResponses(t, ExecutionOptions{Conversation: conversation, Tools: []ToolDefinition{{Name: "fetch"}}}, okResponse,
		PromptSegment{Role: PromptRoleSystem, Content: "Be brief"},
		PromptSegment{Role: PromptRoleTool, ToolUseID: "c1", Content: "fetched", Files: files})
	if err != nil {
		t.Fatal(err)
	}
	input := marshalNoEscape(body["input"])
	for _, want := range []string{
		`{"content":"Be brief","role":"system","type":"message"}`,
		`{"call_id":"c1","output":[{"text":"fetched","type":"input_text"},{"detail":"auto","image_url":"data:image/png;base64,cG5n","type":"input_image"},{"file_data":"data:application/pdf;base64,cGRm","filename":"report.pdf","type":"input_file"},{"text":"plain notes","type":"input_text"}],"type":"function_call_output"}`,
	} {
		if !strings.Contains(input, want) {
			t.Fatalf("input %s\nmissing %s", input, want)
		}
	}
}

// Mirrors the TS named tool choice tests on Responses.
func TestOpenAIResponsesToolChoice(t *testing.T) {
	t.Parallel()

	tools := []ToolDefinition{{Name: "lookup"}}
	body, _, err := executeResponses(t, ExecutionOptions{Tools: tools, ModelOptions: map[string]any{"required_tool_name": "lookup", "parallel_tool_calls": false}}, okResponse)
	if err != nil {
		t.Fatal(err)
	}
	if got := marshalNoEscape(body["tool_choice"]); got != `{"name":"lookup","type":"function"}` || body["parallel_tool_calls"] != false {
		t.Fatalf("body = %#v", body)
	}
	body, _, err = executeResponses(t, ExecutionOptions{Tools: tools, ModelOptions: map[string]any{"tool_choice": "any"}}, okResponse)
	if err != nil || body["tool_choice"] != "required" {
		t.Fatalf("any: %v, %#v", err, body)
	}
	body, _, err = executeResponses(t, ExecutionOptions{ModelOptions: map[string]any{"parallel_tool_calls": true}}, okResponse)
	if err != nil || body["parallel_tool_calls"] != nil || body["tool_choice"] != nil {
		t.Fatalf("no tools: %v, %#v", err, body)
	}
	_, _, err = executeResponses(t, ExecutionOptions{ModelOptions: map[string]any{"tool_choice": "required"}}, okResponse)
	var llumiverseErr *LlumiverseError
	if !errors.As(err, &llumiverseErr) || llumiverseErr.Code != 400 || llumiverseErr.Name != "ToolChoiceConfigurationError" {
		t.Fatalf("error = %#v", err)
	}
}
