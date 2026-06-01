package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go/responses"
)

func TestOpenAICompatibleExecuteUsesResponsesAPI(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("missing bearer key: %q", r.Header.Get("Authorization"))
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_123",
			"object":"response",
			"created_at":1710000000,
			"status":"completed",
			"model":"gpt-test",
			"output":[{
				"id":"msg_1",
				"type":"message",
				"status":"completed",
				"role":"assistant",
				"content":[{"type":"output_text","text":"{\"answer\":\"ok\"}","annotations":[]}]
			}],
			"usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}
		}`))
	}))
	defer server.Close()

	driver, err := NewOpenAICompatibleDriver(OpenAICompatibleOptions{
		APIKey:   "test-key",
		Endpoint: server.URL + "/v1",
	})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := driver.Execute(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "You are terse."},
		{Role: PromptRoleUser, Content: "Answer in JSON."},
	}, ExecutionOptions{
		Model:        "gpt-test",
		ResultSchema: map[string]any{"type": "object"},
		ModelOptions: map[string]any{"temperature": 0.2, "max_tokens": 64},
		Tools: []ToolDefinition{{
			Name:        "lookup",
			Description: "Lookup data",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"q": map[string]any{"type": "string"}},
				"required":   []string{"q"},
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}

	if got["model"] != "gpt-test" {
		t.Fatalf("model = %v", got["model"])
	}
	if got["temperature"] != 0.2 {
		t.Fatalf("temperature = %v", got["temperature"])
	}
	if driver.Provider() != ProviderOpenAICompatible {
		t.Fatalf("provider = %q", driver.Provider())
	}
	if resp.FinishReason != "stop" {
		t.Fatalf("finish reason = %q", resp.FinishReason)
	}
	if resp.TokenUsage == nil || resp.TokenUsage.Total != 7 {
		t.Fatalf("usage = %#v", resp.TokenUsage)
	}
	if len(resp.Result) != 1 || resp.Result[0].Type != ResultTypeJSON {
		t.Fatalf("result = %#v", resp.Result)
	}
}

func TestOpenAICompatibleListModels(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"b","owned_by":"system"},{"id":"a","owned_by":"owner"}]}`))
	}))
	defer server.Close()

	driver, err := NewOpenAICompatibleDriver(OpenAICompatibleOptions{APIKey: "k", Endpoint: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "a" || models[1].ID != "b" {
		t.Fatalf("models not sorted: %#v", models)
	}
}

func TestOpenAIDriverUsesResponsesAPI(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("missing bearer key: %q", r.Header.Get("Authorization"))
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_123",
			"object":"response",
			"created_at":1710000000,
			"status":"completed",
			"model":"gpt-4.1",
			"output":[{
				"id":"msg_1",
				"type":"message",
				"status":"completed",
				"role":"assistant",
				"content":[{"type":"output_text","text":"{\"answer\":\"ok\"}","annotations":[]}]
			}],
			"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5,"input_tokens_details":{"cached_tokens":1}}
		}`))
	}))
	defer server.Close()

	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "test-key", Endpoint: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "You are terse."},
		{Role: PromptRoleUser, Content: "Answer in JSON."},
	}, ExecutionOptions{
		Model:        "gpt-4.1",
		ResultSchema: map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}},
		ModelOptions: map[string]any{"temperature": 0.2, "max_tokens": 64},
		Labels:       map[string]string{"job": "test"},
	})
	if err != nil {
		t.Fatal(err)
	}

	if got["model"] != "gpt-4.1" {
		t.Fatalf("model = %v", got["model"])
	}
	if got["temperature"] != 0.2 {
		t.Fatalf("temperature = %v", got["temperature"])
	}
	metadata, ok := got["metadata"].(map[string]any)
	if !ok || metadata["job"] != "test" {
		t.Fatalf("metadata = %#v", got["metadata"])
	}
	if len(resp.Result) != 1 || resp.Result[0].Type != ResultTypeJSON {
		t.Fatalf("result = %#v", resp.Result)
	}
	if resp.TokenUsage == nil || resp.TokenUsage.PromptCached != 1 || resp.TokenUsage.Total != 5 {
		t.Fatalf("usage = %#v", resp.TokenUsage)
	}
	if resp.Conversation == nil {
		t.Fatal("conversation was not returned")
	}
}

func TestOpenAIResponsesConversationRetentionAndImageDetail(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_123",
			"object":"response",
			"created_at":1710000000,
			"status":"completed",
			"model":"gpt-4.1",
			"output":[{
				"id":"msg_1",
				"type":"message",
				"status":"completed",
				"role":"assistant",
				"content":[{"type":"output_text","text":"done","annotations":[]}]
			}]
		}`))
	}))
	defer server.Close()

	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "test-key", Endpoint: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	stripNow := 0
	resp, err := driver.Execute(context.Background(), []PromptSegment{
		{
			Role:    PromptRoleUser,
			Content: "abcdefghijklmnop",
			Files: []DataSource{BytesDataSource{
				FileName: "image.png",
				MIME:     "image/png",
				Data:     []byte("png"),
			}},
		},
		{Role: PromptRoleAssistant, Content: "<heartbeat>working</heartbeat>"},
	}, ExecutionOptions{
		Model: "gpt-4.1",
		ModelOptions: map[string]any{
			"image_detail": "high",
		},
		StripImagesAfterTurns:     &stripNow,
		StripTextMaxTokens:        2,
		StripHeartbeatsAfterTurns: &stripNow,
	})
	if err != nil {
		t.Fatal(err)
	}

	input := got["input"].([]any)
	content := input[0].(map[string]any)["content"].([]any)
	image := content[0].(map[string]any)
	if image["detail"] != "high" {
		t.Fatalf("image detail = %#v", image)
	}
	conversation, ok := resp.Conversation.(openAIResponsesConversationState)
	if !ok {
		t.Fatalf("conversation = %#v", resp.Conversation)
	}
	if conversation.Turn != 1 {
		t.Fatalf("turn = %d", conversation.Turn)
	}
	texts := openAIResponseInputTexts(conversation.Items)
	if !containsText(texts, conversationImagePlaceholder) {
		t.Fatalf("image placeholder missing: %#v", texts)
	}
	if !containsText(texts, "abcdefgh"+conversationTextTruncatedMarker) {
		t.Fatalf("truncated text missing: %#v", texts)
	}
	if !containsText(texts, conversationHeartbeatPlaceholder) {
		t.Fatalf("heartbeat placeholder missing: %#v", texts)
	}
}

func TestOpenAIResponsesConversationToolItemsBecomeTextWithoutTools(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-4.1","output":[]}`))
	}))
	defer server.Close()

	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "test-key", Endpoint: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "summarize"}}, ExecutionOptions{
		Model: "gpt-4.1",
		Conversation: responses.ResponseInputParam{
			responses.ResponseInputItemParamOfFunctionCall(`{"q":"tokyo"}`, "call_1", "lookup"),
			responses.ResponseInputItemParamOfFunctionCallOutput("call_1", `{"ok":true}`),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	input, ok := got["input"].([]any)
	if !ok {
		t.Fatalf("input = %#v", got["input"])
	}
	for _, item := range input {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if m["type"] == "function_call" || m["type"] == "function_call_output" {
			t.Fatalf("tool item was not converted: %#v", input)
		}
	}
}

func TestOpenAIImageGeneration(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"created":1,"data":[{"b64_json":"aW1hZ2U="}]}`))
	}))
	defer server.Close()

	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "test-key", Endpoint: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "draw a cube"}}, ExecutionOptions{
		Model: "gpt-image-1",
		ModelOptions: map[string]any{
			"size":               "1024x1024",
			"n":                  1,
			"background":         "transparent",
			"output_format":      "webp",
			"output_compression": 80,
			"moderation":         "low",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got["prompt"] != "draw a cube" {
		t.Fatalf("prompt = %v", got["prompt"])
	}
	if got["background"] != "transparent" || got["output_format"] != "webp" || got["output_compression"] != float64(80) || got["moderation"] != "low" {
		t.Fatalf("image options = %#v", got)
	}
	if len(resp.Result) != 1 || resp.Result[0].Type != ResultTypeImage || resp.Result[0].Value != "data:image/png;base64,aW1hZ2U=" {
		t.Fatalf("result = %#v", resp.Result)
	}
}

func openAIResponseInputTexts(items responses.ResponseInputParam) []string {
	var out []string
	for _, item := range items {
		switch {
		case item.OfMessage != nil:
			if item.OfMessage.Content.OfString.Valid() {
				out = append(out, item.OfMessage.Content.OfString.Value)
			}
			for _, part := range item.OfMessage.Content.OfInputItemContentList {
				if part.OfInputText != nil {
					out = append(out, part.OfInputText.Text)
				}
			}
		case item.OfInputMessage != nil:
			for _, part := range item.OfInputMessage.Content {
				if part.OfInputText != nil {
					out = append(out, part.OfInputText.Text)
				}
			}
		case item.OfFunctionCallOutput != nil:
			out = append(out, item.OfFunctionCallOutput.Output)
		}
	}
	return out
}

func containsText(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
