package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
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
			Description: ptr("Lookup data"),
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
	if resp.TokenUsage == nil || count(resp.TokenUsage.Total) != 7 {
		t.Fatalf("usage = %#v", resp.TokenUsage)
	}
	if len(resp.Result) != 1 || resp.Result[0].Type() != ResultTypeJSON {
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

// Mirrors the TS OpenAI-compatible listing: embedding and dedicated-endpoint
// models are left out except file audio, the capability catalogue describes
// known models, and image input marks a model multimodal.
func TestOpenAICompatibleListModelsUsesTheCatalogue(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"meta-llama/Llama-4-Scout"},{"id":"nomic-embed-text"},{"id":"gpt-image-1"},
			{"id":"whisper-large-v3"},{"id":"gpt-realtime"},{"id":"local-model"}]}`))
	}))
	defer server.Close()
	driver, err := NewOpenAICompatibleDriver(OpenAICompatibleOptions{APIKey: "k", Endpoint: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	if got := strings.Join(ids, ","); got != "local-model,meta-llama/Llama-4-Scout,whisper-large-v3" {
		t.Fatalf("models = %s", got)
	}
	local, llama, whisper := models[0], models[1], models[2]
	if !*llama.ToolSupport || !*llama.IsMultimodal || strings.Join(*llama.InputModalities, ",") != "text,image" || *llama.Type != modelTypeText {
		t.Fatalf("llama = %+v", llama)
	}
	if local.ToolSupport != nil || *local.IsMultimodal || strings.Join(*local.OutputModalities, ",") != "text" {
		t.Fatalf("local = %+v", local)
	}
	if *whisper.Type != modelTypeAudio || strings.Join(*whisper.InputModalities, ",") != "audio" {
		t.Fatalf("whisper = %+v", whisper)
	}
}

// Mirrors the TS model listing test: on OpenAI, dedicated and legacy endpoint
// families are left out, and unknown ids stay visible. File audio models such
// as whisper are listed, as audio models.
func TestOpenAIListModelsLeavesOutDedicatedEndpoints(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-5.6"},{"id":"text-embedding-3-small"},{"id":"whisper-1"},{"id":"gpt-realtime"},{"id":"computer-use-preview"},{"id":"davinci-002"},{"id":"gpt-7-future"},{"id":"gpt-5.1-codex"}]}`))
	}))
	defer server.Close()
	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "k", Endpoint: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	if got := strings.Join(ids, ","); got != "gpt-5.1-codex,gpt-5.6,gpt-7-future,whisper-1" {
		t.Fatalf("models = %s", got)
	}
	if whisper := models[3]; *whisper.Type != modelTypeAudio {
		t.Fatalf("whisper = %+v", whisper)
	}
}

// Mirrors the TS model listing test: executable Codex, pro reasoning and file
// audio models are kept, embeddings, moderation and video models are not, and
// the capability catalogue describes each model.
func TestOpenAIListModelsKeepsExecutableModels(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-5.6-codex","owned_by":"system"},{"id":"o1-pro","owned_by":"system"},
			{"id":"gpt-5-audiovisual","owned_by":"system"},{"id":"text-embedding-3-small","owned_by":"system"},
			{"id":"gpt-4o-mini-tts","owned_by":"system"},{"id":"gpt-4o-transcribe","owned_by":"system"},
			{"id":"gpt-audio","owned_by":"system"},{"id":"gpt-image-1","owned_by":"system"},
			{"id":"omni-moderation-latest","owned_by":"system"},{"id":"sora-3","owned_by":"system"},
			{"id":"acme-future","owned_by":"system"}]}`))
	}))
	defer server.Close()
	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "k", Endpoint: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]AIModel{}
	var ids []string
	for _, model := range models {
		byID[model.ID] = model
		ids = append(ids, model.ID)
	}
	want := "acme-future,gpt-4o-mini-tts,gpt-4o-transcribe,gpt-5-audiovisual,gpt-5.6-codex,gpt-audio,gpt-image-1,o1-pro"
	if got := strings.Join(ids, ","); got != want {
		t.Fatalf("models = %s", got)
	}
	codex := byID["gpt-5.6-codex"]
	if *codex.Owner != "openai" || *codex.Type != modelTypeText || !*codex.ToolSupport ||
		strings.Join(*codex.InputModalities, ",") != "text,image" || strings.Join(*codex.OutputModalities, ",") != "text" {
		t.Fatalf("codex = %+v", codex)
	}
	audio := byID["gpt-audio"]
	if *audio.Type != modelTypeAudio || *audio.ToolSupport || strings.Join(*audio.OutputModalities, ",") != "text,audio" {
		t.Fatalf("audio = %+v", audio)
	}
	if image := byID["gpt-image-1"]; *image.Type != modelTypeImage || *image.ToolSupport || strings.Join(*image.OutputModalities, ",") != "image" {
		t.Fatalf("image = %+v", image)
	}
	// The catalogue does not know an unknown model: text in and out, tool support unknown.
	if unknown := byID["acme-future"]; unknown.ToolSupport != nil || strings.Join(*unknown.InputModalities, ",") != "text" {
		t.Fatalf("unknown = %+v", unknown)
	}
}

// Mirrors the TS model listing test: image aliases, snapshots, qualified IDs and
// future generations are listed as image models, DALL-E is not.
func TestOpenAIListModelsListsImageModelsWithoutDallE(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-image-2.5-sunburst"},{"id":"gpt-image-2.5-flare-2026-09-08"},
			{"id":"openai/GPT-IMAGE-3"},{"id":"chatgpt-image-latest"},{"id":"dall-e-2"},{"id":"dall-e-3"}]}`))
	}))
	defer server.Close()
	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "k", Endpoint: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, model := range models {
		ids = append(ids, model.ID)
		if *model.Type != modelTypeImage || strings.Join(*model.OutputModalities, ",") != "image" {
			t.Fatalf("model = %+v", model)
		}
	}
	if got := strings.Join(ids, ","); got != "chatgpt-image-latest,gpt-image-2.5-flare-2026-09-08,gpt-image-2.5-sunburst,openai/GPT-IMAGE-3" {
		t.Fatalf("models = %s", got)
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
	if len(resp.Result) != 1 || resp.Result[0].Type() != ResultTypeJSON {
		t.Fatalf("result = %#v", resp.Result)
	}
	if resp.TokenUsage == nil || count(resp.TokenUsage.PromptCached) != 1 || count(resp.TokenUsage.Total) != 5 {
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
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-4.1","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`))
	}))
	defer server.Close()

	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "test-key", Endpoint: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "summarize"}}, ExecutionOptions{
		Model: "gpt-4.1",
		Conversation: []any{
			map[string]any{"type": "function_call", "arguments": `{"q":"tokyo"}`, "call_id": "call_1", "name": "lookup"},
			map[string]any{"type": "function_call_output", "call_id": "call_1", "output": `{"ok":true}`},
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
	if len(resp.Result) != 1 || resp.Result[0].Type() != ResultTypeImage || resp.Result[0].String() != "data:image/webp;base64,aW1hZ2U=" {
		t.Fatalf("result = %#v", resp.Result)
	}
}

func openAIResponseInputTexts(items []responseItem) []string {
	var out []string
	for _, item := range items {
		switch itemType(item) {
		case "message":
			switch content := item["content"].(type) {
			case string:
				out = append(out, content)
			case []any:
				for _, raw := range content {
					if part, ok := raw.(map[string]any); ok && (part["type"] == "input_text" || part["type"] == "output_text") {
						out = append(out, stringValue(part["text"]))
					}
				}
			}
		case "function_call_output":
			out = append(out, functionCallOutputText(item["output"]))
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
