package openrouter

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/drivers/internal/chatcompat"
	"github.com/vertesia/llumiverse-go/drivers/internal/fakehttp"
)

const chatPath = "/api/v1/chat/completions"

const chatBody = `{"id":"c1","object":"chat.completion","created":1,"model":"m",
	"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"{\"answer\":\"ok\"}","reasoning":"Thinking"}}],
	"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"cost":0.0025,"is_byok":false,
		"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":2}}}`

func newTestDriver(t *testing.T, routes map[string]fakehttp.Response) (*OpenRouterDriver, *fakehttp.Server) {
	t.Helper()
	server := fakehttp.New(t, routes)
	driver, err := NewOpenRouterDriver(OpenRouterOptions{
		APIKey:        "test-key",
		Endpoint:      server.URL + "/api/v1",
		HTTPReferer:   "https://app.example.com",
		AppTitle:      "Example App",
		AppCategories: "programming",
	})
	if err != nil {
		t.Fatal(err)
	}
	return driver, server
}

var answerSchema = map[string]any{
	"type":                 "object",
	"properties":           map[string]any{"answer": map[string]any{"type": "string"}},
	"required":             []any{"answer"},
	"additionalProperties": false,
}

func TestOpenRouterPayloadAndHeaders(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: chatBody}})
	conversation := map[string]any{"messages": []any{
		map[string]any{"role": "user", "content": "First"},
		map[string]any{"role": "assistant", "content": "Earlier", "reasoning_content": "Earlier thoughts"},
	}}
	resp, err := driver.Execute(context.Background(), []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Answer."}}, common.ExecutionOptions{
		Model:        "z-ai/glm-5.3",
		ResultSchema: answerSchema,
		Conversation: conversation,
		ModelOptions: map[string]any{
			"provider_order":           []any{"anthropic", "together"},
			"provider_allow_fallbacks": false,
			"provider_zdr":             true,
			"include_thoughts":         true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	request := server.Last(t, chatPath)
	for header, want := range map[string]string{
		"Authorization":           "Bearer test-key",
		"Http-Referer":            "https://app.example.com",
		"X-Openrouter-Title":      "Example App",
		"X-Openrouter-Categories": "programming",
	} {
		if got := request.Header.Get(header); got != want {
			t.Fatalf("%s = %q, want %q", header, got, want)
		}
	}
	body := request.Body
	if _, ok := body["n"]; ok {
		t.Fatalf("n sent: %v", body)
	}
	provider, _ := body["provider"].(map[string]any)
	if provider["allow_fallbacks"] != false || provider["zdr"] != true || len(provider["order"].([]any)) != 2 {
		t.Fatalf("provider = %v", body["provider"])
	}
	if _, ok := body["response_format"].(map[string]any); !ok {
		t.Fatalf("response_format missing: %v", body)
	}
	messages := body["messages"].([]any)
	schemaInPrompt := false
	for _, raw := range messages {
		message := raw.(map[string]any)
		schemaInPrompt = schemaInPrompt || (message["role"] == "system" && strings.Contains(chatcompat.TextContent(message["content"]), "<response_schema>"))
	}
	if !schemaInPrompt {
		t.Fatalf("GLM 5.3 lacks the prompt schema: %v", messages)
	}
	for _, raw := range messages {
		message := raw.(map[string]any)
		if message["role"] != "assistant" {
			continue
		}
		if message["reasoning"] != "Earlier thoughts" || message["reasoning_content"] != nil {
			t.Fatalf("assistant reasoning = %v", message)
		}
	}
	usage := resp.TokenUsage
	if usage.ProviderCostUsd == nil || *usage.ProviderCostUsd != 0.0025 || *usage.PromptCached != 3 || *usage.PromptCacheWrite != 2 {
		t.Fatalf("usage = %+v", usage)
	}
	if resp.Result[0].Type() != common.ResultTypeThoughts {
		t.Fatalf("results = %+v", resp.Result)
	}
}

func TestOpenRouterPromptSchemaOnlyForUnreliableFamilies(t *testing.T) {
	for model, want := range map[string]bool{
		"z-ai/glm-5.3":          true,
		"z-ai/glm-5.3:free":     true,
		"minimax/minimax-m3":    true,
		"z-ai/glm-5.2":          false,
		"openai/gpt-5.5":        false,
		"meta/minimax-imitator": false,
	} {
		if got := supplementsResultSchemaInPrompt(model); got != want {
			t.Errorf("%s: %v, want %v", model, got, want)
		}
	}
}

func TestOpenRouterStreamErrorChunk(t *testing.T) {
	body := fakehttp.SSE(
		`{"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
		`{"id":"c1","error":{"code":429,"message":"Rate limit exceeded upstream"},"choices":[]}`,
	)
	driver, _ := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: body}})
	stream, err := driver.Stream(context.Background(), []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Hi"}}, common.ExecutionOptions{Model: "openai/gpt-5.5"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()
	var streamErr error
	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			streamErr = err
			break
		}
	}
	var llumiErr *common.LlumiverseError
	if !errors.As(streamErr, &llumiErr) {
		t.Fatalf("err = %v, want a LlumiverseError", streamErr)
	}
	if llumiErr.Code != 429 || llumiErr.Retryable == nil || !*llumiErr.Retryable || llumiErr.Context.Provider != common.ProviderOpenRouter {
		t.Fatalf("err = %+v", llumiErr)
	}
}

func TestOpenRouterRefusesBadMessages(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: chatBody}})
	segments := []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Listen", Files: []common.DataSource{
		&common.BytesDataSource{FileName: "a.mp3", MIME: "audio/mpeg", Data: []byte("ID3")},
	}}}
	if _, err := driver.Execute(context.Background(), segments, common.ExecutionOptions{Model: "openai/gpt-5.5"}); err == nil || !strings.Contains(err.Error(), "audio") {
		t.Fatalf("err = %v, want audio refusal", err)
	}
	if len(server.Requests()) != 0 {
		t.Fatal("request sent")
	}
}

func TestOpenRouterListModels(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{"/api/v1/models": {Body: `{"data":[
		{"id":"openai/gpt-5.5","name":"OpenAI: GPT-5.5","description":"Frontier model","architecture":{"input_modalities":["text","image"],"output_modalities":["text"]},"supported_parameters":["tools","max_tokens"]},
		{"id":"openai/text-embedding-3-large","name":"Embeddings","architecture":{"input_modalities":["text"],"output_modalities":["embeddings"]},"supported_parameters":[]},
		{"id":"google/gemini-image","name":"Image","architecture":{"input_modalities":["text"],"output_modalities":["image"]},"supported_parameters":[]},
		{"id":"openai/gpt-4o-mini-tts","name":"TTS","architecture":{"input_modalities":["text"],"output_modalities":["text","audio"]},"supported_parameters":[]},
		{"id":"meta-llama/llama-4","name":"Llama 4","architecture":{"input_modalities":["text"],"output_modalities":["text"]},"supported_parameters":["max_tokens"]}]}`}})
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "meta-llama/llama-4" || models[1].ID != "openai/gpt-5.5" {
		t.Fatalf("models = %+v", models)
	}
	gpt := models[1]
	if !*gpt.ToolSupport || !*gpt.IsMultimodal || *gpt.Owner != "openai" || *gpt.Description != "Frontier model" || *models[0].ToolSupport {
		t.Fatalf("gpt = %+v", gpt)
	}
	if query := server.Last(t, "/api/v1/models").Query; query != "limit=1000" {
		t.Fatalf("query = %q", query)
	}
}

func TestOpenRouterEmbeddings(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{"/api/v1/embeddings": {Body: `{"model":"google/gemini-embedding-2","data":[
		{"index":1,"embedding":[0.5]},{"index":0,"embedding":[0.1,0.2]}],
		"usage":{"prompt_tokens":20,"prompt_tokens_details":{"text_tokens":4,"image_tokens":16}}}`}})
	result, err := driver.GenerateEmbeddings(context.Background(), common.EmbeddingsOptions{
		Model:      "google/gemini-embedding-2",
		Dimensions: 256,
		Inputs: []common.EmbeddingInput{
			{Text: "a cat"},
			{Type: "image", Source: &common.BytesDataSource{FileName: "cat.png", MIME: "image/png", Data: []byte("PNG")}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 2 || *result.Results[1].Outputs[0].Modality != "image" || len(result.Results[0].Outputs[0].Values) != 2 {
		t.Fatalf("results = %+v", result.Results)
	}
	if *result.Usage.InputTokens != 20 || *result.Usage.InputTextTokens != 4 || *result.Usage.InputImageTokens != 16 {
		t.Fatalf("usage = %+v", result.Usage)
	}
	body := server.Last(t, "/api/v1/embeddings").Body
	inputs := body["input"].([]any)
	image := inputs[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if url := image["image_url"].(map[string]any)["url"].(string); !strings.HasPrefix(url, "data:image/png;base64,") {
		t.Fatalf("image input = %v", image)
	}
	if body["dimensions"] != float64(256) || body["encoding_format"] != "float" {
		t.Fatalf("body = %v", body)
	}
	if _, err := driver.GenerateEmbeddings(context.Background(), common.EmbeddingsOptions{Inputs: []common.EmbeddingInput{{Text: "x"}}}); err == nil {
		t.Fatal("embeddings without a model accepted")
	}
}
