package togetherai

import (
	"context"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
	"github.com/vertesia/llumiverse-go/drivers/internal/fakehttp"
)

const chatPath = "/v1/chat/completions"

const chatBody = `{"id":"c1","object":"chat.completion","created":1,"model":"m",
	"choices":[{"index":0,"finish_reason":"eos","message":{"role":"assistant","content":"Done"}}],
	"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`

func newTestDriver(t *testing.T, routes map[string]fakehttp.Response) (*TogetherAIDriver, *fakehttp.Server) {
	t.Helper()
	server := fakehttp.New(t, routes)
	// A configured /v1 suffix is removed, so requests do not go to /v1/v1.
	driver, err := NewTogetherAIDriver(TogetherAIOptions{APIKey: "test-key", Endpoint: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	return driver, server
}

func TestTogetherPayload(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: chatBody}})
	tool := common.ToolDefinition{Name: "lookup", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}}
	conversation := map[string]any{"messages": []any{
		map[string]any{"role": "user", "content": "Look"},
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "Here"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/a.png"}},
		}},
	}}
	resp, err := driver.Execute(context.Background(), []common.PromptSegment{
		{Role: common.PromptRoleSystem, Content: "Be brief."},
		{Role: common.PromptRoleUser, Content: "Again"},
	}, common.ExecutionOptions{
		Model:        "Qwen/Qwen3.5-397B-A17B",
		Tools:        []common.ToolDefinition{tool},
		Conversation: conversation,
		ModelOptions: map[string]any{"effort": "none", "stop_sequence": "END"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != "stop" {
		t.Fatalf("finish reason = %q, want eos mapped to stop", resp.FinishReason)
	}
	body := server.Last(t, chatPath).Body
	if _, ok := body["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort sent to a toggle-only model: %v", body)
	}
	if reasoning, _ := body["reasoning"].(map[string]any); reasoning["enabled"] != false {
		t.Fatalf("reasoning = %v", body["reasoning"])
	}
	if stop, _ := body["stop"].([]any); len(stop) != 1 || stop[0] != "END" {
		t.Fatalf("stop = %v", body["stop"])
	}
	function := body["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if function["strict"] != false {
		t.Fatalf("tool strict = %v", function["strict"])
	}
	messages := body["messages"].([]any)
	var assistant map[string]any
	for _, raw := range messages {
		if message := raw.(map[string]any); message["role"] == "assistant" {
			assistant = message
		}
	}
	if assistant["content"] != "Here\n[Image: https://example.com/a.png]" {
		t.Fatalf("assistant content = %#v", assistant["content"])
	}
}

func TestTogetherKeepsEffortOnQwen38(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: chatBody}})
	if _, err := driver.Execute(context.Background(), []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Hi"}}, common.ExecutionOptions{
		Model:        "Qwen/Qwen3.8-Flash",
		ModelOptions: map[string]any{"effort": "low"},
	}); err != nil {
		t.Fatal(err)
	}
	body := server.Last(t, chatPath).Body
	if body["reasoning_effort"] != "low" || body["reasoning"] != nil {
		t.Fatalf("body = %v", body)
	}
	if !hasReasoningToggleOnly("Qwen/Qwen3.6-35B") || hasReasoningToggleOnly("Qwen/Qwen3.5-Coder") || hasReasoningToggleOnly("Qwen/Qwen3-235B") {
		t.Fatal("toggle-only detection")
	}
}

func TestTogetherStreamReadsChoiceUsage(t *testing.T) {
	body := fakehttp.SSE(
		`{"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}`,
		`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"eos","usage":{"prompt_tokens":6,"completion_tokens":1,"total_tokens":7}}]}`,
	)
	driver, _ := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: body}})
	stream, err := driver.Stream(context.Background(), []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Hi"}}, common.ExecutionOptions{Model: "moonshotai/Kimi-K3"})
	if err != nil {
		t.Fatal(err)
	}
	_, resp := fakehttp.Drain(t, stream)
	if core.CompletionResultsToText(resp.Result) != "Hi" || resp.FinishReason != "stop" {
		t.Fatalf("resp = %+v", resp.Completion)
	}
	if resp.TokenUsage == nil || *resp.TokenUsage.Prompt != 6 {
		t.Fatalf("usage = %+v", resp.TokenUsage)
	}
}

func TestTogetherListModelsFollowsTheCursor(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{
		"/v2/supported-models?product=PRODUCT_SERVERLESS": {Body: `{"data":[
			{"name":"meta-llama/Llama-4","displayName":"Llama 4","displayType":"chat","publisher":"Meta","status":"SUPPORTED_MODEL_STATUS_RECOMMENDED",
			 "serverlessEndpoint":"meta-llama/Llama-4-Turbo","inputModalities":["MODALITY_TEXT","MODALITY_IMAGE"],"outputModalities":["MODALITY_TEXT"]},
			{"name":"BAAI/bge-large","displayType":"embedding","status":"SUPPORTED_MODEL_STATUS_SUPPORTED"}],
			"next_cursor":"page2"}`},
		"/v2/supported-models?after=page2&product=PRODUCT_SERVERLESS": {Body: `{"data":[
			{"name":"Qwen/Qwen3.8","displayType":"language","publisher":"Qwen","status":"SUPPORTED_MODEL_STATUS_SUPPORTED"},
			{"name":"hidden/model","displayType":"chat","status":"SUPPORTED_MODEL_STATUS_HIDDEN"}],
			"next_cursor":""}`},
	})
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "Qwen/Qwen3.8" || models[1].ID != "meta-llama/Llama-4-Turbo" {
		t.Fatalf("models = %+v", models)
	}
	llama := models[1]
	if llama.Name != "Llama 4" || *llama.Owner != "Meta" || !*llama.IsMultimodal || strings.Join(*llama.InputModalities, ",") != "text,image" {
		t.Fatalf("llama = %+v", llama)
	}
	if *models[0].Type != "text" || *llama.Type != "chat" {
		t.Fatalf("types = %v %v", *models[0].Type, *llama.Type)
	}
	if got := len(server.Requests()); got != 2 {
		t.Fatalf("requests = %d", got)
	}
	if err := driver.ValidateConnection(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestTogetherEmbeddings(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{"/v1/embeddings": {Body: `{"object":"list","model":"m","data":[
		{"object":"embedding","index":1,"embedding":[0.3]},{"object":"embedding","index":0,"embedding":[0.1,0.2]}],
		"usage":{"prompt_tokens":4,"total_tokens":4}}`}})
	result, err := driver.GenerateEmbeddings(context.Background(), common.EmbeddingsOptions{
		Model:  "BAAI/bge-large",
		Inputs: []common.EmbeddingInput{{Text: "a"}, {Text: "b"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 2 || len(result.Results[0].Outputs[0].Values) != 2 {
		t.Fatalf("results = %+v", result.Results)
	}
	if body := server.Last(t, "/v1/embeddings").Body; body["model"] != "BAAI/bge-large" || body["encoding_format"] != "float" {
		t.Fatalf("body = %v", body)
	}
}
