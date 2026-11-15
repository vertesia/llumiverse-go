package mistral

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
	"github.com/vertesia/llumiverse-go/drivers/internal/fakehttp"
)

const chatPath = "/v1/chat/completions"

const thinkingBody = `{"id":"c1","object":"chat.completion","model":"magistral-medium-latest",
	"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":[
		{"type":"thinking","thinking":[{"type":"text","text":"Check the weather."}],"signature":"sig-1"},
		{"type":"text","text":"Looking it up."}],
		"tool_calls":[{"id":"call_1","type":"function","index":0,"function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}}]}}],
	"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}`

func newTestDriver(t *testing.T, routes map[string]fakehttp.Response) (*MistralDriver, *fakehttp.Server) {
	t.Helper()
	server := fakehttp.New(t, routes)
	driver, err := NewMistralDriver(MistralOptions{APIKey: "test-key", Endpoint: server.URL + "/v1/", DefaultMaxTokens: 512})
	if err != nil {
		t.Fatal(err)
	}
	return driver, server
}

func TestMistralPromptAndRequest(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: thinkingBody}})
	description := "Look up the weather"
	tool := common.ToolDefinition{Name: "weather", Description: &description, InputSchema: map[string]any{"type": "object"}}
	png := &common.BytesDataSource{FileName: "map.png", MIME: "image/png", Data: []byte("PNG")}
	resp, err := driver.Execute(context.Background(), []common.PromptSegment{
		{Role: common.PromptRoleSystem, Content: "Be brief."},
		{Role: common.PromptRoleSystem, Content: "Use tools."},
		{Role: common.PromptRoleUser, Content: "Weather here?", Files: []common.DataSource{png}},
	}, common.ExecutionOptions{
		Model:          "magistral-medium-latest",
		Tools:          []common.ToolDefinition{tool},
		PromptCacheKey: "agent-1",
		ModelOptions: map[string]any{
			"temperature":        0.2,
			"random_seed":        7,
			"safe_prompt":        true,
			"prompt_mode":        "reasoning",
			"effort":             "high",
			"required_tool_name": "weather",
			"stop_sequence":      []any{"END"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	request := server.Last(t, chatPath)
	if request.Header.Get("Authorization") != "Bearer test-key" {
		t.Fatalf("authorization = %q", request.Header.Get("Authorization"))
	}
	body := request.Body
	want := map[string]any{
		"model": "magistral-medium-latest", "max_tokens": float64(512), "temperature": 0.2, "random_seed": float64(7),
		"safe_prompt": true, "prompt_mode": "reasoning", "reasoning_effort": "high", "prompt_cache_key": "agent-1",
		"n": float64(1), "stream": false,
	}
	for key, value := range want {
		if body[key] != value {
			t.Errorf("%s = %v, want %v", key, body[key], value)
		}
	}
	if choice, _ := body["tool_choice"].(map[string]any); choice["type"] != "function" {
		t.Errorf("tool_choice = %v", body["tool_choice"])
	}
	messages := body["messages"].([]any)
	if system := messages[0].(map[string]any); system["content"] != "Be brief.\nUse tools." {
		t.Fatalf("system = %v", system)
	}
	user := messages[1].(map[string]any)["content"].([]any)
	if image := user[1].(map[string]any); image["type"] != "image_url" || !strings.HasPrefix(image["image_url"].(string), "data:image/png;base64,") {
		t.Fatalf("user content = %v", user)
	}

	if len(resp.Result) != 2 || resp.Result[0].Type() != common.ResultTypeThoughts || resp.Result[1].String() != "Looking it up." {
		t.Fatalf("results = %+v", resp.Result)
	}
	if resp.FinishReason != "tool_use" || len(resp.ToolUse) != 1 || resp.ToolUse[0].ID != "call_1" || (*resp.ToolUse[0].ToolInput)["city"] != "Paris" {
		t.Fatalf("tool use = %+v, finish = %q", resp.ToolUse, resp.FinishReason)
	}
	if *resp.TokenUsage.Prompt != 12 || *resp.TokenUsage.Total != 17 {
		t.Fatalf("usage = %+v", resp.TokenUsage)
	}
	conversation := resp.Conversation.(Conversation)
	last := conversation.Messages[len(conversation.Messages)-1]
	if last["role"] != "assistant" || last["tool_calls"] == nil || conversation.Turn != 1 {
		t.Fatalf("conversation = %+v", conversation)
	}
}

func TestMistralResultSchemaInPromptAndHiddenThoughts(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: `{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":[
		{"type":"thinking","thinking":[{"type":"text","text":"hmm"}]},{"type":"text","text":"{\"answer\":\"ok\"}"}]}}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`}})
	schema := map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}}
	resp, err := driver.Execute(context.Background(), []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Answer."}}, common.ExecutionOptions{
		Model:        "mistral-small-latest",
		ResultSchema: schema,
		ModelOptions: map[string]any{"include_thoughts": false},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Error != nil || len(resp.Result) != 1 {
		t.Fatalf("resp = %+v", resp.Completion)
	}
	system := server.Last(t, chatPath).Body["messages"].([]any)[0].(map[string]any)
	if !strings.HasPrefix(system["content"].(string), "Only answer with JSON matching this schema: {") {
		t.Fatalf("schema instruction = %v", system)
	}
	if _, ok := server.Last(t, chatPath).Body["response_format"]; ok {
		t.Fatal("response_format sent")
	}
}

func TestMistralReadsStoredConversations(t *testing.T) {
	ts := map[string]any{"messages": []any{
		map[string]any{"role": "user", "content": "Weather?"},
		map[string]any{"role": "assistant", "content": "", "toolCalls": []any{map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "weather", "arguments": "{}"}}}},
		map[string]any{"role": "tool", "toolCallId": "call_1", "content": "Sunny"},
	}}
	legacy := map[string]any{"_is_openai_chat_completions": true, "messages": []any{
		map[string]any{"role": "developer", "content": "Be brief."},
		map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "Look"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/a.png"}},
		}},
		map[string]any{"role": "tool", "tool_call_id": "call_2", "content": "Done"},
	}}
	prompt := Prompt{Messages: []map[string]any{{"role": "user", "content": "Thanks"}}}

	conversation, err := requestConversation(ts, prompt)
	if err != nil {
		t.Fatal(err)
	}
	if conversation.Messages[1]["tool_calls"] == nil || conversation.Messages[2]["tool_call_id"] != "call_1" || len(conversation.Messages) != 4 {
		t.Fatalf("TS conversation = %+v", conversation.Messages)
	}

	conversation, err = requestConversation(legacy, prompt)
	if err != nil {
		t.Fatal(err)
	}
	if conversation.Messages[0]["role"] != "system" || conversation.Messages[2]["tool_call_id"] != "call_2" {
		t.Fatalf("legacy conversation = %+v", conversation.Messages)
	}
	image := conversation.Messages[1]["content"].([]any)[1].(map[string]any)
	if image["image_url"] != "https://example.com/a.png" {
		t.Fatalf("legacy image = %v", image)
	}

	// The driver's own conversation, after a JSON roundtrip.
	data, _ := json.Marshal(Conversation{Messages: []map[string]any{{"role": "user", "content": "Hi"}}, Turn: 3})
	var stored any
	_ = json.Unmarshal(data, &stored)
	conversation, err = requestConversation(stored, prompt)
	if err != nil || conversation.Turn != 3 || len(conversation.Messages) != 2 {
		t.Fatalf("conversation = %+v, err = %v", conversation, err)
	}
}

func TestMistralConversationRetention(t *testing.T) {
	keep := 1
	conversation := Conversation{Turn: 1, Messages: []map[string]any{
		{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "Look"},
			map[string]any{"type": "image_url", "image_url": "data:image/png;base64,AAAA"},
			map[string]any{"type": "input_audio", "input_audio": "UklGRg=="},
		}},
		{"role": "assistant", "content": []any{
			map[string]any{"type": "thinking", "thinking": []any{map[string]any{"type": "text", "text": "t"}}, "signature": "sig"},
			map[string]any{"type": "image_url", "image_url": "data:image/png;base64,BBBB"},
		}},
	}}
	finalized := finalizeConversation(conversation, common.ExecutionOptions{StripImagesAfterTurns: &keep})
	user := finalized.Messages[0]["content"].([]any)
	if user[1].(map[string]any)["text"] != core.ConversationImagePlaceholder || user[2].(map[string]any)["text"] != core.ConversationAudioPlaceholder {
		t.Fatalf("user = %v", user)
	}
	if signed := finalized.Messages[1]["content"].([]any)[1].(map[string]any); signed["type"] != "image_url" {
		t.Fatalf("signed message rewritten: %v", signed)
	}
	if finalized.Turn != 2 {
		t.Fatalf("turn = %d", finalized.Turn)
	}
}

func TestMistralStream(t *testing.T) {
	body := fakehttp.SSE(
		`{"choices":[{"index":0,"delta":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"Plan"}]}]}}]}`,
		`{"choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":" more"}],"signature":"sig","closed":true}]}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"Calling"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":" now","tool_calls":[{"id":"call_9","index":0,"function":{"name":"weather","arguments":"{\"city\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"","arguments":"\"Rome\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}`,
	)
	driver, server := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: body}})
	stream, err := driver.Stream(context.Background(), []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Weather in Rome?"}}, common.ExecutionOptions{
		Model: "magistral-medium-latest",
		Tools: []common.ToolDefinition{{Name: "weather", InputSchema: map[string]any{"type": "object"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, resp := fakehttp.Drain(t, stream)
	if server.Last(t, chatPath).Body["stream"] != true {
		t.Fatal("stream not requested")
	}
	if len(resp.ToolUse) != 1 || resp.ToolUse[0].ID != "call_9" || (*resp.ToolUse[0].ToolInput)["city"] != "Rome" {
		t.Fatalf("tool use = %+v", resp.ToolUse)
	}
	if resp.TokenUsage == nil || *resp.TokenUsage.Total != 13 {
		t.Fatalf("usage = %+v", resp.TokenUsage)
	}
	conversation := resp.Conversation.(Conversation)
	assistant := conversation.Messages[len(conversation.Messages)-1]
	content := assistant["content"].([]any)
	thinking := content[0].(map[string]any)
	if len(content) != 2 || thinking["signature"] != "sig" || len(thinking["thinking"].([]any)) != 2 || content[1].(map[string]any)["text"] != "Calling now" {
		t.Fatalf("assistant content = %v", content)
	}
	call := assistant["tool_calls"].([]any)[0].(map[string]any)
	if call["id"] != "call_9" || call["function"].(map[string]any)["arguments"] != `{"city":"Rome"}` {
		t.Fatalf("tool call = %v", call)
	}
}

func TestMistralErrors(t *testing.T) {
	driver, _ := newTestDriver(t, map[string]fakehttp.Response{
		chatPath:           {Status: 422, Body: `{"object":"error","message":"Invalid model","type":"invalid_request_error"}`},
		"GET /v1/models":   {Status: 429, Body: `{"message":"Requests rate limit exceeded"}`},
		"/v1/embeddings":   {Status: 401, Body: `{"message":"Unauthorized"}`},
		"POST /v1/unknown": {Status: 500},
	})
	_, err := driver.Execute(context.Background(), []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Hi"}}, common.ExecutionOptions{Model: "nope"})
	var llumiErr *common.LlumiverseError
	if !errors.As(err, &llumiErr) || llumiErr.Code != 422 || llumiErr.Retryable == nil || *llumiErr.Retryable || llumiErr.Name != "MistralError" {
		t.Fatalf("execute err = %+v", err)
	}
	_, err = driver.ListModels(context.Background(), nil)
	if !errors.As(err, &llumiErr) || llumiErr.Code != 429 || llumiErr.Retryable == nil || !*llumiErr.Retryable {
		t.Fatalf("list err = %+v", err)
	}
}

func TestMistralListModelsAndEmbeddings(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{
		"/v1/models": {Body: `{"object":"list","data":[
			{"id":"mistral-large-latest","name":"Mistral Large","description":"Flagship","owned_by":"mistralai","capabilities":{"completion_chat":true}},
			{"id":"mistral-embed","owned_by":"mistralai","capabilities":{"completion_chat":false}},
			{"id":"mistral-ocr-latest","owned_by":"mistralai","capabilities":{"completion_chat":false}},
			{"id":"ft:open-mistral-7b:abc","owned_by":"org"}]}`},
		"/v1/embeddings": {Body: `{"model":"mistral-embed","data":[{"index":1,"embedding":[0.3]},{"index":0,"embedding":[0.1,0.2]}],
			"usage":{"prompt_tokens":6,"total_tokens":6}}`},
	})
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].Name != "Mistral Large" || models[1].ID != "ft:open-mistral-7b:abc" || *models[0].Description != "Flagship" {
		t.Fatalf("models = %+v", models)
	}
	result, err := driver.GenerateEmbeddings(context.Background(), common.EmbeddingsOptions{Inputs: []common.EmbeddingInput{{Text: "a"}, {Text: "b"}}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Model != DefaultEmbeddingModel || len(result.Results[0].Outputs[0].Values) != 2 || *result.Usage.InputTokens != 6 {
		t.Fatalf("result = %+v", result)
	}
	body := server.Last(t, "/v1/embeddings").Body
	if inputs := body["input"].([]any); len(inputs) != 2 || body["encoding_format"] != "float" {
		t.Fatalf("body = %v", body)
	}
	if _, err := driver.GenerateEmbeddings(context.Background(), common.EmbeddingsOptions{Inputs: []common.EmbeddingInput{{Type: "image", Source: &common.BytesDataSource{MIME: "image/png"}}}}); err == nil {
		t.Fatal("image embeddings accepted")
	}
}

func TestMistralFormatDebugPromptTruncatesMedia(t *testing.T) {
	driver, err := NewMistralDriver(MistralOptions{APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	image := "data:image/png;base64," + strings.Repeat("A", 4096)
	prompt := Prompt{Messages: []map[string]any{{"role": "user", "content": []any{
		map[string]any{"type": "image_url", "image_url": image},
		map[string]any{"type": "input_audio", "input_audio": strings.Repeat("B", 4096)},
	}}}}
	debug := driver.FormatDebugPrompt(prompt).(Prompt)
	parts := debug.Messages[0]["content"].([]any)
	if url := parts[0].(map[string]any)["image_url"].(string); len(url) >= len(image) {
		t.Fatalf("image not truncated: %d", len(url))
	}
	if audio := parts[1].(map[string]any)["input_audio"].(string); len(audio) >= 4096 {
		t.Fatalf("audio not truncated: %d", len(audio))
	}
	if prompt.Messages[0]["content"].([]any)[0].(map[string]any)["image_url"] != image {
		t.Fatal("prompt modified")
	}
}
