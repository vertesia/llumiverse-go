package groq

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
	"github.com/vertesia/llumiverse-go/drivers/internal/chatcompat"
	"github.com/vertesia/llumiverse-go/drivers/internal/fakehttp"
)

const chatPath = "/openai/v1/chat/completions"

const chatBody = `{"id":"c1","object":"chat.completion","created":1,"model":"m",
	"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"{\"answer\":\"ok\"}"}}],
	"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":4}}}`

var answerSchema = map[string]any{
	"type":                 "object",
	"properties":           map[string]any{"answer": map[string]any{"type": "string"}},
	"required":             []any{"answer"},
	"additionalProperties": false,
}

func newTestDriver(t *testing.T, routes map[string]fakehttp.Response) (*GroqDriver, *fakehttp.Server) {
	t.Helper()
	server := fakehttp.New(t, routes)
	driver, err := NewGroqDriver(GroqOptions{APIKey: "test-key", Endpoint: server.URL + "/openai/v1/"})
	if err != nil {
		t.Fatal(err)
	}
	return driver, server
}

func userPrompt(text string) []common.PromptSegment {
	return []common.PromptSegment{{Role: common.PromptRoleUser, Content: text}}
}

func TestGroqStructuredOutputsByModel(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: chatBody}})
	tool := common.ToolDefinition{Name: "lookup", InputSchema: map[string]any{"type": "object"}}
	cases := []struct {
		name   string
		model  string
		tools  []common.ToolDefinition
		format bool
		strict bool
	}{
		{name: "gpt-oss", model: "openai/gpt-oss-120b", format: true, strict: true},
		{name: "safeguard is not strict", model: "openai/gpt-oss-safeguard-20b", format: true, strict: false},
		{name: "qwen 3.8", model: "qwen/qwen3.8-32b", format: true, strict: true},
		{name: "qwen 3 size is not a minor version", model: "qwen/qwen3-32b"},
		{name: "llama keeps the prompt only", model: "llama-3.3-70b-versatile"},
		{name: "tools keep the prompt only", model: "openai/gpt-oss-120b", tools: []common.ToolDefinition{tool}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := driver.Execute(context.Background(), userPrompt("Answer."), common.ExecutionOptions{
				Model:        tc.model,
				ResultSchema: answerSchema,
				Tools:        tc.tools,
				ModelOptions: map[string]any{"max_tokens": 50, "reasoning_format": "parsed"},
			})
			if err != nil {
				t.Fatal(err)
			}
			if resp.Error != nil {
				t.Fatalf("validation error: %+v", resp.Error)
			}
			body := server.Last(t, chatPath).Body
			if _, ok := body["max_tokens"]; ok || body["max_completion_tokens"] != float64(50) {
				t.Fatalf("max tokens not sent as max_completion_tokens: %v", body)
			}
			if body["reasoning_format"] != "parsed" {
				t.Fatalf("reasoning_format = %v", body["reasoning_format"])
			}
			messages, _ := body["messages"].([]any)
			if first, _ := messages[0].(map[string]any); !strings.Contains(chatcompat.TextContent(first["content"]), "<response_schema>") {
				t.Fatalf("schema instruction missing: %v", messages)
			}
			format, ok := body["response_format"].(map[string]any)
			if ok != tc.format {
				t.Fatalf("response_format = %v, want present %v", body["response_format"], tc.format)
			}
			if ok {
				schema := format["json_schema"].(map[string]any)
				if schema["strict"] != tc.strict {
					t.Fatalf("strict = %v, want %v", schema["strict"], tc.strict)
				}
			}
		})
	}
}

func TestGroqUsageAndAuth(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: chatBody}})
	resp, err := driver.Execute(context.Background(), userPrompt("Hi"), common.ExecutionOptions{Model: "llama-3.3-70b-versatile"})
	if err != nil {
		t.Fatal(err)
	}
	if got := server.Last(t, chatPath).Header.Get("Authorization"); got != "Bearer test-key" {
		t.Fatalf("authorization = %q", got)
	}
	usage := resp.TokenUsage
	if usage == nil || *usage.Prompt != 10 || usage.PromptCached == nil || *usage.PromptCached != 4 || *usage.Result != 2 {
		t.Fatalf("usage = %+v", usage)
	}
	if resp.Prompt == nil || driver.Provider() != common.ProviderGroq {
		t.Fatal("prompt or provider not set")
	}
}

func TestGroqStreamReadsUsageFromXGroq(t *testing.T) {
	body := fakehttp.SSE(
		`{"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
		`{"id":"c1","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}],"x_groq":{"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}}`,
	)
	driver, server := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: body}})
	stream, err := driver.Stream(context.Background(), userPrompt("Hi"), common.ExecutionOptions{Model: "llama-3.3-70b-versatile"})
	if err != nil {
		t.Fatal(err)
	}
	_, resp := fakehttp.Drain(t, stream)
	if text := core.CompletionResultsToText(resp.Result); text != "Hello" {
		t.Fatalf("text = %q", text)
	}
	if resp.TokenUsage == nil || *resp.TokenUsage.Prompt != 5 || *resp.TokenUsage.Total != 7 {
		t.Fatalf("usage = %+v", resp.TokenUsage)
	}
	request := server.Last(t, chatPath).Body
	if options, _ := request["stream_options"].(map[string]any); options["include_usage"] != true {
		t.Fatalf("stream_options = %v", request["stream_options"])
	}
}

func TestGroqRefusesAudioInput(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{chatPath: {Body: chatBody}})
	segments := []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Transcribe", Files: []common.DataSource{
		&common.BytesDataSource{FileName: "a.wav", MIME: "audio/wav", Data: []byte("RIFF")},
	}}}
	_, err := driver.Execute(context.Background(), segments, common.ExecutionOptions{Model: "llama-3.3-70b-versatile"})
	if !errors.Is(err, chatcompat.ErrAudioInput) {
		t.Fatalf("err = %v, want audio refusal", err)
	}
	if len(server.Requests()) != 0 {
		t.Fatal("request sent")
	}
}

func TestGroqListModelsLeavesOutSpeechAndEmbeddings(t *testing.T) {
	driver, _ := newTestDriver(t, map[string]fakehttp.Response{"/openai/v1/models": {Body: `{"data":[
		{"id":"whisper-large-v3","owned_by":"OpenAI"},
		{"id":"llama-3.3-70b-versatile","owned_by":"Meta"},
		{"id":"canopylabs/orpheus-v1-english","owned_by":"Canopy Labs"},
		{"id":"nomic-embed-text","owned_by":"Nomic"},
		{"id":"distil-whisper-large-v3-en","owned_by":"Hugging Face"},
		{"id":"openai/gpt-oss-120b","owned_by":"OpenAI"}]}`}})
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	if strings.Join(ids, ",") != "llama-3.3-70b-versatile,openai/gpt-oss-120b" {
		t.Fatalf("ids = %v", ids)
	}
	if *models[0].Owner != "Meta" || models[0].Provider != "groq" {
		t.Fatalf("model = %+v", models[0])
	}
	if err := driver.ValidateConnection(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestGroqConstructorAndEmbeddings(t *testing.T) {
	if _, err := NewGroqDriver(GroqOptions{}); err == nil {
		t.Fatal("missing api key accepted")
	}
	driver, err := NewGroqDriver(GroqOptions{APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := driver.GenerateEmbeddings(context.Background(), common.EmbeddingsOptions{Inputs: []common.EmbeddingInput{{Text: "x"}}}); err == nil {
		t.Fatal("embeddings accepted")
	}
}
