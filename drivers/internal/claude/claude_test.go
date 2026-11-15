package claude

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vertesia/llumiverse-go/common"
)

func TestAnthropicExecuteFormatsMessages(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "anthropic-key" {
			t.Fatalf("missing anthropic key: %q", r.Header.Get("x-api-key"))
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{
			"id":"msg_1",
			"role":"assistant",
			"content":[
				{"type":"text","text":"hello"},
				{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"kyoto"}}
			],
			"stop_reason":"tool_use",
			"usage":{"input_tokens":5,"output_tokens":6}
		}`))
	}))
	defer server.Close()

	driver, err := NewAnthropicDriver(AnthropicOptions{APIKey: "anthropic-key", BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "You are helpful."},
		{Role: PromptRoleUser, Content: "Hi"},
	}, ExecutionOptions{
		Model:        "claude-test",
		ModelOptions: map[string]any{"max_tokens": 256, "temperature": 0.1},
		Tools: []ToolDefinition{{
			Name:        "lookup",
			Description: ptr("Lookup"),
			InputSchema: map[string]any{"type": "object"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}

	if got["model"] != "claude-test" {
		t.Fatalf("model = %v", got["model"])
	}
	if got["max_tokens"] != float64(256) {
		t.Fatalf("max_tokens = %v", got["max_tokens"])
	}
	if len(resp.ToolUse) != 1 || resp.ToolUse[0].ID != "toolu_1" {
		t.Fatalf("tool use = %#v", resp.ToolUse)
	}
	if resp.TokenUsage == nil || resp.TokenUsage.Total == nil || *resp.TokenUsage.Total != 11 {
		t.Fatalf("usage = %#v", resp.TokenUsage)
	}
}

func TestAnthropicExecuteSupportsConversationDocumentsThinkingAndCache(t *testing.T) {
	t.Parallel()

	var got map[string]any
	var betaHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		betaHeader = r.Header.Get("anthropic-beta")
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_2",
			"role":"assistant",
			"content":[
				{"type":"thinking","thinking":"thinking text","signature":"sig"},
				{"type":"text","text":"final text"}
			],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":7,"output_tokens":8,"cache_read_input_tokens":2,"cache_creation_input_tokens":3}
		}`))
	}))
	defer server.Close()

	driver, err := NewAnthropicDriver(AnthropicOptions{APIKey: "anthropic-key", BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []PromptSegment{{
		Role:    PromptRoleUser,
		Content: "Hi",
		Files: []DataSource{BytesDataSource{
			FileName: "notes.txt",
			MIME:     "text/plain",
			Data:     []byte("file body"),
		}},
	}}, ExecutionOptions{
		Model: "claude-3-7-sonnet-20250219",
		Conversation: claudePrompt{Messages: []claudeMessage{{
			Role:    "user",
			Content: []claudeBlock{{Type: "text", Text: "Earlier"}},
		}}},
		ModelOptions: map[string]any{
			"max_tokens":             65000,
			"thinking_budget_tokens": 1024,
			"include_thoughts":       true,
			"cache_enabled":          true,
			"cache_ttl":              "5m",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if betaHeader != "output-128k-2025-02-19" {
		t.Fatalf("anthropic-beta = %q", betaHeader)
	}
	if got["temperature"] != nil || got["top_p"] != nil || got["top_k"] != nil {
		t.Fatalf("sampling options should be omitted while thinking is active: %#v", got)
	}
	thinking, ok := got["thinking"].(map[string]any)
	if !ok || thinking["type"] != "enabled" || thinking["budget_tokens"] != float64(1024) {
		t.Fatalf("thinking = %#v", got["thinking"])
	}
	messages, ok := got["messages"].([]any)
	if !ok || len(messages) != 1 {
		t.Fatalf("messages = %#v", got["messages"])
	}
	content := messages[0].(map[string]any)["content"].([]any)
	var sawDocument bool
	for _, raw := range content {
		block := raw.(map[string]any)
		if block["type"] == "document" {
			sawDocument = true
			source := block["source"].(map[string]any)
			if source["type"] != "text" || source["data"] != "file body" {
				t.Fatalf("document source = %#v", source)
			}
		}
	}
	if !sawDocument {
		t.Fatalf("document block missing: %#v", content)
	}
	if len(resp.Result) != 2 || resp.Result[0].Type() != common.ResultTypeThoughts || resp.Result[0].String() != "thinking text" || resp.Result[1].String() != "final text" {
		t.Fatalf("result = %#v", resp.Result)
	}
	if resp.TokenUsage == nil || resp.TokenUsage.Prompt == nil || *resp.TokenUsage.Prompt != 12 || resp.TokenUsage.PromptCached == nil || *resp.TokenUsage.PromptCached != 2 || resp.TokenUsage.PromptCacheWrite == nil || *resp.TokenUsage.PromptCacheWrite != 3 {
		t.Fatalf("usage = %#v", resp.TokenUsage)
	}
	conversation, ok := resp.Conversation.(claudePrompt)
	if !ok || len(conversation.Messages) != 2 || conversation.Messages[1].Role != "assistant" {
		t.Fatalf("conversation = %#v", resp.Conversation)
	}
}

func TestAnthropicHTTPTimeoutConfiguration(t *testing.T) {
	t.Parallel()

	driver, err := NewAnthropicDriver(AnthropicOptions{
		APIKey: "anthropic-key",
		DriverOptions: DriverOptions{HTTPTimeout: HTTPTimeoutOptions{
			BodyTimeout:    3 * time.Second,
			HeadersTimeout: 2 * time.Second,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if driver.client.Timeout != 3*time.Second {
		t.Fatalf("constructor timeout = %s", driver.client.Timeout)
	}
	override := driver.httpClient(HTTPTimeoutOptions{BodyTimeout: time.Second})
	if override == driver.client || override.Timeout != time.Second {
		t.Fatalf("override client = %#v", override)
	}
	if got := driver.httpClient(HTTPTimeoutOptions{}); got != driver.client {
		t.Fatal("empty per-request timeout should reuse the driver client")
	}
}

func TestAnthropicModelsValidationUnsupportedEmbeddingsAndErrors(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_, _ = w.Write([]byte(`{"data":[{"id":"claude-3-5-sonnet","display_name":"Claude 3.5 Sonnet"},{"id":"claude-3-haiku"},{"id":"claude-3-5-haiku-20241022"}]}`))
		case "/v1/messages":
			http.Error(w, "overloaded", http.StatusServiceUnavailable)
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer server.Close()

	driver, err := NewAnthropicDriver(AnthropicOptions{APIKey: "anthropic-key", BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	if driver.Provider() != ProviderAnthropic {
		t.Fatalf("provider = %s", driver.Provider())
	}
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 3 || models[0].ID != "claude-3-5-sonnet" || models[0].Name != "Claude 3.5 Sonnet" || models[1].Name != "claude-3-haiku" {
		t.Fatalf("models = %#v", models)
	}
	// The capability catalogue describes each model: Claude 3.5 Haiku takes no images.
	sonnet, haiku := models[0], models[2]
	if !*sonnet.ToolSupport || !*sonnet.IsMultimodal || strings.Join(*sonnet.InputModalities, ",") != "text,image" ||
		strings.Join(*sonnet.OutputModalities, ",") != "text" {
		t.Fatalf("sonnet = %#v", sonnet)
	}
	if !*haiku.ToolSupport || *haiku.IsMultimodal || strings.Join(*haiku.InputModalities, ",") != "text" {
		t.Fatalf("haiku = %#v", haiku)
	}
	if err := driver.ValidateConnection(context.Background()); err != nil {
		t.Fatal(err)
	}
	_, err = driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{Model: "claude-3-5-sonnet"})
	var llumiErr *common.LlumiverseError
	if !errors.As(err, &llumiErr) || llumiErr.Name != "UnsupportedOperation" || llumiErr.Retryable == nil || *llumiErr.Retryable {
		t.Fatalf("embeddings error = %#v", err)
	}
	_, err = driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{Model: "claude-test"})
	if !errors.As(err, &llumiErr) || llumiErr.Code != http.StatusServiceUnavailable || llumiErr.Context.Operation != "execute" {
		t.Fatalf("execute error = %#v", err)
	}
}

func TestAnthropicStreamParsesSSEAndFinalizesConversation(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w, `{"type":"message_start","message":{"role":"assistant","content":[],"usage":{"input_tokens":2,"cache_read_input_tokens":1}}}`)
		writeSSE(t, w, `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`)
		writeSSE(t, w, `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"thinking"}}`)
		writeSSE(t, w, `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`)
		writeSSE(t, w, `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`)
		writeSSE(t, w, `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hello "}}`)
		writeSSE(t, w, `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"world"}}`)
		writeSSE(t, w, `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tool_1","name":"lookup","input":{}}}`)
		writeSSE(t, w, `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\""}}`)
		writeSSE(t, w, `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":":\"tokyo\"}"}}`)
		writeSSE(t, w, `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}`)
		writeSSE(t, w, `{"type":"message_stop"}`)
	}))
	defer server.Close()

	driver, err := NewAnthropicDriver(AnthropicOptions{APIKey: "anthropic-key", BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{
		Model:        "claude-test",
		ModelOptions: map[string]any{"include_thoughts": true},
		Tools: []ToolDefinition{{
			Name:        "lookup",
			Description: ptr("Lookup"),
			InputSchema: map[string]any{"type": "object"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	completion := stream.Completion()
	if got["stream"] != true {
		t.Fatalf("payload = %#v", got)
	}
	if completion.TokenUsage == nil || completion.TokenUsage.Prompt == nil || *completion.TokenUsage.Prompt != 3 || completion.TokenUsage.Result == nil || *completion.TokenUsage.Result != 3 {
		t.Fatalf("usage = %#v", completion.TokenUsage)
	}
	if len(completion.Result) != 2 || completion.Result[0].String() != "thinking" || completion.Result[1].String() != "hello world" {
		t.Fatalf("result = %#v", completion.Result)
	}
	if len(completion.ToolUse) != 1 || completion.ToolUse[0].ToolName != "lookup" {
		t.Fatalf("tool use = %#v", completion.ToolUse)
	}
	input := completion.ToolUse[0].ToolInput
	if input == nil || (*input)["q"] != "tokyo" {
		t.Fatalf("tool input = %#v", completion.ToolUse[0].ToolInput)
	}
	conversation, ok := completion.Conversation.(claudePrompt)
	if !ok || len(conversation.Messages) != 2 || conversation.Messages[1].Role != "assistant" {
		t.Fatalf("conversation = %#v", completion.Conversation)
	}
}

func writeSSE(t *testing.T, w http.ResponseWriter, data string) {
	t.Helper()
	_, err := w.Write([]byte("data: " + data + "\n\n"))
	if err != nil {
		t.Fatal(err)
	}
}
