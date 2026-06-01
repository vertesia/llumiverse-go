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
	"github.com/vertesia/llumiverse-go/core"
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
			Description: "Lookup",
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
	if resp.TokenUsage == nil || resp.TokenUsage.Total != 11 {
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
				{"type":"thinking","thinking":"thinking text"},
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
	if !strings.Contains(resp.Result[0].Value.(string), "thinking text") || !strings.Contains(resp.Result[0].Value.(string), "final text") {
		t.Fatalf("result = %#v", resp.Result)
	}
	if resp.TokenUsage == nil || resp.TokenUsage.Prompt != 12 || resp.TokenUsage.PromptCached != 2 || resp.TokenUsage.PromptCacheWrite != 3 {
		t.Fatalf("usage = %#v", resp.TokenUsage)
	}
	conversation, ok := resp.Conversation.(claudePrompt)
	if !ok || len(conversation.Messages) != 2 || conversation.Messages[1].Role != "assistant" {
		t.Fatalf("conversation = %#v", resp.Conversation)
	}
}

func TestClaudeNewVersionNamesUseAdaptiveThinking(t *testing.T) {
	t.Parallel()

	payload, _, err := claudePayload(claudePrompt{Messages: []claudeMessage{{
		Role:    "user",
		Content: []claudeBlock{{Type: "text", Text: "Hi"}},
	}}}, ExecutionOptions{
		Model: "claude-sonnet-4-6-20260601",
		ModelOptions: map[string]any{
			"effort":           "high",
			"include_thoughts": true,
			"temperature":      0.4,
			"top_p":            0.8,
			"top_k":            40,
		},
	}, false, map[string]string{})
	if err != nil {
		t.Fatal(err)
	}
	thinking, ok := payload["thinking"].(map[string]any)
	if !ok || thinking["type"] != "adaptive" || thinking["display"] != "summarized" {
		t.Fatalf("thinking = %#v", payload["thinking"])
	}
	outputConfig, ok := payload["output_config"].(map[string]any)
	if !ok || outputConfig["effort"] != "high" {
		t.Fatalf("output_config = %#v", payload["output_config"])
	}
	if payload["temperature"] != nil || payload["top_p"] != nil || payload["top_k"] != nil {
		t.Fatalf("sampling options should be omitted while adaptive thinking is active: %#v", payload)
	}
	if !isClaudeVersionGTE("claude-opus-4-7-20260601", 4, 7) {
		t.Fatal("new Claude version name was not parsed")
	}
}

func TestClaudePromptFormattingFileCacheAndRetentionBranches(t *testing.T) {
	t.Parallel()

	prompt, err := formatClaudePrompt(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "System"},
		{Role: PromptRoleAssistant, Content: "Previous answer"},
		{
			Role:    PromptRoleUser,
			Content: "Use these",
			Files: []DataSource{
				BytesDataSource{FileName: "brief.pdf", MIME: "application/pdf", Data: []byte("%PDF")},
				BytesDataSource{FileName: "notes.txt", MIME: "text/plain", Data: []byte("notes")},
				BytesDataSource{FileName: "image.webp", MIME: "image/webp", Data: []byte("webp")},
			},
		},
		{
			Role:      PromptRoleTool,
			ToolUseID: "tool_1",
			Content:   "tool output",
			Files: []DataSource{
				BytesDataSource{FileName: "tool.png", MIME: "image/png", Data: []byte("png")},
				BytesDataSource{FileName: "tool.txt", MIME: "text/plain", Data: []byte("skipped")},
			},
		},
		{Role: PromptRoleSafety, Content: "Safety"},
		{Role: PromptRoleNegative, Content: "ignored"},
		{Role: PromptRoleMask, Content: "ignored"},
	}, ExecutionOptions{
		ResultSchema: map[string]any{"type": "object"},
		Tools:        []ToolDefinition{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(prompt.System) != 2 || !strings.Contains(prompt.System[1].Text, "When not calling tools") {
		t.Fatalf("system = %#v", prompt.System)
	}
	if len(prompt.Messages) != 4 || prompt.Messages[0].Role != "assistant" || prompt.Messages[3].Content[0].Text != "Safety" {
		t.Fatalf("messages = %#v", prompt.Messages)
	}
	userBlocks := prompt.Messages[1].Content
	var sawPDF, sawTextDoc, sawImage bool
	for _, block := range userBlocks {
		if block.Type == "document" && block.Source != nil && block.Source.Type == "base64" {
			sawPDF = true
		}
		if block.Type == "document" && block.Source != nil && block.Source.Type == "text" && block.Source.Data == "notes" {
			sawTextDoc = true
		}
		if block.Type == "image" && block.Source != nil && block.Source.MediaType == "image/webp" {
			sawImage = true
		}
	}
	if !sawPDF || !sawTextDoc || !sawImage {
		t.Fatalf("user file blocks = %#v", userBlocks)
	}
	toolContent := prompt.Messages[2].Content[0].Content
	if len(toolContent) != 2 || toolContent[0].Type != "text" || toolContent[1].Type != "image" {
		t.Fatalf("restricted tool content = %#v", toolContent)
	}
	if _, err := formatClaudePrompt(context.Background(), []PromptSegment{{Role: PromptRoleTool, Content: "missing id"}}, ExecutionOptions{}); err == nil {
		t.Fatal("expected missing ToolUseID error")
	}
	if !isClaudeImageMIME("image/jpeg") || isClaudeImageMIME("image/svg+xml") {
		t.Fatal("Claude image MIME helper returned unexpected result")
	}

	cache := claudeCacheControl(ExecutionOptions{ModelOptions: map[string]any{"cache_ttl": "1h"}})
	if cache["type"] != "ephemeral" || cache["ttl"] != "1h" {
		t.Fatalf("cache control = %#v", cache)
	}
	if _, ok := claudeCacheControl(ExecutionOptions{})["ttl"]; ok {
		t.Fatal("cache control should omit empty ttl")
	}
	if claudeNeedsOutput128KBeta(ExecutionOptions{Model: "claude-3-5-sonnet"}, 65000) {
		t.Fatal("non-3.7 model should not need output beta")
	}
	if claudeNeedsOutput128KBeta(ExecutionOptions{Model: "claude-3-7-sonnet-20250219"}, 64000) {
		t.Fatal("64k max tokens should not need output beta")
	}
	if !claudeNeedsOutput128KBeta(ExecutionOptions{
		Model:        "claude-3-7-sonnet-20250219",
		ModelOptions: map[string]any{"thinking_budget_tokens": 64001},
	}, 1000) {
		t.Fatal("large thinking budget should need output beta")
	}

	processed := processClaudeBlocksForConversation([]claudeBlock{
		{Type: "image"},
		{Type: "document"},
		{Type: "text", Text: "abcdefghijklmnop"},
		{Type: "tool_result", Content: []claudeBlock{
			{Type: "image"},
			{Type: "text", Text: "<heartbeat>working</heartbeat>"},
		}},
	}, ExecutionOptions{StripTextMaxTokens: 2}, true, true)
	if processed[0].Text != core.ConversationImagePlaceholder || processed[1].Text != core.ConversationDocumentPlaceholder {
		t.Fatalf("media placeholders = %#v", processed)
	}
	if processed[2].Text != "abcdefgh"+core.ConversationTextTruncatedMarker {
		t.Fatalf("truncated text = %#v", processed[2])
	}
	nested := processed[3].Content
	if nested[0].Text != core.ConversationImagePlaceholder || nested[1].Text != core.ConversationHeartbeatPlaceholder {
		t.Fatalf("nested processed blocks = %#v", nested)
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
			_, _ = w.Write([]byte(`{"data":[{"id":"claude-3-5-sonnet","display_name":"Claude 3.5 Sonnet"},{"id":"claude-3-haiku"}]}`))
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
	if len(models) != 2 || models[0].ID != "claude-3-5-sonnet" || models[0].Name != "Claude 3.5 Sonnet" || models[1].Name != "claude-3-haiku" {
		t.Fatalf("models = %#v", models)
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
		writeSSE(t, w, `{"type":"message_start","message":{"usage":{"input_tokens":2,"cache_read_input_tokens":1}}}`)
		writeSSE(t, w, `{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"thinking"}}`)
		writeSSE(t, w, `{"type":"content_block_delta","delta":{"type":"signature_delta"}}`)
		writeSSE(t, w, `{"type":"content_block_delta","delta":{"type":"text_delta","text":"hello "}}`)
		writeSSE(t, w, `{"type":"content_block_delta","delta":{"type":"text_delta","text":"world"}}`)
		writeSSE(t, w, `{"type":"content_block_start","content_block":{"type":"tool_use","id":"tool_1","name":"lookup","input":{}}}`)
		writeSSE(t, w, `{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"q\""}}`)
		writeSSE(t, w, `{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":":\"tokyo\"}"}}`)
		writeSSE(t, w, `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}`)
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
			Description: "Lookup",
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
	if completion.TokenUsage == nil || completion.TokenUsage.Prompt != 3 || completion.TokenUsage.Result != 3 {
		t.Fatalf("usage = %#v", completion.TokenUsage)
	}
	if len(completion.Result) != 1 || !strings.Contains(completion.Result[0].Value.(string), "thinking") || !strings.Contains(completion.Result[0].Value.(string), "hello world") {
		t.Fatalf("result = %#v", completion.Result)
	}
	if len(completion.ToolUse) != 1 || completion.ToolUse[0].ToolName != "lookup" {
		t.Fatalf("tool use = %#v", completion.ToolUse)
	}
	input, ok := completion.ToolUse[0].ToolInput.(map[string]any)
	if !ok || input["q"] != "tokyo" {
		t.Fatalf("tool input = %#v", completion.ToolUse[0].ToolInput)
	}
	conversation, ok := completion.Conversation.(claudePrompt)
	if !ok || len(conversation.Messages) != 2 || conversation.Messages[1].Role != "assistant" {
		t.Fatalf("conversation = %#v", completion.Conversation)
	}
}

func TestClaudeConversationRepairAndToolTextHelpers(t *testing.T) {
	t.Parallel()

	messages := []claudeMessage{
		{
			Role: "assistant",
			Content: []claudeBlock{{
				Type:  "tool_use",
				ID:    "tool_1",
				Name:  "lookup",
				Input: map[string]any{"q": "tokyo"},
			}},
		},
		{Role: "user", Content: []claudeBlock{{Type: "text", Text: "next request"}}},
	}
	repaired := fixClaudeOrphanedToolUse(messages)
	if repaired[1].Content[0].Type != "tool_result" || repaired[1].Content[0].ToolUseID != "tool_1" {
		t.Fatalf("repaired = %#v", repaired)
	}
	if !claudeMessagesContainToolBlocks(repaired) {
		t.Fatal("expected tool blocks")
	}
	asText := convertClaudeToolBlocksToText(repaired)
	if !strings.Contains(asText[0].Content[0].Text, "[Tool call: lookup") || !strings.Contains(asText[1].Content[0].Text, "[Tool result:") {
		t.Fatalf("as text = %#v", asText)
	}
	if got := toolInputString(nil); got != "" {
		t.Fatalf("toolInputString nil = %q", got)
	}
	if got := toolInputString("raw"); got != "raw" {
		t.Fatalf("toolInputString string = %q", got)
	}
	if got := claudeToolResultText(claudeBlock{}); got != "No content" {
		t.Fatalf("empty tool result = %q", got)
	}
	if got := claudeToolResultText(claudeBlock{Content: []claudeBlock{{Type: "text", Text: "one"}, {Type: "text", Text: "two"}}}); got != "one\ntwo" {
		t.Fatalf("nested tool result = %q", got)
	}
}

func TestClaudeExportedHelpersAndSSEBranches(t *testing.T) {
	t.Parallel()

	prompt, err := FormatPrompt(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	payload, headers, err := Payload(prompt, ExecutionOptions{
		Model: "claude-3-7-sonnet-20250219",
		ModelOptions: map[string]any{
			"max_tokens": 65001,
			"cache_ttl":  "1h",
		},
	}, true, map[string]string{})
	if err != nil {
		t.Fatal(err)
	}
	if payload["stream"] != true || headers["anthropic-beta"] != "output-128k-2025-02-19" {
		t.Fatalf("payload = %#v headers = %#v", payload, headers)
	}
	withConversation := ConversationInput(&Prompt{Messages: []Message{{Role: "user", Content: []Block{{Type: "text", Text: "old"}}}}}, prompt)
	if len(withConversation.Messages) != 1 || len(withConversation.Messages[0].Content) != 2 {
		t.Fatalf("conversation input = %#v", withConversation)
	}
	response := Response{
		Role:       "assistant",
		StopReason: "max_tokens",
		Content: []claudeResponseBlock{
			{Type: "thinking", Thinking: "hidden"},
			{Type: "redacted_thinking", Data: "sealed"},
			{Type: "text", Text: "done"},
			{Type: "tool_use", ID: "tool_1", Name: "lookup", Input: json.RawMessage(`{"q":"tokyo"}`)},
		},
	}
	completion := ExtractCompletion(response, ExecutionOptions{ModelOptions: map[string]any{"include_thoughts": true}, IncludeOriginalResponse: true})
	if completion.FinishReason != "tool_use" || completion.OriginalResponse == nil || len(completion.ToolUse) != 1 {
		t.Fatalf("completion = %#v", completion)
	}
	appended := AppendResponseToConversation(prompt, response)
	streaming := BuildStreamingConversation(appended, completion)
	if len(streaming.Messages) == 0 {
		t.Fatalf("streaming = %#v", streaming)
	}
	finalized := FinalizeConversation(streaming, ExecutionOptions{})
	if len(finalized.Messages) == 0 {
		t.Fatalf("finalized = %#v", finalized)
	}

	messageStart := SSEToChunk(rawEnvelope(`{"type":"message_start","message":{"usage":{"input_tokens":1}}}`), ExecutionOptions{})
	if messageStart.TokenUsage == nil || messageStart.TokenUsage.Prompt != 1 {
		t.Fatalf("message_start = %#v", messageStart)
	}
	redactedStart := SSEToChunk(rawEnvelope(`{"type":"content_block_start","content_block":{"type":"redacted_thinking","data":"secret"}}`), ExecutionOptions{ModelOptions: map[string]any{"include_thoughts": true}})
	if len(redactedStart.Result) != 1 || !strings.Contains(redactedStart.Result[0].Value.(string), "secret") {
		t.Fatalf("redacted start = %#v", redactedStart)
	}
	messageDelta := SSEToChunk(rawEnvelope(`{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":4}}`), ExecutionOptions{})
	if messageDelta.FinishReason != "length" || messageDelta.TokenUsage.Result != 4 {
		t.Fatalf("message_delta = %#v", messageDelta)
	}
	if chunk := SSEToChunk(rawEnvelope(`{"type":"content_block_delta","delta":{"type":"signature_delta"}}`), ExecutionOptions{}); len(chunk.Result) != 0 {
		t.Fatalf("signature without thoughts = %#v", chunk)
	}
	if claudeUsage(0, 0, 0, 0) != nil {
		t.Fatal("zero usage should be nil")
	}
	if got := claudeFinishReason("end_turn", false); got != "stop" {
		t.Fatalf("finish reason = %q", got)
	}
}

func TestClaudeConversationInputUsesJSONRestoredConversation(t *testing.T) {
	t.Parallel()

	stored := Prompt{
		System: []Block{{Type: "text", Text: "stored system"}},
		Messages: []Message{
			{Role: "user", Content: []Block{{Type: "text", Text: "old question"}}},
			{Role: "assistant", Content: []Block{{Type: "text", Text: "old answer"}}},
		},
	}
	data, err := json.Marshal(stored)
	if err != nil {
		t.Fatal(err)
	}
	var restored any
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatal(err)
	}
	prompt, err := FormatPrompt(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "new question"}}, ExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}

	withConversation := ConversationInput(restored, prompt)
	if len(withConversation.System) != 1 || withConversation.System[0].Text != "stored system" {
		t.Fatalf("system = %#v", withConversation.System)
	}
	if len(withConversation.Messages) != 3 {
		t.Fatalf("messages = %#v", withConversation.Messages)
	}
	if withConversation.Messages[0].Content[0].Text != "old question" ||
		withConversation.Messages[1].Content[0].Text != "old answer" ||
		withConversation.Messages[2].Content[0].Text != "new question" {
		t.Fatalf("messages = %#v", withConversation.Messages)
	}
}

func writeSSE(t *testing.T, w http.ResponseWriter, data string) {
	t.Helper()
	_, err := w.Write([]byte("data: " + data + "\n\n"))
	if err != nil {
		t.Fatal(err)
	}
}

func rawEnvelope(data string) map[string]json.RawMessage {
	var envelope map[string]json.RawMessage
	_ = json.Unmarshal([]byte(data), &envelope)
	return envelope
}
