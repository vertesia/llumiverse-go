package openai

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

	oa "github.com/openai/openai-go"
	"github.com/vertesia/llumiverse-go/common"
)

func TestOpenAIEmbeddingsUseSDKAndSortResults(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"object":"list",
			"data":[
				{"object":"embedding","index":1,"embedding":[0.3,0.4]},
				{"object":"embedding","index":0,"embedding":[0.1,0.2]}
			],
			"model":"text-embedding-3-small",
			"usage":{"prompt_tokens":5,"total_tokens":5}
		}`))
	}))
	defer server.Close()

	driver, err := NewOpenAICompatibleDriver(OpenAICompatibleOptions{APIKey: "test-key", Endpoint: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{
		Dimensions: 128,
		Inputs: []EmbeddingInput{
			{Text: "first"},
			{Type: embeddingInputText, Text: "second"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got["model"] != defaultOpenAIEmbeddingModel || got["encoding_format"] != "float" || got["dimensions"] != float64(128) {
		t.Fatalf("request = %#v", got)
	}
	inputs := got["input"].([]any)
	if len(inputs) != 2 || inputs[0] != "first" || inputs[1] != "second" {
		t.Fatalf("input = %#v", inputs)
	}
	if len(result.Results) != 2 || result.Results[0].Outputs[0].Values[0] != 0.1 || result.Results[1].Outputs[0].Values[0] != 0.3 {
		t.Fatalf("result = %#v", result)
	}
	if result.Usage == nil || count(result.Usage.InputTokens) != 5 || count(result.Usage.InputTextTokens) != 5 {
		t.Fatalf("usage = %#v", result.Usage)
	}

	_, err = driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{Inputs: []EmbeddingInput{{Type: "image", Source: BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte("img")}}}})
	if err == nil {
		t.Fatal("expected image input error")
	}
}

func TestOpenAIHTTPTimeoutConfiguration(t *testing.T) {
	t.Parallel()

	driver, err := NewOpenAICompatibleDriver(OpenAICompatibleOptions{
		APIKey:   "test-key",
		Endpoint: "https://example.test/v1",
		DriverOptions: DriverOptions{HTTPTimeout: HTTPTimeoutOptions{
			BodyTimeout:      3 * time.Second,
			HeadersTimeout:   2 * time.Second,
			ConnectTimeout:   time.Second,
			KeepAliveTimeout: 4 * time.Second,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if driver.client.Timeout != 3*time.Second {
		t.Fatalf("constructor timeout = %s", driver.client.Timeout)
	}
	if len(driver.requestOptions(HTTPTimeoutOptions{BodyTimeout: time.Second})) != 1 {
		t.Fatal("per-request timeout option was not generated")
	}
	if opts := driver.requestOptions(HTTPTimeoutOptions{}); opts != nil {
		t.Fatalf("empty timeout options = %#v", opts)
	}
}

func TestOpenAICompatiblePromptContentAndValidationBranches(t *testing.T) {
	t.Parallel()

	driver := &OpenAICompatibleDriver{options: OpenAICompatibleOptions{Provider: ProviderOpenAICompatible}}
	promptAny, err := driver.CreatePrompt(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "System"},
		{Role: PromptRoleSafety, Content: "Safety"},
		{Role: PromptRoleAssistant, Content: "Assistant"},
		{Role: PromptRoleUser, Content: "User", Files: []DataSource{BytesDataSource{
			FileName: "image.png",
			MIME:     "image/png",
			Data:     []byte("img"),
		}}},
		{Role: PromptRoleTool, ToolUseID: "call_1", Content: "Tool output"},
		{Role: PromptRoleNegative, Content: "ignored"},
		{Role: PromptRoleMask, Content: "ignored"},
	}, ExecutionOptions{ModelOptions: map[string]any{"image_detail": "low"}})
	if err != nil {
		t.Fatal(err)
	}
	prompt := promptAny.(openAIResponsesPrompt)
	texts := openAIResponseInputTexts(prompt.Items)
	for _, want := range []string{"System", "DO NOT IGNORE - IMPORTANT: Safety", "Assistant", "User", "Tool output"} {
		if !containsText(texts, want) {
			t.Fatalf("missing %q in %#v", want, texts)
		}
	}
	raw, err := json.Marshal(prompt.Items)
	if err != nil {
		t.Fatal(err)
	}
	serialized := string(raw)
	if !strings.Contains(serialized, "data:image/png;base64,aW1n") || !strings.Contains(serialized, "low") {
		t.Fatalf("serialized prompt = %s", serialized)
	}
	if _, err := driver.CreatePrompt(context.Background(), []PromptSegment{{Role: PromptRoleTool, Content: "missing id"}}, ExecutionOptions{}); err == nil {
		t.Fatal("expected missing ToolUseID error")
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"gpt-test"}]}`))
	}))
	defer server.Close()
	validating, err := NewOpenAICompatibleDriver(OpenAICompatibleOptions{APIKey: "test-key", Endpoint: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	if err := validating.ValidateConnection(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestOpenAIResponsesStreamingRequestPath(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		writeOpenAISSE(t, w, `{"type":"response.output_text.delta","delta":"hello "}`)
		writeOpenAISSE(t, w, `{"type":"response.output_text.delta","delta":"world"}`)
		writeOpenAISSE(t, w, `{"type":"response.completed","response":{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-4.1","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"hello world","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}`)
	}))
	defer server.Close()

	driver, err := NewOpenAICompatibleDriver(OpenAICompatibleOptions{APIKey: "test-key", Endpoint: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{Model: "gpt-4.1"})
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
	if got["model"] != "gpt-4.1" || got["stream"] != true {
		t.Fatalf("payload = %#v", got)
	}
	completion := stream.Completion()
	if len(completion.Result) != 1 || completion.Result[0].Type() != ResultTypeText || completion.Result[0].String() != "hello world" {
		t.Fatalf("stream completion = %#v", completion.Result)
	}
	if completion.TokenUsage == nil || count(completion.TokenUsage.Total) != 5 || completion.Conversation == nil {
		t.Fatalf("stream completion = %#v", completion)
	}
}

func TestOpenAIHelperBranches(t *testing.T) {
	t.Parallel()

	if openAIUsage(0, 0, 0, 0) != nil {
		t.Fatal("zero usage should be nil")
	}
	usage := openAIUsage(7, 3, 10, 2)
	if count(usage.PromptNew) != 5 || count(usage.PromptCached) != 2 {
		t.Fatalf("usage = %#v", usage)
	}
	if got := safeJSONParse(""); got != nil {
		t.Fatalf("empty json parse = %#v", got)
	}
	if got := safeJSONParse(`{"ok":true}`).(map[string]any); got["ok"] != true {
		t.Fatalf("json parse = %#v", got)
	}
	if got := safeJSONParse("plain"); got != "plain" {
		t.Fatalf("plain parse = %#v", got)
	}
	if !isLikelyNonToolOpenAIModel("text-embedding-3-small") || isLikelyNonToolOpenAIModel("gpt-4.1") {
		t.Fatal("non-tool model heuristic returned unexpected result")
	}
	if openAIRetryable(nil) != nil {
		t.Fatal("nil OpenAI error should have unknown retryability")
	}
	assertRetryable(t, openAIRetryable(&oa.Error{Code: "timeout"}), true)
	assertRetryable(t, openAIRetryable(&oa.Error{Code: "invalid_tool"}), false)
	assertRetryable(t, openAIRetryable(&oa.Error{Type: "authentication_error"}), false)
	assertRetryable(t, openAIRetryable(&oa.Error{StatusCode: http.StatusServiceUnavailable, Message: "overloaded"}), true)

	driver := &OpenAICompatibleDriver{options: OpenAICompatibleOptions{Provider: ProviderOpenAI}}
	wrapped := driver.formatError(&oa.Error{StatusCode: http.StatusTooManyRequests, Message: "slow down"}, "gpt-test", "execute")
	var llumiErr *common.LlumiverseError
	if !errors.As(wrapped, &llumiErr) || llumiErr.Context.Provider != ProviderOpenAI || llumiErr.Code != http.StatusTooManyRequests {
		t.Fatalf("wrapped error = %#v", wrapped)
	}
	if err := driver.formatError(errors.New("plain"), "gpt-test", "execute"); err.Error() != "plain" {
		t.Fatalf("plain error = %v", err)
	}
}

func TestOpenAIResponsesConversationHelperBranches(t *testing.T) {
	t.Parallel()

	turnOf := func(conversation any) int {
		_, turn, _ := openAIResponsesConversation(conversation)
		return turn
	}
	if got := turnOf(openAIResponsesConversationState{Turn: 2}); got != 2 {
		t.Fatalf("turn value = %d", got)
	}
	if got := turnOf(&openAIResponsesConversationState{Turn: 3}); got != 3 {
		t.Fatalf("turn pointer = %d", got)
	}
	if got := turnOf((*openAIResponsesConversationState)(nil)); got != 0 {
		t.Fatalf("nil turn = %d", got)
	}
	stored := openAIResponsesConversationState{
		Items: []responseItem{{"role": "user", "content": "old question"}},
		Turn:  6,
	}
	data, err := json.Marshal(stored)
	if err != nil {
		t.Fatal(err)
	}
	var restored any
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatal(err)
	}
	if got := turnOf(restored); got != 6 {
		t.Fatalf("json-restored turn = %d", got)
	}
	restoredInput := openAIResponsesInput(restored, []responseItem{{"role": "user", "content": "new question"}}, true)
	restoredTexts := openAIResponseInputTexts(restoredInput)
	if !containsText(restoredTexts, "old question") || !containsText(restoredTexts, "new question") {
		t.Fatalf("json-restored input texts = %#v", restoredTexts)
	}
	if got := completionResultsToText([]CompletionResult{
		newTextResult("hello"),
		common.NewJSONResult(`{"ok":true}`),
		common.NewJSONResult(map[string]any{"n": 1}),
		newImageResult("ignored"),
	}); got != `hello{"ok":true}{"n":1}` {
		t.Fatalf("completion text = %q", got)
	}

	prior := []responseItem{{"type": "function_call", "arguments": `{"q":"tokyo"}`, "call_id": "call_1", "name": "lookup"}}
	combined := openAIResponsesInput(&openAIResponsesConversationState{Items: prior}, []responseItem{{"role": "user", "content": "next"}}, false)
	if openAIResponsesInputHasToolItems(combined) {
		t.Fatalf("tool items were not converted: %#v", combined)
	}
	texts := openAIResponseInputTexts(combined)
	if len(texts) < 3 || !containsText(texts, `[Tool call: lookup({"q":"tokyo"})]`) {
		t.Fatalf("combined texts = %#v", texts)
	}

	state := buildOpenAIResponsesConversation([]responseItem{{"role": "user", "content": "question"}}, nil, &Completion{
		Result: []CompletionResult{common.NewJSONResult(map[string]any{"answer": "ok"})},
		ToolUse: []ToolUse{{
			ID:        "call_2",
			ToolName:  "lookup",
			ToolInput: toolInputObject(map[string]any{"q": "kyoto"}),
		}},
	}, ExecutionOptions{Conversation: openAIResponsesConversationState{Turn: 4}})
	if state.Turn != 5 || !openAIResponsesInputHasToolItems(state.Items) {
		t.Fatalf("conversation state = %#v", state)
	}
	if got := itemType(responseItem{"role": "assistant", "content": "x"}); got != "message" {
		t.Fatalf("role item type = %q", got)
	}

	processed := processOpenAIResponsesItemsForConversation([]responseItem{
		{"type": "message", "role": "user", "content": []any{
			map[string]any{"type": "input_image", "detail": "high", "image_url": "data:image/png;base64,aW1n"},
			map[string]any{"type": "input_text", "text": "abcdefghijklmnop"},
		}},
		{"type": "function_call", "arguments": "<heartbeat>working</heartbeat>", "call_id": "call_3", "name": "work"},
		{"type": "function_call_output", "call_id": "call_3", "output": "abcdefghijklmnop"},
	}, ExecutionOptions{StripTextMaxTokens: 2}, true, true)
	texts = openAIResponseInputTexts(processed)
	if !containsText(texts, conversationImagePlaceholder) || !containsText(texts, "abcdefgh"+conversationTextTruncatedMarker) {
		t.Fatalf("processed texts = %#v", texts)
	}
	if processed[1]["arguments"] != conversationHeartbeatPlaceholder {
		t.Fatalf("processed function call = %#v", processed[1])
	}
}

func TestOpenAISchemaAndResponseHelpers(t *testing.T) {
	t.Parallel()

	if got := openAIResponseFinishReason(&responsesResponse{Status: "completed"}, false); got != "stop" {
		t.Fatalf("completed finish = %q", got)
	}
	if got := openAIResponseFinishReason(&responsesResponse{Status: "incomplete", IncompleteDetails: &responsesIncompleteDetails{Reason: "max_output_tokens"}}, false); got != "length" {
		t.Fatalf("incomplete finish = %q", got)
	}
	if got := openAIResponseFinishReason(&responsesResponse{Status: "incomplete", IncompleteDetails: &responsesIncompleteDetails{Reason: "content_filter"}}, false); got != "content_filter" {
		t.Fatalf("content filter finish = %q", got)
	}
	if got := openAIResponseFinishReason(&responsesResponse{Status: "failed"}, false); got != "failed" {
		t.Fatalf("failed finish = %q", got)
	}
	if got := openAIResponseFinishReason(&responsesResponse{}, true); got != "tool_use" {
		t.Fatalf("tool finish = %q", got)
	}

	var response responsesResponse
	if err := json.Unmarshal([]byte(`{
		"id":"resp_1",
		"object":"response",
		"created_at":1,
		"status":"completed",
		"model":"gpt-4.1",
		"output":[
			{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]}]},
			{"id":"img_1","type":"image_generation_call","status":"completed","result":"aW1n"},
			{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"tokyo\"}"}
		],
		"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}
	}`), &response); err != nil {
		t.Fatal(err)
	}
	completion := extractOpenAIResponseCompletion(&response, true)
	if len(completion.Result) != 2 || completion.Result[1].Type() != ResultTypeImage || completion.Result[1].String() != "data:image/png;base64,aW1n" {
		t.Fatalf("completion result = %#v", completion)
	}
	if len(completion.ToolUse) != 1 || completion.ToolUse[0].ID != "call_1" {
		t.Fatalf("tool use = %#v", completion.ToolUse)
	}
	if empty := extractOpenAIResponseCompletion(&responsesResponse{Status: "completed"}, true); len(empty.Result) != 0 || empty.FinishReason != "stop" {
		t.Fatalf("empty completion = %#v", empty)
	}
	responseTools := openAIResponseTools([]ToolDefinition{{Name: "lookup", Description: ptr("Lookup"), InputSchema: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}}})
	if len(responseTools) != 1 || responseTools[0]["name"] != "lookup" {
		t.Fatalf("response tools = %#v", responseTools)
	}
	if role := openAIResponsesSystemRole("o1-mini"); role != "user" {
		t.Fatalf("o1-mini role = %s", role)
	}
	if role := openAIResponsesSystemRole("o3-mini"); role != "developer" {
		t.Fatalf("o3 role = %s", role)
	}
	for model, want := range map[string]bool{"o1": true, "o3-mini": true, "openai/o4-mini": true, "gpt-5": true, "gpt-5.6-sol": true, "gpt-6": true, "gpt-4.1": false, "gpt-4o": false, "pro1": false} {
		if got := isOpenAIReasoningModel(model); got != want {
			t.Fatalf("isOpenAIReasoningModel(%q) = %v", model, got)
		}
	}
	if got := openAIReasoning(ExecutionOptions{Model: "gpt-4.1", ModelOptions: map[string]any{"effort": "max", "reasoning_effort": "low"}}); got["effort"] != "max" || got["summary"] != "auto" {
		t.Fatalf("compatible effort = %#v", got)
	}
	if got := openAIReasoning(ExecutionOptions{Model: "gpt-4.1"}); got != nil {
		t.Fatalf("non-reasoning reasoning = %#v", got)
	}
	if got := openAIReasoning(ExecutionOptions{Model: "gpt-5.6", ModelOptions: map[string]any{"reasoning_context": "all_turns"}}); got["context"] != "all_turns" || got["effort"] != nil {
		t.Fatalf("reasoning context = %#v", got)
	}
	if got := openAIImageDetail(ExecutionOptions{ModelOptions: map[string]any{"image_detail": "invalid"}}); got != "auto" {
		t.Fatalf("image detail = %q", got)
	}
}

func TestOpenAIResponsesStreamState(t *testing.T) {
	t.Parallel()

	state := newOpenAIResponsesStreamState(false, LlumiverseErrorContext{})
	start, _ := state.chunk(openAIStreamEvent(t, `{
		"type":"response.output_item.added",
		"output_index":0,
		"item":{"id":"item_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":""}
	}`))
	if len(start.ToolUse) != 1 || start.ToolUse[0].ID != "call_1" || start.ToolUse[0].ToolName != "lookup" {
		t.Fatalf("start = %#v", start)
	}
	delta, _ := state.chunk(openAIStreamEvent(t, `{
		"type":"response.function_call_arguments.delta",
		"item_id":"item_1",
		"output_index":0,
		"delta":"{\"q\""
	}`))
	if len(delta.ToolUse) != 1 || delta.ToolUse[0].ID != "call_1" || delta.ToolUse[0].ToolInput != `{"q"` {
		t.Fatalf("delta = %#v", delta)
	}
	text, _ := state.chunk(openAIStreamEvent(t, `{"type":"response.output_text.delta","delta":"hello"}`))
	if len(text.Result) != 1 || text.Result[0].Type() != ResultTypeText || text.Result[0].String() != "hello" {
		t.Fatalf("text = %#v", text)
	}
	completed, _ := state.chunk(openAIStreamEvent(t, `{
		"type":"response.completed",
		"response":{
			"id":"resp_1",
			"object":"response",
			"created_at":1,
			"status":"completed",
			"model":"gpt-4.1",
			"output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}],
			"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}
		}
	}`))
	if completed.FinishReason != "tool_use" || completed.TokenUsage == nil || count(completed.TokenUsage.Total) != 5 {
		t.Fatalf("completed = %#v", completed)
	}
	incomplete, _ := state.chunk(openAIStreamEvent(t, `{
		"type":"response.incomplete",
		"response":{
			"id":"resp_2",
			"object":"response",
			"created_at":1,
			"status":"incomplete",
			"incomplete_details":{"reason":"max_output_tokens"},
			"model":"gpt-4.1",
			"output":[],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
		}
	}`))
	if incomplete.FinishReason != "length" {
		t.Fatalf("incomplete = %#v", incomplete)
	}
	failed, _ := state.chunk(openAIStreamEvent(t, `{
		"type":"response.failed",
		"response":{"id":"resp_3","object":"response","created_at":1,"status":"failed","model":"gpt-4.1","output":[]}
	}`))
	if failed.FinishReason != "failed" {
		t.Fatalf("failed = %#v", failed)
	}
	if empty, _ := state.chunk(responsesStreamEvent{Type: "response.output_text.delta"}); len(empty.Result) != 0 {
		t.Fatalf("empty = %#v", empty)
	}
}

func assertRetryable(t *testing.T, got *bool, want bool) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("retryable = %#v, want %v", got, want)
	}
}

func openAIStreamEvent(t *testing.T, data string) responsesStreamEvent {
	t.Helper()
	var event responsesStreamEvent
	if err := json.Unmarshal([]byte(data), &event); err != nil {
		t.Fatal(err)
	}
	return event
}

func writeOpenAISSE(t *testing.T, w http.ResponseWriter, data string) {
	t.Helper()
	_, err := w.Write([]byte("data: " + data + "\n\n"))
	if err != nil {
		t.Fatal(err)
	}
}
