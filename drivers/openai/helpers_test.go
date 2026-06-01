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
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
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
	if result.Usage == nil || result.Usage.InputTokens != 5 || result.Usage.InputTextTokens != 5 {
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
	if len(completion.Result) != 1 || completion.Result[0].Value != "hello world" {
		t.Fatalf("stream completion = %#v", completion.Result)
	}
	if completion.TokenUsage == nil || completion.TokenUsage.Total != 5 || completion.Conversation == nil {
		t.Fatalf("stream completion = %#v", completion)
	}
}

func TestOpenAIHelperBranches(t *testing.T) {
	t.Parallel()

	if openAIUsage(0, 0, 0, 0) != nil {
		t.Fatal("zero usage should be nil")
	}
	usage := openAIUsage(7, 3, 10, 2)
	if usage.PromptNew != 5 || usage.PromptCached != 2 {
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

	if got := openAIResponsesTurn(openAIResponsesConversationState{Turn: 2}); got != 2 {
		t.Fatalf("turn value = %d", got)
	}
	if got := openAIResponsesTurn(&openAIResponsesConversationState{Turn: 3}); got != 3 {
		t.Fatalf("turn pointer = %d", got)
	}
	if got := openAIResponsesTurn((*openAIResponsesConversationState)(nil)); got != 0 {
		t.Fatalf("nil turn = %d", got)
	}
	if got := completionResultsToText([]CompletionResult{
		{Type: ResultTypeText, Value: "hello"},
		{Type: ResultTypeJSON, Value: `{"ok":true}`},
		{Type: ResultTypeJSON, Value: map[string]any{"n": 1}},
		{Type: ResultTypeImage, Value: "ignored"},
	}); got != `hello{"ok":true}{"n":1}` {
		t.Fatalf("completion text = %q", got)
	}

	prior := responses.ResponseInputParam{
		responses.ResponseInputItemParamOfFunctionCall(`{"q":"tokyo"}`, "call_1", "lookup"),
	}
	combined := openAIResponsesInput(&openAIResponsesConversationState{Items: prior}, responses.ResponseInputParam{
		responses.ResponseInputItemParamOfMessage("next", responses.EasyInputMessageRoleUser),
	}, false)
	if openAIResponsesInputHasToolItems(combined) {
		t.Fatalf("tool items were not converted: %#v", combined)
	}
	texts := openAIResponseInputTexts(combined)
	if len(texts) < 3 || !containsText(texts, `[Tool call: lookup({"q":"tokyo"})]`) {
		t.Fatalf("combined texts = %#v", texts)
	}

	state := buildOpenAIResponsesConversation(responses.ResponseInputParam{
		responses.ResponseInputItemParamOfMessage("question", responses.EasyInputMessageRoleUser),
	}, &Completion{
		Result: []CompletionResult{{Type: ResultTypeJSON, Value: map[string]any{"answer": "ok"}}},
		ToolUse: []ToolUse{{
			ID:        "call_2",
			ToolName:  "lookup",
			ToolInput: map[string]any{"q": "kyoto"},
		}},
	}, ExecutionOptions{Conversation: openAIResponsesConversationState{Turn: 4}})
	if state.Turn != 5 || !openAIResponsesInputHasToolItems(state.Items) {
		t.Fatalf("conversation state = %#v", state)
	}
	if got := openAIResponsesItemType(responses.ResponseInputItemParamOfOutputMessage(nil, "msg_1", responses.ResponseOutputMessageStatusCompleted)); got != "message" {
		t.Fatalf("output message item type = %q", got)
	}

	imagePart := responses.ResponseInputContentParamOfInputImage(responses.ResponseInputImageDetailHigh)
	imagePart.OfInputImage.ImageURL = param.NewOpt("data:image/png;base64,aW1n")
	processed := processOpenAIResponsesItemsForConversation(responses.ResponseInputParam{
		responses.ResponseInputItemParamOfInputMessage(responses.ResponseInputMessageContentListParam{
			imagePart,
			responses.ResponseInputContentParamOfInputText("abcdefghijklmnop"),
		}, "user"),
		responses.ResponseInputItemParamOfFunctionCall("<heartbeat>working</heartbeat>", "call_3", "work"),
		responses.ResponseInputItemParamOfFunctionCallOutput("call_3", "abcdefghijklmnop"),
	}, ExecutionOptions{StripTextMaxTokens: 2}, true, true)
	texts = openAIResponseInputTexts(processed)
	if !containsText(texts, conversationImagePlaceholder) || !containsText(texts, "abcdefgh"+conversationTextTruncatedMarker) {
		t.Fatalf("processed texts = %#v", texts)
	}
	if processed[1].OfFunctionCall.Arguments != conversationHeartbeatPlaceholder {
		t.Fatalf("processed function call = %#v", processed[1].OfFunctionCall)
	}
}

func TestOpenAISchemaAndResponseHelpers(t *testing.T) {
	t.Parallel()

	strict, ok := openAISchemaForResponses(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "string", "default": "ignored"},
		},
	})
	if !ok || strict["additionalProperties"] != false {
		t.Fatalf("strict schema = %#v, %v", strict, ok)
	}
	props := strict["properties"].(map[string]any)
	if _, exists := props["answer"].(map[string]any)["default"]; exists {
		t.Fatalf("default was not dropped: %#v", props)
	}
	if _, err := openAIStrictSchema(map[string]any{"type": "object"}, 0); err == nil {
		t.Fatal("expected empty strict object error")
	}
	if _, err := openAIStrictSchema(map[string]any{"properties": map[string]any{"bad": map[string]any{}}}, 0); err == nil {
		t.Fatal("expected missing property type error")
	}
	if _, err := openAIStrictSchema(map[string]any{"type": "string"}, 6); err == nil {
		t.Fatal("expected depth error")
	}
	limited, ok := openAISchemaForResponses(map[string]any{
		"properties": map[string]any{
			"nested": map[string]any{"default": "ignored"},
		},
	})
	if ok || limited["type"] != "object" {
		t.Fatalf("limited schema = %#v, %v", limited, ok)
	}
	nested := limited["properties"].(map[string]any)["nested"].(map[string]any)
	if nested["type"] != "object" {
		t.Fatalf("nested limited schema = %#v", nested)
	}

	if got := openAIResponseFinishReason(responses.Response{Status: "completed"}, false); got != "stop" {
		t.Fatalf("completed finish = %q", got)
	}
	if got := openAIResponseFinishReason(responses.Response{Status: "incomplete", IncompleteDetails: responses.ResponseIncompleteDetails{Reason: "max_output_tokens"}}, false); got != "length" {
		t.Fatalf("incomplete finish = %q", got)
	}
	if got := openAIResponseFinishReason(responses.Response{Status: "incomplete", IncompleteDetails: responses.ResponseIncompleteDetails{Reason: "content_filter"}}, false); got != "content_filter" {
		t.Fatalf("content filter finish = %q", got)
	}
	if got := openAIResponseFinishReason(responses.Response{Status: "failed"}, false); got != "failed" {
		t.Fatalf("failed finish = %q", got)
	}
	if got := openAIResponseFinishReason(responses.Response{}, true); got != "tool_use" {
		t.Fatalf("tool finish = %q", got)
	}

	var response responses.Response
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
	if completion.OriginalResponse == nil || len(completion.Result) != 2 || completion.Result[1].Value != "data:image/png;base64,aW1n" {
		t.Fatalf("completion result = %#v", completion)
	}
	if len(completion.ToolUse) != 1 || completion.ToolUse[0].ID != "call_1" {
		t.Fatalf("tool use = %#v", completion.ToolUse)
	}
	if empty := extractOpenAIResponseCompletion(&responses.Response{Status: "completed"}, false); len(empty.Result) != 1 || empty.Result[0].Value != "" {
		t.Fatalf("empty completion = %#v", empty)
	}
	responseTools := openAIResponseTools([]ToolDefinition{{Name: "lookup", Description: "Lookup", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}}})
	if len(responseTools) != 1 || responseTools[0].OfFunction == nil || responseTools[0].OfFunction.Name != "lookup" {
		t.Fatalf("response tools = %#v", responseTools)
	}
	if role := openAIResponsesSystemRole("o1-mini"); role != responses.EasyInputMessageRoleUser {
		t.Fatalf("o1-mini role = %s", role)
	}
	if role := openAIResponsesSystemRole("o3-mini"); role != responses.EasyInputMessageRoleDeveloper {
		t.Fatalf("o3 role = %s", role)
	}
	if got := openAIReasoningEffort("gpt-5-pro", "low"); got != "high" {
		t.Fatalf("gpt-5-pro effort = %q", got)
	}
	if got := openAIReasoningEffort("gpt-5", "medium"); got != "medium" {
		t.Fatalf("gpt-5 effort = %q", got)
	}
	if got := openAIReasoningEffort("gpt-4.1", "high"); got != "" {
		t.Fatalf("non-reasoning effort = %q", got)
	}
	if got := openAIReasoningEffort("gpt-5", "invalid"); got != "" {
		t.Fatalf("invalid effort = %q", got)
	}
	if got := openAIImageDetail(ExecutionOptions{ModelOptions: map[string]any{"image_detail": "invalid"}}); got != "auto" {
		t.Fatalf("image detail = %q", got)
	}
}

func TestOpenAIResponsesStreamState(t *testing.T) {
	t.Parallel()

	state := openAIResponsesStreamState{tools: map[string]openAIResponsesStreamTool{}}
	start := state.chunk(openAIStreamEvent(t, `{
		"type":"response.output_item.added",
		"output_index":0,
		"item":{"id":"item_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":""}
	}`))
	if len(start.ToolUse) != 1 || start.ToolUse[0].ID != "call_1" || start.ToolUse[0].ToolName != "lookup" {
		t.Fatalf("start = %#v", start)
	}
	delta := state.chunk(openAIStreamEvent(t, `{
		"type":"response.function_call_arguments.delta",
		"item_id":"item_1",
		"output_index":0,
		"delta":"{\"q\""
	}`))
	if len(delta.ToolUse) != 1 || delta.ToolUse[0].ID != "call_1" || delta.ToolUse[0].ToolInput != `{"q"` {
		t.Fatalf("delta = %#v", delta)
	}
	text := state.chunk(openAIStreamEvent(t, `{"type":"response.output_text.delta","delta":"hello"}`))
	if len(text.Result) != 1 || text.Result[0].Value != "hello" {
		t.Fatalf("text = %#v", text)
	}
	completed := state.chunk(openAIStreamEvent(t, `{
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
	if completed.FinishReason != "tool_use" || completed.TokenUsage == nil || completed.TokenUsage.Total != 5 {
		t.Fatalf("completed = %#v", completed)
	}
	incomplete := state.chunk(openAIStreamEvent(t, `{
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
	failed := state.chunk(openAIStreamEvent(t, `{
		"type":"response.failed",
		"response":{"id":"resp_3","object":"response","created_at":1,"status":"failed","model":"gpt-4.1","output":[]}
	}`))
	if failed.FinishReason != "failed" {
		t.Fatalf("failed = %#v", failed)
	}
	if empty := state.chunk(responses.ResponseStreamEventUnion{Type: "response.output_text.delta"}); len(empty.Result) != 0 {
		t.Fatalf("empty = %#v", empty)
	}
}

func assertRetryable(t *testing.T, got *bool, want bool) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("retryable = %#v, want %v", got, want)
	}
}

func openAIStreamEvent(t *testing.T, data string) responses.ResponseStreamEventUnion {
	t.Helper()
	var event responses.ResponseStreamEventUnion
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
