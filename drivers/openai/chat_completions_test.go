package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeChatTransport records the payload and replies with a fixed response or
// SSE stream.
type fakeChatTransport struct {
	payload  map[string]any
	response string
	events   []string
}

func (f *fakeChatTransport) PostChatCompletion(_ context.Context, payload map[string]any, _ ExecutionOptions) (json.RawMessage, error) {
	f.payload = payload
	return json.RawMessage(f.response), nil
}

func (f *fakeChatTransport) PostChatCompletionStream(_ context.Context, payload map[string]any, _ ExecutionOptions) (io.ReadCloser, error) {
	f.payload = payload
	var b strings.Builder
	for _, event := range f.events {
		b.WriteString("data: " + event + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return io.NopCloser(strings.NewReader(b.String())), nil
}

func (f *fakeChatTransport) FormatChatCompletionsError(err error, _ string, _ string) error {
	return err
}

func chatExecute(t *testing.T, protocol ChatCompletionsProtocol, response string, options ExecutionOptions, segments ...PromptSegment) (*Completion, *fakeChatTransport) {
	t.Helper()
	transport := &fakeChatTransport{response: response}
	if options.Model == "" {
		options.Model = "test-model"
	}
	if len(segments) == 0 {
		segments = []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}
	}
	prompt, err := protocol.CreatePrompt(context.Background(), segments, options)
	if err != nil {
		t.Fatal(err)
	}
	completion, err := protocol.Execute(context.Background(), transport, prompt, options)
	if err != nil {
		t.Fatal(err)
	}
	return completion, transport
}

func chatStream(t *testing.T, protocol ChatCompletionsProtocol, options ExecutionOptions, events ...string) (*ExecutionResponse, *fakeChatTransport) {
	t.Helper()
	transport := &fakeChatTransport{events: events}
	if options.Model == "" {
		options.Model = "test-model"
	}
	prompt, err := protocol.CreatePrompt(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, options)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := protocol.Stream(context.Background(), transport, prompt, options)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()
	for {
		if _, err := stream.Recv(); errors.Is(err, io.EOF) {
			return stream.Completion(), transport
		} else if err != nil {
			t.Fatal(err)
		}
	}
}

func resultsOfType(results []CompletionResult, kind string) []string {
	var out []string
	for _, result := range results {
		if string(result.Type()) == kind {
			out = append(out, result.String())
		}
	}
	return out
}

func conversationMessages(t *testing.T, conversation any) []map[string]any {
	t.Helper()
	state, ok := conversation.(chatCompletionsConversation)
	if !ok {
		t.Fatalf("conversation = %#v", conversation)
	}
	return state.Messages
}

func TestChatCompletionsCutOffToolCall(t *testing.T) {
	t.Parallel()

	completion, _ := chatExecute(t, ChatCompletionsProtocol{}, `{"choices":[{"finish_reason":"length","message":{"role":"assistant","content":null,"tool_calls":[
		{"id":"a","type":"function","function":{"name":"ping","arguments":""}},
		{"id":"b","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}`,
		ExecutionOptions{Tools: []ToolDefinition{{Name: "lookup"}}})
	if completion.FinishReason != "length" || len(completion.ToolUse) != 1 || completion.ToolUse[0].ID != "a" {
		t.Fatalf("completion = %#v", completion)
	}

	completion, _ = chatExecute(t, ChatCompletionsProtocol{}, `{"choices":[{"finish_reason":"content_filter","message":{"role":"assistant","content":null,"tool_calls":[
		{"id":"b","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}`,
		ExecutionOptions{Tools: []ToolDefinition{{Name: "lookup"}}})
	if completion.FinishReason != "content_filter" || len(completion.ToolUse) != 0 {
		t.Fatalf("content filter completion = %#v", completion)
	}
}

func TestChatCompletionsPayload(t *testing.T) {
	t.Parallel()

	protocol := ChatCompletionsProtocol{Options: ChatCompletionsProtocolOptions{ModelName: "pub/model-maas", DefaultMaxTokens: 4096, ExtraBody: map[string]any{"a": 1, "b": 1}}}
	_, transport := chatExecute(t, protocol, `{"service_tier":"flex","choices":[{"finish_reason":"stop","message":{"content":"ok"}}]}`, ExecutionOptions{
		ModelOptions: map[string]any{"extra_body": map[string]any{"b": 2}, "service_tier": "flex", "effort": "high", "reasoning_effort": "low", "seed": 7},
	})
	payload := transport.payload
	if payload["model"] != "pub/model-maas" || payload["max_tokens"] != 4096 || payload["n"] != 1 || payload["service_tier"] != "flex" || payload["reasoning_effort"] != "high" || payload["seed"] != 7 {
		t.Fatalf("payload = %#v", payload)
	}
	extra := payload["extra_body"].(map[string]any)
	if extra["a"] != 1 || extra["b"] != 2 {
		t.Fatalf("extra_body = %#v", extra)
	}
	for _, key := range []string{"temperature", "top_p", "tools", "tool_choice", "parallel_tool_calls", "response_format"} {
		if _, ok := payload[key]; ok {
			t.Fatalf("payload sends %s: %#v", key, payload)
		}
	}
	flat := MergeChatCompletionsExtraBody(map[string]any{"model": "m", "extra_body": map[string]any{"model": "x", "c": 3}})
	if flat["model"] != "m" || flat["c"] != 3 || flat["extra_body"] != nil {
		t.Fatalf("flattened = %#v", flat)
	}
}

func TestChatCompletionsToolChoice(t *testing.T) {
	t.Parallel()

	tools := []ToolDefinition{{Name: "lookup", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}}}
	_, transport := chatExecute(t, ChatCompletionsProtocol{}, `{"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"id":"c","type":"function","function":{"name":"lookup","arguments":"{}"}}]}}]}`,
		ExecutionOptions{Tools: tools, ModelOptions: map[string]any{"required_tool_name": "lookup", "parallel_tool_calls": false}})
	choice, _ := json.Marshal(transport.payload["tool_choice"])
	if string(choice) != `{"function":{"name":"lookup"},"type":"function"}` || transport.payload["parallel_tool_calls"] != false {
		t.Fatalf("payload = %#v", transport.payload)
	}

	_, transport = chatExecute(t, ChatCompletionsProtocol{}, `{"choices":[{"finish_reason":"stop","message":{"content":"x"}}]}`,
		ExecutionOptions{Tools: tools, ModelOptions: map[string]any{"tool_choice": "any"}})
	if transport.payload["tool_choice"] != "required" {
		t.Fatalf("tool_choice = %#v", transport.payload["tool_choice"])
	}

	_, err := ChatCompletionsProtocol{}.BuildPayload(nil, ExecutionOptions{Model: "m", ModelOptions: map[string]any{"tool_choice": "required"}}, false)
	var llumiverseErr *LlumiverseError
	if !errors.As(err, &llumiverseErr) || llumiverseErr.Code != 400 || llumiverseErr.Name != "ToolChoiceConfigurationError" || llumiverseErr.Retryable == nil || *llumiverseErr.Retryable {
		t.Fatalf("error = %#v", err)
	}
}

func TestChatCompletionsReasoningAndThinkBlocks(t *testing.T) {
	t.Parallel()

	response := `{"choices":[{"finish_reason":"stop","message":{"content":"<think>inline</think> answer","reasoning_content":"native"}}]}`
	completion, _ := chatExecute(t, ChatCompletionsProtocol{}, response, ExecutionOptions{})
	if got := resultsOfType(completion.Result, "thoughts"); len(got) != 2 || got[0] != "native" || got[1] != "inline" {
		t.Fatalf("thoughts = %#v", got)
	}
	if got := completionText(completion.Result); got != "answer" {
		t.Fatalf("text = %q", got)
	}
	messages := conversationMessages(t, completion.Conversation)
	last := messages[len(messages)-1]
	if last["content"] != "<think>inline</think> answer" || last["reasoning_content"] != "native" {
		t.Fatalf("assistant = %#v", last)
	}

	completion, _ = chatExecute(t, ChatCompletionsProtocol{}, response, ExecutionOptions{ModelOptions: map[string]any{"include_thoughts": false}})
	if got := resultsOfType(completion.Result, "thoughts"); len(got) != 0 {
		t.Fatalf("hidden thoughts = %#v", got)
	}

	completion, _ = chatExecute(t, ChatCompletionsProtocol{}, response, ExecutionOptions{Model: "deepseek-r1"})
	messages = conversationMessages(t, completion.Conversation)
	if _, ok := messages[len(messages)-1]["reasoning_content"]; ok || len(resultsOfType(completion.Result, "thoughts")) != 2 {
		t.Fatalf("deepseek-r1 replay = %#v", messages)
	}
}

func TestChatCompletionsDeepSeekReasoningReplay(t *testing.T) {
	t.Parallel()

	messages := []map[string]any{
		{"role": "user", "content": "first"},
		{"role": "assistant", "content": "a", "reasoning_content": "old"},
		{"role": "user", "content": "second"},
		{"role": "assistant", "content": nil, "reasoning_content": "current", "tool_calls": []any{map[string]any{"id": "t", "type": "function", "function": map[string]any{"name": "x", "arguments": "{}"}}}},
		{"role": "tool", "tool_call_id": "t", "content": "ok"},
	}
	out := prepareChatCompletionsReasoning(cloneMessages(messages), "deepseek-v3.2")
	if _, ok := out[1]["reasoning_content"]; ok || out[3]["reasoning_content"] != "current" {
		t.Fatalf("v3.2 replay = %#v", out)
	}
	out = prepareChatCompletionsReasoning(cloneMessages(messages), "deepseek-v4.1-flash")
	if out[1]["reasoning_content"] != "old" {
		t.Fatalf("v4 replay = %#v", out)
	}
	state := finalizeChatCompletionsConversation(chatCompletionsConversation{Messages: cloneMessages(messages)}, ExecutionOptions{Model: "deepseek-v4"})
	if state.Messages[1]["reasoning_content"] != "old" || state.Turn != 1 {
		t.Fatalf("v4 finalize = %#v", state)
	}
}

func TestChatCompletionsConversationRepair(t *testing.T) {
	t.Parallel()

	legacy := []map[string]any{
		{"role": "user", "content": "go"},
		{"role": "assistant", "content": nil, "tool_calls": []any{map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "lookup", "arguments": "{}"}}}},
		{"role": "tool", "tool_call_id": "gone", "content": "stale"},
	}
	options := ExecutionOptions{Conversation: legacy, Tools: []ToolDefinition{{Name: "lookup"}}}
	_, transport := chatExecute(t, ChatCompletionsProtocol{}, `{"choices":[{"finish_reason":"stop","message":{"content":"done"}}]}`, options)
	sent, _ := json.Marshal(transport.payload["messages"])
	if !strings.Contains(string(sent), `Tool interrupted: The user stopped the operation before \"lookup\" could execute.`) || strings.Contains(string(sent), "stale") {
		t.Fatalf("messages = %s", sent)
	}

	options = ExecutionOptions{Conversation: legacy}
	_, transport = chatExecute(t, ChatCompletionsProtocol{}, `{"choices":[{"finish_reason":"stop","message":{"content":"done"}}]}`, options)
	sent, _ = json.Marshal(transport.payload["messages"])
	if !strings.Contains(string(sent), "[Tool call: lookup({})]") || strings.Contains(string(sent), "tool_calls") {
		t.Fatalf("messages without tools = %s", sent)
	}
}

// Mirrors the TS rule that tool-result images travel in a following user
// message, since tool messages accept only text.
func TestChatCompletionsToolImagesMoveToUserMessage(t *testing.T) {
	t.Parallel()

	converted := convertToChatCompletionsMessages([]map[string]any{
		{"role": "tool", "tool_call_id": "t1", "content": []any{
			map[string]any{"type": "text", "text": "chart"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,AA"}},
		}},
		{"role": "user", "content": ""},
	})
	data, _ := json.Marshal(converted)
	want := `[{"content":"chart\n[Image 1 attached below]","role":"tool","tool_call_id":"t1"},{"content":[{"text":"Image 1 from tool result t1:","type":"text"},{"image_url":{"url":"data:image/png;base64,AA"},"type":"image_url"}],"role":"user"},{"content":null,"role":"user"}]`
	if string(data) != want {
		t.Fatalf("converted = %s", data)
	}
}

func TestChatCompletionsSchemaInPromptAndNoData(t *testing.T) {
	t.Parallel()

	schema := map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}}
	protocol := ChatCompletionsProtocol{Options: ChatCompletionsProtocolOptions{ResultSchemaInPrompt: true}}
	_, transport := chatExecute(t, protocol, `{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}`, ExecutionOptions{ResultSchema: schema})
	sent := fmt.Sprint(transport.payload["messages"])
	if !strings.Contains(sent, "<response_schema>") || transport.payload["response_format"] != nil {
		t.Fatalf("payload = %#v", transport.payload)
	}

	_, transport = chatExecute(t, ChatCompletionsProtocol{}, `{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}`, ExecutionOptions{ResultSchema: schema})
	format, _ := json.Marshal(transport.payload["response_format"])
	if !strings.Contains(string(format), `"name":"output"`) || !strings.Contains(string(format), `"strict":true`) {
		t.Fatalf("response_format = %s", format)
	}

	prompt, _ := ChatCompletionsProtocol{}.CreatePrompt(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{})
	_, err := ChatCompletionsProtocol{}.Execute(context.Background(), &fakeChatTransport{response: `{"choices":[{"finish_reason":"stop","message":{"content":""}}]}`}, prompt, ExecutionOptions{Model: "m"})
	if err == nil || !strings.Contains(err.Error(), "no data") {
		t.Fatalf("error = %v", err)
	}
}

func TestChatCompletionsStream(t *testing.T) {
	t.Parallel()

	resp, _ := chatStream(t, ChatCompletionsProtocol{}, ExecutionOptions{Tools: []ToolDefinition{{Name: "lookup"}}},
		`{"choices":[{"delta":{"reasoning_content":"plan"}}]}`,
		`{"choices":[{"delta":{"content":"<thi"}}]}`,
		`{"choices":[{"delta":{"content":"nk>inner</think> hello"}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_9","type":"function","function":{"name":"lookup","arguments":"{\"q\""}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"x\"}"}}]}}]}`,
		`{"service_tier":"default","choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":2},"cost":0.5}}`)
	if got := completionText(resp.Result); got != "hello" {
		t.Fatalf("text = %q (%#v)", got, resp.Result)
	}
	if got := strings.Join(resultsOfType(resp.Result, "thoughts"), "|"); got != "planinner" && got != "plan|inner" {
		t.Fatalf("thoughts = %q", got)
	}
	if len(resp.ToolUse) != 1 || resp.ToolUse[0].ID != "call_9" || (*resp.ToolUse[0].ToolInput)["q"] != "x" || resp.FinishReason != "tool_use" {
		t.Fatalf("tool use = %#v finish %q", resp.ToolUse, resp.FinishReason)
	}
	if resp.ServiceTier != "default" {
		t.Fatalf("service tier = %q", resp.ServiceTier)
	}
	usage := resp.TokenUsage
	if count(usage.PromptNew) != 4 || count(usage.PromptCached) != 4 || count(usage.PromptCacheWrite) != 2 || usage.ProviderCostUsd == nil || *usage.ProviderCostUsd != 0.5 {
		t.Fatalf("usage = %#v", usage)
	}
	messages := conversationMessages(t, resp.Conversation)
	last := marshalNoEscape(messages[len(messages)-1])
	if last != `{"content":"<think>inner</think> hello","reasoning_content":"plan","role":"assistant","tool_calls":[{"function":{"arguments":"{\"q\":\"x\"}","name":"lookup"},"id":"call_9","type":"function"}]}` {
		t.Fatalf("assistant = %s", last)
	}
}

func marshalNoEscape(value any) string {
	var b strings.Builder
	encoder := json.NewEncoder(&b)
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(value)
	return strings.TrimSpace(b.String())
}

func TestThinkStreamProjectorHoldsPartialTags(t *testing.T) {
	t.Parallel()

	var projector thinkStreamProjector
	var text, thoughts strings.Builder
	for i, piece := range []string{"Hi <", "think>a</th", "ink>  b", ""} {
		for _, result := range projector.push(piece, i == 3) {
			if result.Type() == ResultTypeThoughts {
				thoughts.WriteString(result.String())
			} else {
				text.WriteString(result.String())
			}
		}
	}
	if text.String() != "Hi b" || thoughts.String() != "a" {
		t.Fatalf("text %q thoughts %q", text.String(), thoughts.String())
	}
}

func TestOpenAIChatCompletionsDriverSDKTransport(t *testing.T) {
	t.Parallel()

	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		bodies = append(bodies, body)
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			writeOpenAISSE(t, w, `{"choices":[{"delta":{"content":"hey"},"finish_reason":"stop"}]}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"stop","message":{"content":"hello"}}]}`))
	}))
	defer server.Close()
	driver, err := NewOpenAIChatCompletionsDriver(OpenAIChatCompletionsOptions{
		OpenAICompatibleOptions:        OpenAICompatibleOptions{APIKey: "k", Endpoint: server.URL},
		ChatCompletionsProtocolOptions: ChatCompletionsProtocolOptions{ExtraBody: map[string]any{"top_k": 5}},
	})
	if err != nil {
		t.Fatal(err)
	}
	segments := []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}
	resp, err := driver.Execute(context.Background(), segments, ExecutionOptions{Model: "m"})
	if err != nil || completionText(resp.Result) != "hello" {
		t.Fatalf("execute = %#v, %v", resp, err)
	}
	stream, err := driver.Stream(context.Background(), segments, ExecutionOptions{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := stream.Recv(); err != nil {
			break
		}
	}
	_ = stream.Close()
	if got := completionText(stream.Completion().Result); got != "hey" {
		t.Fatalf("stream text = %q", got)
	}
	if len(bodies) != 2 || bodies[0]["top_k"] != float64(5) || bodies[0]["extra_body"] != nil {
		t.Fatalf("bodies = %#v", bodies)
	}
	if options, _ := bodies[1]["stream_options"].(map[string]any); options["include_usage"] != true {
		t.Fatalf("stream body = %#v", bodies[1])
	}
}
