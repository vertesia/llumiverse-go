package azurefoundry

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

func TestChatPayloadOptionsAndToolCalls(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	f.respond = func(w http.ResponseWriter, r recordedRequest) bool {
		writeRaw(w, `{
			"id":"foundry-1","created":1,"model":"llama-deployment",
			"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,
				"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Paris\"}"}}]}}],
			"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}
		}`)
		return true
	}
	driver := newKeyDriver(t, f, nil)
	description := "Lookup"
	resp, err := driver.Execute(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "Be brief."},
		{Role: PromptRoleUser, Content: "Weather?"},
		{Role: PromptRoleSafety, Content: "No secrets."},
	}, ExecutionOptions{
		Model: "llama-deployment::llama",
		ModelOptions: map[string]any{
			"max_tokens":        16,
			"presence_penalty":  0.1,
			"frequency_penalty": 0.2,
			"stop_sequence":     []string{"END"},
			"seed":              42,
			"tool_choice":       "any",
		},
		Tools: []ToolDefinition{{Name: "lookup", Description: &description, InputSchema: map[string]any{"type": "object"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := f.recorded()[0].Body
	want := map[string]any{
		"model": "llama-deployment", "stream": false, "n": float64(1), "max_tokens": float64(16),
		"presence_penalty": 0.1, "frequency_penalty": 0.2, "stop": []any{"END"}, "seed": float64(42), "tool_choice": "required",
	}
	for key, value := range want {
		if !reflect.DeepEqual(body[key], value) {
			t.Fatalf("%s = %#v, want %#v", key, body[key], value)
		}
	}
	tools, _ := body["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v", body["tools"])
	}
	function := tools[0].(map[string]any)["function"].(map[string]any)
	if function["name"] != "lookup" || function["description"] != "Lookup" || function["parameters"].(map[string]any)["type"] != "object" {
		t.Fatalf("tool = %v", tools[0])
	}
	messages := body["messages"].([]any)
	roles := make([]string, len(messages))
	for i, message := range messages {
		roles[i] = message.(map[string]any)["role"].(string)
	}
	if !reflect.DeepEqual(roles, []string{"system", "system", "user"}) {
		t.Fatalf("roles = %v", roles)
	}
	if got := messages[1].(map[string]any)["content"]; got != "DO NOT IGNORE - IMPORTANT: No secrets." {
		t.Fatalf("safety = %v", got)
	}
	if len(resp.ToolUse) != 1 || resp.ToolUse[0].ID != "call_1" || resp.ToolUse[0].ToolName != "lookup" || (*resp.ToolUse[0].ToolInput)["city"] != "Paris" {
		t.Fatalf("tool use = %+v", resp.ToolUse)
	}
	if resp.FinishReason != "tool_use" {
		t.Fatalf("finish reason = %q", resp.FinishReason)
	}
	conversation := resp.Conversation.(chatPrompt)
	last := conversation.Messages[len(conversation.Messages)-1]
	encoded, _ := json.Marshal(last)
	if !strings.Contains(string(encoded), `"content":null`) || !strings.Contains(string(encoded), `"tool_calls"`) {
		t.Fatalf("assistant message = %s", encoded)
	}
	if conversation.Turn != 1 {
		t.Fatalf("turn = %d", conversation.Turn)
	}
}

func TestChatRequiredToolChoiceWithoutTools(t *testing.T) {
	t.Parallel()
	driver := newKeyDriver(t, newFakeAzure(t), nil)
	_, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "llama::llama", ModelOptions: map[string]any{"tool_choice": "required"}})
	var llumiErr *LlumiverseError
	if !errors.As(err, &llumiErr) || llumiErr.Retryable == nil || *llumiErr.Retryable {
		t.Fatalf("error = %#v", err)
	}
}

func TestChatResultSchemaUsesResponseFormatAndInstruction(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	f.respond = func(w http.ResponseWriter, r recordedRequest) bool {
		writeRaw(w, `{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"<think>pondering</think>{\"answer\":\"green\"}"}}]}`)
		return true
	}
	driver := newKeyDriver(t, f, nil)
	schema := map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}}
	resp, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "mistral::mistral-large", ResultSchema: schema})
	if err != nil {
		t.Fatal(err)
	}
	body := f.recorded()[0].Body
	format := body["response_format"].(map[string]any)
	jsonSchema := format["json_schema"].(map[string]any)
	if format["type"] != "json_schema" || jsonSchema["name"] != "output" || jsonSchema["strict"] != true {
		t.Fatalf("response_format = %v", format)
	}
	system := body["messages"].([]any)[0].(map[string]any)
	if !strings.Contains(system["content"].(string), "<response_schema>") {
		t.Fatalf("system = %v", system)
	}
	if len(resp.Result) != 2 || resp.Result[0].Type() != ResultTypeThoughts || resp.Result[0].String() != "pondering" || resp.Result[1].Type() != ResultTypeJSON {
		t.Fatalf("result = %+v", resp.Result)
	}
}

func TestChatReasoningContentAndIncludeThoughts(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	f.respond = func(w http.ResponseWriter, r recordedRequest) bool {
		writeRaw(w, `{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"Answer","reasoning_content":"Because"}}]}`)
		return true
	}
	driver := newKeyDriver(t, f, nil)
	resp, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "deepseek::DeepSeek-V4-Pro"})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Result) != 2 || resp.Result[0].Type() != ResultTypeThoughts || resp.Result[0].String() != "Because" || resp.Result[1].String() != "Answer" {
		t.Fatalf("result = %+v", resp.Result)
	}
	resp, err = driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "deepseek::DeepSeek-V4-Pro", ModelOptions: map[string]any{"include_thoughts": false}})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Result) != 1 || resp.Result[0].String() != "Answer" {
		t.Fatalf("result = %+v", resp.Result)
	}
}

func TestChatCutOffDropsAnIncompleteLastToolCall(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	f.respond = func(w http.ResponseWriter, r recordedRequest) bool {
		writeRaw(w, `{"choices":[{"finish_reason":"length","message":{"role":"assistant","content":null,"tool_calls":[
			{"id":"a","type":"function","function":{"name":"lookup","arguments":"{\"q\":1}"}},
			{"id":"b","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}`)
		return true
	}
	driver := newKeyDriver(t, f, nil)
	resp, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "llama::llama", Tools: []ToolDefinition{{Name: "lookup"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.ToolUse) != 1 || resp.ToolUse[0].ID != "a" || resp.FinishReason != "length" {
		t.Fatalf("completion = %+v", resp.Completion)
	}
}

func TestChatStreamSplitsThinkBlocksAndToolCalls(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	events := []string{
		`{"choices":[{"index":0,"delta":{"role":"assistant","content":"<thi"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"nk>hidden</th"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"ink>Green"}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\""}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}`,
	}
	f.respond = func(w http.ResponseWriter, r recordedRequest) bool {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			_, _ = io.WriteString(w, "data: "+event+"\n\n")
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		return true
	}
	driver := newKeyDriver(t, f, nil)
	stream, err := driver.Stream(context.Background(), helloSegments, ExecutionOptions{Model: "chat-deployment::mistral", Tools: []ToolDefinition{{Name: "lookup"}}})
	if err != nil {
		t.Fatal(err)
	}
	drain(t, stream)
	body := f.recorded()[0].Body
	if body["stream"] != true || body["model"] != "chat-deployment" {
		t.Fatalf("body = %v", body)
	}
	if _, ok := body["stream_options"]; ok {
		t.Fatalf("stream_options sent: %v", body)
	}
	completion := stream.Completion()
	var thoughts, text string
	for _, result := range completion.Result {
		if result.Type() == ResultTypeThoughts {
			thoughts += result.String()
		} else {
			text += result.String()
		}
	}
	if thoughts != "hidden" || text != "Green" {
		t.Fatalf("thoughts = %q, text = %q", thoughts, text)
	}
	if len(completion.ToolUse) != 1 || completion.ToolUse[0].ID != "call_1" || (*completion.ToolUse[0].ToolInput)["q"] != float64(1) {
		t.Fatalf("tool use = %+v", completion.ToolUse)
	}
	if completion.FinishReason != "tool_use" || common.Count(completion.TokenUsage.Total) != 7 {
		t.Fatalf("completion = %+v", completion.Completion)
	}
	conversation := completion.Conversation.(chatPrompt)
	last := conversation.Messages[len(conversation.Messages)-1]
	if last.Role != "assistant" || len(last.ToolCalls) != 1 || last.ToolCalls[0].Function.Arguments != `{"q":1}` {
		t.Fatalf("assistant = %+v", last)
	}
}

func TestChatConversationRoundTripsThroughJSON(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	driver := newKeyDriver(t, f, nil)
	keep := 1
	first, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Look", Files: []common.DataSource{
		common.BytesDataSource{FileName: "a.png", MIME: "image/png", Data: []byte("img")},
	}}}, ExecutionOptions{Model: "llama::llama", StripImagesAfterTurns: &keep})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(first.Conversation)
	if err != nil {
		t.Fatal(err)
	}
	var stored any
	if err := json.Unmarshal(encoded, &stored); err != nil {
		t.Fatal(err)
	}
	if _, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Again"}}, ExecutionOptions{Model: "llama::llama", Conversation: stored}); err != nil {
		t.Fatal(err)
	}
	messages := f.recorded()[1].Body["messages"].([]any)
	if len(messages) != 3 {
		t.Fatalf("messages = %v", messages)
	}
	// The first turn's image expired after one turn.
	parts := messages[0].(map[string]any)["content"].([]any)
	if parts[0].(map[string]any)["text"] != conversationImagePlaceholder {
		t.Fatalf("parts = %v", parts)
	}
	if messages[1].(map[string]any)["role"] != "assistant" || messages[2].(map[string]any)["content"] != "Again" {
		t.Fatalf("messages = %v", messages)
	}
}

func TestChatConversationRepairsToolHistory(t *testing.T) {
	t.Parallel()
	messages := []chatMessage{
		{Role: "user", Text: "go"},
		{Role: "assistant", ToolCalls: []chatToolCall{{ID: "a", Type: "function", Function: chatToolFunction{Name: "lookup", Arguments: "{}"}}}},
		{Role: "user", Text: "next"},
		{Role: "tool", ToolCallID: "gone", Text: "orphan"},
	}
	repaired := chatConversationInput(chatPrompt{Messages: messages}, chatPrompt{}, "llama", true)
	var roles []string
	for _, message := range repaired.Messages {
		roles = append(roles, message.Role)
	}
	if !reflect.DeepEqual(roles, []string{"user", "assistant", "tool", "user"}) || !strings.Contains(repaired.Messages[2].Text, "Tool interrupted") {
		t.Fatalf("repaired = %+v", repaired.Messages)
	}
	textOnly := chatConversationInput(chatPrompt{Messages: repaired.Messages}, chatPrompt{}, "llama", false)
	if textOnly.Messages[1].Text != "[Tool call: lookup({})]" || textOnly.Messages[2].Role != "user" || !strings.HasPrefix(textOnly.Messages[2].Text, "[Tool result: ") {
		t.Fatalf("text only = %+v", textOnly.Messages)
	}
}

func TestChatReasoningReplayPolicy(t *testing.T) {
	t.Parallel()
	reasoning := "thinking"
	messages := []chatMessage{
		{Role: "user", Text: "q1"},
		{Role: "assistant", Text: "a1", ReasoningContent: &reasoning},
		{Role: "user", Text: "q2"},
		{Role: "assistant", Text: "a2", ReasoningContent: &reasoning},
	}
	if out := prepareChatReasoning(messages, "DeepSeek-R1"); out[1].ReasoningContent != nil || out[3].ReasoningContent != nil {
		t.Fatal("R1 keeps replayed reasoning")
	}
	if out := prepareChatReasoning(messages, "deepseek-v3.2"); out[1].ReasoningContent != nil || out[3].ReasoningContent == nil {
		t.Fatal("V3.2 must keep the active turn's reasoning only")
	}
	if out := prepareChatReasoning(messages, "DeepSeek-V4-Pro"); out[1].ReasoningContent == nil {
		t.Fatal("V4 drops reasoning")
	}
}

func TestSplitThinkBlocks(t *testing.T) {
	t.Parallel()
	results := splitThinkBlocks("before <think>a</think> middle <THINK>b", true)
	var got []string
	for _, result := range results {
		got = append(got, string(result.Type())+":"+result.String())
	}
	want := []string{"text:before", "thoughts:a", "text:middle", "thoughts:b"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	if results := splitThinkBlocks("<think>a</think>text", false); len(results) != 1 || results[0].String() != "text" {
		t.Fatalf("results = %+v", results)
	}
}

func TestChatDebugPromptTruncatesImages(t *testing.T) {
	t.Parallel()
	driver := newKeyDriver(t, newFakeAzure(t), nil)
	url := "data:image/png;base64," + strings.Repeat("A", 500)
	prompt := chatPrompt{Messages: []chatMessage{{Role: "user", Parts: []chatPart{{Type: "image_url", ImageURL: &chatImageURL{URL: url}}}}}}
	debug := driver.FormatDebugPrompt(prompt).(chatPrompt)
	if got := debug.Messages[0].Parts[0].ImageURL.URL; len(got) > 80 || !strings.Contains(got, "truncated") {
		t.Fatalf("debug url = %q", got)
	}
	if prompt.Messages[0].Parts[0].ImageURL.URL != url {
		t.Fatal("FormatDebugPrompt modified the prompt")
	}
}
