package openai

import (
	"strings"
	"testing"
)

const reasoningResponse = `{"id":"r","status":"completed","output":[
	{"type":"reasoning","id":"rs_1","encrypted_content":"opaque","summary":[{"type":"summary_text","text":"thinking"}]},
	{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}]}`

// Mirrors the TS native thinking tests: reasoning summaries become thoughts,
// encrypted reasoning is requested and replayed unchanged in the conversation.
func TestOpenAIResponsesNativeReasoning(t *testing.T) {
	t.Parallel()

	body, resp, err := executeResponses(t, ExecutionOptions{Model: "gpt-5.6", ModelOptions: map[string]any{"effort": "max"}}, reasoningResponse)
	if err != nil {
		t.Fatal(err)
	}
	if got := marshalNoEscape(body["reasoning"]); got != `{"effort":"max","summary":"auto"}` {
		t.Fatalf("reasoning = %s", got)
	}
	if got := marshalNoEscape(body["include"]); got != `["reasoning.encrypted_content"]` {
		t.Fatalf("include = %s", got)
	}
	if got := resultsOfType(resp.Result, "thoughts"); len(got) != 1 || got[0] != "thinking" {
		t.Fatalf("thoughts = %#v", got)
	}
	if completionText(resp.Result) != "answer" {
		t.Fatalf("text = %#v", resp.Result)
	}
	stored := marshalNoEscape(resp.Conversation)
	if !strings.Contains(stored, `"encrypted_content":"opaque"`) || !strings.Contains(stored, `"type":"reasoning"`) {
		t.Fatalf("conversation = %s", stored)
	}

	_, resp, err = executeResponses(t, ExecutionOptions{Model: "gpt-5.6", ModelOptions: map[string]any{"include_thoughts": false}}, reasoningResponse)
	if err != nil {
		t.Fatal(err)
	}
	if got := resultsOfType(resp.Result, "thoughts"); len(got) != 0 {
		t.Fatalf("hidden thoughts = %#v", got)
	}

	body, _, err = executeResponses(t, ExecutionOptions{Model: "gpt-4.1", ModelOptions: map[string]any{"temperature": 0.2}}, okResponse)
	if err != nil {
		t.Fatal(err)
	}
	if body["reasoning"] != nil || body["include"] != nil || body["temperature"] != 0.2 {
		t.Fatalf("non-reasoning body = %#v", body)
	}
}

func TestOpenAIResponsesStreamReasoningAndConversation(t *testing.T) {
	t.Parallel()

	resp, err := streamResponses(t, ExecutionOptions{Model: "gpt-5"},
		`{"type":"response.reasoning_summary_text.delta","delta":"plan"}`,
		`{"type":"response.output_text.delta","delta":"done"}`,
		`{"type":"response.completed","response":{"id":"r","status":"completed","output":[{"type":"reasoning","id":"rs","encrypted_content":"e","summary":[]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}}`)
	if err != nil {
		t.Fatal(err)
	}
	if got := resultsOfType(resp.Result, "thoughts"); len(got) != 1 || got[0] != "plan" {
		t.Fatalf("thoughts = %#v", got)
	}
	if stored := marshalNoEscape(resp.Conversation); !strings.Contains(stored, `"encrypted_content":"e"`) {
		t.Fatalf("conversation = %s", stored)
	}

	_, err = streamResponses(t, ExecutionOptions{}, `{"type":"error","code":"server_error","message":"overloaded"}`)
	if err == nil || !strings.Contains(err.Error(), "overloaded (server_error)") {
		t.Fatalf("error event = %v", err)
	}
}
