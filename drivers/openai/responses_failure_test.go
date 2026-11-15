package openai

import (
	"errors"
	"strings"
	"testing"
)

// Mirrors the TS failed-response tests: a failed response is a typed error
// whose status follows its code, and unknown codes are not retryable.
func TestOpenAIResponsesFailedResponse(t *testing.T) {
	t.Parallel()

	for code, want := range map[string]int{"rate_limit_exceeded": 429, "server_error": 500, "vector_store_timeout": 408, "image_file_not_found": 404, "invalid_prompt": 400, "brand_new": 400} {
		_, _, err := executeResponses(t, ExecutionOptions{}, `{"id":"resp_9","status":"failed","error":{"code":"`+code+`","message":"boom"},"output":[]}`)
		var llumiverseErr *LlumiverseError
		if !errors.As(err, &llumiverseErr) || llumiverseErr.Code != want || llumiverseErr.Name != "OpenAIResponseError" {
			t.Fatalf("%s: error = %#v", code, err)
		}
		if retryable := llumiverseErr.Retryable != nil && *llumiverseErr.Retryable; retryable != (want != 400 && want != 404) {
			t.Errorf("%s: retryable = %v", code, retryable)
		}
		if !strings.Contains(err.Error(), "[OpenAI Responses API] boom ("+code+") [response resp_9]") {
			t.Errorf("%s: message = %q", code, err.Error())
		}
	}

	_, _, err := executeResponses(t, ExecutionOptions{}, `{"id":"r","status":"completed","output":[]}`)
	if err == nil || !strings.Contains(err.Error(), "no data") {
		t.Fatalf("empty response error = %v", err)
	}

	resp, err := streamResponses(t, ExecutionOptions{},
		`{"type":"response.failed","response":{"id":"r","status":"failed","error":{"code":"server_error","message":"down"},"output":[],"usage":{"input_tokens":3,"output_tokens":0,"total_tokens":3}}}`)
	var llumiverseErr *LlumiverseError
	if !errors.As(err, &llumiverseErr) || llumiverseErr.Code != 500 || count(resp.TokenUsage.Prompt) != 3 {
		t.Fatalf("stream failure = %#v, usage %#v", err, resp.TokenUsage)
	}
}

// Mirrors the TS cut-off tool call tests: an incomplete response reports its
// stop rather than tool_use, and drops a last call whose arguments were cut off.
func TestOpenAIResponsesIncompleteToolCall(t *testing.T) {
	t.Parallel()

	_, resp, err := executeResponses(t, ExecutionOptions{}, `{"id":"r","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[
		{"type":"function_call","call_id":"c1","name":"a","arguments":"{\"x\":1}"},
		{"type":"function_call","call_id":"c2","name":"b","arguments":"{\"y\":"}]}`)
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != "length" || len(resp.ToolUse) != 1 || resp.ToolUse[0].ID != "c1" {
		t.Fatalf("finish %q, tools %#v", resp.FinishReason, resp.ToolUse)
	}

	_, resp, err = executeResponses(t, ExecutionOptions{}, `{"id":"r","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]}`)
	if err != nil || resp.FinishReason != "length" {
		t.Fatalf("empty cut-off: %v, %#v", err, resp)
	}
}

// Mirrors the TS tests for arguments sent without deltas: arguments in the
// done events or only in the terminal response reach the caller once, and a
// call cut off inside its arguments is dropped.
func TestOpenAIResponsesStreamArgumentsWithoutDeltas(t *testing.T) {
	t.Parallel()

	resp, err := streamResponses(t, ExecutionOptions{},
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"c1","name":"a"}}`,
		`{"type":"response.function_call_arguments.done","output_index":0,"item_id":"fc_1","arguments":"{\"x\":1}"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"c1","name":"a","arguments":"{\"x\":1}"}}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"c2","name":"b"}}`,
		`{"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"fc_3","call_id":"c3","name":"c"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":2,"item_id":"fc_3","delta":"{\"z\":3}"}`,
		`{"type":"response.completed","response":{"id":"r","status":"completed","output":[
			{"type":"function_call","id":"fc_1","call_id":"c1","name":"a","arguments":"{\"x\":1}"},
			{"type":"function_call","id":"fc_2","call_id":"c2","name":"b","arguments":"{\"y\":2}"},
			{"type":"function_call","id":"fc_3","call_id":"c3","name":"c","arguments":"{\"z\":3}"}]}}`)
	if err != nil {
		t.Fatal(err)
	}
	if got := marshalNoEscape(resp.ToolUse); got != `[{"id":"c1","tool_input":{"x":1},"tool_name":"a"},{"id":"c2","tool_input":{"y":2},"tool_name":"b"},{"id":"c3","tool_input":{"z":3},"tool_name":"c"}]` || resp.FinishReason != "tool_use" {
		t.Fatalf("tools %s, finish %q", got, resp.FinishReason)
	}

	resp, err = streamResponses(t, ExecutionOptions{},
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"c1","name":"a"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"{\"x\":1}"}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"c2","name":"b"}}`,
		`{"type":"response.incomplete","response":{"id":"r","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[
			{"type":"function_call","id":"fc_1","call_id":"c1","name":"a","arguments":"{\"x\":1}"},
			{"type":"function_call","id":"fc_2","call_id":"c2","name":"b","arguments":"{\"y\":"}]}}`)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.ToolUse) != 1 || resp.ToolUse[0].ID != "c1" || resp.FinishReason != "length" {
		t.Fatalf("cut-off tools %#v, finish %q", resp.ToolUse, resp.FinishReason)
	}
}

func TestOpenAIResponsesStreamImages(t *testing.T) {
	t.Parallel()

	image := `{"type":"image_generation_call","id":"ig_1","status":"completed","result":"aW1n"}`
	resp, err := streamResponses(t, ExecutionOptions{},
		`{"type":"response.output_item.done","output_index":0,"item":`+image+`}`,
		`{"type":"response.completed","response":{"id":"r","status":"completed","output":[`+image+`,{"type":"image_generation_call","id":"ig_2","status":"completed","result":"bW9yZQ=="}]}}`)
	if err != nil {
		t.Fatal(err)
	}
	if got := resultsOfType(resp.Result, "image"); len(got) != 2 || got[0] != "data:image/png;base64,aW1n" || got[1] != "data:image/png;base64,bW9yZQ==" {
		t.Fatalf("images = %#v", got)
	}
}
