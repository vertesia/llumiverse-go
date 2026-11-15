package openai

import "testing"

// Mirrors the TS fixOrphanedToolResults: an output whose call is gone is dropped,
// an answered call is kept, and an unanswered call still gets a synthetic output.
func TestOpenAIResponsesDropsOrphanedToolResults(t *testing.T) {
	t.Parallel()

	input := openAIResponsesInput(nil, []responseItem{
		{"type": "function_call_output", "call_id": "call_gone", "output": "stale"},
		{"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
		{"type": "function_call_output", "call_id": "call_1", "output": "ok"},
		{"type": "function_call", "call_id": "call_2", "name": "search", "arguments": "{}"},
		{"role": "user", "content": "next"},
	}, true)

	var outputs []string
	for _, item := range input {
		if itemType(item) == "function_call_output" {
			outputs = append(outputs, stringValue(item["call_id"]))
		}
	}
	if len(outputs) != 2 || outputs[0] != "call_1" || outputs[1] != "call_2" {
		t.Fatalf("function_call_output call ids = %#v (input %#v)", outputs, input)
	}
}
