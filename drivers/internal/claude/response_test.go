package claude

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

// count reads an optional usage count.
func count(v *float64) float64 {
	if v == nil {
		return -1
	}
	return *v
}

func TestClaudeTokenUsage(t *testing.T) {
	t.Parallel()

	var usage claudeUsage
	if err := json.Unmarshal([]byte(`{"input_tokens":7,"output_tokens":8,"cache_read_input_tokens":2,"cache_creation_input_tokens":3,"cache_creation":{"ephemeral_1h_input_tokens":2,"ephemeral_5m_input_tokens":1}}`), &usage); err != nil {
		t.Fatal(err)
	}
	got := claudeTokenUsage(usage)
	if count(got.Prompt) != 12 || count(got.PromptNew) != 7 || count(got.Result) != 8 || count(got.Total) != 20 || count(got.PromptCached) != 2 || count(got.PromptCacheWrite) != 3 || count(got.PromptCacheWrite1H) != 2 {
		t.Fatalf("usage = %+v", got)
	}
	plain := claudeTokenUsage(claudeUsage{InputTokens: 1, OutputTokens: 1})
	if plain.PromptCached != nil || plain.PromptCacheWrite != nil || plain.PromptCacheWrite1H != nil {
		t.Fatalf("unreported cache counts stay unset: %+v", plain)
	}

	// Compaction iterations are billed apart from the top-level usage and counted with the request.
	usage = claudeUsage{}
	if err := json.Unmarshal([]byte(`{"input_tokens":100,"output_tokens":10,"cache_read_input_tokens":5,"iterations":[
		{"type":"compaction","input_tokens":1000,"output_tokens":50,"cache_read_input_tokens":20,"cache_creation_input_tokens":4,"cache_creation":{"ephemeral_1h_input_tokens":4}},
		{"type":"message","input_tokens":100,"output_tokens":10}]}`), &usage); err != nil {
		t.Fatal(err)
	}
	got = claudeTokenUsage(usage)
	if count(got.PromptNew) != 1100 || count(got.Result) != 60 || count(got.PromptCached) != 25 || count(got.PromptCacheWrite) != 4 || count(got.PromptCacheWrite1H) != 4 || count(got.Prompt) != 1129 {
		t.Fatalf("compaction usage = %+v", got)
	}
	if claudeServiceTier(claudeUsage{ServiceTier: "standard", Speed: "fast"}) != "fast" || claudeServiceTier(claudeUsage{ServiceTier: "priority"}) != "priority" {
		t.Fatal("service tier")
	}
}

func TestClaudeFinishReasons(t *testing.T) {
	t.Parallel()

	for reason, want := range map[string]string{"end_turn": "stop", "max_tokens": "length", "model_context_window_exceeded": "length", "tool_use": "tool_use", "refusal": "refusal", "compaction": "compaction", "": ""} {
		if got := claudeFinishReason(reason); got != want {
			t.Fatalf("%q: %q, want %q", reason, got, want)
		}
	}
}

func TestClaudeStopsAfterToolCalls(t *testing.T) {
	t.Parallel()

	response := func(stop string, content string) claudeResponse {
		return claudeResponse{Role: "assistant", StopReason: stop, Content: decodeBlocks(t, content)}
	}
	twoCalls := `[{"type":"tool_use","id":"a","name":"lookup","input":{"q":1}},{"type":"tool_use","id":"b","name":"lookup","input":{"q":2}}]`
	refused := extractClaudeCompletion(response("refusal", twoCalls), ExecutionOptions{})
	if refused.FinishReason != "refusal" || len(refused.ToolUse) != 1 || refused.ToolUse[0].ID != "a" {
		t.Fatalf("refusal = %+v", refused)
	}
	cut := extractClaudeCompletion(response("max_tokens", `[{"type":"tool_use","id":"a","name":"lookup","input":{"q":1}}]`), ExecutionOptions{})
	if cut.FinishReason != "length" || len(cut.ToolUse) != 0 {
		t.Fatalf("cut off = %+v", cut)
	}
	normal := extractClaudeCompletion(response("tool_use", twoCalls), ExecutionOptions{})
	if normal.FinishReason != "tool_use" || len(normal.ToolUse) != 2 || (*normal.ToolUse[1].ToolInput)["q"] != float64(2) {
		t.Fatalf("tool_use = %+v", normal)
	}
	// The conversation keeps the refused calls, answered as interrupted on the next request.
	request, err := NewRequest(userPrompt("go"), ExecutionOptions{Model: "claude-sonnet-4-5", Tools: []ToolDefinition{namedTool("lookup")}}, RequestSettings{Provider: ProviderNameAnthropic})
	if err != nil {
		t.Fatal(err)
	}
	conversation := request.Complete(response("refusal", twoCalls)).Conversation.(claudePrompt)
	if last := conversation.Messages[len(conversation.Messages)-1]; len(last.Content) != 2 {
		t.Fatalf("conversation = %#v", conversation.Messages)
	}
}

func TestClaudeResultsThoughtsAndCompaction(t *testing.T) {
	t.Parallel()

	content := decodeBlocks(t, `[{"type":"thinking","thinking":"plan","signature":"s"},{"type":"redacted_thinking","data":"sealed"},{"type":"text","text":"answer"}]`)
	with := extractClaudeCompletion(claudeResponse{Content: content}, ExecutionOptions{ModelOptions: map[string]any{"include_thoughts": true}, IncludeOriginalResponse: true})
	if len(with.Result) != 2 || with.Result[0].Type() != common.ResultTypeThoughts || with.Result[0].String() != "plan" || with.Result[1].String() != "answer" || with.OriginalResponse == nil {
		t.Fatalf("with thoughts = %+v", with.Result)
	}
	without := extractClaudeCompletion(claudeResponse{Content: content}, ExecutionOptions{})
	if len(without.Result) != 1 || without.Result[0].String() != "answer" {
		t.Fatalf("without thoughts = %+v", without.Result)
	}
	empty := extractClaudeCompletion(claudeResponse{}, ExecutionOptions{})
	if len(empty.Result) != 1 || empty.Result[0].String() != "" {
		t.Fatalf("empty = %+v", empty.Result)
	}
	compacted := extractClaudeCompletion(claudeResponse{StopReason: "compaction", Content: decodeBlocks(t, `[{"type":"compaction","content":"the summary"}]`)}, ExecutionOptions{CompactConversation: &CompactConversationOptions{}})
	if compacted.FinishReason != "compaction" || compacted.Result[0].String() != "the summary" {
		t.Fatalf("compaction = %+v", compacted)
	}
}

// eventSource replays stream events.
func eventSource(events ...string) EventSource {
	return func(yield func([]byte) error) error {
		for _, event := range events {
			if err := yield([]byte(event)); err != nil {
				return err
			}
		}
		return io.EOF
	}
}

// drain reads a stream to its end and returns the completion and the error that ended it.
func drain(t *testing.T, stream CompletionStream) (*ExecutionResponse, []CompletionChunk, error) {
	t.Helper()
	var chunks []CompletionChunk
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return stream.Completion(), chunks, nil
		}
		if err != nil {
			_ = stream.Close()
			return stream.Completion(), chunks, err
		}
		chunks = append(chunks, chunk)
	}
}

func TestClaudeStreamRebuildsTheFinalMessage(t *testing.T) {
	t.Parallel()

	options := ExecutionOptions{Model: "claude-sonnet-4-6", ModelOptions: map[string]any{"include_thoughts": true}, Tools: []ToolDefinition{namedTool("lookup")}}
	request, err := NewRequest(userPrompt("Hi"), options, RequestSettings{Provider: ProviderNameAnthropic, Operation: "stream"})
	if err != nil {
		t.Fatal(err)
	}
	stream := request.Stream(nil, eventSource(
		`{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"usage":{"input_tokens":2,"cache_read_input_tokens":1,"service_tier":"priority"}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hello "}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"world"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tool_1","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\""}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":":\"tokyo\"}"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"ping"}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}`,
		`{"type":"message_stop"}`,
	), func() error { return nil }, func(err error) error { return err })
	completion, _, err := drain(t, stream)
	if err != nil {
		t.Fatal(err)
	}
	if len(completion.Result) != 2 || completion.Result[0].Type() != common.ResultTypeThoughts || completion.Result[0].String() != "let me think" || completion.Result[1].String() != "hello world" {
		t.Fatalf("result = %+v", completion.Result)
	}
	if completion.FinishReason != "tool_use" || len(completion.ToolUse) != 1 || (*completion.ToolUse[0].ToolInput)["q"] != "tokyo" {
		t.Fatalf("completion = %+v", completion.Completion)
	}
	if count(completion.TokenUsage.Prompt) != 3 || count(completion.TokenUsage.Result) != 3 || completion.ServiceTier != "priority" {
		t.Fatalf("usage = %+v tier = %q", completion.TokenUsage, completion.ServiceTier)
	}
	conversation := completion.Conversation.(claudePrompt)
	data, _ := json.Marshal(conversation.Messages[len(conversation.Messages)-1])
	want := `{"role":"assistant","content":[{"signature":"sig-1","thinking":"let me think","type":"thinking"},{"text":"hello world","type":"text"},{"id":"tool_1","input":{"q":"tokyo"},"name":"lookup","type":"tool_use"}]}`
	if string(data) != want {
		t.Fatalf("stored message:\n%s\nwant\n%s", data, want)
	}
}

func TestClaudeStreamCompaction(t *testing.T) {
	t.Parallel()

	request, err := NewRequest(userPrompt("Hi"), ExecutionOptions{Model: "claude-opus-4-8", Compaction: &ConversationCompactionOptions{}}, RequestSettings{Provider: ProviderNameAnthropic})
	if err != nil {
		t.Fatal(err)
	}
	stream := request.Stream(nil, eventSource(
		`{"type":"message_start","message":{"role":"assistant","content":[],"usage":{"input_tokens":50,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"compaction","content":null}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"the summary","encrypted_content":"opaque"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5,"iterations":[{"type":"compaction","input_tokens":1000,"output_tokens":40},{"type":"message","input_tokens":50,"output_tokens":5}]}}`,
		`{"type":"message_stop"}`,
	), func() error { return nil }, func(err error) error { return err })
	completion, _, err := drain(t, stream)
	if err != nil {
		t.Fatal(err)
	}
	if count(completion.TokenUsage.PromptNew) != 1050 || count(completion.TokenUsage.Result) != 45 {
		t.Fatalf("usage = %+v", completion.TokenUsage)
	}
	conversation := completion.Conversation.(claudePrompt)
	if len(conversation.Messages) != 1 {
		t.Fatalf("the conversation continues from the summary: %#v", conversation.Messages)
	}
	if summary, ok := claudeCompactionSummary(conversation.Messages[0].Content[0]); !ok || summary != "the summary" {
		t.Fatalf("first block = %#v", conversation.Messages[0].Content[0])
	}
	if data, _ := json.Marshal(conversation.Messages[0].Content[0]); !strings.Contains(string(data), `"encrypted_content":"opaque"`) {
		t.Fatalf("compaction block = %s", data)
	}

	// On demand: the summary alone is the answer.
	onDemand, err := NewRequest(userPrompt("Hi"), ExecutionOptions{Model: "claude-opus-4-8", CompactConversation: &CompactConversationOptions{}}, RequestSettings{Provider: ProviderNameAnthropic})
	if err != nil {
		t.Fatal(err)
	}
	completion, _, err = drain(t, onDemand.Stream(nil, eventSource(
		`{"type":"message_start","message":{"role":"assistant","content":[],"usage":{"input_tokens":5}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"compaction","content":"short summary"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"compaction"},"usage":{"output_tokens":9}}`,
		`{"type":"message_stop"}`,
	), func() error { return nil }, func(err error) error { return err }))
	if err != nil {
		t.Fatal(err)
	}
	if completion.FinishReason != "compaction" || len(completion.Result) != 1 || completion.Result[0].String() != "short summary" {
		t.Fatalf("on-demand = %+v", completion.Completion)
	}
}

func TestClaudeStreamStopsInsideAToolCall(t *testing.T) {
	t.Parallel()

	request, err := NewRequest(userPrompt("Hi"), ExecutionOptions{Model: "claude-sonnet-4-5", Tools: []ToolDefinition{namedTool("lookup")}}, RequestSettings{Provider: ProviderNameAnthropic})
	if err != nil {
		t.Fatal(err)
	}
	completion, _, err := drain(t, request.Stream(nil, eventSource(
		`{"type":"message_start","message":{"role":"assistant","content":[],"usage":{"input_tokens":5}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":1}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"b","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		`{"type":"message_delta","delta":{"stop_reason":"refusal"},"usage":{"output_tokens":9}}`,
		`{"type":"message_stop"}`,
	), func() error { return nil }, func(err error) error { return err }))
	if err != nil {
		t.Fatal(err)
	}
	if completion.FinishReason != "refusal" || len(completion.ToolUse) != 1 || completion.ToolUse[0].ID != "a" {
		t.Fatalf("completion = %+v", completion.Completion)
	}
	stored := completion.Conversation.(claudePrompt)
	if last := stored.Messages[len(stored.Messages)-1]; len(last.Content) != 2 {
		t.Fatalf("the conversation keeps the refused turn: %#v", last)
	}
}

func TestClaudeStreamErrors(t *testing.T) {
	t.Parallel()

	request, err := NewRequest(userPrompt("Hi"), ExecutionOptions{Model: "claude-sonnet-4-5"}, RequestSettings{Provider: ProviderNameVertexAI})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = drain(t, request.Stream(nil, eventSource(
		`{"type":"message_start","message":{"role":"assistant","content":[],"usage":{"input_tokens":5}}}`,
		`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
	), func() error { return nil }, func(err error) error { return err }))
	var llumiErr *common.LlumiverseError
	if !errors.As(err, &llumiErr) || llumiErr.Retryable == nil || !*llumiErr.Retryable || !strings.Contains(llumiErr.Message, "overloaded_error: Overloaded") || llumiErr.Context.Provider != common.ProviderVertexAI {
		t.Fatalf("error = %#v", err)
	}

	broken := func(yield func([]byte) error) error {
		if err := yield([]byte(`{"type":"message_start","message":{"role":"assistant","content":[],"usage":{}}}`)); err != nil {
			return err
		}
		if err := yield([]byte(`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)); err != nil {
			return err
		}
		if err := yield([]byte(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`)); err != nil {
			return err
		}
		return errors.New("connection reset")
	}
	stream := request.Stream(nil, broken, func() error { return nil }, func(err error) error { return err })
	completion, _, err := drain(t, stream)
	if !errors.As(err, &llumiErr) || llumiErr.Retryable == nil || !*llumiErr.Retryable {
		t.Fatalf("a failure after the stream started is retryable: %#v", err)
	}
	// The stream never completed, so the conversation is rebuilt from the streamed text.
	conversation := completion.Conversation.(claudePrompt)
	if last := conversation.Messages[len(conversation.Messages)-1]; last.Role != "assistant" || last.Content[0].Text != "partial" {
		t.Fatalf("conversation = %#v", conversation.Messages)
	}
}
