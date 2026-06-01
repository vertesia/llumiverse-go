package core

import (
	"errors"
	"io"
	"reflect"
	"testing"
)

func TestConversationRetentionHelpers(t *testing.T) {
	t.Parallel()

	if ShouldStripConversationMedia(10, nil) {
		t.Fatal("nil media retention should never strip")
	}
	keepTwo := 2
	if ShouldStripConversationMedia(1, &keepTwo) {
		t.Fatal("media stripped before keep window")
	}
	if !ShouldStripConversationMedia(2, &keepTwo) {
		t.Fatal("media not stripped at keep window")
	}
	if ShouldStripConversationHeartbeats(0, nil) {
		t.Fatal("default heartbeat retention stripped current turn")
	}
	if !ShouldStripConversationHeartbeats(1, nil) {
		t.Fatal("default heartbeat retention did not strip next turn")
	}
	zero := 0
	if !ShouldStripConversationHeartbeats(0, &zero) {
		t.Fatal("zero heartbeat retention should strip immediately")
	}
	text := ProcessConversationText("<heartbeat>working</heartbeat>", ExecutionOptions{}, true)
	if text != ConversationHeartbeatPlaceholder {
		t.Fatalf("heartbeat text = %q", text)
	}
	text = ProcessConversationText("abcdefghijkl", ExecutionOptions{StripTextMaxTokens: 2}, false)
	if text != "abcdefgh"+ConversationTextTruncatedMarker {
		t.Fatalf("truncated text = %q", text)
	}
}

func TestEmbeddingNormalization(t *testing.T) {
	t.Parallel()

	if _, err := NormalizeEmbeddingsOptions(EmbeddingsOptions{}); err == nil {
		t.Fatal("expected empty input error")
	}
	normalized, err := NormalizeEmbeddingsOptions(EmbeddingsOptions{
		TaskType: EmbeddingTaskQuery,
		Inputs: []EmbeddingInput{
			{Text: "query"},
			{Type: EmbeddingInputImage, TaskType: EmbeddingTaskDocument},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if normalized.Inputs[0].Type != EmbeddingInputText || normalized.Inputs[0].TaskType != EmbeddingTaskQuery {
		t.Fatalf("text input = %#v", normalized.Inputs[0])
	}
	if normalized.Inputs[1].TaskType != EmbeddingTaskDocument {
		t.Fatalf("image input task type changed: %#v", normalized.Inputs[1])
	}
}

func TestResultAndMapHelpers(t *testing.T) {
	t.Parallel()

	in := map[string]any{
		"keep": "x",
		"drop": nil,
		"nested": map[string]any{
			"drop": nil,
			"keep": 1,
		},
		"empty": map[string]any{"drop": nil},
	}
	got := RemoveNilMapValues(in)
	want := map[string]any{"keep": "x", "nested": map[string]any{"keep": 1}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("RemoveNilMapValues = %#v", got)
	}

	completion := &Completion{Result: []CompletionResult{
		{Type: ResultTypeText, Value: "not json"},
		{Type: ResultTypeText, Value: "```json\n{\"ok\":true}\n```"},
	}}
	NormalizeJSONResult(completion)
	if completion.Result[0].Type != ResultTypeText || completion.Result[1].Type != ResultTypeJSON {
		t.Fatalf("NormalizeJSONResult = %#v", completion.Result)
	}
	text := CompletionResultsToText([]CompletionResult{
		{Type: ResultTypeText, Value: "hello"},
		{Type: ResultTypeJSON, Value: map[string]any{"ok": true}},
		{Type: ResultTypeImage, Value: "ignored"},
	})
	if text != `hello{"ok":true}` {
		t.Fatalf("CompletionResultsToText = %q", text)
	}
	if TruncateForConversation("abcdef", 3) != "abc..." {
		t.Fatal("TruncateForConversation did not append ellipsis")
	}
	if TruncateForConversation("abcdef", 0) != "abcdef" {
		t.Fatal("TruncateForConversation should not cut when max length is disabled")
	}
	if text, err := ReadAllString(io.NopCloser(assertionReader{err: errors.New("read failed")})); err == nil || text != "" {
		t.Fatalf("ReadAllString error path = %q, %v", text, err)
	}
}

func TestClaudeModelVersionHelpers(t *testing.T) {
	t.Parallel()

	if !IsClaudeVersionGTE("claude-3-7-sonnet-20250219", 3, 7) {
		t.Fatal("legacy Claude version was not parsed")
	}
	if !ClaudeSupportsAdaptiveThinking("claude-sonnet-4-6-20260601") {
		t.Fatal("Claude 4.6 Sonnet should support adaptive thinking")
	}
	if ClaudeSupportsAdaptiveThinking("claude-haiku-4-6-20260601") {
		t.Fatal("Claude Haiku should not support adaptive thinking")
	}
	if !ClaudeHasSamplingParameterRestriction("claude-opus-4-7-20260601") {
		t.Fatal("Claude 4.7 should have sampling restriction")
	}
	if ClaudeHasSamplingParameterRestriction("not-claude") {
		t.Fatal("unparseable model should not have sampling restriction")
	}
}

func TestRetryableAndGenericErrorHelpers(t *testing.T) {
	t.Parallel()

	if v := RetryableFromStatusAndMessage(429, ""); v == nil || !*v {
		t.Fatalf("429 retryable = %#v", v)
	}
	if v := RetryableFromStatusAndMessage(400, "invalid request"); v == nil || *v {
		t.Fatalf("400 retryable = %#v", v)
	}
	if v := RetryableFromStatusAndMessage(0, "rate limit exceeded"); v == nil || !*v {
		t.Fatalf("rate limit retryable = %#v", v)
	}
	if v := RetryableFromStatusAndMessage(0, "plain error"); v != nil {
		t.Fatalf("plain retryable = %#v", v)
	}
	err := FormatGenericError(assertionError("boom"), LlumiverseErrorContext{Provider: ProviderOpenAI, Model: "m", Operation: "execute"})
	llumiErr, ok := err.(*LlumiverseError)
	if !ok {
		t.Fatalf("FormatGenericError type = %T", err)
	}
	if llumiErr.Context.Provider != ProviderOpenAI || llumiErr.Context.Model != "m" {
		t.Fatalf("FormatGenericError = %#v", llumiErr)
	}
	if FormatGenericError(nil, LlumiverseErrorContext{}) != nil {
		t.Fatal("FormatGenericError(nil) should return nil")
	}
}

type assertionError string

func (e assertionError) Error() string { return string(e) }

type assertionReader struct {
	err error
}

func (r assertionReader) Read([]byte) (int, error) {
	return 0, r.err
}
