package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

// Mirrors the TS Responses compaction tests: context_management carries the
// threshold, and the stored conversation starts at the latest compaction item.
func TestOpenAIResponsesCompaction(t *testing.T) {
	t.Parallel()

	conversation := []responseItem{
		{"type": "message", "role": "user", "content": "old"},
		{"type": "compaction", "id": "cmp_1", "encrypted_content": "summary"},
		{"type": "message", "role": "assistant", "content": "later"},
	}
	body, resp, err := executeResponses(t, ExecutionOptions{Conversation: conversation, Compaction: &common.ConversationCompactionOptions{TriggerTokens: 50000, Instructions: "ignored"}},
		`{"id":"r","status":"completed","output":[{"type":"compaction","id":"cmp_2","encrypted_content":"newer"},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`)
	if err != nil {
		t.Fatal(err)
	}
	if got := marshalNoEscape(body["context_management"]); got != `[{"compact_threshold":50000,"type":"compaction"}]` {
		t.Fatalf("context_management = %s", got)
	}
	state := resp.Conversation.(openAIResponsesConversationState)
	if len(state.Items) != 2 || state.Items[0]["id"] != "cmp_2" {
		t.Fatalf("conversation = %s", marshalNoEscape(state.Items))
	}

	body, _, err = executeResponses(t, ExecutionOptions{Compaction: &common.ConversationCompactionOptions{}}, okResponse)
	if err != nil || marshalNoEscape(body["context_management"]) != `[{"type":"compaction"}]` {
		t.Fatalf("default threshold: %v, %#v", err, body["context_management"])
	}
}

// Mirrors the TS on-demand compaction tests: responses/compact gets the
// conversation and cache key, and its output replaces the conversation with no
// answer, executed or streamed.
func TestOpenAIResponsesCompactConversation(t *testing.T) {
	t.Parallel()

	var paths, bodies []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		paths = append(paths, r.URL.Path)
		bodies = append(bodies, string(data))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"output":[{"type":"message","role":"user","content":[{"type":"input_text","text":"first"}]},{"type":"compaction","id":"cmp","encrypted_content":"opaque"}],"usage":{"input_tokens":40,"output_tokens":5,"total_tokens":45}}`))
	}))
	defer server.Close()
	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "k", Endpoint: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	options := ExecutionOptions{
		Model:               "gpt-5.6",
		PromptCacheKey:      "tenant",
		Conversation:        []responseItem{{"type": "message", "role": "user", "content": "first"}, {"type": "message", "role": "assistant", "content": "reply"}},
		CompactConversation: &common.CompactConversationOptions{},
	}
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "next"}}, options)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(paths[0], "/responses/compact") || !strings.Contains(bodies[0], `"prompt_cache_key":"tenant"`) || !strings.Contains(bodies[0], `"next"`) {
		t.Fatalf("request %s %s", paths[0], bodies[0])
	}
	state := resp.Conversation.(openAIResponsesConversationState)
	if resp.FinishReason != "compaction" || len(resp.Result) != 0 || len(state.Items) != 2 || state.Items[1]["type"] != "compaction" || count(resp.TokenUsage.Prompt) != 40 {
		t.Fatalf("response %#v, conversation %s", resp, marshalNoEscape(state.Items))
	}

	stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "next"}}, options)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()
	for {
		if _, err := stream.Recv(); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
	}
	if streamed := stream.Completion(); streamed.FinishReason != "compaction" || streamed.Conversation == nil {
		t.Fatalf("stream completion = %#v", streamed)
	}
}

// Mirrors the TS conversation-shape tests: a conversation of an unknown shape
// is refused as a non-retryable 400 before any request.
func TestOpenAIResponsesRefusesInvalidConversation(t *testing.T) {
	t.Parallel()

	for _, conversation := range []any{"text", 42, map[string]any{"messages": []any{}}} {
		_, _, err := executeResponses(t, ExecutionOptions{Conversation: conversation}, okResponse)
		var llumiverseErr *LlumiverseError
		if !errors.As(err, &llumiverseErr) || llumiverseErr.Code != 400 || llumiverseErr.Name != "InvalidConversationError" || llumiverseErr.Retryable == nil || *llumiverseErr.Retryable {
			t.Errorf("%#v: error = %#v", conversation, err)
		}
	}
	if _, err := streamResponses(t, ExecutionOptions{Conversation: "text"}); err == nil || !strings.Contains(err.Error(), "must be an array") {
		t.Fatalf("stream error = %v", err)
	}
}
