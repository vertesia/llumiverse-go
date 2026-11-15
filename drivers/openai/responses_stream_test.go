package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// streamResponses runs a Responses stream against SSE events and returns the
// completion and the error that ended the stream, if any.
func streamResponses(t *testing.T, options ExecutionOptions, events ...string) (*ExecutionResponse, error) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			writeOpenAISSE(t, w, event)
		}
	}))
	defer server.Close()
	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "k", Endpoint: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	if options.Model == "" {
		options.Model = "gpt-4.1"
	}
	stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, options)
	if err != nil {
		return nil, err
	}
	defer func() { _ = stream.Close() }()
	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return stream.Completion(), nil
		}
		if err != nil {
			return stream.Completion(), err
		}
	}
}

func completionText(results []CompletionResult) string {
	var b strings.Builder
	for _, result := range results {
		if result.Type() == ResultTypeText {
			b.WriteString(result.String())
		}
	}
	return b.String()
}

// Mirrors the TS fallback for models that send their text only in output_text.done.
func TestOpenAIResponsesStreamUsesTextDoneWithoutDeltas(t *testing.T) {
	t.Parallel()

	completed := `{"type":"response.completed","response":{"id":"r","status":"completed","output":[]}}`
	resp, err := streamResponses(t, ExecutionOptions{},
		`{"type":"response.output_text.done","text":"{\"answer\":1}"}`, completed)
	if err != nil {
		t.Fatal(err)
	}
	if got := completionText(resp.Result); got != `{"answer":1}` {
		t.Fatalf("text = %q", got)
	}

	resp, err = streamResponses(t, ExecutionOptions{},
		`{"type":"response.output_text.delta","delta":"hel"}`,
		`{"type":"response.output_text.delta","delta":"lo"}`,
		`{"type":"response.output_text.done","text":"hello"}`, completed)
	if err != nil {
		t.Fatal(err)
	}
	if got := completionText(resp.Result); got != "hello" {
		t.Fatalf("text with deltas = %q", got)
	}
}

func TestOpenAIResponsesStreamRefusalAndErrorEvents(t *testing.T) {
	t.Parallel()

	_, err := streamResponses(t, ExecutionOptions{},
		`{"type":"response.refusal.delta","delta":"I can't "}`,
		`{"type":"response.refusal.delta","delta":"help"}`,
		`{"type":"response.refusal.done","refusal":""}`)
	if err == nil || !strings.Contains(err.Error(), "Model refused: I can't help") {
		t.Fatalf("refusal error = %v", err)
	}

	_, err = streamResponses(t, ExecutionOptions{},
		`{"type":"response.error","code":"server_error","message":"boom"}`)
	if err == nil || !strings.Contains(err.Error(), "boom (server_error)") {
		t.Fatalf("error event = %v", err)
	}
}
