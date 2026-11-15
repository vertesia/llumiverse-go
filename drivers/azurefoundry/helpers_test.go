package azurefoundry

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
)

// recordedRequest is one request the fake Azure server received.
type recordedRequest struct {
	Method string
	Path   string
	Query  url.Values
	Header http.Header
	Body   map[string]any
}

// fakeAzure is an httptest server that answers like Foundry and Azure OpenAI
// and records every request. respond overrides the default answers.
type fakeAzure struct {
	t           *testing.T
	server      *httptest.Server
	mu          sync.Mutex
	requests    []recordedRequest
	deployments map[string]foundryDeployment
	respond     func(w http.ResponseWriter, r recordedRequest) bool
}

const responsesBody = `{
	"id":"response-1","object":"response","created_at":1,"model":"chat","status":"completed",
	"output":[{"id":"message-1","type":"message","role":"assistant","status":"completed",
		"content":[{"type":"output_text","text":"Hello","annotations":[]}]}],
	"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}
}`

const chatBody = `{
	"id":"chat-1","object":"chat.completion","created":1,"model":"chat",
	"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Hello"}}],
	"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}
}`

const claudeBody = `{
	"id":"msg-test","type":"message","role":"assistant","model":"claude",
	"content":[{"type":"text","text":"Green"}],"stop_reason":"end_turn",
	"usage":{"input_tokens":2,"output_tokens":1}
}`

func claudeStreamBody() string {
	events := []string{
		`{"type":"message_start","message":{"id":"msg-1","type":"message","role":"assistant","model":"claude","content":[],"usage":{"input_tokens":2,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Green"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
		`{"type":"message_stop"}`,
	}
	var b strings.Builder
	for _, event := range events {
		var envelope struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(event), &envelope)
		fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", envelope.Type, event)
	}
	return b.String()
}

func newFakeAzure(t *testing.T) *fakeAzure {
	t.Helper()
	f := &fakeAzure{t: t, deployments: map[string]foundryDeployment{}}
	f.server = httptest.NewServer(http.HandlerFunc(f.handle))
	t.Cleanup(f.server.Close)
	return f
}

func (f *fakeAzure) handle(w http.ResponseWriter, r *http.Request) {
	data, _ := io.ReadAll(r.Body)
	request := recordedRequest{Method: r.Method, Path: r.URL.Path, Query: r.URL.Query(), Header: r.Header.Clone()}
	if len(data) > 0 {
		_ = json.Unmarshal(data, &request.Body)
	}
	f.mu.Lock()
	f.requests = append(f.requests, request)
	respond := f.respond
	f.mu.Unlock()
	if respond != nil && respond(w, request) {
		return
	}
	path := r.URL.Path
	switch {
	case strings.Contains(path, "/deployments/"):
		name := path[strings.LastIndex(path, "/")+1:]
		deployment, ok := f.deployments[name]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			_, _ = io.WriteString(w, `{"error":{"code":"NotFound","message":"deployment not found"}}`)
			return
		}
		writeJSON(w, deployment)
	case strings.HasSuffix(path, "/deployments"):
		values := make([]foundryDeployment, 0, len(f.deployments))
		for _, deployment := range f.deployments {
			values = append(values, deployment)
		}
		writeJSON(w, map[string]any{"value": values})
	case strings.HasSuffix(path, "/anthropic/v1/messages"):
		if request.Body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, claudeStreamBody())
			return
		}
		writeRaw(w, claudeBody)
	case strings.HasSuffix(path, "/responses"):
		writeRaw(w, responsesBody)
	case strings.HasSuffix(path, "/chat/completions"):
		writeRaw(w, chatBody)
	case strings.HasSuffix(path, "/models"):
		writeRaw(w, `{"object":"list","data":[]}`)
	case strings.HasSuffix(path, "/images/generations"):
		writeRaw(w, `{"created":1,"data":[{"b64_json":"YQ=="}]}`)
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func writeRaw(w http.ResponseWriter, body string) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, body)
}

func (f *fakeAzure) recorded() []recordedRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]recordedRequest(nil), f.requests...)
}

// inference returns the recorded requests other than the project API's.
func (f *fakeAzure) inference() []recordedRequest {
	var out []recordedRequest
	for _, request := range f.recorded() {
		if !strings.Contains(request.Path, "/deployments") {
			out = append(out, request)
		}
	}
	return out
}

func (f *fakeAzure) deploymentLookups() int {
	count := 0
	for _, request := range f.recorded() {
		if strings.Contains(request.Path, "/deployments") {
			count++
		}
	}
	return count
}

func (f *fakeAzure) project() string {
	return f.server.URL + "/api/projects/project"
}

// tokenRecorder is a TokenProvider that records the scopes it was asked for.
type tokenRecorder struct {
	mu     sync.Mutex
	scopes []string
}

func (r *tokenRecorder) provider() TokenProvider {
	return func(_ context.Context, scope string) (string, error) {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.scopes = append(r.scopes, scope)
		return "test-token", nil
	}
}

func (r *tokenRecorder) calls() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.scopes...)
}

var helloSegments = []PromptSegment{{Role: PromptRoleUser, Content: "Hello"}}

func expectAPIKeyHeaders(t *testing.T, request recordedRequest) {
	t.Helper()
	if got := request.Header.Get("api-key"); got != "test-key" {
		t.Fatalf("api-key header = %q", got)
	}
	if got := request.Header.Get("Authorization"); got != "" {
		t.Fatalf("Authorization header = %q, want none", got)
	}
}

func resultText(results []CompletionResult) string {
	var b strings.Builder
	for _, result := range results {
		b.WriteString(result.String())
	}
	return b.String()
}

func drain(t *testing.T, stream CompletionStream) []CompletionChunk {
	t.Helper()
	var chunks []CompletionChunk
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	return chunks
}
