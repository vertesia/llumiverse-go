package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
	"golang.org/x/oauth2"
)

// claudeTestDriver is a Vertex AI driver on a test server, with a project unique to the test so structured-output
// denials do not leak between tests.
func claudeTestDriver(t *testing.T, project string, handler http.HandlerFunc) *VertexAIDriver {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     project,
		Region:      "global",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	return driver
}

// closedSchema is a result schema within the structured-output subset.
var closedSchema = map[string]any{
	"type":                 "object",
	"properties":           map[string]any{"answer": map[string]any{"type": "string"}},
	"required":             []any{"answer"},
	"additionalProperties": false,
}

const policyDenial = `{"error":{"code":400,"message":"Organization Policy constraint constraints/vertexai.allowedPartnerModelFeatures violated for a disallowed feature structured_outputs","status":"FAILED_PRECONDITION"}}`

func TestVertexClaudeRoutesABareClaudeID(t *testing.T) {
	t.Parallel()

	var path string
	driver := claudeTestDriver(t, "bare-id", func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_, _ = w.Write([]byte(`{"role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	prompt, err := driver.CreatePrompt(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{Model: "claude-sonnet-4-5"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := prompt.(internalclaude.Prompt); !ok {
		t.Fatalf("prompt = %T", prompt)
	}
	if _, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{Model: "claude-sonnet-4-5"}); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(path, "/locations/global/publishers/anthropic/models/claude-sonnet-4-5:rawPredict") {
		t.Fatalf("path = %s", path)
	}
}

func TestVertexClaudeSendsNativeResultSchemas(t *testing.T) {
	t.Parallel()

	var got map[string]any
	driver := claudeTestDriver(t, "native-schema", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"role":"assistant","content":[{"type":"text","text":"{\"answer\":\"yes\"}"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Answer"}}, ExecutionOptions{Model: "claude-sonnet-4-5@20250929", ResultSchema: closedSchema})
	if err != nil {
		t.Fatal(err)
	}
	format, _ := got["output_config"].(map[string]any)["format"].(map[string]any)
	if format["type"] != "json_schema" || got["system"] != nil || got["model"] != nil || got["anthropic_version"] != vertexAnthropicVersion {
		t.Fatalf("body = %#v", got)
	}
	if resp.Result[0].String() == "" {
		t.Fatalf("result = %#v", resp.Result)
	}
}

func TestVertexClaudeFallsBackAfterAStructuredOutputDenial(t *testing.T) {
	t.Parallel()

	var mu sync.Mutex
	var bodies []map[string]any
	driver := claudeTestDriver(t, "denied-execute", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		if body["output_config"] != nil {
			http.Error(w, policyDenial, http.StatusBadRequest)
			return
		}
		_, _ = w.Write([]byte(`{"role":"assistant","content":[{"type":"text","text":"{\"answer\":\"yes\"}"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	options := ExecutionOptions{Model: "claude-sonnet-4-5", ResultSchema: closedSchema}
	segments := []PromptSegment{{Role: PromptRoleUser, Content: "Answer"}}
	if _, err := driver.Execute(context.Background(), segments, options); err != nil {
		t.Fatal(err)
	}
	if len(bodies) != 2 || bodies[0]["output_config"] == nil || bodies[1]["output_config"] != nil {
		t.Fatalf("bodies = %#v", bodies)
	}
	system := bodies[1]["system"].([]any)
	if !strings.Contains(system[0].(map[string]any)["text"].(string), "JSON Schema") {
		t.Fatalf("the fallback carries the instructions: %#v", system)
	}
	// The denial is remembered: the next request goes straight to the prompt path, with the schema in its prompt.
	prompt, err := driver.CreatePrompt(context.Background(), segments, options)
	if err != nil {
		t.Fatal(err)
	}
	if len(prompt.(internalclaude.Prompt).System) != 1 {
		t.Fatalf("prompt = %#v", prompt)
	}
	if _, err := driver.Execute(context.Background(), segments, options); err != nil {
		t.Fatal(err)
	}
	if len(bodies) != 3 || bodies[2]["output_config"] != nil {
		t.Fatalf("bodies = %d", len(bodies))
	}
	// Another 400 is not a denial.
	other := claudeTestDriver(t, "other-400", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"max_tokens: too large"}}`, http.StatusBadRequest)
	})
	if _, err := other.Execute(context.Background(), segments, options); err == nil {
		t.Fatal("expected the 400")
	}
}

func TestVertexClaudeStreamFallsBackAfterAStructuredOutputDenial(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32
	driver := claudeTestDriver(t, "denied-stream", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		calls.Add(1)
		if body["output_config"] != nil {
			http.Error(w, policyDenial, http.StatusBadRequest)
			return
		}
		if body["stream"] != true || r.URL.Query().Get("alt") != "sse" {
			t.Errorf("stream request = %#v %s", body, r.URL)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range []string{
			`{"type":"message_start","message":{"role":"assistant","content":[],"usage":{"input_tokens":3}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\"answer\":\"yes\"}"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
			`{"type":"message_stop"}`,
		} {
			_, _ = w.Write([]byte("data: " + event + "\n\n"))
		}
	})
	stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Answer"}}, ExecutionOptions{Model: "claude-sonnet-4-5", ResultSchema: closedSchema})
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := stream.Recv(); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
	}
	completion := stream.Completion()
	if calls.Load() != 2 || completion.FinishReason != "stop" || completion.Conversation == nil {
		t.Fatalf("calls = %d, completion = %#v", calls.Load(), completion.Completion)
	}
}

func TestVertexClaudeFastModeAndBetaHeader(t *testing.T) {
	t.Parallel()

	var got map[string]any
	var betas []string
	driver := claudeTestDriver(t, "fast-mode", func(w http.ResponseWriter, r *http.Request) {
		betas = r.Header.Values("anthropic-beta")
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1,"speed":"fast"}}`))
	})
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{
		Model:        "claude-fable-5-2",
		ModelOptions: map[string]any{"speed": "fast"},
		Compaction:   &common.ConversationCompactionOptions{TriggerTokens: 50000},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got["speed"] != "fast" || resp.ServiceTier != "fast" {
		t.Fatalf("body = %#v", got)
	}
	// One comma-separated header: Google Cloud reads only the first anthropic-beta header.
	if len(betas) != 1 || betas[0] != "compact-2026-01-12,thinking-binding-controls-2026-08-01" && betas[0] != "thinking-binding-controls-2026-08-01,compact-2026-01-12" {
		t.Fatalf("anthropic-beta = %v", betas)
	}
	if !internalclaude.SupportsVertexAIFastMode("claude-fable-5-2") || internalclaude.SupportsVertexAIFastMode("claude-opus-5-5") {
		t.Fatal("Vertex AI fast mode is Fable and Mythos 5.2+")
	}
}
