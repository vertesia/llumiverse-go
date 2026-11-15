package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"golang.org/x/oauth2"
)

// Mirrors the TS open-maas-models routing tests.
func TestVertexMaaSRoute(t *testing.T) {
	t.Parallel()

	cases := []struct {
		model, modelName, region, apiVersion string
		maxTokens                            int
	}{
		{"locations/us-east5/publishers/meta/models/llama-4-maverick-17b-128e", "meta/llama-4-maverick-17b-128e-instruct-maas", "us-east5", "v1beta1", 0},
		{"publishers/meta/models/llama-4-scout-17b-16e-instruct-maas", "meta/llama-4-scout-17b-16e-instruct-maas", "us-east5", "v1beta1", 8192},
		{"locations/global/publishers/openai/models/gpt-oss-120b-maas", "openai/gpt-oss-120b-maas", "global", "", 16384},
		{"publishers/deepseek-ai/models/deepseek-v4.1-flash-maas", "deepseek-ai/deepseek-v4.1-flash-maas", "global", "", 0},
		{"locations/global/publishers/zaiorg/models/glm-5.3-maas", "zai-org/glm-5.3-maas", "global", "", 0},
		{"locations/us/publishers/xai/models/grok-4.7", "xai/grok-4.7", "us", "", 0},
		{"publishers/xai/models/grok-5", "xai/grok-5", "global", "", 0},
	}
	for _, tc := range cases {
		route, ok := vertexMaaSRoute(tc.model)
		if !ok || route.modelName != tc.modelName || route.region != tc.region || route.apiVersion != tc.apiVersion || route.defaultMaxTokens != tc.maxTokens {
			t.Errorf("%s: route = %#v, %v", tc.model, route, ok)
		}
	}
	for _, model := range []string{"gemini-2.5-pro", "publishers/google/models/gemini-3-pro", "publishers/qwen/models/qwen3-unlisted", "publishers/anthropic/models/claude-opus-5"} {
		if route, ok := vertexMaaSRoute(model); ok {
			t.Errorf("%s routed to MaaS: %#v", model, route)
		}
	}
	if got := (&VertexAIDriver{}).maaSBaseURL("us", "v1"); got != "https://aiplatform.us.rep.googleapis.com/v1" {
		t.Errorf("us base = %s", got)
	}
}

func TestVertexMaaSExecuteAndStream(t *testing.T) {
	t.Parallel()

	var paths []string
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		if r.Header.Get("Authorization") != "Bearer vertex-token" {
			t.Errorf("authorization = %q", r.Header.Get("Authorization"))
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		bodies = append(bodies, body)
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"streamed\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"finish_reason":"stop","message":{"content":"hello"}}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`)
	}))
	defer server.Close()
	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "p",
		Region:      "us-central1",
		BaseURL:     server.URL + "/v1",
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	segments := []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}
	options := ExecutionOptions{Model: "publishers/meta/models/llama-3.3-70b-instruct-maas", Tools: []ToolDefinition{{Name: "lookup", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}}}}
	resp, err := driver.Execute(context.Background(), segments, options)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Result[0].String() != "hello" || resp.TokenUsage == nil {
		t.Fatalf("response = %#v", resp)
	}
	stream, err := driver.Stream(context.Background(), segments, options)
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
	_ = stream.Close()
	if got := stream.Completion().Result[0].String(); got != "streamed" {
		t.Fatalf("stream text = %q", got)
	}
	for _, path := range paths {
		if path != "/v1beta1/projects/p/locations/us-central1/endpoints/openapi/chat/completions" {
			t.Fatalf("path = %s", path)
		}
	}
	body := bodies[0]
	extra, _ := json.Marshal(body["extra_body"])
	if body["model"] != "meta/llama-3.3-70b-instruct-maas" || !strings.Contains(string(extra), `"model_safety_settings"`) {
		t.Fatalf("body = %#v", body)
	}
	tool, _ := json.Marshal(body["tools"])
	if strings.Contains(string(tool), `"strict"`) {
		t.Fatalf("MaaS tools must use plain schemas: %s", tool)
	}
}
