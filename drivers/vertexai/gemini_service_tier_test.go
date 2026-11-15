package vertexai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// TestGeminiServiceTier mirrors the resolveVertexAIServiceTier and
// normalizeVertexAIResolvedServiceTier cases of gemini-conversation-mutation.test.ts.
func TestGeminiServiceTier(t *testing.T) {
	t.Parallel()

	if got := geminiServiceTier(map[string]any{"flex": true}); got != "flex" {
		t.Fatalf("flex alias = %q", got)
	}
	if got := geminiServiceTier(map[string]any{"flex": false}); got != "" {
		t.Fatalf("flex false = %q", got)
	}
	if got := geminiServiceTier(map[string]any{"service_tier": "future-tier", "flex": true}); got != "future-tier" {
		t.Fatalf("service_tier precedence = %q", got)
	}
	for trafficType, want := range map[string]string{"ON_DEMAND": "default", "ON_DEMAND_PRIORITY": "priority", "ON_DEMAND_FLEX": "flex", "PROVISIONED_THROUGHPUT": "provisioned", "TRAFFIC_TYPE_UNSPECIFIED": ""} {
		if got := normalizeVertexServiceTier(trafficType); got != want {
			t.Errorf("%s = %q, want %q", trafficType, got, want)
		}
	}
}

// TestGeminiServiceTierHeadersAndResolvedTier checks the shared request-type
// headers and that the tier Vertex reports reaches the completion and the stream.
func TestGeminiServiceTierHeadersAndResolvedTier(t *testing.T) {
	t.Parallel()

	var (
		mu      sync.Mutex
		headers []http.Header
	)
	body := `{"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2,"trafficType":"ON_DEMAND_FLEX"},"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"text":"ok"}]}}]}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		headers = append(headers, r.Header.Clone())
		mu.Unlock()
		if strings.Contains(r.URL.Path, "streamGenerateContent") {
			_, _ = w.Write([]byte("data: " + body + "\n\n"))
			return
		}
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()
	driver := newTestVertexDriver(t, server.URL)
	segments := []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}

	completion, err := driver.Execute(context.Background(), segments, ExecutionOptions{Model: "gemini-2.5-flash", ModelOptions: map[string]any{"flex": true}})
	if err != nil {
		t.Fatal(err)
	}
	if completion.ServiceTier != "flex" {
		t.Fatalf("completion service tier = %q", completion.ServiceTier)
	}
	stream, err := driver.Stream(context.Background(), segments, ExecutionOptions{Model: "gemini-2.5-flash", ModelOptions: map[string]any{"service_tier": "priority"}})
	if err != nil {
		t.Fatal(err)
	}
	if streamed := drainStream(t, stream); streamed.ServiceTier != "flex" {
		t.Fatalf("stream service tier = %q", streamed.ServiceTier)
	}
	if _, err := driver.Execute(context.Background(), segments, ExecutionOptions{Model: "gemini-2.5-flash", ModelOptions: map[string]any{"service_tier": "default"}}); err != nil {
		t.Fatal(err)
	}

	mu.Lock()
	defer mu.Unlock()
	for i, want := range []string{"flex", "priority", ""} {
		requestType, shared := headers[i].Get("X-Vertex-AI-LLM-Request-Type"), headers[i].Get("X-Vertex-AI-LLM-Shared-Request-Type")
		if want == "" {
			if requestType != "" || shared != "" {
				t.Errorf("request %d: unexpected tier headers %q / %q", i, requestType, shared)
			}
			continue
		}
		if requestType != "shared" || shared != want {
			t.Errorf("request %d: tier headers %q / %q, want shared / %s", i, requestType, shared, want)
		}
	}
}
