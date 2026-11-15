package vertexai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"golang.org/x/oauth2"
)

// TestGeminiRegionResolution mirrors gemini-region.test.ts.
func TestGeminiRegionResolution(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ model, explicit, want string }{
		{"gemini-4-argon", "", "global"},
		{"gemini-4-argon", "us", "us"},
		{"gemini-4-argon", "eu", "eu"},
		{"gemini-4-tts", "", ""},
		{"gemini-3.8-flash", "", ""},
		{"gemini-3.8-flash", "us-central1", "us-central1"},
		{"gemini-nano-banana-2.1", "us-central1", "global"},
		{"gemini-2.5-flash-image", "", ""},
		{"gemini-2.5-flash-image", "us-central1", "us-central1"},
	} {
		if got := resolveGeminiRegion(tc.model, tc.explicit); got != tc.want {
			t.Errorf("resolveGeminiRegion(%q, %q) = %q, want %q", tc.model, tc.explicit, got, tc.want)
		}
	}
}

// TestGeminiEndpointRouting checks the request URL: v1beta1 on the regional or
// global host, Nano Banana routed globally even from a named region (the "routes
// image requests globally" cases of gemini-conversation-mutation.test.ts).
func TestGeminiEndpointRouting(t *testing.T) {
	t.Parallel()

	driver, err := NewVertexAIDriver(VertexAIOptions{Project: "p", Region: "us-central1", TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "t"})})
	if err != nil {
		t.Fatal(err)
	}
	for model, want := range map[string]string{
		"gemini-2.5-flash": "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent",
		"gemini-4-argon":   "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/publishers/google/models/gemini-4-argon:generateContent",
		"locations/us/publishers/google/models/gemini-4-argon":                  "https://us-aiplatform.googleapis.com/v1beta1/projects/p/locations/us/publishers/google/models/gemini-4-argon:generateContent",
		"locations/us-central1/publishers/google/models/gemini-nano-banana-2.1": "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/publishers/google/models/gemini-nano-banana-2.1:generateContent",
	} {
		got, err := driver.geminiEndpoint(model, "generateContent")
		if err != nil || got != want {
			t.Errorf("%s: endpoint = %q (%v), want %q", model, got, err, want)
		}
	}
}

// TestVertexListingFiltersSpecialPurposeModels mirrors the embedding (420fd65) and
// Live / Nano Banana (98f733a) listing cases of index.test.ts.
func TestVertexListingFiltersSpecialPurposeModels(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/publishers/anthropic/models") {
			_, _ = w.Write([]byte(`{"publisherModels":[]}`))
			return
		}
		_, _ = w.Write([]byte(`{"publisherModels":[` +
			`{"name":"publishers/google/models/gemini-2.5-flash"},` +
			`{"name":"publishers/google/models/gemini-4-argon"},` +
			`{"name":"publishers/google/models/gemini-nano-banana-2.1"},` +
			`{"name":"publishers/google/models/gemini-live-future"},` +
			`{"name":"publishers/google/models/gemini-3.8-live"},` +
			`{"name":"publishers/google/models/gemini-4-live-preview"},` +
			`{"name":"publishers/google/models/gemini-embedding-001"},` +
			`{"name":"publishers/google/models/text-embedding-005"},` +
			`{"name":"publishers/google/models/gemini-2.5-flash-native-audio"},` +
			`{"name":"publishers/google/models/gemini-robotics-er"},` +
			`{"name":"publishers/google/models/gemini-2.5-computer-use-preview"},` +
			`{"name":"publishers/google/models/imagen-4.0-generate-001"},` +
			`{"name":"publishers/google/models/veo-3.0-generate"}]}`))
	}))
	defer server.Close()
	driver := newTestVertexDriver(t, server.URL)
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	want := []string{
		"locations/global/publishers/google/models/gemini-2.5-flash",
		"locations/global/publishers/google/models/gemini-4-argon",
		"locations/global/publishers/google/models/gemini-nano-banana-2.1",
		"publishers/google/models/gemini-2.5-flash",
		"publishers/google/models/imagen-4.0-generate-001",
	}
	if strings.Join(ids, ",") != strings.Join(want, ",") {
		t.Fatalf("listed = %v, want %v", ids, want)
	}
}

// longDecodeHops mirrors the two hops of gemini-long-decode.test.ts: the first
// stops at the per-request cap with a continuation token, the second finishes.
var longDecodeHops = []string{
	`{"candidates":[{"content":{"role":"model","parts":[{"text":"part one "}]},"finishReason":"CONTINUATION","continuationToken":"tok-1"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":100,"totalTokenCount":110}}`,
	`{"candidates":[{"content":{"role":"model","parts":[{"text":"part two"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":110,"candidatesTokenCount":50,"totalTokenCount":160}}`,
}

func startLongDecodeServer(t *testing.T, hops []string) (*httptest.Server, *[]map[string]any, *sync.Mutex) {
	t.Helper()
	var (
		mu     sync.Mutex
		bodies []map[string]any
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		hop := hops[0]
		if body["continuationToken"] == "tok-1" {
			hop = hops[1]
		}
		if strings.Contains(r.URL.Path, "streamGenerateContent") {
			_, _ = w.Write([]byte("data: " + hop + "\n\n"))
			return
		}
		_, _ = w.Write([]byte(hop))
	}))
	return server, &bodies, &mu
}

// TestGeminiLongDecodeContinues mirrors "continues across requests" of
// gemini-long-decode.test.ts.
func TestGeminiLongDecodeContinues(t *testing.T) {
	t.Parallel()

	for _, stream := range []bool{false, true} {
		server, bodies, mu := startLongDecodeServer(t, longDecodeHops)
		driver := newTestVertexDriver(t, server.URL)
		segments := []PromptSegment{{Role: PromptRoleUser, Content: "Write a long proof."}}
		options := ExecutionOptions{Model: "gemini-4-argon", ModelOptions: map[string]any{"max_tokens": 200_000}}
		var response *ExecutionResponse
		if stream {
			completionStream, err := driver.Stream(context.Background(), segments, options)
			if err != nil {
				t.Fatal(err)
			}
			response = drainStream(t, completionStream)
		} else {
			var err error
			if response, err = driver.Execute(context.Background(), segments, options); err != nil {
				t.Fatal(err)
			}
		}
		server.Close()

		var text strings.Builder
		for _, result := range response.Result {
			text.WriteString(result.String())
		}
		if text.String() != "part one part two" || response.FinishReason != "stop" {
			t.Fatalf("stream=%v: text %q, finish %q", stream, text.String(), response.FinishReason)
		}
		// Each request's usage is summed, which is what Vertex bills.
		usage := response.TokenUsage
		if usage == nil || *usage.Prompt != 120 || *usage.Result != 150 || *usage.Total != 270 {
			t.Fatalf("stream=%v: usage = %#v", stream, usage)
		}
		mu.Lock()
		if len(*bodies) != 2 {
			t.Fatalf("stream=%v: %d requests", stream, len(*bodies))
		}
		first, second := (*bodies)[0], (*bodies)[1]
		mu.Unlock()
		if _, ok := first["continuationToken"]; ok || second["continuationToken"] != "tok-1" {
			t.Fatalf("stream=%v: continuation tokens %v / %v", stream, first["continuationToken"], second["continuationToken"])
		}
		// The continuation resends the original request instead of appending the partial output.
		if a, _ := json.Marshal(first["contents"]); string(a) != mustJSON(t, second["contents"]) {
			t.Fatalf("stream=%v: continuation contents differ", stream)
		}
		if first["generationConfig"].(map[string]any)["maxOutputTokens"] != float64(200_000) {
			t.Fatalf("stream=%v: generationConfig = %#v", stream, first["generationConfig"])
		}
		if conversation := response.Conversation.(geminiPrompt); len(conversation.Contents) != 2 {
			t.Fatalf("stream=%v: conversation = %#v", stream, conversation)
		}
	}
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// TestGeminiLongDecodeWithoutTokenReturnsLength mirrors "returns the partial
// output as length when it cannot be continued".
func TestGeminiLongDecodeWithoutTokenReturnsLength(t *testing.T) {
	t.Parallel()

	partial := `{"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]},"finishReason":"CONTINUATION"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`
	for _, stream := range []bool{false, true} {
		server, bodies, mu := startLongDecodeServer(t, []string{partial, partial})
		driver := newTestVertexDriver(t, server.URL)
		segments := []PromptSegment{{Role: PromptRoleUser, Content: "Write a long proof."}}
		options := ExecutionOptions{Model: "gemini-4-argon", ModelOptions: map[string]any{"max_tokens": 200_000}}
		var response *ExecutionResponse
		if stream {
			completionStream, err := driver.Stream(context.Background(), segments, options)
			if err != nil {
				t.Fatal(err)
			}
			response = drainStream(t, completionStream)
		} else {
			var err error
			if response, err = driver.Execute(context.Background(), segments, options); err != nil {
				t.Fatal(err)
			}
		}
		server.Close()
		if len(response.Result) != 1 || response.Result[0].String() != "partial" || response.FinishReason != "length" {
			t.Fatalf("stream=%v: result %#v, finish %q", stream, response.Result, response.FinishReason)
		}
		mu.Lock()
		if len(*bodies) != 1 {
			t.Fatalf("stream=%v: %d requests", stream, len(*bodies))
		}
		mu.Unlock()
	}
}
