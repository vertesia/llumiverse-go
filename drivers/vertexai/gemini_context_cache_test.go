package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/wire"
)

// The tests in this file mirror gemini-context-cache.test.ts.

const (
	cacheTestModel = "gemini-3.7-flash"
	cacheTestName  = "projects/project-1/locations/us-central1/cachedContents/1"
	cacheTestPeer  = "projects/project-1/locations/us-central1/cachedContents/peer"
)

var cacheTestCatalog = strings.Repeat("catalog entry. ", 40)

// routePrompt is the shape the feature exists for: a stable system instruction,
// a large static catalog, and a per-photo turn that carries the image.
func routePrompt() geminiPrompt {
	return geminiPrompt{
		System: &geminiContent{Role: "user", Parts: []geminiPart{{Text: "Route each photo to a domain."}}},
		Contents: []geminiContent{
			{Role: "user", Parts: []geminiPart{{Text: cacheTestCatalog}}},
			{Role: "user", Parts: []geminiPart{{Text: "Photo 42: route it."}, {InlineData: &geminiInlineData{MIMEType: "image/jpeg", Data: "aW1hZ2U="}}}},
		},
	}
}

func cachedOptions() ExecutionOptions {
	return ExecutionOptions{Model: cacheTestModel, PromptCacheKey: "route-catalog"}
}

// cacheHarness is a fake Vertex endpoint serving cachedContents and Gemini
// generation.
type cacheHarness struct {
	t        *testing.T
	server   *httptest.Server
	mu       sync.Mutex
	creates  []map[string]any
	updates  []map[string]any
	lists    int
	requests []map[string]any
	// listed is what the cachedContents listing returns.
	listed []map[string]any
	// createStatus, when set, fails creates with that status and createMessage.
	createStatus  int
	createMessage string
	createDelay   time.Duration
	// deadCaches answer generation with 404 when named as cachedContent.
	deadCaches map[string]bool
	// generateStatus, when set, fails cache-routed generation with that status.
	generateStatus int
	usage          string
	nextCache      int
}

func newCacheHarness(t *testing.T) *cacheHarness {
	t.Helper()
	h := &cacheHarness{t: t, deadCaches: map[string]bool{}, usage: `{"promptTokenCount":100,"candidatesTokenCount":10,"totalTokenCount":110}`}
	h.server = httptest.NewServer(http.HandlerFunc(h.serve))
	t.Cleanup(h.server.Close)
	return h
}

func (h *cacheHarness) driver(options ...func(*VertexAIOptions)) *VertexAIDriver {
	h.t.Helper()
	driver := newTestVertexDriver(h.t, h.server.URL)
	for _, option := range options {
		option(&driver.options)
	}
	if driver.options.DisableGeminiContextCache {
		driver.contextCache = nil
	} else {
		driver.contextCache = newGeminiContextCacheManager(driver.options.GeminiContextCacheTTLSeconds)
	}
	return driver
}

func (h *cacheHarness) serve(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	if r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&body)
	}
	switch {
	case strings.HasSuffix(r.URL.Path, "/cachedContents") && r.Method == http.MethodPost:
		if h.createDelay > 0 {
			time.Sleep(h.createDelay)
		}
		h.mu.Lock()
		h.creates = append(h.creates, body)
		status, message := h.createStatus, h.createMessage
		h.nextCache++
		name := cacheTestName
		if h.nextCache > 1 {
			name = fmt.Sprintf("projects/project-1/locations/us-central1/cachedContents/%d", h.nextCache)
		}
		h.mu.Unlock()
		if status != 0 {
			w.WriteHeader(status)
			_, _ = fmt.Fprintf(w, `{"error":{"code":%d,"message":%q}}`, status, message)
			return
		}
		ttl, _ := time.ParseDuration(body["ttl"].(string))
		_ = json.NewEncoder(w).Encode(map[string]any{"name": name, "displayName": body["displayName"], "expireTime": time.Now().Add(ttl).Format(time.RFC3339Nano)})
	case strings.HasSuffix(r.URL.Path, "/cachedContents") && r.Method == http.MethodGet:
		h.mu.Lock()
		h.lists++
		listed := h.listed
		h.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{"cachedContents": listed})
	case strings.Contains(r.URL.Path, "/cachedContents/") && r.Method == http.MethodPatch:
		h.mu.Lock()
		h.updates = append(h.updates, body)
		h.mu.Unlock()
		ttl, _ := time.ParseDuration(body["ttl"].(string))
		_ = json.NewEncoder(w).Encode(map[string]any{"name": strings.TrimPrefix(r.URL.Path, "/"), "expireTime": time.Now().Add(ttl).Format(time.RFC3339Nano)})
	case strings.Contains(r.URL.Path, "enerateContent"):
		h.mu.Lock()
		h.requests = append(h.requests, body)
		cache, _ := body["cachedContent"].(string)
		dead := h.deadCaches[cache]
		status := h.generateStatus
		usage := h.usage
		h.mu.Unlock()
		if dead {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":404,"message":"CachedContent not found"}}`))
			return
		}
		if status != 0 && cache != "" {
			w.WriteHeader(status)
			_, _ = fmt.Fprintf(w, `{"error":{"code":%d,"message":"Request contains an invalid argument."}}`, status)
			return
		}
		response := `{"candidates":[{"content":{"role":"model","parts":[{"text":"places"}]},"finishReason":"STOP"}],"usageMetadata":` + usage + `}`
		if strings.Contains(r.URL.Path, "streamGenerateContent") {
			_, _ = w.Write([]byte("data: " + response + "\n\n"))
			return
		}
		_, _ = w.Write([]byte(response))
	default:
		h.t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusNotFound)
	}
}

func (h *cacheHarness) counts() (creates, lists, requests int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.creates), h.lists, len(h.requests)
}

func (h *cacheHarness) request(i int) map[string]any {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.requests[i]
}

func (h *cacheHarness) create(i int) map[string]any {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.creates[i]
}

func (h *cacheHarness) execute(driver *VertexAIDriver, prompt geminiPrompt, options ExecutionOptions) (*Completion, error) {
	h.t.Helper()
	return driver.requestGemini(context.Background(), prompt, options)
}

func (h *cacheHarness) mustExecute(driver *VertexAIDriver, prompt geminiPrompt, options ExecutionOptions) *Completion {
	h.t.Helper()
	completion, err := h.execute(driver, prompt, options)
	if err != nil {
		h.t.Fatal(err)
	}
	return completion
}

// canonicalJSON encodes value with sorted object keys, so a decoded request body
// and the struct it was encoded from compare equal.
func canonicalJSON(t *testing.T, value any) string {
	t.Helper()
	var decoded any
	if err := json.Unmarshal([]byte(mustJSON(t, value)), &decoded); err != nil {
		t.Fatal(err)
	}
	return mustJSON(t, decoded)
}

func cachePath(t *testing.T, completion *Completion) wire.PromptCachePath {
	t.Helper()
	if completion.PromptCacheDiagnostic == nil {
		t.Fatal("no prompt cache diagnostic")
	}
	return completion.PromptCacheDiagnostic.Path
}

// TestGeminiCachePrefixDerivation mirrors the deriveGeminiCachePrefix,
// stripLeadingParts and cache identity cases.
func TestGeminiCachePrefixDerivation(t *testing.T) {
	t.Parallel()

	prefix, ok := deriveGeminiCachePrefix(routePrompt())
	if !ok || prefix.system == nil || canonicalJSON(t, prefix.contents) != canonicalJSON(t, routePrompt().Contents[:1]) || prefix.partCount != 1 {
		t.Fatalf("route prefix = %#v", prefix)
	}
	// Stops at the first content holding a non-text part.
	prefix, _ = deriveGeminiCachePrefix(geminiPrompt{Contents: []geminiContent{
		{Role: "user", Parts: []geminiPart{{Text: "instructions"}}},
		{Role: "user", Parts: []geminiPart{{Text: "exhibit"}, {FileData: &geminiFileData{FileURI: "gs://b/o"}}}},
		{Role: "model", Parts: []geminiPart{{Text: "noted"}}},
		{Role: "user", Parts: []geminiPart{{Text: "question"}}},
	}})
	if len(prefix.contents) != 1 || prefix.contents[0].Parts[0].Text != "instructions" {
		t.Fatalf("mixed prefix = %#v", prefix)
	}
	// Signed thoughts are never cached.
	if _, ok := deriveGeminiCachePrefix(geminiPrompt{Contents: []geminiContent{
		{Role: "model", Parts: []geminiPart{{Text: "plan", Thought: true, ThoughtSignature: "sig"}}},
		{Role: "user", Parts: []geminiPart{{Text: "go"}}},
	}}); ok {
		t.Fatal("signed thought prefix was cacheable")
	}
	if _, ok := deriveGeminiCachePrefix(geminiPrompt{Contents: []geminiContent{{Role: "user", Parts: []geminiPart{{Text: "hello"}}}}}); ok {
		t.Fatal("single-turn prompt was cacheable")
	}

	stripped, ok := stripLeadingGeminiParts([]geminiContent{{Role: "user", Parts: []geminiPart{{Text: "a"}, {Text: "b"}, {Text: "c"}}}}, 2)
	if !ok || len(stripped) != 1 || len(stripped[0].Parts) != 1 || stripped[0].Parts[0].Text != "c" {
		t.Fatalf("stripped = %#v", stripped)
	}
	if _, ok := stripLeadingGeminiParts([]geminiContent{{Role: "user", Parts: []geminiPart{{Text: "a"}}}}, 3); ok {
		t.Fatal("stripped more parts than the request holds")
	}

	catalog := geminiCachePrefix{contents: []geminiContent{{Role: "user", Parts: []geminiPart{{Text: cacheTestCatalog}}}}, partCount: 1}
	other := geminiCachePrefix{contents: []geminiContent{{Role: "user", Parts: []geminiPart{{Text: "other catalog"}}}}, partCount: 1}
	hash := geminiCacheContentHash(cacheTestModel, catalog, nil, nil)
	name := geminiCacheDisplayName(cacheTestModel, hash)
	if name != geminiCacheDisplayName(cacheTestModel, geminiCacheContentHash(cacheTestModel, catalog, nil, nil)) ||
		!regexp.MustCompile(`^llmv-cache:gemini-3\.7-flash:[0-9a-f]{16}$`).MatchString(name) {
		t.Fatalf("display name = %q", name)
	}
	withSystem := catalog
	withSystem.system = &geminiContent{Role: "user", Parts: []geminiPart{{Text: "s"}}}
	for label, differing := range map[string]string{
		"content": geminiCacheContentHash(cacheTestModel, other, nil, nil),
		"model":   geminiCacheContentHash("gemini-3.7-pro", catalog, nil, nil),
		"tools":   geminiCacheContentHash(cacheTestModel, catalog, []map[string]any{{"functionDeclarations": []map[string]any{{"name": "lookup"}}}}, nil),
		"system":  geminiCacheContentHash(cacheTestModel, withSystem, nil, nil),
	} {
		if differing == hash {
			t.Errorf("%s does not change the cache identity", label)
		}
	}
	long := geminiCacheDisplayName("publishers/google/models/gemini-9.9-flash-lite-preview-experimental-09-2031", strings.Repeat("a", 64))
	if long != "llmv-cache:gemini-9.9-flash-lite-preview-experimental-09-2031:"+strings.Repeat("a", 16) || len(long) > 128 {
		t.Fatalf("long display name = %q", long)
	}

	if isGeminiMinimumTokenCountError(&fakeStatusError{code: 400, body: "TTL must be at least the minimum of 60 seconds."}) {
		t.Error("a minimum TTL error was classified as a minimum token error")
	}
	if !isGeminiMinimumTokenCountError(&fakeStatusError{code: 400, body: "The cached content is of 12 tokens. The minimum token count to start caching is 1024."}) {
		t.Error("a minimum token error was not classified")
	}
	if !isGeminiExplicitCacheUnsupportedError(&fakeStatusError{code: 400, body: "Model gemini-x does not support explicit context caching."}) ||
		isGeminiExplicitCacheUnsupportedError(&fakeStatusError{code: 400, body: "Cached content is too large."}) ||
		isGeminiExplicitCacheUnsupportedError(&fakeStatusError{code: 500, body: "does not support explicit caching"}) {
		t.Error("unsupported-model classification is wrong")
	}
}

// TestGeminiContextCacheSendsDynamicTurn mirrors "caches the system instruction
// and catalog, and sends only the dynamic turn" and the system-only prefix case.
func TestGeminiContextCacheSendsDynamicTurn(t *testing.T) {
	t.Parallel()

	h := newCacheHarness(t)
	completion := h.mustExecute(h.driver(), routePrompt(), cachedOptions())
	diagnostic := completion.PromptCacheDiagnostic
	if diagnostic == nil || diagnostic.Path != wire.PromptCachePathCreated || !diagnostic.ExplicitCacheUsed ||
		diagnostic.Model != cacheTestModel || *diagnostic.CacheablePartCount != 1 || len(*diagnostic.ContentHashPrefix) != 12 {
		t.Fatalf("diagnostic = %#v", diagnostic)
	}
	create := h.create(0)
	if create["model"] != "projects/project-1/locations/us-central1/publishers/google/models/"+cacheTestModel ||
		create["ttl"] != "1800s" || !strings.HasPrefix(create["displayName"].(string), "llmv-cache:"+cacheTestModel+":") ||
		canonicalJSON(t, create["contents"]) != canonicalJSON(t, routePrompt().Contents[:1]) ||
		canonicalJSON(t, create["systemInstruction"]) != canonicalJSON(t, routePrompt().System) {
		t.Fatalf("create = %v", create)
	}
	request := h.request(0)
	if request["cachedContent"] != cacheTestName || request["systemInstruction"] != nil ||
		canonicalJSON(t, request["contents"]) != canonicalJSON(t, routePrompt().Contents[1:]) {
		t.Fatalf("request = %v", request)
	}

	// A system-only prefix caches no contents.
	h = newCacheHarness(t)
	prompt := routePrompt()
	prompt.Contents = prompt.Contents[1:]
	h.mustExecute(h.driver(), prompt, cachedOptions())
	if _, ok := h.create(0)["contents"]; ok || h.create(0)["systemInstruction"] == nil {
		t.Fatalf("system-only create = %v", h.create(0))
	}
	if canonicalJSON(t, h.request(0)["contents"]) != canonicalJSON(t, prompt.Contents) {
		t.Fatalf("system-only request = %v", h.request(0))
	}
}

// TestGeminiContextCacheReuse mirrors "reuses the memoized cache", "collapses
// sharded cache keys onto one cache" and the 20 concurrent creates case.
func TestGeminiContextCacheReuse(t *testing.T) {
	t.Parallel()

	h := newCacheHarness(t)
	driver := h.driver()
	h.mustExecute(driver, routePrompt(), cachedOptions())
	sharded := cachedOptions()
	sharded.PromptCacheKey = "route-catalog:shard-7"
	second := h.mustExecute(driver, routePrompt(), sharded)
	if creates, lists, _ := h.counts(); creates != 1 || lists != 1 || cachePath(t, second) != wire.PromptCachePathLocalMemoHit {
		t.Fatalf("creates %d, lists %d, path %s", creates, lists, cachePath(t, second))
	}
	if h.request(1)["cachedContent"] != cacheTestName {
		t.Fatalf("second request = %v", h.request(1))
	}

	h = newCacheHarness(t)
	h.createDelay = 50 * time.Millisecond
	driver = h.driver()
	var wg sync.WaitGroup
	paths := make(chan wire.PromptCachePath, 20)
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			completion, err := h.execute(driver, routePrompt(), cachedOptions())
			if err != nil {
				t.Error(err)
				return
			}
			paths <- completion.PromptCacheDiagnostic.Path
		}()
	}
	wg.Wait()
	close(paths)
	if creates, _, requests := h.counts(); creates != 1 || requests != 20 {
		t.Fatalf("concurrent: %d creates, %d requests", creates, requests)
	}
	for path := range paths {
		if path != wire.PromptCachePathCreated && path != wire.PromptCachePathWaitedForCreator && path != wire.PromptCachePathLocalMemoHit {
			t.Errorf("concurrent path %s", path)
		}
	}
}

// TestGeminiContextCacheAdoptsPeer mirrors "adopts a live cache a peer instance
// already created" and "ignores listed caches that are expired, undated, or hold
// other content".
func TestGeminiContextCacheAdoptsPeer(t *testing.T) {
	t.Parallel()

	prefix, _ := deriveGeminiCachePrefix(routePrompt())
	displayName := geminiCacheDisplayName(cacheTestModel, geminiCacheContentHash(cacheTestModel, prefix, nil, nil))
	live := time.Now().Add(time.Hour).Format(time.RFC3339Nano)

	h := newCacheHarness(t)
	h.listed = []map[string]any{{"name": cacheTestPeer, "displayName": displayName, "expireTime": live}}
	completion := h.mustExecute(h.driver(), routePrompt(), cachedOptions())
	if creates, _, _ := h.counts(); creates != 0 || cachePath(t, completion) != wire.PromptCachePathProviderListRecovery || h.request(0)["cachedContent"] != cacheTestPeer {
		t.Fatalf("creates %d, path %s, request %v", creates, cachePath(t, completion), h.request(0))
	}

	h = newCacheHarness(t)
	h.listed = []map[string]any{
		{"name": "expired", "displayName": displayName, "expireTime": time.Now().Add(-time.Minute).Format(time.RFC3339Nano)},
		{"name": "undated", "displayName": displayName},
		{"name": "other", "displayName": "llmv-cache:" + cacheTestModel + ":0000000000000000", "expireTime": live},
	}
	completion = h.mustExecute(h.driver(), routePrompt(), cachedOptions())
	if creates, _, _ := h.counts(); creates != 1 || cachePath(t, completion) != wire.PromptCachePathCreated {
		t.Fatalf("creates %d, path %s", creates, cachePath(t, completion))
	}
}

// TestGeminiContextCacheToolsAndTTL mirrors "moves declared tools into the cache
// and out of the request" and the two TTL cases.
func TestGeminiContextCacheToolsAndTTL(t *testing.T) {
	t.Parallel()

	h := newCacheHarness(t)
	options := cachedOptions()
	options.Tools = []ToolDefinition{{Name: "lookup", InputSchema: JSONObject{"type": "object"}}}
	options.PromptCacheTTLSeconds = 600
	h.mustExecute(h.driver(), routePrompt(), options)
	if h.create(0)["tools"] == nil || h.create(0)["toolConfig"] == nil || h.create(0)["ttl"] != "600s" {
		t.Fatalf("create = %v", h.create(0))
	}
	if request := h.request(0); request["tools"] != nil || request["toolConfig"] != nil {
		t.Fatalf("request kept tools: %v", request)
	}

	h = newCacheHarness(t)
	h.mustExecute(h.driver(func(o *VertexAIOptions) { o.GeminiContextCacheTTLSeconds = 900 }), routePrompt(), cachedOptions())
	if h.create(0)["ttl"] != "900s" {
		t.Fatalf("driver TTL create = %v", h.create(0))
	}
}

// TestGeminiContextCacheCreateFailures mirrors "sends the full un-cached request
// when cache creation fails", "stops retrying a prefix Vertex refuses to cache",
// the unsupported-model recheck and "leaves transient create retry policy to the
// host workflow".
func TestGeminiContextCacheCreateFailures(t *testing.T) {
	t.Parallel()

	h := newCacheHarness(t)
	h.createStatus, h.createMessage = 500, "internal"
	completion := h.mustExecute(h.driver(), routePrompt(), cachedOptions())
	if diagnostic := completion.PromptCacheDiagnostic; diagnostic.Path != wire.PromptCachePathFallbackProviderError || diagnostic.ExplicitCacheUsed || *diagnostic.ProviderStatus != 500 {
		t.Fatalf("diagnostic = %#v", diagnostic)
	}
	if request := h.request(0); request["cachedContent"] != nil || request["systemInstruction"] == nil || len(request["contents"].([]any)) != 2 {
		t.Fatalf("fallback request = %v", request)
	}

	h = newCacheHarness(t)
	h.createStatus, h.createMessage = 400, "The cached content is of 12 tokens. The minimum token count to start caching is 1024."
	driver := h.driver()
	h.mustExecute(driver, routePrompt(), cachedOptions())
	second := h.mustExecute(driver, routePrompt(), cachedOptions())
	if creates, _, _ := h.counts(); creates != 1 || cachePath(t, second) != wire.PromptCachePathMinimumTokenRejection {
		t.Fatalf("minimum tokens: creates %d, path %s", creates, cachePath(t, second))
	}

	h = newCacheHarness(t)
	h.createStatus, h.createMessage = 400, "Model gemini-3.7-flash does not support explicit context caching."
	driver = h.driver()
	now := time.Now()
	driver.contextCache.now = func() time.Time { return now }
	first := h.mustExecute(driver, routePrompt(), cachedOptions())
	second = h.mustExecute(driver, routePrompt(), cachedOptions())
	if creates, _, _ := h.counts(); creates != 1 || cachePath(t, first) != wire.PromptCachePathUnsupportedModel || cachePath(t, second) != wire.PromptCachePathUnsupportedModel {
		t.Fatalf("unsupported: creates %d, paths %s / %s", creates, cachePath(t, first), cachePath(t, second))
	}
	now = now.Add(time.Hour + time.Second)
	h.mustExecute(driver, routePrompt(), cachedOptions())
	if creates, _, _ := h.counts(); creates != 2 {
		t.Fatalf("unsupported model was not rechecked: %d creates", creates)
	}

	h = newCacheHarness(t)
	h.createStatus, h.createMessage = 429, "Resource exhausted"
	driver = h.driver()
	first = h.mustExecute(driver, routePrompt(), cachedOptions())
	second = h.mustExecute(driver, routePrompt(), cachedOptions())
	if creates, _, requests := h.counts(); creates != 1 || requests != 2 ||
		cachePath(t, first) != wire.PromptCachePathFallbackQuota || cachePath(t, second) != wire.PromptCachePathFallbackQuota {
		t.Fatalf("quota: creates %d, requests %d, paths %s / %s", creates, requests, cachePath(t, first), cachePath(t, second))
	}
}

// TestGeminiContextCacheUnusableResource mirrors "rediscovers, then recreates
// once, when the resource has gone" and "does not swallow errors that have
// nothing to do with the cache".
func TestGeminiContextCacheUnusableResource(t *testing.T) {
	t.Parallel()

	h := newCacheHarness(t)
	h.deadCaches[cacheTestName] = true
	completion := h.mustExecute(h.driver(), routePrompt(), cachedOptions())
	if creates, lists, requests := h.counts(); creates != 2 || lists != 2 || requests != 2 {
		t.Fatalf("creates %d, lists %d, requests %d", creates, lists, requests)
	}
	if cachePath(t, completion) != wire.PromptCachePathUnusableResourceRecreated || h.request(1)["cachedContent"] == cacheTestName {
		t.Fatalf("path %s, retry %v", cachePath(t, completion), h.request(1))
	}

	// Both resources gone: the full request goes out uncached.
	h = newCacheHarness(t)
	h.deadCaches[cacheTestName] = true
	h.deadCaches["projects/project-1/locations/us-central1/cachedContents/2"] = true
	completion = h.mustExecute(h.driver(), routePrompt(), cachedOptions())
	if diagnostic := completion.PromptCacheDiagnostic; diagnostic.Path != wire.PromptCachePathFallbackProviderError || *diagnostic.ProviderStatus != 404 || h.request(2)["cachedContent"] != nil {
		t.Fatalf("diagnostic %#v, last request %v", diagnostic, h.request(2))
	}

	h = newCacheHarness(t)
	h.generateStatus = 400
	if _, err := h.execute(h.driver(), routePrompt(), cachedOptions()); err == nil {
		t.Fatal("an unrelated error was swallowed")
	}
	if _, _, requests := h.counts(); requests != 1 {
		t.Fatalf("an unrelated error was retried: %d requests", requests)
	}
}

// TestGeminiContextCacheStreamingAndUsage mirrors "routes the streaming path
// through the same cache" and "reports explicit cache reads as prompt_cached".
func TestGeminiContextCacheStreamingAndUsage(t *testing.T) {
	t.Parallel()

	h := newCacheHarness(t)
	h.usage = `{"promptTokenCount":5000,"cachedContentTokenCount":4800,"candidatesTokenCount":10,"totalTokenCount":5010}`
	driver := h.driver()
	stream, err := driver.requestGeminiStream(context.Background(), routePrompt(), cachedOptions())
	if err != nil {
		t.Fatal(err)
	}
	response := drainStream(t, stream)
	if response.PromptCacheDiagnostic == nil || response.PromptCacheDiagnostic.Path != wire.PromptCachePathCreated || h.request(0)["cachedContent"] != cacheTestName {
		t.Fatalf("stream diagnostic %#v, request %v", response.PromptCacheDiagnostic, h.request(0))
	}
	completion := h.mustExecute(driver, routePrompt(), cachedOptions())
	if usage := completion.TokenUsage; usage == nil || usage.PromptCached == nil || *usage.PromptCached != 4800 {
		t.Fatalf("usage = %#v", usage)
	}
}

// TestGeminiContextCacheSegments mirrors the PromptSegment cache breakpoint cases:
// stable user text goes into the cache and the image task stays in the request,
// and a trailing model segment stays out of the cache.
func TestGeminiContextCacheSegments(t *testing.T) {
	t.Parallel()

	segments := []PromptSegment{
		{Role: PromptRoleSystem, Content: "Route each photo to a domain."},
		{Role: PromptRoleUser, Content: cacheTestCatalog},
		{Role: PromptRoleUser, Content: "Route this photo.", Files: []DataSource{gcsSource{t: t, uri: "gs://bucket/photo.jpg", mime: "image/jpeg"}}},
	}
	for _, stream := range []bool{false, true} {
		h := newCacheHarness(t)
		driver := h.driver()
		if stream {
			completionStream, err := driver.Stream(context.Background(), segments, cachedOptions())
			if err != nil {
				t.Fatal(err)
			}
			drainStream(t, completionStream)
		} else if _, err := driver.Execute(context.Background(), segments, cachedOptions()); err != nil {
			t.Fatal(err)
		}
		if got := canonicalJSON(t, h.create(0)["contents"]); got != `[{"parts":[{"text":"`+cacheTestCatalog+`"}],"role":"user"}]` {
			t.Fatalf("stream=%v: cached contents = %s", stream, got)
		}
		if got := canonicalJSON(t, h.request(0)["contents"]); got != `[{"parts":[{"text":"Route this photo."},{"fileData":{"fileUri":"gs://bucket/photo.jpg","mimeType":"image/jpeg"}}],"role":"user"}]` {
			t.Fatalf("stream=%v: request contents = %s", stream, got)
		}
	}

	h := newCacheHarness(t)
	prompt := routePrompt()
	prompt.Contents = append(prompt.Contents[:1], append([]geminiContent{{Role: "model", Parts: []geminiPart{{Text: "Ready to route a photo."}}}}, prompt.Contents[1:]...)...)
	h.mustExecute(h.driver(), prompt, cachedOptions())
	if canonicalJSON(t, h.create(0)["contents"]) != canonicalJSON(t, routePrompt().Contents[:1]) || canonicalJSON(t, h.request(0)["contents"]) != canonicalJSON(t, prompt.Contents[1:]) {
		t.Fatalf("create %v, request %v", h.create(0)["contents"], h.request(0)["contents"])
	}
}

// TestGeminiContextCacheOptOut mirrors the opt-out cases: prompt_cache_mode off,
// the driver kill switch and a missing key leave the payload unchanged, and
// required surfaces a cache that cannot be used.
func TestGeminiContextCacheOptOut(t *testing.T) {
	t.Parallel()

	off := cachedOptions()
	off.PromptCacheMode = common.PromptCacheModeOff
	noKey := cachedOptions()
	noKey.PromptCacheKey = ""
	for label, tc := range map[string]struct {
		options ExecutionOptions
		disable bool
		path    wire.PromptCachePath
	}{
		"off":         {off, false, wire.PromptCachePathDisabled},
		"kill switch": {cachedOptions(), true, wire.PromptCachePathDisabled},
		"no key":      {noKey, false, wire.PromptCachePathNoKey},
	} {
		h := newCacheHarness(t)
		completion := h.mustExecute(h.driver(func(o *VertexAIOptions) { o.DisableGeminiContextCache = tc.disable }), routePrompt(), tc.options)
		creates, lists, _ := h.counts()
		if creates != 0 || lists != 0 || cachePath(t, completion) != tc.path {
			t.Fatalf("%s: creates %d, lists %d, path %s", label, creates, lists, cachePath(t, completion))
		}
		if request := h.request(0); request["cachedContent"] != nil || request["systemInstruction"] == nil || len(request["contents"].([]any)) != 2 {
			t.Fatalf("%s: request = %v", label, request)
		}
	}
	if driver, _ := NewVertexAIDriver(VertexAIOptions{Project: "p", Region: "r", DisableGeminiContextCache: true}); driver.contextCache != nil {
		t.Fatal("the kill switch kept a cache registry")
	}

	h := newCacheHarness(t)
	h.createStatus, h.createMessage = 500, "internal"
	required := cachedOptions()
	required.PromptCacheMode = common.PromptCacheModeRequired
	_, err := h.execute(h.driver(), routePrompt(), required)
	var llumiErr *LlumiverseError
	if !errors.As(err, &llumiErr) || llumiErr.Name != "GeminiContextCacheRequiredError" ||
		!strings.Contains(err.Error(), "Gemini explicit context cache was required but unavailable (fallback_provider_error)") {
		t.Fatalf("required error = %v", err)
	}
	if _, _, requests := h.counts(); requests != 0 {
		t.Fatalf("required mode sent %d uncached requests", requests)
	}
}
