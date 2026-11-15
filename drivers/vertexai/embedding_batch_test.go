package vertexai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"

	"github.com/vertesia/llumiverse-go/core"
)

// The tests in this file mirror batch.test.ts, batch-result.test.ts and
// embedding-batch.test.ts.

// TestVertexEmbeddingBatchCapability mirrors "uses an explicit model and
// modality matrix" and "opts in explicitly and resolves model-specific location
// and limits".
func TestVertexEmbeddingBatchCapability(t *testing.T) {
	t.Parallel()

	driver := newTestVertexDriver(t, "")
	if _, ok := core.EmbeddingBatchAdapterFor(driver); !ok {
		t.Fatal("the Vertex driver is not an embedding batch adapter")
	}
	for _, tc := range []struct {
		model    string
		modality core.EmbeddingBatchModality
		want     string
	}{
		{"gemini-embedding-2", core.EmbeddingBatchModalityImage, "gemini-embedding-2/gemini/1000000/global"},
		{"publishers/google/models/gemini-embedding-2", core.EmbeddingBatchModalityText, "gemini-embedding-2/gemini/1000000/global"},
		{"gemini-embedding-001", core.EmbeddingBatchModalityText, "gemini-embedding-001/gemini/1000000/us-central1"},
		{"gemini-embedding-001", core.EmbeddingBatchModalityImage, ""},
		{"text-embedding-004", core.EmbeddingBatchModalityText, "text-embedding-004/legacy/30000/us-central1"},
		{"text-embedding-005", core.EmbeddingBatchModalityText, "text-embedding-005/legacy/30000/us-central1"},
		{"text-embedding-005", core.EmbeddingBatchModalityImage, ""},
		{"textembedding-gecko@003", core.EmbeddingBatchModalityText, ""},
		{"multimodalembedding@001", core.EmbeddingBatchModalityImage, ""},
		{"future-gemini-embedding-2", core.EmbeddingBatchModalityText, ""},
	} {
		capability, ok := driver.EmbeddingBatchCapability(tc.model, tc.modality, "us-central1")
		got := ""
		if ok {
			got = strings.Join([]string{capability.Model, capability.InputFormat, itoa(capability.MaxRows), capability.Location}, "/")
		}
		if got != tc.want {
			t.Errorf("%s/%s: capability = %q, want %q", tc.model, tc.modality, got, tc.want)
		}
	}
}

func itoa(n int) string {
	data, _ := json.Marshal(n)
	return string(data)
}

// TestVertexEmbeddingBatchRows mirrors the "Vertex embedding batch rows" cases and
// "rejects unsupported batch media locations before downloading or copying".
func TestVertexEmbeddingBatchRows(t *testing.T) {
	t.Parallel()

	driver := newTestVertexDriver(t, "")
	ctx := context.Background()
	row, key, err := driver.FormatEmbeddingBatchRow(ctx, core.EmbeddingBatchRowOptions{
		Key: "text:1:etag", Model: "gemini-embedding-001", Dimensions: 768, InputFormat: "gemini",
		Input: EmbeddingInput{Type: "text", Text: "hello", TaskType: EmbeddingTaskDocument, Title: "Greeting"},
	})
	if err != nil || key != "text:1:etag" || mustJSON(t, row) != `{"key":"text:1:etag","request":{"content":{"parts":[{"text":"hello"}]},"embed_content_config":{"output_dimensionality":768,"task_type":"RETRIEVAL_DOCUMENT","title":"Greeting"}}}` {
		t.Fatalf("gemini 001 row = %s, %q (%v)", mustJSON(t, row), key, err)
	}
	row, _, err = driver.FormatEmbeddingBatchRow(ctx, core.EmbeddingBatchRowOptions{
		Key: "text:1:etag", Model: "gemini-embedding-2", Dimensions: 1024, InputFormat: "gemini",
		Input: EmbeddingInput{Type: "text", Text: "hello", TaskType: EmbeddingTaskDocument},
	})
	if err != nil || mustJSON(t, row) != `{"key":"text:1:etag","request":{"content":{"parts":[{"text":"title: none | text: hello"}]},"embed_content_config":{"output_dimensionality":1024}}}` {
		t.Fatalf("gemini 2 row = %s (%v)", mustJSON(t, row), err)
	}
	row, _, err = driver.FormatEmbeddingBatchRow(ctx, core.EmbeddingBatchRowOptions{
		Key: "image:1:etag", Model: "gemini-embedding-2", Dimensions: 768, InputFormat: "gemini",
		Input: EmbeddingInput{Type: "image", Source: gcsSource{t: t, uri: "gs://bucket/rendition.jpg", mime: "image/jpeg"}},
	})
	if err != nil || mustJSON(t, row) != `{"key":"image:1:etag","request":{"content":{"parts":[{"fileData":{"fileUri":"gs://bucket/rendition.jpg","mimeType":"image/jpeg"}}]},"embed_content_config":{"output_dimensionality":768}}}` {
		t.Fatalf("gemini 2 image row = %s (%v)", mustJSON(t, row), err)
	}
	row, key, err = driver.FormatEmbeddingBatchRow(ctx, core.EmbeddingBatchRowOptions{
		Key: "properties:1:etag", Model: "text-embedding-005", Dimensions: 768, InputFormat: "legacy",
		Input: EmbeddingInput{Type: "text", Text: `{"name":"Ada"}`},
	})
	if err != nil || mustJSON(t, row) != `{"content":"{\"name\":\"Ada\"}","key":"properties:1:etag"}` || !regexp.MustCompile(`^input:[a-f0-9]{64}$`).MatchString(key) {
		t.Fatalf("legacy row = %s, %q (%v)", mustJSON(t, row), key, err)
	}
	if _, _, err := driver.FormatEmbeddingBatchRow(ctx, core.EmbeddingBatchRowOptions{
		Key: "text:1:etag", Model: "gemini-embedding-001", Dimensions: 768, InputFormat: "legacy",
		Input: EmbeddingInput{Type: "text", Text: "hello"},
	}); err == nil || !strings.Contains(err.Error(), "uses gemini batch rows") {
		t.Fatalf("mismatched format error = %v", err)
	}
	// A non-GCS image is rejected without being read (gcsSource fails the test on Open).
	if _, _, err := driver.FormatEmbeddingBatchRow(ctx, core.EmbeddingBatchRowOptions{
		Key: "row", Model: "gemini-embedding-2", Dimensions: 2, InputFormat: "gemini",
		Input: EmbeddingInput{Type: "image", Source: gcsSource{t: t, uri: "https://example.com/image.jpg", mime: "image/jpeg"}},
	}); err == nil || !strings.Contains(err.Error(), "gs://") {
		t.Fatalf("non-GCS image error = %v", err)
	}
}

// TestVertexEmbeddingBatchResults mirrors batch-result.test.ts and "owns
// input/output correlation" of embedding-batch.test.ts.
func TestVertexEmbeddingBatchResults(t *testing.T) {
	t.Parallel()

	driver := newTestVertexDriver(t, "")
	parse := func(record string) core.ParsedEmbeddingBatchResult {
		var decoded map[string]any
		if err := json.Unmarshal([]byte(record), &decoded); err != nil {
			t.Fatal(err)
		}
		return driver.ParseEmbeddingBatchResult(decoded)
	}
	if got := parse(`{"key":42,"response":{"embeddings":[{"values":[1,2]}]}}`); got.Key != "" || mustJSON(t, got.Vector) != "[1,2]" || got.ProviderError {
		t.Errorf("embeddings array = %#v", got)
	}
	for _, status := range []string{`"{\"code\":7,\"message\":\"private details\"}"`, `{"code":7}`} {
		got := parse(`{"key":"row","status":` + status + `,"response":{}}`)
		if got.Key != "row" || got.Vector != nil || !got.ProviderError || got.FailureCategory != "provider_permission_denied" {
			t.Errorf("status %s = %#v", status, got)
		}
	}
	for _, status := range []string{`""`, `{"code":0}`, `"{\"code\":0}"`} {
		if parse(`{"status":` + status + `}`).ProviderError {
			t.Errorf("status %s was an error", status)
		}
	}
	if got := parse(`{"status":"unexpected failure"}`); !got.ProviderError || got.FailureCategory != "provider_row_error" {
		t.Errorf("unparseable status = %#v", got)
	}
	if got := parse(`{"key":"gemini","response":{"embedding":{"values":[1,2]}}}`); got.Key != "gemini" || mustJSON(t, got.Vector) != "[1,2]" {
		t.Errorf("gemini output = %#v", got)
	}
	omitted, _ := vertexEmbeddingBatchCorrelationKey(map[string]any{"content": "omitted"})
	if got := parse(`{"instance":{"key":"legacy","content":"omitted"},"predictions":[{"embeddings":{"values":[3,4]}}]}`); got.Key != omitted || mustJSON(t, got.Vector) != "[3,4]" {
		t.Errorf("legacy output = %#v", got)
	}
	if got := parse(`{"key":"failed","error":{"message":"sensitive provider detail"}}`); !got.ProviderError || got.FailureCategory != "" || strings.Contains(mustJSON(t, got), "sensitive") {
		t.Errorf("row error = %#v", got)
	}

	// Legacy correlation is independent of field order and keeps every input field.
	text := "Café\nsecond line 🐈"
	_, key, err := driver.FormatEmbeddingBatchRow(context.Background(), core.EmbeddingBatchRowOptions{Key: "object-snapshot", Model: "text-embedding-004", Dimensions: 256, InputFormat: "legacy", Input: EmbeddingInput{Text: text}})
	if err != nil {
		t.Fatal(err)
	}
	if got := driver.ParseEmbeddingBatchResult(map[string]any{"instance": map[string]any{"content": text}, "status": "", "predictions": []any{map[string]any{"embeddings": map[string]any{"values": []any{1.0, 2.0}}}}}); got.Key != key || got.ProviderError {
		t.Errorf("legacy round trip = %#v, want key %s", got, key)
	}
	if other, _ := vertexEmbeddingBatchCorrelationKey(map[string]any{"key": "another-object", "content": text}); other != key {
		t.Error("the legacy key depends on the custom key")
	}
	for _, changed := range []map[string]any{{"content": "different"}, {"content": text, "task_type": "RETRIEVAL_QUERY"}, {"content": text, "title": "title"}} {
		if other, _ := vertexEmbeddingBatchCorrelationKey(changed); other == key {
			t.Errorf("%v shares the key", changed)
		}
	}
	if _, ok := vertexEmbeddingBatchCorrelationKey(map[string]any{"content": "x", "task_type": 42.0}); ok {
		t.Error("a numeric task type was correlated")
	}
	if _, ok := vertexEmbeddingBatchCorrelationKey(map[string]any{"content": 42.0}); ok {
		t.Error("numeric content was correlated")
	}
	// The key matches JavaScript's JSON.stringify of the same fields.
	if got := string(jsStringify([]any{"a<b>&\u2028", nil, nil})); got != "[\"a<b>&\u2028\",null,null]" {
		t.Errorf("jsStringify = %q", got)
	}
}

// batchServer is a fake batchPredictionJobs API.
type batchServer struct {
	mu       sync.Mutex
	requests []string
	bodies   []map[string]any
	listed   string
	job      string
}

func (s *batchServer) handler(w http.ResponseWriter, r *http.Request) {
	data, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(data, &body)
	s.mu.Lock()
	s.requests = append(s.requests, r.Method+" "+r.URL.Path+"?"+r.URL.RawQuery)
	s.bodies = append(s.bodies, body)
	listed, job := s.listed, s.job
	s.mu.Unlock()
	switch {
	case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/batchPredictionJobs"):
		_, _ = w.Write([]byte(`{"batchPredictionJobs":[` + listed + `]}`))
	case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, ":cancel"), r.Method == http.MethodDelete:
		_, _ = w.Write([]byte(`{}`))
	default:
		_, _ = w.Write([]byte(job))
	}
}

func newBatchServer(t *testing.T, listed, job string) (*VertexAIDriver, *batchServer) {
	t.Helper()
	fake := &batchServer{listed: listed, job: job}
	server := httptest.NewServer(http.HandlerFunc(fake.handler))
	t.Cleanup(server.Close)
	return newTestVertexDriver(t, server.URL), fake
}

func batchCreateOptions(model string) core.EmbeddingBatchCreateOptions {
	return core.EmbeddingBatchCreateOptions{Model: model, Modality: core.EmbeddingBatchModalityText, DisplayName: "stable-name", InputURI: "gs://bucket/input.jsonl", OutputURI: "gs://bucket/output/"}
}

const batchExistingJob = `{"name":"projects/p/locations/global/batchPredictionJobs/1","displayName":"stable-name","state":"JOB_STATE_RUNNING","model":"publishers/google/models/gemini-embedding-2","inputConfig":{"gcsSource":{"uris":["gs://bucket/input.jsonl"]}},"outputConfig":{"gcsDestination":{"outputUriPrefix":"gs://bucket/output/"}}}`

// TestVertexEmbeddingBatchLifecycle mirrors the "Vertex embedding batch
// lifecycle" cases.
func TestVertexEmbeddingBatchLifecycle(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Adopts an exact match of the deterministic display name before creating.
	driver, fake := newBatchServer(t, batchExistingJob, `{}`)
	job, err := driver.CreateEmbeddingBatch(ctx, batchCreateOptions("gemini-embedding-2"), nil)
	if err != nil || job.Name != "projects/p/locations/global/batchPredictionJobs/1" || job.State != core.EmbeddingBatchStateRunning || len(fake.requests) != 1 {
		t.Fatalf("adopt: job %#v (%v), requests %v", job, err, fake.requests)
	}
	if fake.requests[0] != "GET /projects/project-1/locations/global/batchPredictionJobs?filter=displayName%3D%22stable-name%22&pageSize=10" {
		t.Fatalf("list request = %s", fake.requests[0])
	}

	// A listed job with another display name is not adopted.
	other := strings.Replace(batchExistingJob, `"displayName":"stable-name"`, `"displayName":"other-name"`, 1)
	driver, fake = newBatchServer(t, other, `{"name":"jobs/new","model":"publishers/google/models/gemini-embedding-2"}`)
	if job, err := driver.CreateEmbeddingBatch(ctx, batchCreateOptions("gemini-embedding-2"), nil); err != nil || job.Name != "jobs/new" || len(fake.requests) != 2 {
		t.Fatalf("other name: job %#v (%v), requests %v", job, err, fake.requests)
	}

	// Creates, cancels, polls and deletes in the model's location.
	driver, fake = newBatchServer(t, ``, `{"name":"projects/project-1/locations/us-central1/batchPredictionJobs/1","state":"JOB_STATE_PENDING","model":"publishers/google/models/gemini-embedding-001"}`)
	job, err = driver.CreateEmbeddingBatch(ctx, batchCreateOptions("gemini-embedding-001"), nil)
	if err != nil || job.State != core.EmbeddingBatchStatePending {
		t.Fatalf("create: %#v (%v)", job, err)
	}
	if got := mustJSON(t, fake.bodies[1]); got != `{"displayName":"stable-name","inputConfig":{"gcsSource":{"uris":["gs://bucket/input.jsonl"]},"instancesFormat":"jsonl"},"model":"publishers/google/models/gemini-embedding-001","outputConfig":{"gcsDestination":{"outputUriPrefix":"gs://bucket/output/"},"predictionsFormat":"jsonl"}}` {
		t.Fatalf("create body = %s", got)
	}
	jobOptions := core.EmbeddingBatchJobOptions{Model: "gemini-embedding-001", Modality: core.EmbeddingBatchModalityText, Name: "1"}
	if _, err := driver.CancelEmbeddingBatch(ctx, jobOptions); err != nil {
		t.Fatal(err)
	}
	deleted, err := driver.DeleteEmbeddingBatch(ctx, jobOptions)
	if err != nil || deleted.State != core.EmbeddingBatchStateCancelled {
		t.Fatalf("delete: %#v (%v)", deleted, err)
	}
	want := []string{
		"GET /projects/project-1/locations/us-central1/batchPredictionJobs?filter=displayName%3D%22stable-name%22&pageSize=10",
		"POST /projects/project-1/locations/us-central1/batchPredictionJobs?",
		"POST /projects/project-1/locations/us-central1/batchPredictionJobs/1:cancel?",
		"GET /projects/project-1/locations/us-central1/batchPredictionJobs/1?",
		"DELETE /projects/project-1/locations/us-central1/batchPredictionJobs/1?",
	}
	if strings.Join(fake.requests, "\n") != strings.Join(want, "\n") {
		t.Fatalf("requests = %v", fake.requests)
	}

	// Partial provider success is terminal so successful rows can be applied.
	driver, _ = newBatchServer(t, ``, `{"name":"jobs/1","state":"JOB_STATE_PARTIALLY_SUCCEEDED","model":"gemini-embedding-2"}`)
	if job, err := driver.GetEmbeddingBatch(ctx, core.EmbeddingBatchJobOptions{Model: "gemini-embedding-2", Modality: core.EmbeddingBatchModalityImage, Name: "jobs/1"}); err != nil || job.State != core.EmbeddingBatchStateFailed {
		t.Fatalf("partial: %#v (%v)", job, err)
	}

	// Legacy dimensions are sent as integer strings.
	for _, model := range []string{"text-embedding-004", "text-embedding-005"} {
		driver, fake = newBatchServer(t, ``, `{"name":"jobs/1","model":"`+model+`"}`)
		options := batchCreateOptions(model)
		options.Dimensions = 256
		if _, err := driver.CreateEmbeddingBatch(ctx, options, nil); err != nil {
			t.Fatal(err)
		}
		if got := mustJSON(t, fake.bodies[1]["modelParameters"]); got != `{"outputDimensionality":"256"}` {
			t.Fatalf("%s: modelParameters = %s", model, got)
		}
	}

	// A provider job that belongs to another model is rejected.
	driver, _ = newBatchServer(t, ``, `{"name":"jobs/1","state":"JOB_STATE_RUNNING","model":"publishers/google/models/text-embedding-005"}`)
	if _, err := driver.GetEmbeddingBatch(ctx, jobOptions); err == nil || !strings.Contains(err.Error(), "belongs to model") {
		t.Fatalf("model mismatch error = %v", err)
	}
}

// fakeArtifacts records artifact store calls.
type fakeArtifacts struct {
	listed []string
	calls  []string
}

func (f *fakeArtifacts) List(_ context.Context, prefix string) ([]string, error) {
	f.calls = append(f.calls, "list "+prefix)
	return f.listed, nil
}

func (f *fakeArtifacts) Read(context.Context, string) (io.ReadCloser, error) {
	f.calls = append(f.calls, "read")
	return nil, nil
}

func (f *fakeArtifacts) Write(context.Context, string, io.Reader) error {
	f.calls = append(f.calls, "write")
	return nil
}

// TestVertexEmbeddingBatchOutputArtifacts mirrors "discovers all shards without
// copying".
func TestVertexEmbeddingBatchOutputArtifacts(t *testing.T) {
	t.Parallel()

	driver := newTestVertexDriver(t, "")
	for _, state := range []core.EmbeddingBatchState{core.EmbeddingBatchStateSucceeded, core.EmbeddingBatchStateFailed, core.EmbeddingBatchStateCancelled} {
		artifacts := &fakeArtifacts{listed: []string{"gs://bucket/output/a/predictions.jsonl", "gs://bucket/output/b/errors.jsonl"}}
		shards, err := driver.EmbeddingBatchOutputArtifacts(context.Background(), core.EmbeddingBatchJob{Name: "jobs/1", State: state, OutputURI: "gs://bucket/output/"}, artifacts)
		if err != nil || len(shards) != 2 || strings.Join(artifacts.calls, ",") != "list gs://bucket/output/" {
			t.Fatalf("%s: shards %v (%v), calls %v", state, shards, err, artifacts.calls)
		}
	}
	if _, err := driver.EmbeddingBatchOutputArtifacts(context.Background(), core.EmbeddingBatchJob{Name: "jobs/1"}, &fakeArtifacts{}); err == nil {
		t.Fatal("a job without an output prefix was listed")
	}
}
