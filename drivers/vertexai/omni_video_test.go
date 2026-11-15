package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
)

const omniOutputPrefix = "gs://project-bucket/runs/run-1/media/"

// gcsSource is a media input exposed only by its Cloud Storage URI; reading its
// bytes fails the test.
type gcsSource struct {
	t    *testing.T
	uri  string
	mime string
}

func (s gcsSource) Name() string     { return "media" }
func (s gcsSource) MIMEType() string { return s.mime }
func (s gcsSource) Open(context.Context) (io.ReadCloser, error) {
	s.t.Error("Omni media bytes were read")
	return nil, errors.New("not readable")
}
func (s gcsSource) URL(context.Context) (string, error) { return s.uri, nil }

func omniImage(t *testing.T, uri string) DataSource {
	return gcsSource{t: t, uri: uri, mime: "image/png"}
}

func omniVideo(t *testing.T, uri string) DataSource {
	return gcsSource{t: t, uri: uri, mime: "video/mp4"}
}

func omniCompleted(videos ...string) string {
	content := make([]string, len(videos))
	for i, video := range videos {
		if strings.HasPrefix(video, "text:") {
			content[i] = `{"type":"text","text":` + mustQuote(strings.TrimPrefix(video, "text:")) + `}`
			continue
		}
		content[i] = `{"type":"video","uri":` + mustQuote(video) + `,"mime_type":"video/mp4"}`
	}
	return `{"id":"interaction-1","status":"completed","usage":{"total_tokens":9,"total_input_tokens":3,"total_output_tokens":6},` +
		`"steps":[{"type":"thought","summary":[{"type":"text","text":"hidden"}]},{"type":"model_output","content":[` + strings.Join(content, ",") + `]}]}`
}

func mustQuote(value string) string {
	data, _ := json.Marshal(value)
	return string(data)
}

// omniServer answers interactions requests with body and records the requests.
type omniServer struct {
	mu       sync.Mutex
	paths    []string
	payloads []map[string]any
	body     string
}

func (s *omniServer) start(t *testing.T) *VertexAIDriver {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		s.mu.Lock()
		s.paths = append(s.paths, r.URL.Path)
		s.payloads = append(s.payloads, payload)
		body := s.body
		s.mu.Unlock()
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return newTestVertexDriver(t, server.URL)
}

func omniOptions(model string, modelOptions map[string]any) ExecutionOptions {
	return ExecutionOptions{Model: "publishers/google/models/" + model, ModelOptions: modelOptions, OutputStorageURI: omniOutputPrefix}
}

// TestOmniVideoRouting mirrors the routing case of omni-video.test.ts.
func TestOmniVideoRouting(t *testing.T) {
	t.Parallel()

	for _, model := range []string{geminiOmniVideoModel, geminiOmni11VideoModel, geminiOmni12VideoModel, "gemini-omni-1.3-flash"} {
		if !isGeminiOmniVideoModel("publishers/google/models/" + model) {
			t.Errorf("%s is not routed to Omni video", model)
		}
	}
	for _, model := range []string{"gemini-2.5-flash", "gemini-4-argon", "imagen-4.0-generate-001"} {
		if isGeminiOmniVideoModel(model) {
			t.Errorf("%s is routed to Omni video", model)
		}
	}
}

// TestOmniVideoTextToVideo mirrors "constructs a text-to-video request with
// defaults and the global beta endpoint".
func TestOmniVideoTextToVideo(t *testing.T) {
	t.Parallel()

	server := &omniServer{body: omniCompleted(omniOutputPrefix + "video.mp4")}
	driver := server.start(t)
	response, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleSystem, Content: "Cinematic"}, {Role: PromptRoleUser, Content: "A fox in snow"}}, omniOptions(geminiOmniVideoModel, nil))
	if err != nil {
		t.Fatal(err)
	}
	if server.paths[0] != "/projects/project-1/locations/global/interactions" {
		t.Fatalf("path = %s", server.paths[0])
	}
	want := map[string]any{
		"model":             geminiOmniVideoModel,
		"input":             []any{map[string]any{"type": "text", "text": "Cinematic\nA fox in snow"}},
		"response_format":   []any{map[string]any{"type": "video", "delivery": "uri", "gcs_uri": omniOutputPrefix, "duration": "5s"}},
		"generation_config": map[string]any{"video_config": map[string]any{"task": "text_to_video"}},
	}
	if !reflect.DeepEqual(server.payloads[0], want) {
		t.Fatalf("payload = %#v", server.payloads[0])
	}
	if len(response.Result) != 1 || response.Result[0].Type() != "video" || response.Result[0].String() != omniOutputPrefix+"video.mp4" || response.FinishReason != "stop" {
		t.Fatalf("result = %#v", response.Completion)
	}
	if usage := response.TokenUsage; usage == nil || *usage.Total != 9 || *usage.Prompt != 3 || *usage.Result != 6 {
		t.Fatalf("usage = %#v", response.TokenUsage)
	}
}

// TestOmniVideoMediaRequests mirrors the media mapping cases of
// omni-video.test.ts.
func TestOmniVideoMediaRequests(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name           string
		model          string
		modelOptions   map[string]any
		files          []DataSource
		wantInputTypes []string
		wantFormat     map[string]any
		wantGeneration map[string]any
	}{
		{"first frame", geminiOmniVideoModel, map[string]any{"task": "image_to_video", "aspect_ratio": "9:16", "duration_seconds": 10}, []DataSource{omniImage(t, "gs://project-bucket/input/frame.png")}, []string{"image"},
			map[string]any{"type": "video", "delivery": "uri", "gcs_uri": omniOutputPrefix, "duration": "10s", "aspect_ratio": "9:16"}, map[string]any{"video_config": map[string]any{"task": "image_to_video"}}},
		{"first and last frames", geminiOmni11VideoModel, map[string]any{"task": "image_to_video", "resolution": "4k"}, []DataSource{omniImage(t, "gs://inputs/first.png"), omniImage(t, "gs://inputs/last.png")}, []string{"image", "image"},
			map[string]any{"type": "video", "delivery": "uri", "gcs_uri": omniOutputPrefix, "duration": "5s", "resolution": "4k"}, map[string]any{"video_config": map[string]any{"task": "image_to_video"}}},
		{"references", geminiOmniVideoModel, map[string]any{"task": "reference_to_video"}, []DataSource{omniImage(t, "gs://inputs/1.png"), omniImage(t, "gs://inputs/2.png"), omniImage(t, "gs://inputs/3.png")}, []string{"image", "image", "image"}, nil, nil},
		{"1.1 references", geminiOmni11VideoModel, map[string]any{"task": "reference_to_video"}, []DataSource{omniImage(t, "gs://inputs/reference.png"), omniVideo(t, "gs://inputs/reference.mp4")}, []string{"image", "video"}, nil, nil},
		{"1.1 edit", geminiOmni11VideoModel, map[string]any{"task": "edit"}, []DataSource{omniVideo(t, "gs://inputs/source.mp4")}, []string{"video"}, nil, nil},
		{"1.1 extend", geminiOmni11VideoModel, map[string]any{"task": "extend"}, []DataSource{omniVideo(t, "gs://inputs/source.mp4")}, []string{"video"}, nil, nil},
		{"1.0 edit", geminiOmniVideoModel, map[string]any{"task": "edit"}, []DataSource{omniVideo(t, "gs://project-bucket/input/clip.mp4")}, []string{"video"}, nil, nil},
		{"1.2 upscale", geminiOmni12VideoModel, map[string]any{"task": "upscale", "resolution": "1080p", "seed": 42}, []DataSource{omniVideo(t, "gs://inputs/source.mp4")}, []string{"video"},
			map[string]any{"type": "video", "delivery": "uri", "gcs_uri": omniOutputPrefix, "resolution": "1080p"}, map[string]any{"video_config": map[string]any{"task": "upscale"}, "seed": 42.0}},
		{"1.2 duration and seed", geminiOmni12VideoModel, map[string]any{"duration_seconds": 15, "seed": 0}, nil, nil,
			map[string]any{"type": "video", "delivery": "uri", "gcs_uri": omniOutputPrefix, "duration": "15s"}, map[string]any{"video_config": map[string]any{"task": "text_to_video"}, "seed": 0.0}},
	}
	for _, model := range []string{geminiOmni12VideoModel, "gemini-omni-1.3-flash"} {
		for _, task := range []string{"edit", "extend"} {
			cases = append(cases, struct {
				name           string
				model          string
				modelOptions   map[string]any
				files          []DataSource
				wantInputTypes []string
				wantFormat     map[string]any
				wantGeneration map[string]any
			}{model + " " + task + " with references", model, map[string]any{"task": task}, []DataSource{omniVideo(t, "gs://inputs/clip.mp4"), omniImage(t, "gs://inputs/one.png"), omniImage(t, "gs://inputs/two.png")}, []string{"video", "image", "image"}, nil, nil})
		}
	}
	for _, tc := range cases {
		server := &omniServer{body: omniCompleted(omniOutputPrefix+"one.mp4", omniOutputPrefix+"two.mp4")}
		driver := server.start(t)
		_, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Transform this", Files: tc.files}}, omniOptions(tc.model, tc.modelOptions))
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		payload := server.payloads[0]
		input := payload["input"].([]any)
		var types []string
		for i, item := range input[1:] {
			entry := item.(map[string]any)
			types = append(types, entry["type"].(string))
			want, _ := tc.files[i].URL(context.Background())
			if entry["uri"] != want || entry["mime_type"] != tc.files[i].MIMEType() {
				t.Errorf("%s: input %d = %#v", tc.name, i, entry)
			}
		}
		if !reflect.DeepEqual(types, tc.wantInputTypes) {
			t.Errorf("%s: input types = %v, want %v", tc.name, types, tc.wantInputTypes)
		}
		if tc.wantFormat != nil && !reflect.DeepEqual(payload["response_format"], []any{tc.wantFormat}) {
			t.Errorf("%s: response_format = %#v", tc.name, payload["response_format"])
		}
		if tc.wantGeneration != nil && !reflect.DeepEqual(payload["generation_config"], tc.wantGeneration) {
			t.Errorf("%s: generation_config = %#v", tc.name, payload["generation_config"])
		}
	}
}

// TestOmniVideoRejectsInvalidRequests mirrors the rejection cases of
// omni-video.test.ts: every request error is a non-retryable 400 sent nowhere.
func TestOmniVideoRejectsInvalidRequests(t *testing.T) {
	t.Parallel()

	images := func(n int) []DataSource {
		out := make([]DataSource, n)
		for i := range out {
			out[i] = omniImage(t, "gs://inputs/"+string(rune('a'+i))+".png")
		}
		return out
	}
	videos := func(n int) []DataSource {
		out := make([]DataSource, n)
		for i := range out {
			out[i] = omniVideo(t, "gs://inputs/"+string(rune('a'+i))+".mp4")
		}
		return out
	}
	cases := []struct {
		model        string
		modelOptions map[string]any
		segments     []PromptSegment
		options      func(*ExecutionOptions)
		message      string
	}{
		{geminiOmniVideoModel, map[string]any{"task": "extend"}, nil, nil, "does not support extend"},
		{geminiOmniVideoModel, map[string]any{"task": "image_to_video"}, []PromptSegment{{Content: "p", Files: images(2)}}, nil, "exactly one"},
		{geminiOmni11VideoModel, map[string]any{"task": "edit"}, []PromptSegment{{Content: "p", Files: images(1)}}, nil, "exactly one video"},
		{geminiOmni11VideoModel, map[string]any{"task": "extend"}, []PromptSegment{{Content: "p", Files: videos(2)}}, nil, "exactly one video"},
		{geminiOmni12VideoModel, map[string]any{"task": "edit"}, []PromptSegment{{Content: "p", Files: append(videos(1), images(3)...)}}, nil, "at most 2 reference images"},
		{geminiOmni12VideoModel, map[string]any{"task": "upscale"}, []PromptSegment{{Content: "p", Files: append(videos(1), images(1)...)}}, nil, "upscale requires exactly one video"},
		{geminiOmni12VideoModel, map[string]any{"task": "upscale"}, []PromptSegment{{Content: "p", Files: images(1)}}, nil, "upscale requires exactly one video"},
		{geminiOmni12VideoModel, map[string]any{"task": "upscale"}, []PromptSegment{{Content: "p", Files: videos(1)}}, nil, "upscale requires a target resolution"},
		{geminiOmni12VideoModel, map[string]any{"task": "upscale", "resolution": "360p"}, []PromptSegment{{Content: "p", Files: videos(1)}}, nil, "upscale requires a target resolution: 720p, 1080p or 4k"},
		{geminiOmni11VideoModel, map[string]any{"task": "upscale"}, []PromptSegment{{Content: "p", Files: videos(1)}}, nil, "does not support upscale"},
		{geminiOmni11VideoModel, map[string]any{"task": "reference_to_video"}, []PromptSegment{{Content: "p", Files: images(11)}}, nil, "ten images"},
		{geminiOmni11VideoModel, map[string]any{"task": "reference_to_video"}, []PromptSegment{{Content: "p", Files: videos(4)}}, nil, "three videos"},
		{geminiOmniVideoModel, map[string]any{"resolution": "1080p"}, nil, nil, "only supports 720p"},
		{geminiOmniVideoModel, nil, []PromptSegment{{Content: "Use this", Files: images(1)}}, nil, "requires an explicit"},
		{geminiOmniVideoModel, nil, []PromptSegment{{Content: "  "}}, nil, "non-empty"},
		{geminiOmniVideoModel, map[string]any{"task": "image_to_video"}, []PromptSegment{{Content: "p", Files: []DataSource{gcsSource{t: t, uri: "gs://inputs/a.gif", mime: "image/gif"}}}}, nil, "MIME type"},
		{geminiOmniVideoModel, map[string]any{"task": "edit"}, []PromptSegment{{Content: "p", Files: []DataSource{gcsSource{t: t, uri: "gs://inputs/a.wav", mime: "audio/wav"}}}}, nil, "MIME type"},
		{geminiOmniVideoModel, map[string]any{"task": "image_to_video"}, []PromptSegment{{Content: "p", Files: []DataSource{omniImage(t, "https://example.com/a.png")}}}, nil, "GCS object URI"},
		{geminiOmniVideoModel, nil, nil, func(o *ExecutionOptions) {
			o.Tools = []ToolDefinition{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}}
		}, "tools"},
		{geminiOmniVideoModel, nil, nil, func(o *ExecutionOptions) { o.ResultSchema = map[string]any{"type": "object"} }, "result schemas"},
		{geminiOmniVideoModel, nil, nil, func(o *ExecutionOptions) { o.Conversation = map[string]any{"id": "interaction-0"} }, "conversation resume"},
		{geminiOmniVideoModel, nil, nil, func(o *ExecutionOptions) { o.OutputStorageURI = "https://example.com/out/" }, "valid GCS output prefix"},
	}
	server := &omniServer{body: omniCompleted(omniOutputPrefix + "video.mp4")}
	driver := server.start(t)
	for _, tc := range cases {
		segments := tc.segments
		if segments == nil {
			segments = []PromptSegment{{Content: "prompt"}}
		}
		for i := range segments {
			segments[i].Role = PromptRoleUser
		}
		options := omniOptions(tc.model, tc.modelOptions)
		if tc.options != nil {
			tc.options(&options)
		}
		_, err := driver.Execute(context.Background(), segments, options)
		got := asLlumiverseError(t, err)
		if got.Code != 400 || retryableString(got.Retryable) != "false" || !strings.Contains(got.Message, tc.message) {
			t.Errorf("%s %v: error = %#v, want %q", tc.model, tc.modelOptions, got, tc.message)
		}
	}
	if len(server.payloads) != 0 {
		t.Fatalf("%d rejected requests were sent", len(server.payloads))
	}
	_, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "prompt"}}, omniOptions(geminiOmniVideoModel, nil))
	if got := asLlumiverseError(t, err); got.Code != 400 || !strings.Contains(got.Message, "does not support streaming") {
		t.Fatalf("stream error = %#v", got)
	}
}

// TestOmniVideoOutput mirrors the output and interaction-status cases of
// omni-video.test.ts.
func TestOmniVideoOutput(t *testing.T) {
	t.Parallel()

	run := func(body string, options ExecutionOptions) (*ExecutionResponse, error) {
		server := &omniServer{body: body}
		return server.start(t).Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "prompt"}}, options)
	}

	response, err := run(omniCompleted("text:Generated video", omniOutputPrefix+"result.mp4"), omniOptions(geminiOmni11VideoModel, nil))
	if err != nil || len(response.Result) != 2 || response.Result[0].String() != "Generated video" || response.Result[1].String() != omniOutputPrefix+"result.mp4" {
		t.Fatalf("text and video = %#v, %v", response, err)
	}
	original := omniOptions(geminiOmniVideoModel, nil)
	original.IncludeOriginalResponse = true
	if response, err := run(omniCompleted(omniOutputPrefix+"result.mp4"), original); err != nil || response.OriginalResponse.(map[string]any)["id"] != "interaction-1" {
		t.Fatalf("original response = %#v, %v", response, err)
	}

	for body, message := range map[string]string{
		`{"status":"in_progress"}`:                     "did not complete",
		omniCompleted():                                "without a video",
		omniCompleted("gs://foreign-bucket/video.mp4"): "outside the requested output prefix",
		omniCompleted("https://example.com/video.mp4"): "invalid GCS video URI",
		`{"id":"interaction-1","status":"completed","steps":[{"type":"model_output","content":[{"type":"video","data":"AAAA","mime_type":"video/mp4"}]}]}`: "inline video data",
	} {
		_, err := run(body, omniOptions(geminiOmniVideoModel, nil))
		got := asLlumiverseError(t, err)
		if !strings.Contains(got.Message, message) {
			t.Errorf("%s: error = %#v, want %q", body, got, message)
		}
		if message != "did not complete" && (got.Code != 0 || retryableString(got.Retryable) != "false") {
			t.Errorf("%s: invalid output error = %#v", body, got)
		}
	}

	for _, tc := range []struct {
		status, errors, retryable string
	}{
		{"in_progress", "", "true"},
		{"incomplete", "", "true"},
		{"requires_action", "", "false"},
		{"cancelled", "", "false"},
		{"budget_exceeded", "", "false"},
		{"failed", `[{"code":"RESOURCE_EXHAUSTED","message":"Try later"}]`, "true"},
		{"failed", `[{"code":"PERMISSION_DENIED","message":"Forbidden"}]`, "false"},
		{"failed", `[{"code":"UNKNOWN","message":"Unknown failure"}]`, "unknown"},
	} {
		body := `{"id":"interaction-1","status":"` + tc.status + `"`
		if tc.errors != "" {
			body += `,"errors":` + tc.errors
		}
		_, err := run(body+"}", omniOptions(geminiOmniVideoModel, nil))
		if got := asLlumiverseError(t, err); retryableString(got.Retryable) != tc.retryable {
			t.Errorf("%s %s: retryable = %s, want %s", tc.status, tc.errors, retryableString(got.Retryable), tc.retryable)
		}
	}

	if got := asLlumiverseError(t, formatOmniVideoError(contextCanceledError{}, geminiOmniVideoModel, "execute")); retryableString(got.Retryable) != "false" {
		t.Fatalf("cancellation = %#v", got)
	}
	if got := asLlumiverseError(t, formatOmniVideoError(&fakeStatusError{code: 504, body: "gateway timeout"}, geminiOmniVideoModel, "execute")); got.Code != 504 || retryableString(got.Retryable) != "true" {
		t.Fatalf("transport failure = %#v", got)
	}
}

// contextCanceledError wraps context.Canceled, as an aborted request returns it.
type contextCanceledError struct{}

func (contextCanceledError) Error() string { return "request cancelled" }
func (contextCanceledError) Unwrap() error { return context.Canceled }
