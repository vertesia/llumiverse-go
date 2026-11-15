package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// imageServer answers the Images API and records each request.
func imageServer(t *testing.T, response string) (*OpenAICompatibleDriver, *[]string, *[]string) {
	t.Helper()
	var paths, bodies []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		paths = append(paths, r.URL.Path)
		bodies = append(bodies, string(data))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(response))
	}))
	t.Cleanup(server.Close)
	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "k", Endpoint: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	return driver, &paths, &bodies
}

func decodeBody(t *testing.T, body string) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal([]byte(body), &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// Mirrors the TS image generation tests: per-family defaults, width and
// height as a size, GPT Image 2.6 options passed through, and SVG output with
// its authoring usage.
func TestOpenAIImageGenerationOptions(t *testing.T) {
	t.Parallel()

	prompt := []PromptSegment{{Role: PromptRoleUser, Content: "a fox"}}
	driver, _, bodies := imageServer(t, `{"data":[{"b64_json":"aW1n"}],"output_format":"png"}`)
	if _, err := driver.Execute(context.Background(), prompt, ExecutionOptions{Model: "gpt-image-2.6"}); err != nil {
		t.Fatal(err)
	}
	if got := marshalNoEscape(decodeBody(t, (*bodies)[0])); got != `{"model":"gpt-image-2.6","n":1,"output_format":"png","prompt":"a fox","quality":"auto"}` {
		t.Fatalf("gpt image defaults = %s", got)
	}
	if _, err := driver.Execute(context.Background(), prompt, ExecutionOptions{Model: "dall-e-3"}); err != nil {
		t.Fatal(err)
	}
	if got := marshalNoEscape(decodeBody(t, (*bodies)[1])); got != `{"model":"dall-e-3","n":1,"prompt":"a fox","quality":"standard","response_format":"b64_json","size":"1024x1024"}` {
		t.Fatalf("dall-e defaults = %s", got)
	}
	if _, err := driver.Execute(context.Background(), prompt, ExecutionOptions{Model: "gpt-image-2.6", ModelOptions: map[string]any{
		"width": 1536, "height": 1024, "output_format": "gif", "frame_count": 4, "frame_duration_ms": 120, "complexity": "high",
	}}); err != nil {
		t.Fatal(err)
	}
	body := decodeBody(t, (*bodies)[2])
	if body["size"] != "1536x1024" || body["output_format"] != "gif" || body["frame_count"] != float64(4) || body["frame_duration_ms"] != float64(120) || body["complexity"] != "high" {
		t.Fatalf("gif body = %#v", body)
	}
	if _, err := driver.Execute(context.Background(), prompt, ExecutionOptions{Model: "gpt-image-2.6", ModelOptions: map[string]any{"width": 1024}}); err == nil || !strings.Contains(err.Error(), "width and height") {
		t.Fatalf("lone width error = %v", err)
	}

	driver, _, bodies = imageServer(t, `{"data":[{"b64_json":"PHN2Zz4="}],"output_format":"svg","usage":{"input_tokens":10,"output_tokens":300,"total_tokens":310,"output_tokens_details":{"image_tokens":290}},"svg_usage":{"model":"gpt-5.6","input_tokens":5,"output_tokens":80,"total_tokens":85}}`)
	resp, err := driver.Execute(context.Background(), prompt, ExecutionOptions{Model: "gpt-image-2.6", ModelOptions: map[string]any{"output_format": "svg", "svg": map[string]any{"model": "gpt-5.6"}}})
	if err != nil {
		t.Fatal(err)
	}
	if decodeBody(t, (*bodies)[0])["svg"] == nil {
		t.Fatalf("svg settings missing: %s", (*bodies)[0])
	}
	if got := resultsOfType(resp.Result, "image"); len(got) != 1 || got[0] != "data:image/svg+xml;base64,PHN2Zz4=" {
		t.Fatalf("svg result = %#v", got)
	}
	if got := marshalNoEscape(resp.TokenUsage); got != `{"prompt":10,"prompt_new":10,"result":300,"result_image":290,"tool_usage":{"svg":{"model":"gpt-5.6","prompt":5,"result":80,"total":85}},"total":310}` {
		t.Fatalf("usage = %s", got)
	}

	driver, _, _ = imageServer(t, `{"data":[]}`)
	if _, err := driver.Execute(context.Background(), prompt, ExecutionOptions{Model: "gpt-image-2.6"}); err == nil || !strings.Contains(err.Error(), "completed image") {
		t.Fatalf("empty image error = %v", err)
	}
}

// Mirrors the TS image edit tests: reference images select the edit endpoint
// as multipart with the mask, and a mask needs a reference image.
func TestOpenAIImageEdit(t *testing.T) {
	t.Parallel()

	driver, paths, bodies := imageServer(t, `{"data":[{"b64_json":"aW1n"}]}`)
	source := &BytesDataSource{FileName: "room.png", MIME: "image/png", Data: []byte("room-bytes")}
	mask := &BytesDataSource{FileName: "mask.png", MIME: "image/png", Data: []byte("mask-bytes")}
	segments := []PromptSegment{
		{Role: PromptRoleUser, Content: "add a lamp", Files: []DataSource{source}},
		{Role: PromptRoleMask, Files: []DataSource{mask}},
	}
	if _, err := driver.Execute(context.Background(), segments, ExecutionOptions{Model: "gpt-image-1.5", ModelOptions: map[string]any{"moderation": "low", "input_fidelity": "high", "image_quality": "hd"}}); err != nil {
		t.Fatal(err)
	}
	body := (*bodies)[0]
	if !strings.HasSuffix((*paths)[0], "/images/edits") {
		t.Fatalf("path = %s", (*paths)[0])
	}
	for _, want := range []string{`name="image[]"; filename="room.png"`, "room-bytes", `name="mask"; filename="mask.png"`, "mask-bytes", "add a lamp", `name="input_fidelity"`} {
		if !strings.Contains(body, want) {
			t.Fatalf("edit body missing %q:\n%s", want, body)
		}
	}
	if strings.Contains(body, `name="moderation"`) || strings.Contains(body, "\r\nhd\r\n") {
		t.Fatalf("edit body kept generation-only fields:\n%s", body)
	}

	if _, err := driver.Execute(context.Background(), segments, ExecutionOptions{Model: "gpt-image-2", ModelOptions: map[string]any{"input_fidelity": "high"}}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains((*bodies)[1], `name="input_fidelity"`) {
		t.Fatal("gpt-image-2 edit kept input_fidelity")
	}

	_, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "x"}, {Role: PromptRoleMask, Files: []DataSource{mask}}}, ExecutionOptions{Model: "gpt-image-1.5"})
	if err == nil || !strings.Contains(err.Error(), "reference image") {
		t.Fatalf("mask without image error = %v", err)
	}

	stream, err := driver.Stream(context.Background(), segments[:1], ExecutionOptions{Model: "gpt-image-1.5"})
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
	if got := resultsOfType(stream.Completion().Result, "image"); len(got) != 1 {
		t.Fatalf("streamed images = %#v", got)
	}
}

// Mirrors the TS Responses image_generation tool tests: the tool carries the
// settings and the mask, and force selects it.
func TestOpenAIResponsesImageGenerationTool(t *testing.T) {
	t.Parallel()

	mask := &BytesDataSource{FileName: "mask.png", MIME: "image/png", Data: []byte("m")}
	body, resp, err := executeResponses(t, ExecutionOptions{Model: "gpt-5.6", ModelOptions: map[string]any{
		"image_generation": map[string]any{"model": "deployment::gpt-image-2.6", "force": true, "output_format": "webp", "quality": "high"},
	}}, `{"id":"r","status":"completed","output":[{"type":"image_generation_call","id":"ig","status":"completed","result":"aW1n"}]}`,
		PromptSegment{Role: PromptRoleUser, Content: "draw"}, PromptSegment{Role: PromptRoleMask, Files: []DataSource{mask}})
	if err != nil {
		t.Fatal(err)
	}
	if got := marshalNoEscape(body["tools"]); got != `[{"input_image_mask":{"image_url":"data:image/png;base64,bQ=="},"model":"gpt-image-2.6","output_format":"webp","quality":"high","type":"image_generation"}]` {
		t.Fatalf("tools = %s", got)
	}
	if got := marshalNoEscape(body["tool_choice"]); got != `{"type":"image_generation"}` {
		t.Fatalf("tool_choice = %s", got)
	}
	if got := resultsOfType(resp.Result, "image"); len(got) != 1 || got[0] != "data:image/webp;base64,aW1n" {
		t.Fatalf("images = %#v", got)
	}

	_, _, err = executeResponses(t, ExecutionOptions{ModelOptions: map[string]any{"image_generation": map[string]any{"quality": "high"}}}, okResponse)
	if err == nil || !strings.Contains(err.Error(), "requires an image model") {
		t.Fatalf("missing model error = %v", err)
	}
}
