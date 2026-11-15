package vertexai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

// streamingSource is a DataSource whose stream yields size zero bytes and counts
// what was read; it fails the test if a Cloud Storage URI is read.
type streamingSource struct {
	t      *testing.T
	name   string
	mime   string
	uri    string
	size   int64
	read   *int64
	opened *bool
}

func (s streamingSource) Name() string     { return s.name }
func (s streamingSource) MIMEType() string { return s.mime }
func (s streamingSource) URL(context.Context) (string, error) {
	if s.uri == "" {
		return "", errors.New("no URL")
	}
	return s.uri, nil
}
func (s streamingSource) Open(context.Context) (io.ReadCloser, error) {
	if s.opened != nil {
		*s.opened = true
	}
	return io.NopCloser(&countingReader{remaining: s.size, read: s.read}), nil
}

type countingReader struct {
	remaining int64
	read      *int64
}

func (r *countingReader) Read(p []byte) (int, error) {
	if r.remaining <= 0 {
		return 0, io.EOF
	}
	if int64(len(p)) > r.remaining {
		p = p[:r.remaining]
	}
	clear(p)
	r.remaining -= int64(len(p))
	if r.read != nil {
		*r.read += int64(len(p))
	}
	return len(p), nil
}

func toolSegmentWith(files ...DataSource) PromptSegment {
	return PromptSegment{Role: PromptRoleTool, ToolUseID: "view_image", Content: `{"artifact_path":"out/plot.png"}`, Files: files}
}

// TestGeminiToolResultAttachments mirrors gemini-tool-result-files.test.ts.
func TestGeminiToolResultAttachments(t *testing.T) {
	t.Parallel()
	driver := newTestVertexDriver(t, "http://127.0.0.1:1")
	ctx := context.Background()

	opened := false
	gcsAudio := streamingSource{t: t, name: "recording.mp3", mime: "audio/mpeg", uri: "gs://bucket/recording.mp3", opened: &opened}
	prompt, err := driver.formatGeminiPrompt(ctx, []PromptSegment{{Role: PromptRoleUser, Content: "Transcribe", Files: []DataSource{gcsAudio}}}, ExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if file := prompt.Contents[0].Parts[1].FileData; file == nil || file.FileURI != "gs://bucket/recording.mp3" || file.MIMEType != "audio/mpeg" || opened {
		t.Fatalf("GCS audio part = %#v (opened=%v)", prompt.Contents[0].Parts[1], opened)
	}

	image := BytesDataSource{FileName: "plot.png", MIME: "image/jpeg", Data: []byte{1, 2, 3}, URI: "https://signed.example/plot.png"}
	prompt, err = driver.formatGeminiPrompt(ctx, []PromptSegment{toolSegmentWith(image)}, ExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	response := prompt.Contents[0].Parts[0].FunctionResponse
	if response.Name != "view_image" || response.Response["artifact_path"] != "out/plot.png" || len(response.Parts) != 1 ||
		response.Parts[0].InlineData == nil || response.Parts[0].InlineData.Data != "AQID" || response.Parts[0].InlineData.MIMEType != "image/jpeg" {
		t.Fatalf("function response = %#v", response)
	}

	prompt, err = driver.formatGeminiPrompt(ctx, []PromptSegment{toolSegmentWith(BytesDataSource{FileName: "plot.png", MIME: "image/png", URI: "gs://bucket/plot.png"})}, ExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if parts := prompt.Contents[0].Parts[0].FunctionResponse.Parts; len(parts) != 1 || parts[0].FileData == nil || parts[0].FileData.FileURI != "gs://bucket/plot.png" {
		t.Fatalf("GCS attachment = %#v", parts)
	}

	prompt, err = driver.formatGeminiPrompt(ctx, []PromptSegment{toolSegmentWith()}, ExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if parts := prompt.Contents[0].Parts[0].FunctionResponse.Parts; parts != nil {
		t.Fatalf("function response without attachments has parts: %#v", parts)
	}
}

func TestGeminiInlineMediaPerFileLimit(t *testing.T) {
	t.Parallel()
	driver := newTestVertexDriver(t, "http://127.0.0.1:1")

	var read int64
	video := streamingSource{t: t, name: "video.mp4", mime: "video/mp4", uri: "https://bucket.s3.amazonaws.com/video.mp4?X-Amz-Signature=x", size: 1 << 40, read: &read}
	_, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Describe", Files: []DataSource{video}}}, ExecutionOptions{Model: "gemini-3-pro"})
	got := asLlumiverseError(t, err)
	if got.Code != http.StatusRequestEntityTooLarge || retryableString(got.Retryable) != "false" || got.Name != "GeminiInlineMediaLimitError" {
		t.Fatalf("error = %#v", got)
	}
	if !strings.Contains(got.Message, "video.mp4 exceeds the 100000000 byte limit for inline Gemini input") {
		t.Fatalf("message = %q", got.Message)
	}
	// Reading stops just past the limit instead of buffering the whole source.
	if read > geminiInlineMediaMaxBytes+64*1024 {
		t.Fatalf("read %d bytes", read)
	}
}

func TestGeminiInlineMediaPerRequestLimit(t *testing.T) {
	t.Parallel()
	driver := newTestVertexDriver(t, "http://127.0.0.1:1")

	video := func(name string, size int64) DataSource {
		return streamingSource{t: t, name: name, mime: "video/mp4", size: size}
	}
	segments := []PromptSegment{{Role: PromptRoleUser, Content: "Describe", Files: []DataSource{video("a.mp4", 60_000_000), video("b.mp4", 60_000_000), video("c.mp4", 60_000_000)}}}
	_, err := driver.Stream(context.Background(), segments, ExecutionOptions{Model: "gemini-3-pro"})
	got := asLlumiverseError(t, err)
	if got.Code != http.StatusRequestEntityTooLarge || got.Context.Operation != "stream" || !strings.Contains(got.Message, "exceeds the 150000000 byte limit at c.mp4") {
		t.Fatalf("error = %#v", got)
	}

	// Inline media a resumed conversation already carries counts toward the total.
	history := strings.Repeat("A", 150_000_000) // 112.5 MB decoded
	conversation := geminiPrompt{Contents: []geminiContent{{Role: "user", Parts: []geminiPart{{InlineData: &geminiInlineData{MIMEType: "video/mp4", Data: history}}}}}}
	_, err = driver.formatGeminiPrompt(context.Background(), []PromptSegment{{Role: PromptRoleUser, Files: []DataSource{video("d.mp4", 40_000_000)}}}, ExecutionOptions{Conversation: conversation})
	var limitErr *geminiInlineMediaLimitError
	if !errors.As(err, &limitErr) {
		t.Fatalf("error = %v", err)
	}
}

func TestGeminiInlineAudioLimit(t *testing.T) {
	t.Parallel()
	driver := newTestVertexDriver(t, "http://127.0.0.1:1")

	audio := streamingSource{t: t, name: "speech.wav", mime: "audio/wav", size: geminiInlineAudioMaxBytes + 1}
	_, err := driver.formatGeminiPrompt(context.Background(), []PromptSegment{{Role: PromptRoleUser, Files: []DataSource{audio}}}, ExecutionOptions{})
	if err == nil || !strings.Contains(err.Error(), "audio exceeds the 25000000 byte limit") {
		t.Fatalf("error = %v", err)
	}
	empty := streamingSource{t: t, name: "empty.wav", mime: "audio/wav"}
	if _, err := driver.formatGeminiPrompt(context.Background(), []PromptSegment{{Role: PromptRoleUser, Files: []DataSource{empty}}}, ExecutionOptions{}); err == nil {
		t.Fatal("expected an empty audio error")
	}
}

func TestGeminiConversationDropsExpiredToolResultMedia(t *testing.T) {
	t.Parallel()

	parts := processGeminiPartsForConversation([]geminiPart{{FunctionResponse: &geminiFunctionResult{
		Name:     "view_image",
		Response: map[string]any{"ok": true},
		Parts: []geminiFunctionMediaPart{
			{InlineData: &geminiInlineData{MIMEType: "image/png", Data: "AQID"}},
			{FileData: &geminiFileData{MIMEType: "image/png", FileURI: "gs://bucket/a.png"}},
		},
	}}}, ExecutionOptions{}, true, false)
	if media := parts[0].FunctionResponse.Parts; len(media) != 1 || media[0].FileData == nil {
		t.Fatalf("media = %#v", media)
	}
}
