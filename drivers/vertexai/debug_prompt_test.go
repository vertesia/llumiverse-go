package vertexai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/core"
	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
	openaidriver "github.com/vertesia/llumiverse-go/drivers/openai"
	"golang.org/x/oauth2"
)

func newTestVertexDriver(t *testing.T, baseURL string) *VertexAIDriver {
	t.Helper()
	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     baseURL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	return driver
}

func TestVertexGeminiDebugPromptTruncatesInlineMedia(t *testing.T) {
	t.Parallel()

	image := bytes.Repeat([]byte("p"), 300)
	encoded := base64.StdEncoding.EncodeToString(image)
	var sent string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body bytes.Buffer
		_, _ = body.ReadFrom(r.Body)
		sent = body.String()
		_, _ = w.Write([]byte(`{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"text":"ok"}]}}]}`))
	}))
	defer server.Close()
	driver := newTestVertexDriver(t, server.URL)

	segments := []PromptSegment{{Role: PromptRoleUser, Content: "Describe", Files: []DataSource{BytesDataSource{FileName: "a.png", MIME: "image/png", Data: image}}}}
	resp, err := driver.Execute(context.Background(), segments, ExecutionOptions{Model: "gemini-2.5-flash"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sent, encoded) {
		t.Fatal("request did not carry the full inline media")
	}
	debug, ok := resp.Prompt.(geminiPrompt)
	if !ok {
		t.Fatalf("prompt = %T", resp.Prompt)
	}
	data := debug.Contents[0].Parts[1].InlineData.Data
	if data != core.TruncateBinaryForDebug(encoded) || !strings.Contains(data, core.BinaryTruncatedMarker) {
		t.Fatalf("debug inline data = %q", data)
	}

	// The formatter never modifies the prompt it is given.
	prompt, err := driver.CreatePrompt(context.Background(), segments, ExecutionOptions{Model: "gemini-2.5-flash"})
	if err != nil {
		t.Fatal(err)
	}
	_ = driver.FormatDebugPrompt(prompt)
	if prompt.(geminiPrompt).Contents[0].Parts[1].InlineData.Data != encoded {
		t.Fatal("FormatDebugPrompt modified the prompt")
	}
}

func TestVertexImagenDebugPromptTruncatesReferenceImages(t *testing.T) {
	t.Parallel()

	encoded := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte("r"), 300))
	prompt := vertexImagenPrompt{Prompt: "edit", ReferenceImages: []map[string]any{
		{"referenceType": "REFERENCE_TYPE_RAW", "referenceId": 1, "referenceImage": map[string]any{"bytesBase64Encoded": encoded}},
		{"referenceType": "REFERENCE_TYPE_MASK", "referenceId": 2},
	}}
	before, _ := json.Marshal(prompt)
	debug := (&VertexAIDriver{}).FormatDebugPrompt(prompt).(vertexImagenPrompt)
	after, _ := json.Marshal(prompt)
	if !bytes.Equal(before, after) {
		t.Fatal("FormatDebugPrompt modified the Imagen prompt")
	}
	got := debug.ReferenceImages[0]["referenceImage"].(map[string]any)["bytesBase64Encoded"]
	if got != core.TruncateBinaryForDebug(encoded) {
		t.Fatalf("debug reference image = %v", got)
	}
	if debug.ReferenceImages[1]["referenceId"] != 2 {
		t.Fatalf("reference without image = %#v", debug.ReferenceImages[1])
	}
}

func TestVertexDebugPromptLeavesOtherPromptsUnchanged(t *testing.T) {
	t.Parallel()

	driver := &VertexAIDriver{}
	claudePrompt := internalclaude.Prompt{}
	if _, ok := driver.FormatDebugPrompt(claudePrompt).(internalclaude.Prompt); !ok {
		t.Fatal("Claude prompt was not formatted as a Claude prompt")
	}
	if got := driver.FormatDebugPrompt("raw"); got != "raw" {
		t.Fatalf("other prompt = %#v", got)
	}

	// Claude prompts have their inline images truncated too.
	encoded := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte("p"), 300))
	prompt, err := internalclaude.FormatPrompt(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Describe", Files: []DataSource{BytesDataSource{FileName: "a.png", MIME: "image/png", Data: bytes.Repeat([]byte("p"), 300)}}}}, ExecutionOptions{Model: "claude-sonnet-4-6"})
	if err != nil {
		t.Fatal(err)
	}
	debug, _ := json.Marshal(driver.FormatDebugPrompt(prompt))
	sent, _ := json.Marshal(prompt)
	if strings.Contains(string(debug), encoded) || !strings.Contains(string(debug), core.BinaryTruncatedMarker) || !strings.Contains(string(sent), encoded) {
		t.Fatalf("Claude debug prompt = %s", debug)
	}

	// Open MaaS (Chat Completions) prompts go to the OpenAI formatter.
	maas := openaidriver.ChatCompletionsPrompt{Messages: []map[string]any{{"role": "user", "content": []any{map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64," + encoded}}}}}}
	debug, _ = json.Marshal(driver.FormatDebugPrompt(maas))
	if strings.Contains(string(debug), encoded) || !strings.Contains(string(debug), core.BinaryTruncatedMarker) {
		t.Fatalf("MaaS debug prompt = %s", debug)
	}
}
