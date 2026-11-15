package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/core"
)

// Mirrors the TS debug-prompt test: OpenAI input images and files keep only the
// edges of their base64 payload, and the prompt being sent is untouched.
func TestOpenAIFormatDebugPromptTruncatesDataURLs(t *testing.T) {
	t.Parallel()

	image := "data:image/png;base64," + strings.Repeat("b", 80)
	file := "data:application/pdf;base64," + strings.Repeat("f", 80)
	prompt := openAIResponsesPrompt{Items: []responseItem{{
		"role": "user",
		"content": []any{
			map[string]any{"type": "input_image", "image_url": image, "detail": "auto"},
			map[string]any{"type": "input_file", "filename": "document.pdf", "file_data": file},
		},
	}}}

	debug := formatOpenAIDebugPrompt(prompt).(openAIResponsesPrompt)
	content := debug.Items[0]["content"].([]any)
	if got := content[0].(map[string]any)["image_url"]; got != "data:image/png;base64,bbbbbbbbbb"+core.BinaryTruncatedMarker+"bbbbbbbbbb" {
		t.Fatalf("image_url = %v", got)
	}
	if got := content[1].(map[string]any)["file_data"]; got != "data:application/pdf;base64,ffffffffff"+core.BinaryTruncatedMarker+"ffffffffff" {
		t.Fatalf("file_data = %v", got)
	}
	original := prompt.Items[0]["content"].([]any)
	if original[0].(map[string]any)["image_url"] != image || original[1].(map[string]any)["file_data"] != file {
		t.Fatalf("original prompt was modified: %#v", original)
	}
}

func TestOpenAIExecuteReturnsTruncatedDebugPrompt(t *testing.T) {
	t.Parallel()

	var sent string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		data, _ := json.Marshal(body["input"])
		sent = string(data)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`))
	}))
	defer server.Close()

	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "k", Endpoint: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	payload := strings.Repeat("x", 300)
	resp, err := driver.Execute(context.Background(), []PromptSegment{{
		Role:    PromptRoleUser,
		Content: "describe",
		Files:   []DataSource{BytesDataSource{FileName: "a.png", MIME: "image/png", Data: []byte(payload)}},
	}}, ExecutionOptions{Model: "gpt-4.1"})
	if err != nil {
		t.Fatal(err)
	}
	debug, _ := json.Marshal(resp.Prompt)
	if !strings.Contains(string(debug), core.BinaryTruncatedMarker) {
		t.Fatalf("debug prompt was not truncated: %s", debug)
	}
	if strings.Contains(sent, core.BinaryTruncatedMarker) {
		t.Fatalf("request was truncated: %s", sent)
	}
}
