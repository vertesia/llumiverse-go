package claude

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/core"
)

func TestAnthropicExecuteSendsHeadersBetasAndADebugPrompt(t *testing.T) {
	t.Parallel()

	var got map[string]any
	var header http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header = r.Header.Clone()
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1,"speed":"fast"}}`))
	}))
	defer server.Close()

	driver, err := NewAnthropicDriver(AnthropicOptions{APIKey: "anthropic-key", BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	image := strings.Repeat("i", 300)
	resp, err := driver.Execute(context.Background(), []PromptSegment{{
		Role:    PromptRoleUser,
		Content: "Describe",
		Files:   []DataSource{BytesDataSource{FileName: "a.png", MIME: "image/png", Data: []byte(image)}},
	}}, ExecutionOptions{
		Model:        "claude-opus-5",
		ModelOptions: map[string]any{"speed": "fast"},
		Compaction:   &ConversationCompactionOptions{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if header.Get("anthropic-version") != "2023-06-01" || header.Get("anthropic-beta") != "fast-mode-2026-02-01,compact-2026-01-12" || len(header.Values("anthropic-beta")) != 1 {
		t.Fatalf("headers = %#v", header)
	}
	if got["model"] != "claude-opus-5" || got["speed"] != "fast" || got["stream"] != nil {
		t.Fatalf("body = %#v", got)
	}
	if resp.FinishReason != "stop" || resp.ServiceTier != "fast" {
		t.Fatalf("completion = %#v", resp.Completion)
	}
	prompt := resp.Prompt.(claudePrompt)
	if data := prompt.Messages[0].Content[1].Source.Data; !strings.Contains(data, core.BinaryTruncatedMarker) {
		t.Fatalf("debug prompt = %q", data)
	}
	// The stored conversation keeps the full payload, and restores from JSON.
	data, _ := json.Marshal(resp.Conversation)
	if !strings.Contains(string(data), base64.StdEncoding.EncodeToString([]byte(image))) {
		t.Fatal("the conversation lost the image")
	}
	var restored any
	_ = json.Unmarshal(data, &restored)
	again, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "More"}}, ExecutionOptions{Model: "claude-opus-5", Conversation: restored})
	if err != nil {
		t.Fatal(err)
	}
	if messages := got["messages"].([]any); len(messages) != 3 {
		t.Fatalf("messages = %#v", messages)
	}
	if conversation := again.Conversation.(claudePrompt); conversation.turn() != 2 {
		t.Fatalf("turn = %d", conversation.turn())
	}
}
