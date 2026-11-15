package claude

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/core"
)

func TestClaudePromptFormatting(t *testing.T) {
	t.Parallel()

	prompt, err := formatClaudePrompt(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "System"},
		{Role: PromptRoleAssistant, Content: "Previous answer"},
		{Role: PromptRoleUser, Content: "Use these", Files: []DataSource{
			BytesDataSource{FileName: "brief.pdf", MIME: "application/pdf", Data: []byte("%PDF")},
			BytesDataSource{FileName: "notes.txt", MIME: "text/plain", Data: []byte("notes")},
			BytesDataSource{FileName: "data.json", MIME: "application/vnd.api+json; charset=utf-8", Data: []byte("{}")},
			BytesDataSource{FileName: "photo.jpg", MIME: "image/jpg", Data: []byte("jpg")},
			BytesDataSource{FileName: "clip.mp4", MIME: "video/mp4", Data: []byte("mp4")},
			BytesDataSource{FileName: "blob.bin", MIME: "application/octet-stream", Data: []byte("bin")},
		}},
		{Role: PromptRoleTool, ToolUseID: "tool_1", Content: "tool output", Files: []DataSource{
			BytesDataSource{FileName: "tool.png", MIME: "image/png", Data: []byte("png")},
			BytesDataSource{FileName: "tool.txt", MIME: "text/plain", Data: []byte("tool text")},
			BytesDataSource{FileName: "tool.pdf", MIME: "application/pdf", Data: []byte("%PDF")},
		}},
		{Role: PromptRoleSafety, Content: "Safety"},
		{Role: PromptRoleNegative, Content: "ignored"},
	}, ExecutionOptions{ResultSchema: map[string]any{"type": "object"}, Tools: []ToolDefinition{namedTool("lookup")}}, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(prompt.System) != 2 || prompt.System[1].Text != claudeToolAwareJSONSchemaInstructionPrefix+"\n"+`{"type":"object"}` {
		t.Fatalf("system = %#v", prompt.System)
	}
	if len(prompt.Messages) != 4 || prompt.Messages[0].Role != "assistant" || prompt.Messages[3].Content[0].Text != "Safety" {
		t.Fatalf("messages = %#v", prompt.Messages)
	}
	var kinds []string
	for _, block := range prompt.Messages[1].Content {
		kind := block.Type
		if block.Source != nil {
			kind += ":" + block.Source.Type + ":" + block.Source.MediaType
		}
		kinds = append(kinds, kind)
	}
	if strings.Join(kinds, ",") != "text,document:base64:application/pdf,document:text:text/plain,document:text:text/plain,image:base64:image/jpeg" {
		t.Fatalf("user blocks = %v", kinds)
	}
	// Tool results take the same file types as user turns.
	toolContent := prompt.Messages[2].Content[0].Content
	if len(toolContent) != 4 || toolContent[1].Type != "image" || toolContent[2].Type != "document" || toolContent[3].Source.MediaType != "application/pdf" {
		t.Fatalf("tool content = %#v", toolContent)
	}

	for mime, want := range map[string]string{"audio/wav": "audio input is not supported by Claude; supply a transcript instead", "image/svg+xml": "unsupported image type: image/svg+xml"} {
		_, err := formatClaudePrompt(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "x", Files: []DataSource{BytesDataSource{FileName: "f", MIME: mime, Data: []byte("x")}}}}, ExecutionOptions{}, false)
		if err == nil || err.Error() != want {
			t.Fatalf("%s: error = %v", mime, err)
		}
	}
	if _, err := formatClaudePrompt(context.Background(), []PromptSegment{{Role: PromptRoleTool, Content: "missing id"}}, ExecutionOptions{}, false); err == nil {
		t.Fatal("expected a missing ToolUseID error")
	}
}

func TestClaudePromptSchemaPlacement(t *testing.T) {
	t.Parallel()

	schema := map[string]any{"type": "object", "description": "a <b> & c"}
	instructions := claudeJSONSchemaInstructionPrefix + "\n" + `{"description":"a <b> & c","type":"object"}`
	segments := []PromptSegment{{Role: PromptRoleSystem, Content: "System"}, {Role: PromptRoleUser, Content: "Task"}}

	plain, _ := formatClaudePrompt(context.Background(), segments, ExecutionOptions{ResultSchema: schema}, false)
	if len(plain.System) != 2 || plain.System[1].Text != instructions {
		t.Fatalf("no cache key: %#v", plain.System)
	}
	agent, _ := formatClaudePrompt(context.Background(), segments, ExecutionOptions{ResultSchema: schema, PromptCacheKey: "agent-1"}, false)
	if len(agent.System) != 2 || agent.System[1].Text != instructions || agent.Messages[0].Content[0].Text != "Task" {
		t.Fatalf("agent key: %#v", agent)
	}
	routed, _ := formatClaudePrompt(context.Background(), segments, ExecutionOptions{ResultSchema: schema, PromptCacheKey: "route-1"}, false)
	if len(routed.System) != 1 || routed.Messages[0].Content[0].Text != "Task\n\n"+instructions {
		t.Fatalf("routed key: %#v", routed)
	}
	noTask, _ := formatClaudePrompt(context.Background(), segments[:1], ExecutionOptions{ResultSchema: schema, PromptCacheKey: "route-1"}, false)
	if len(noTask.System) != 2 {
		t.Fatalf("routed without a task block: %#v", noTask.System)
	}
	native, _ := formatClaudePrompt(context.Background(), segments, ExecutionOptions{ResultSchema: schema}, true)
	if len(native.System) != 1 {
		t.Fatalf("a native schema needs no instructions: %#v", native.System)
	}
}

func TestClaudeFormatDebugPrompt(t *testing.T) {
	t.Parallel()

	data := strings.Repeat("A", 200)
	prompt := claudePrompt{Messages: []claudeMessage{{Role: "user", Content: []claudeBlock{
		text("hello"),
		{Type: "image", Source: &claudeSource{Type: "base64", MediaType: "image/png", Data: data}},
		{Type: "document", Source: &claudeSource{Type: "base64", MediaType: "application/pdf", Data: data}},
		{Type: "document", Source: &claudeSource{Type: "text", MediaType: "text/plain", Data: data}},
	}}}}
	before, _ := json.Marshal(prompt)
	debug := FormatDebugPrompt(prompt)
	if after, _ := json.Marshal(prompt); string(after) != string(before) {
		t.Fatal("the prompt was modified")
	}
	blocks := debug.Messages[0].Content
	if blocks[1].Source.Data != core.TruncateBinaryForDebug(data) || blocks[2].Source.Data != core.TruncateBinaryForDebug(data) || blocks[3].Source.Data != data || blocks[0].Text != "hello" {
		t.Fatalf("debug = %#v", blocks)
	}
	driver := &AnthropicDriver{}
	if got := driver.FormatDebugPrompt(prompt).(claudePrompt); got.Messages[0].Content[1].Source.Data == data {
		t.Fatal("the driver truncates binary payloads")
	}
	if got := driver.FormatDebugPrompt("other"); got != "other" {
		t.Fatalf("other prompts pass through: %#v", got)
	}
}
