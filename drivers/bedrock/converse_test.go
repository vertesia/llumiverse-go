package bedrock

import (
	"context"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func bedrockToolUseBlock(id string) brtypes.ContentBlock {
	return &brtypes.ContentBlockMemberToolUse{Value: brtypes.ToolUseBlock{
		ToolUseId: aws.String(id),
		Name:      aws.String("search"),
		Input:     brdoc.NewLazyDocument(map[string]any{}),
	}}
}

func bedrockToolResultBlock(id string, text string) brtypes.ContentBlock {
	return &brtypes.ContentBlockMemberToolResult{Value: brtypes.ToolResultBlock{
		ToolUseId: aws.String(id),
		Content:   []brtypes.ToolResultContentBlock{&brtypes.ToolResultContentBlockMemberText{Value: text}},
	}}
}

func bedrockToolResultID(t *testing.T, block brtypes.ContentBlock) string {
	t.Helper()
	result, ok := block.(*brtypes.ContentBlockMemberToolResult)
	if !ok {
		t.Fatalf("block = %T, want a toolResult", block)
	}
	return aws.ToString(result.Value.ToolUseId)
}

func bedrockBlockText(t *testing.T, block brtypes.ContentBlock) string {
	t.Helper()
	text, ok := block.(*brtypes.ContentBlockMemberText)
	if !ok {
		t.Fatalf("block = %T, want text", block)
	}
	return text.Value
}

// Mirrors the TS "fixOrphanedToolResults - Bedrock" suite (drivers/test/orphaned-tool-use.test.ts).
func TestBedrockFixOrphanedToolResults(t *testing.T) {
	if got := fixBedrockOrphanedToolResults(nil); len(got) != 0 {
		t.Fatalf("empty input = %#v", got)
	}

	paired := []brtypes.Message{
		{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{bedrockToolUseBlock("tool_1")}},
		{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{bedrockToolResultBlock("tool_1", "ok")}},
	}
	if got := fixBedrockOrphanedToolResults(paired); len(got) != 2 || len(got[1].Content) != 1 {
		t.Fatalf("paired result changed: %#v", got)
	}

	partial := []brtypes.Message{
		{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{bedrockToolUseBlock("tool_1")}},
		{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{bedrockToolResultBlock("tool_1", "r1"), bedrockToolResultBlock("tool_2", "orphan")}},
	}
	got := fixBedrockOrphanedToolResults(partial)
	if len(got[1].Content) != 1 || bedrockToolResultID(t, got[1].Content[0]) != "tool_1" {
		t.Fatalf("orphan kept: %#v", got[1].Content)
	}
	if len(partial[1].Content) != 2 {
		t.Fatal("input messages were modified")
	}

	emptied := []brtypes.Message{
		{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "[summary of prior work]"}}},
		{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{bedrockToolResultBlock("gone", "orphan")}},
	}
	got = fixBedrockOrphanedToolResults(emptied)
	if len(got) != 1 || bedrockBlockText(t, got[0].Content[0]) != "[summary of prior work]" {
		t.Fatalf("emptied message kept: %#v", got)
	}

	mixed := []brtypes.Message{
		{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "thinking"}}},
		{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{bedrockToolResultBlock("orphan", "x"), &brtypes.ContentBlockMemberText{Value: "continue please"}}},
	}
	got = fixBedrockOrphanedToolResults(mixed)
	if len(got[1].Content) != 1 || bedrockBlockText(t, got[1].Content[0]) != "continue please" {
		t.Fatalf("mixed content = %#v", got[1].Content)
	}
}

// A conversation whose assistant toolUse turn was trimmed must not send the
// dangling toolResult to Converse.
func TestBedrockExecuteDropsOrphanedToolResultFromConversation(t *testing.T) {
	runtime := &fakeBedrockRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	conversation := bedrockPrompt{Messages: []brtypes.Message{
		{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "[summary]"}}},
		{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{bedrockToolResultBlock("gone", "orphan")}},
		{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "ok"}}},
	}}
	_, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "next"}}, ExecutionOptions{
		Model:        "mistral.mistral-large-2407-v1:0",
		Conversation: conversation,
		Tools:        []ToolDefinition{{Name: "search", InputSchema: map[string]any{"type": "object"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, message := range runtime.got.Messages {
		if bedrockHasToolResult(message.Content) {
			t.Fatalf("orphaned toolResult sent: %#v", runtime.got.Messages)
		}
	}
	if len(runtime.got.Messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(runtime.got.Messages))
	}
}
