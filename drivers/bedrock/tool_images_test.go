package bedrock

import (
	"context"
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

var bedrockToolImage = brtypes.ImageBlock{Format: brtypes.ImageFormatJpeg, Source: &brtypes.ImageSourceMemberBytes{Value: []byte{1, 2, 3}}}

var bedrockPreviewTools = []ToolDefinition{
	{Name: "preview_artifact_officexml", InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}},
	{Name: "other_preview", InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}},
}

// bedrockToolImageHistory mirrors the TS tool-result-images.test.ts messages().
func bedrockToolImageHistory() []brtypes.Message {
	return []brtypes.Message{
		{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "Review the preview."}}},
		{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{
			&brtypes.ContentBlockMemberToolUse{Value: brtypes.ToolUseBlock{ToolUseId: aws.String("preview"), Name: aws.String("preview_artifact_officexml"), Input: brdoc.NewLazyDocument(map[string]any{})}},
			&brtypes.ContentBlockMemberToolUse{Value: brtypes.ToolUseBlock{ToolUseId: aws.String("other"), Name: aws.String("other_preview"), Input: brdoc.NewLazyDocument(map[string]any{})}},
		}},
		{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{
			&brtypes.ContentBlockMemberToolResult{Value: brtypes.ToolResultBlock{ToolUseId: aws.String("preview"), Content: []brtypes.ToolResultContentBlock{
				&brtypes.ToolResultContentBlockMemberText{Value: "Pages 1-8"},
				&brtypes.ToolResultContentBlockMemberImage{Value: bedrockToolImage},
			}}},
			&brtypes.ContentBlockMemberToolResult{Value: brtypes.ToolResultBlock{ToolUseId: aws.String("other"), Status: brtypes.ToolResultStatusSuccess, Content: []brtypes.ToolResultContentBlock{
				&brtypes.ToolResultContentBlockMemberImage{Value: bedrockToolImage},
			}}},
		}},
	}
}

func expectBedrockToolImagesRelocated(t *testing.T, messages []brtypes.Message) {
	t.Helper()
	want := []brtypes.ContentBlock{
		&brtypes.ContentBlockMemberToolResult{Value: brtypes.ToolResultBlock{ToolUseId: aws.String("preview"), Content: []brtypes.ToolResultContentBlock{
			&brtypes.ToolResultContentBlockMemberText{Value: "Pages 1-8"},
		}}},
		&brtypes.ContentBlockMemberToolResult{Value: brtypes.ToolResultBlock{ToolUseId: aws.String("other"), Status: brtypes.ToolResultStatusSuccess, Content: []brtypes.ToolResultContentBlock{
			&brtypes.ToolResultContentBlockMemberText{Value: "See the attached tool-result image(s)."},
		}}},
		&brtypes.ContentBlockMemberText{Value: "Image from tool result preview:"},
		&brtypes.ContentBlockMemberImage{Value: bedrockToolImage},
		&brtypes.ContentBlockMemberText{Value: "Image from tool result other:"},
		&brtypes.ContentBlockMemberImage{Value: bedrockToolImage},
	}
	if len(messages) < 3 || !reflect.DeepEqual(messages[2].Content, want) {
		t.Fatalf("tool-result message = %#v", messages)
	}
}

// Mirrors the TS "preserves images, tool pairing and source history for %s".
func TestBedrockRelocatesToolImagesForVisionModels(t *testing.T) {
	for _, model := range []string{
		"arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.openai.gpt-6-astra",
		"global.openai.gpt-7",
		"mistral.pixtral-large-2502-v1:0",
	} {
		original := bedrockToolImageHistory()
		before := bedrockToolImageHistory()
		relocated := relocateBedrockToolImages(original, model)
		expectBedrockToolImagesRelocated(t, relocated)
		if !reflect.DeepEqual(original, before) {
			t.Fatalf("%s: source history modified", model)
		}
		if again := relocateBedrockToolImages(relocated, model); !reflect.DeepEqual(again, relocated) {
			t.Fatalf("%s: relocation is not idempotent", model)
		}
	}
}

// Mirrors the TS "keeps supported nested images for %s on Converse" and "preserves ordinary user images".
func TestBedrockKeepsNestedToolImagesForClaudeAndNova(t *testing.T) {
	for _, model := range []string{"us.anthropic.claude-sonnet-4-6", "amazon.nova-pro-v1:0"} {
		if got := relocateBedrockToolImages(bedrockToolImageHistory(), model); !reflect.DeepEqual(got, bedrockToolImageHistory()) {
			t.Fatalf("%s: messages changed: %#v", model, got)
		}
	}
	plain := []brtypes.Message{{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "Describe this."}, &brtypes.ContentBlockMemberImage{Value: bedrockToolImage}}}}
	if got := relocateBedrockToolImages(plain, "global.openai.gpt-7"); !reflect.DeepEqual(got, plain) {
		t.Fatalf("ordinary images changed: %#v", got)
	}
}

// Mirrors the TS "normalizes images at the API boundary" for stored conversations.
func TestBedrockRelocatesToolImagesAtTheAPIBoundary(t *testing.T) {
	model := "global.openai.gpt-7"
	history := append(bedrockToolImageHistory(), brtypes.Message{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "Reviewed."}}})
	runtime := &fakeBedrockRuntime{streamEvents: []brtypes.ConverseStreamOutput{
		&brtypes.ConverseStreamOutputMemberMessageStop{Value: brtypes.MessageStopEvent{StopReason: brtypes.StopReasonEndTurn}},
	}}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	segments := []PromptSegment{{Role: PromptRoleUser, Content: "Review once more."}}
	options := ExecutionOptions{Model: model, Tools: bedrockPreviewTools, Conversation: bedrockPrompt{Messages: history}}
	resp, err := driver.Execute(context.Background(), segments, options)
	if err != nil {
		t.Fatal(err)
	}
	expectBedrockToolImagesRelocated(t, runtime.got.Messages)
	// The stored conversation keeps the images nested in their tool results.
	stored := resp.Conversation.(bedrockPrompt)
	if !bedrockToolResultHasImage(stored.Messages[2].Content[1].(*brtypes.ContentBlockMemberToolResult).Value.Content) {
		t.Fatalf("stored conversation changed: %#v", stored.Messages[2])
	}

	stream, err := driver.Stream(context.Background(), segments, options)
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.Close()
	expectBedrockToolImagesRelocated(t, runtime.gotStream.Messages)
}
