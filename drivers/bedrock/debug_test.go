package bedrock

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/vertesia/llumiverse-go/core"
)

// Mirrors the TS debug-prompt tests for the Converse prompt: inline bytes are
// truncated in the returned copy, nested tool results included, and the prompt
// being sent is left untouched.
func TestBedrockFormatDebugPromptTruncatesInlineBytes(t *testing.T) {
	image := bytes.Repeat([]byte{0xAB}, 300)
	pdf := bytes.Repeat([]byte("%PDF"), 100)
	prompt := bedrockPrompt{
		System: []brtypes.SystemContentBlock{&brtypes.SystemContentBlockMemberText{Value: "Be brief."}},
		Messages: []brtypes.Message{
			{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{
				&brtypes.ContentBlockMemberText{Value: "Describe these."},
				&brtypes.ContentBlockMemberImage{Value: brtypes.ImageBlock{Format: brtypes.ImageFormatPng, Source: &brtypes.ImageSourceMemberBytes{Value: image}}},
				&brtypes.ContentBlockMemberVideo{Value: brtypes.VideoBlock{Format: brtypes.VideoFormatMp4, Source: &brtypes.VideoSourceMemberS3Location{Value: brtypes.S3Location{Uri: aws.String("s3://bucket/clip.mp4")}}}},
			}},
			{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{bedrockToolUseBlock("call-1")}},
			{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{
				&brtypes.ContentBlockMemberToolResult{Value: brtypes.ToolResultBlock{ToolUseId: aws.String("call-1"), Content: []brtypes.ToolResultContentBlock{
					&brtypes.ToolResultContentBlockMemberText{Value: "attached"},
					&brtypes.ToolResultContentBlockMemberDocument{Value: brtypes.DocumentBlock{Name: aws.String("brief"), Format: brtypes.DocumentFormatPdf, Source: &brtypes.DocumentSourceMemberBytes{Value: pdf}}},
				}}},
			}},
		},
	}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, &fakeBedrockRuntime{}, nil)
	debug := driver.FormatDebugPrompt(prompt)
	data, err := json.Marshal(debug)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	imageBase64 := base64.StdEncoding.EncodeToString(image)
	for _, want := range []string{
		core.TruncateBinaryForDebug(imageBase64),
		core.TruncateBinaryForDebug(base64.StdEncoding.EncodeToString(pdf)),
		"Describe these.", "s3://bucket/clip.mp4", "Be brief.", "attached", "call-1",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("debug prompt lacks %q: %s", want, text)
		}
	}
	if strings.Contains(text, imageBase64) {
		t.Fatal("debug prompt carries the full image")
	}
	// The prompt being sent keeps its bytes.
	source := prompt.Messages[0].Content[1].(*brtypes.ContentBlockMemberImage).Value.Source.(*brtypes.ImageSourceMemberBytes)
	if !bytes.Equal(source.Value, image) {
		t.Fatal("prompt modified")
	}

	// The shared lifecycle returns the debug copy as ExecutionResponse.Prompt.
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "hello"}}, ExecutionOptions{Model: "amazon.nova-pro-v1:0"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := resp.Prompt.(map[string]any); !ok {
		t.Fatalf("response prompt = %#v", resp.Prompt)
	}
}

func TestBedrockFormatDebugPromptImageGenerationAndOtherPrompts(t *testing.T) {
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, &fakeBedrockRuntime{}, nil)
	raw := strings.Repeat("A", 200)
	prompt := bedrockImagePrompt{Text: "A cube", Images: []string{raw, "data:image/png;base64," + raw}, Masks: []string{raw}}
	debug := driver.FormatDebugPrompt(prompt).(bedrockImagePrompt)
	want := []string{core.TruncateBinaryForDebug(raw), "data:image/png;base64," + core.TruncateBinaryForDebug(raw)}
	if !reflect.DeepEqual(debug.Images, want) || debug.Masks[0] != core.TruncateBinaryForDebug(raw) || debug.Text != "A cube" {
		t.Fatalf("debug image prompt = %#v", debug)
	}
	if prompt.Images[0] != raw {
		t.Fatal("image prompt modified")
	}
	other := map[string]any{"messages": []any{}}
	if got := driver.FormatDebugPrompt(other); !reflect.DeepEqual(got, other) {
		t.Fatalf("other prompt = %#v", got)
	}
}
