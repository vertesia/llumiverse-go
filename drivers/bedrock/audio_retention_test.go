package bedrock

import (
	"context"
	"testing"

	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/vertesia/llumiverse-go/core"
)

// Mirrors the TS stripAudioPayloads for Converse: inline audio leaves the
// stored conversation on the turn it is sent, while the request keeps it and
// S3-referenced audio stays.
func TestBedrockConversationOmitsInlineAudio(t *testing.T) {
	runtime := &fakeBedrockRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	resp, err := driver.Execute(context.Background(), []PromptSegment{{
		Role:    PromptRoleUser,
		Content: "Transcribe these.",
		Files: []DataSource{
			BytesDataSource{FileName: "speech.wav", MIME: "audio/wav", Data: []byte{0, 1, 255}},
			BytesDataSource{FileName: "call.mp3", MIME: "audio/mpeg", URI: "s3://bucket/call.mp3"},
		},
	}}, ExecutionOptions{Model: "amazon.nova-pro-v1:0"})
	if err != nil {
		t.Fatal(err)
	}
	sent := runtime.got.Messages[0].Content
	if _, ok := sent[1].(*brtypes.ContentBlockMemberAudio); !ok {
		t.Fatalf("sent content = %#v", sent)
	}
	stored := resp.Conversation.(bedrockPrompt).Messages[0].Content
	if text, ok := stored[1].(*brtypes.ContentBlockMemberText); !ok || text.Value != core.ConversationAudioPlaceholder {
		t.Fatalf("stored inline audio = %#v", stored[1])
	}
	audio, ok := stored[2].(*brtypes.ContentBlockMemberAudio)
	if !ok {
		t.Fatalf("stored s3 audio = %#v", stored[2])
	}
	if _, ok := audio.Value.Source.(*brtypes.AudioSourceMemberS3Location); !ok {
		t.Fatalf("stored s3 audio source = %#v", audio.Value.Source)
	}
}
