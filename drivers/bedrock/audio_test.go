package bedrock

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// Mirrors the TS Converse processFile audio branch: binary audio becomes an
// audio block (inline or by S3 location), bounded, and refused in tool results.
func TestBedrockAudioFileBlocks(t *testing.T) {
	ctx := context.Background()
	block, err := bedrockFileBlock(ctx, BytesDataSource{FileName: "speech.wav", MIME: "audio/wav", Data: []byte{0, 1, 255}})
	if err != nil {
		t.Fatal(err)
	}
	audio, ok := block.(*brtypes.ContentBlockMemberAudio)
	if !ok || audio.Value.Format != brtypes.AudioFormatWav {
		t.Fatalf("block = %#v", block)
	}
	if source, ok := audio.Value.Source.(*brtypes.AudioSourceMemberBytes); !ok || !bytes.Equal(source.Value, []byte{0, 1, 255}) {
		t.Fatalf("source = %#v", audio.Value.Source)
	}

	block, err = bedrockFileBlock(ctx, BytesDataSource{FileName: "call.m4a", MIME: "audio/x-m4a", URI: "https://bucket.s3.us-east-1.amazonaws.com/calls/call.m4a"})
	if err != nil {
		t.Fatal(err)
	}
	audio = block.(*brtypes.ContentBlockMemberAudio)
	if source, ok := audio.Value.Source.(*brtypes.AudioSourceMemberS3Location); !ok || aws.ToString(source.Value.Uri) != "s3://bucket/calls/call.m4a" || audio.Value.Format != brtypes.AudioFormatM4a {
		t.Fatalf("s3 audio = %#v", audio.Value)
	}

	if got := bedrockAudioFormat("audio/x-unknown"); got != brtypes.AudioFormatMp3 {
		t.Fatalf("fallback format = %s", got)
	}
	if _, err := bedrockFileBlock(ctx, BytesDataSource{FileName: "empty.mp3", MIME: "audio/mpeg"}); err == nil || !strings.Contains(err.Error(), "empty") {
		t.Fatalf("empty audio error = %v", err)
	}
	large := BytesDataSource{FileName: "long.mp3", MIME: "audio/mpeg", Data: make([]byte, bedrockMaxInlineAudioBytes+1)}
	if _, err := bedrockFileBlock(ctx, large); err == nil || !strings.Contains(err.Error(), "byte limit") {
		t.Fatalf("oversized audio error = %v", err)
	}
	if _, err := bedrockToolResultFileBlock(ctx, BytesDataSource{FileName: "speech.wav", MIME: "audio/wav", Data: []byte{1}}); err == nil || !strings.Contains(err.Error(), "tool results") {
		t.Fatalf("tool-result audio error = %v", err)
	}
}
