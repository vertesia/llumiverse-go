package bedrock

import (
	"context"
	"io"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrockruntime "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// Mirrors the TS service-tier.test.ts "uses the Converse service tier object for text models".
func TestBedrockSendsServiceTier(t *testing.T) {
	runtime := &fakeBedrockRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	segments := []PromptSegment{{Role: PromptRoleUser, Content: "hello"}}
	if _, err := driver.Execute(context.Background(), segments, ExecutionOptions{
		Model:        "amazon.nova-pro-v1:0",
		ModelOptions: map[string]any{"service_tier": "future-tier"},
	}); err != nil {
		t.Fatal(err)
	}
	if runtime.got.ServiceTier == nil || runtime.got.ServiceTier.Type != "future-tier" {
		t.Fatalf("service tier = %#v", runtime.got.ServiceTier)
	}

	stream, err := driver.Stream(context.Background(), segments, ExecutionOptions{
		Model:        "amazon.nova-pro-v1:0",
		ModelOptions: map[string]any{"service_tier": "flex"},
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.Close()
	if runtime.gotStream.ServiceTier == nil || runtime.gotStream.ServiceTier.Type != brtypes.ServiceTierTypeFlex {
		t.Fatalf("stream service tier = %#v", runtime.gotStream.ServiceTier)
	}

	// No tier is sent unless the caller sets one.
	if _, err := driver.Execute(context.Background(), segments, ExecutionOptions{Model: "amazon.nova-pro-v1:0"}); err != nil {
		t.Fatal(err)
	}
	if runtime.got.ServiceTier != nil {
		t.Fatalf("default service tier = %#v", runtime.got.ServiceTier)
	}
}

// fakeBedrockTierRuntime reports a service tier on Converse responses.
type fakeBedrockTierRuntime struct {
	fakeBedrockRuntime
}

func (f *fakeBedrockTierRuntime) Converse(ctx context.Context, input *bedrockruntime.ConverseInput, opts ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	output, err := f.fakeBedrockRuntime.Converse(ctx, input, opts...)
	if err == nil {
		output.ServiceTier = &brtypes.ServiceTier{Type: brtypes.ServiceTierTypePriority}
	}
	return output, err
}

// Mirrors the TS "returns the processing tier reported by Converse", for Converse and ConverseStream.
func TestBedrockCapturesServiceTier(t *testing.T) {
	runtime := &fakeBedrockTierRuntime{fakeBedrockRuntime: fakeBedrockRuntime{streamEvents: []brtypes.ConverseStreamOutput{
		&brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(0), Delta: &brtypes.ContentBlockDeltaMemberText{Value: "answer"}}},
		&brtypes.ConverseStreamOutputMemberMessageStop{Value: brtypes.MessageStopEvent{StopReason: brtypes.StopReasonEndTurn}},
		&brtypes.ConverseStreamOutputMemberMetadata{Value: brtypes.ConverseStreamMetadataEvent{
			Usage:       &brtypes.TokenUsage{InputTokens: aws.Int32(1), OutputTokens: aws.Int32(1), TotalTokens: aws.Int32(2)},
			ServiceTier: &brtypes.ServiceTier{Type: brtypes.ServiceTierTypeFlex},
		}},
	}}}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	segments := []PromptSegment{{Role: PromptRoleUser, Content: "hello"}}
	resp, err := driver.Execute(context.Background(), segments, ExecutionOptions{Model: "amazon.nova-pro-v1:0"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.ServiceTier != "priority" {
		t.Fatalf("service tier = %q", resp.ServiceTier)
	}

	stream, err := driver.Stream(context.Background(), segments, ExecutionOptions{Model: "amazon.nova-pro-v1:0"})
	if err != nil {
		t.Fatal(err)
	}
	var chunkTier string
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if chunk.ServiceTier != "" {
			chunkTier = chunk.ServiceTier
		}
	}
	if chunkTier != "flex" || stream.Completion().ServiceTier != "flex" {
		t.Fatalf("stream service tier: chunk %q, completion %q", chunkTier, stream.Completion().ServiceTier)
	}
}
