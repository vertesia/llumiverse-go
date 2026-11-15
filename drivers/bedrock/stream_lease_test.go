package bedrock

import (
	"context"
	"errors"
	"testing"
	"time"

	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/vertesia/llumiverse-go/core"
)

// An unread stream is closed once DriverOptions.StreamStartTimeout elapses.
func TestBedrockStreamIsLeasedUntilRead(t *testing.T) {
	runtime := &fakeBedrockRuntime{streamEvents: []brtypes.ConverseStreamOutput{
		&brtypes.ConverseStreamOutputMemberMessageStop{Value: brtypes.MessageStopEvent{StopReason: brtypes.StopReasonEndTurn}},
	}}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1", DriverOptions: DriverOptions{StreamStartTimeout: 5 * time.Millisecond}}, runtime, nil)
	stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "hello"}}, ExecutionOptions{Model: "amazon.nova-pro-v1:0"})
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)
	if _, err := stream.Recv(); !errors.Is(err, core.ErrCompletionStreamNotConsumed) {
		t.Fatalf("Recv error = %v", err)
	}
}
