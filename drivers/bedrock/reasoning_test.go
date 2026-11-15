package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrockruntime "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// fakeBedrockReasoningRuntime returns a fixed Converse response and records every request.
type fakeBedrockReasoningRuntime struct {
	fakeBedrockRuntime
	output *bedrockruntime.ConverseOutput
	calls  []*bedrockruntime.ConverseInput
}

func (f *fakeBedrockReasoningRuntime) Converse(_ context.Context, input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	f.calls = append(f.calls, input)
	return f.output, nil
}

const bedrockReasoningModel = "anthropic.claude-sonnet-4-6-v1:0"

func bedrockReasoningText(text string, signature string) brtypes.ContentBlock {
	block := brtypes.ReasoningTextBlock{Text: aws.String(text)}
	if signature != "" {
		block.Signature = aws.String(signature)
	}
	return &brtypes.ContentBlockMemberReasoningContent{Value: &brtypes.ReasoningContentBlockMemberReasoningText{Value: block}}
}

func bedrockRedactedReasoning(data []byte) brtypes.ContentBlock {
	return &brtypes.ContentBlockMemberReasoningContent{Value: &brtypes.ReasoningContentBlockMemberRedactedContent{Value: data}}
}

// Mirrors the TS reasoning-replay.test.ts "preserves reasoning signatures and
// redacted bytes across a JSON roundtrip".
func TestBedrockReplaysSignedReasoningAcrossJSONRoundtrip(t *testing.T) {
	redacted := []byte{1, 2, 255}
	runtime := &fakeBedrockReasoningRuntime{output: &bedrockruntime.ConverseOutput{
		Output: &brtypes.ConverseOutputMemberMessage{Value: brtypes.Message{
			Role: brtypes.ConversationRoleAssistant,
			Content: []brtypes.ContentBlock{
				bedrockReasoningText("plan", "signed-chain"),
				bedrockRedactedReasoning(redacted),
				&brtypes.ContentBlockMemberText{Value: "answer"},
			},
		}},
		StopReason: brtypes.StopReasonEndTurn,
	}}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	zero := 0
	first, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "question"}}, ExecutionOptions{
		Model:                 bedrockReasoningModel,
		StripTextMaxTokens:    1,
		StripImagesAfterTurns: &zero,
	})
	if err != nil {
		t.Fatal(err)
	}
	if text := completionResultsToText(first.Result); text != "answer" {
		t.Fatalf("result = %q", text)
	}
	data, err := json.Marshal(first.Conversation)
	if err != nil {
		t.Fatal(err)
	}
	var persisted any
	if err := json.Unmarshal(data, &persisted); err != nil {
		t.Fatal(err)
	}

	if _, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "continue"}}, ExecutionOptions{
		Model:        bedrockReasoningModel,
		Conversation: persisted,
	}); err != nil {
		t.Fatal(err)
	}
	replay := runtime.calls[1].Messages
	// The signed chain is replayed untouched: no text truncation despite StripTextMaxTokens.
	if got := bedrockBlockText(t, replay[0].Content[0]); got != "question" {
		t.Fatalf("first message = %q", got)
	}
	assistant := replay[1].Content
	if len(assistant) != 3 {
		t.Fatalf("assistant content = %#v", assistant)
	}
	reasoning := assistant[0].(*brtypes.ContentBlockMemberReasoningContent).Value.(*brtypes.ReasoningContentBlockMemberReasoningText).Value
	if aws.ToString(reasoning.Text) != "plan" || aws.ToString(reasoning.Signature) != "signed-chain" {
		t.Fatalf("reasoning = %#v", reasoning)
	}
	gotRedacted := assistant[1].(*brtypes.ContentBlockMemberReasoningContent).Value.(*brtypes.ReasoningContentBlockMemberRedactedContent).Value
	if !bytes.Equal(gotRedacted, redacted) {
		t.Fatalf("redacted = %v, want %v", gotRedacted, redacted)
	}
}

// Mirrors the TS "applies caller stripping to unsigned reasoning without modifying the reasoning block".
func TestBedrockStripsUnsignedReasoningConversation(t *testing.T) {
	runtime := &fakeBedrockReasoningRuntime{output: &bedrockruntime.ConverseOutput{
		Output: &brtypes.ConverseOutputMemberMessage{Value: brtypes.Message{
			Role:    brtypes.ConversationRoleAssistant,
			Content: []brtypes.ContentBlock{bedrockReasoningText("unsigned reasoning text", ""), &brtypes.ContentBlockMemberText{Value: "answer"}},
		}},
		StopReason: brtypes.StopReasonEndTurn,
	}}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: strings.Repeat("a deliberately long prior message ", 20)}}, ExecutionOptions{
		Model:              bedrockReasoningModel,
		StripTextMaxTokens: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	conversation := resp.Conversation.(bedrockPrompt)
	if got := bedrockBlockText(t, conversation.Messages[0].Content[0]); !strings.Contains(got, conversationTextTruncatedMarker) {
		t.Fatalf("prior message not truncated: %q", got)
	}
	reasoning := conversation.Messages[1].Content[0].(*brtypes.ContentBlockMemberReasoningContent).Value.(*brtypes.ReasoningContentBlockMemberReasoningText).Value
	if aws.ToString(reasoning.Text) != "unsigned reasoning text" {
		t.Fatalf("reasoning modified: %#v", reasoning)
	}
}

// Mirrors the TS "reconstructs fragmented reasoning, text, redaction, and tool blocks by native index".
func TestBedrockStreamConversationReplaysNativeBlocks(t *testing.T) {
	runtime := &fakeBedrockRuntime{streamEvents: []brtypes.ConverseStreamOutput{
		bedrockReasoningDelta(0, &brtypes.ReasoningContentBlockDeltaMemberText{Value: "plan "}),
		bedrockReasoningDelta(0, &brtypes.ReasoningContentBlockDeltaMemberText{Value: "more"}),
		bedrockReasoningDelta(0, &brtypes.ReasoningContentBlockDeltaMemberSignature{Value: "stream-signature"}),
		&brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(1), Delta: &brtypes.ContentBlockDeltaMemberText{Value: "answer"}}},
		bedrockToolStart(2, "call-1"),
		bedrockToolDelta(2, `{"city":`),
		bedrockToolDelta(2, `"Paris"}`),
		bedrockToolStart(3, "call-truncated"),
		bedrockToolDelta(3, `{"city":`),
		bedrockReasoningDelta(4, &brtypes.ReasoningContentBlockDeltaMemberRedactedContent{Value: []byte{9, 8}}),
		&brtypes.ConverseStreamOutputMemberMessageStop{Value: brtypes.MessageStopEvent{StopReason: brtypes.StopReasonMaxTokens}},
	}}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "question"}}, ExecutionOptions{Model: bedrockReasoningModel})
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := stream.Recv(); err == io.EOF {
			break
		} else if err != nil {
			t.Fatal(err)
		}
	}
	resp := stream.Completion()
	if text := completionResultsToText(resp.Result); text != "answer" {
		t.Fatalf("result = %q", text)
	}
	conversation := resp.Conversation.(bedrockPrompt)
	content := conversation.Messages[len(conversation.Messages)-1].Content
	if len(content) != 4 {
		t.Fatalf("assistant content = %#v", content)
	}
	reasoning := content[0].(*brtypes.ContentBlockMemberReasoningContent).Value.(*brtypes.ReasoningContentBlockMemberReasoningText).Value
	if aws.ToString(reasoning.Text) != "plan more" || aws.ToString(reasoning.Signature) != "stream-signature" {
		t.Fatalf("reasoning = %#v", reasoning)
	}
	if bedrockBlockText(t, content[1]) != "answer" {
		t.Fatalf("text = %#v", content[1])
	}
	tool := content[2].(*brtypes.ContentBlockMemberToolUse).Value
	input, err := tool.Input.MarshalSmithyDocument()
	if err != nil {
		t.Fatal(err)
	}
	if aws.ToString(tool.ToolUseId) != "call-1" || string(input) != `{"city":"Paris"}` {
		t.Fatalf("tool = %#v input %s", tool, input)
	}
	redacted := content[3].(*brtypes.ContentBlockMemberReasoningContent).Value.(*brtypes.ReasoningContentBlockMemberRedactedContent).Value
	if !bytes.Equal(redacted, []byte{9, 8}) {
		t.Fatalf("redacted = %v", redacted)
	}
}

// Mirrors the TS "keeps the DeepSeek replay exclusion narrow".
func TestBedrockExcludesReasoningReplay(t *testing.T) {
	for model, want := range map[string]bool{
		"deepseek.r1-v1:0":              true,
		"us.deepseek.r1-v1:0":           true,
		"deepseek.v3-v1:0":              false,
		"vendor.deepseek-r1-compatible": false,
	} {
		if got := bedrockExcludesReasoningReplay(model); got != want {
			t.Errorf("%s = %v, want %v", model, got, want)
		}
	}
	content := bedrockReplayContent([]brtypes.ContentBlock{bedrockReasoningText("r", ""), &brtypes.ContentBlockMemberText{Value: "t"}}, "us.deepseek.r1-v1:0")
	if len(content) != 1 {
		t.Fatalf("DeepSeek R1 replay = %#v", content)
	}
}

func TestBedrockContextWindowStopIsLength(t *testing.T) {
	if got := bedrockFinishReason(brtypes.StopReasonModelContextWindowExceeded); got != "length" {
		t.Fatalf("finish reason = %q", got)
	}
}

func bedrockReasoningDelta(index int32, delta brtypes.ReasoningContentBlockDelta) brtypes.ConverseStreamOutput {
	return &brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{
		ContentBlockIndex: aws.Int32(index),
		Delta:             &brtypes.ContentBlockDeltaMemberReasoningContent{Value: delta},
	}}
}

func bedrockToolStart(index int32, id string) brtypes.ConverseStreamOutput {
	return &brtypes.ConverseStreamOutputMemberContentBlockStart{Value: brtypes.ContentBlockStartEvent{
		ContentBlockIndex: aws.Int32(index),
		Start:             &brtypes.ContentBlockStartMemberToolUse{Value: brtypes.ToolUseBlockStart{ToolUseId: aws.String(id), Name: aws.String("lookup")}},
	}}
}

func bedrockToolDelta(index int32, input string) brtypes.ConverseStreamOutput {
	return &brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{
		ContentBlockIndex: aws.Int32(index),
		Delta:             &brtypes.ContentBlockDeltaMemberToolUse{Value: brtypes.ToolUseBlockDelta{Input: aws.String(input)}},
	}}
}
