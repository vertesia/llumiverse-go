package bedrock

import (
	"context"
	"encoding/json"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrockruntime "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

const bedrockEmptyTurnModel = "anthropic.claude-sonnet-4-6-v1:0"

func bedrockEmptyTurnToolTurn() []brtypes.Message {
	return []brtypes.Message{
		{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "question"}}},
		{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{bedrockToolUseBlock("call-1")}},
		{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{bedrockToolResultBlock("call-1", "Thought recorded.")}},
	}
}

// expectValidConverseMessages mirrors the TS helper: every message has content,
// no text block is blank, and roles alternate.
func expectValidConverseMessages(t *testing.T, messages []brtypes.Message) {
	t.Helper()
	if len(messages) == 0 {
		t.Fatal("no messages")
	}
	for i, message := range messages {
		if len(message.Content) == 0 {
			t.Fatalf("message %d is empty: %#v", i, messages)
		}
		for _, block := range message.Content {
			if text, ok := block.(*brtypes.ContentBlockMemberText); ok && strings.TrimSpace(text.Value) == "" {
				t.Fatalf("message %d has blank text: %#v", i, messages)
			}
		}
		if i > 0 && message.Role == messages[i-1].Role {
			t.Fatalf("messages %d and %d share a role: %#v", i-1, i, messages)
		}
	}
}

// Mirrors the TS sanitizeConverseMessages suite.
func TestBedrockSanitizeMessages(t *testing.T) {
	tool := bedrockEmptyTurnToolTurn()
	messages := append(bedrockEmptyTurnToolTurn(),
		brtypes.Message{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{}},
		brtypes.Message{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "continue"}}},
		brtypes.Message{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "  \n"}, &brtypes.ContentBlockMemberText{Value: "answer"}}},
	)
	want := []brtypes.Message{
		tool[0],
		tool[1],
		{Role: brtypes.ConversationRoleUser, Content: append(append([]brtypes.ContentBlock(nil), tool[2].Content...), &brtypes.ContentBlockMemberText{Value: "continue"})},
		{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "answer"}}},
	}
	if got := sanitizeBedrockMessages(messages); !reflect.DeepEqual(got, want) {
		t.Fatalf("sanitized = %#v", got)
	}
	// The merge must not write into the caller's tool-result message.
	if len(messages[2].Content) != 1 {
		t.Fatalf("input modified: %#v", messages[2])
	}
	valid := append(bedrockEmptyTurnToolTurn(), brtypes.Message{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "done"}}})
	if got := sanitizeBedrockMessages(valid); !reflect.DeepEqual(got, valid) {
		t.Fatalf("valid conversation changed: %#v", got)
	}
}

// fakeBedrockFilteredRuntime answers Converse with no message, as after a
// content_filtered stop.
type fakeBedrockFilteredRuntime struct {
	fakeBedrockRuntime
}

func (f *fakeBedrockFilteredRuntime) Converse(_ context.Context, input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	f.got = input
	return &bedrockruntime.ConverseOutput{StopReason: brtypes.StopReasonContentFiltered, Usage: &brtypes.TokenUsage{InputTokens: aws.Int32(1), OutputTokens: aws.Int32(0), TotalTokens: aws.Int32(1)}}, nil
}

// Mirrors the TS "Bedrock empty assistant turns" suite.
func TestBedrockNeverReplaysAnEmptyAssistantTurn(t *testing.T) {
	segments := []PromptSegment{{Role: PromptRoleUser, Content: "continue"}}

	t.Run("content-filtered stream is not persisted", func(t *testing.T) {
		runtime := &fakeBedrockRuntime{streamEvents: []brtypes.ConverseStreamOutput{
			&brtypes.ConverseStreamOutputMemberMessageStart{Value: brtypes.MessageStartEvent{Role: brtypes.ConversationRoleAssistant}},
			&brtypes.ConverseStreamOutputMemberMessageStop{Value: brtypes.MessageStopEvent{StopReason: brtypes.StopReasonContentFiltered}},
		}}
		driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
		options := ExecutionOptions{Model: bedrockEmptyTurnModel, Conversation: bedrockPrompt{Messages: bedrockEmptyTurnToolTurn()[:2]}}
		stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "next"}}, options)
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
		_ = stream.Close()
		conversation := resp.Conversation.(bedrockPrompt)
		if conversation.Messages[len(conversation.Messages)-1].Role != brtypes.ConversationRoleUser {
			t.Fatalf("conversation = %#v", conversation.Messages)
		}
		expectValidConverseMessages(t, conversation.Messages)

		raw, err := json.Marshal(conversation)
		if err != nil {
			t.Fatal(err)
		}
		var restored map[string]any
		if err := json.Unmarshal(raw, &restored); err != nil {
			t.Fatal(err)
		}
		if _, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "The plan is not completed."}}, ExecutionOptions{Model: bedrockEmptyTurnModel, Conversation: restored}); err != nil {
			t.Fatal(err)
		}
		expectValidConverseMessages(t, runtime.got.Messages)
		last := runtime.got.Messages[len(runtime.got.Messages)-1].Content
		if bedrockBlockText(t, last[len(last)-1]) != "The plan is not completed." {
			t.Fatalf("last message = %#v", last)
		}
	})

	t.Run("stored empty turn is repaired", func(t *testing.T) {
		runtime := &fakeBedrockRuntime{}
		driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
		stored := append(bedrockEmptyTurnToolTurn(), brtypes.Message{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{}})
		resp, err := driver.Execute(context.Background(), segments, ExecutionOptions{Model: bedrockEmptyTurnModel, Conversation: bedrockPrompt{Messages: stored}})
		if err != nil {
			t.Fatal(err)
		}
		expectValidConverseMessages(t, runtime.got.Messages)
		expectValidConverseMessages(t, resp.Conversation.(bedrockPrompt).Messages)
	})

	t.Run("tool result separated by an empty turn is kept", func(t *testing.T) {
		runtime := &fakeBedrockRuntime{}
		driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
		tool := bedrockEmptyTurnToolTurn()
		stored := []brtypes.Message{tool[0], tool[1], {Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{}}, tool[2]}
		tools := []ToolDefinition{{Name: "search", InputSchema: map[string]any{"type": "object"}}}
		if _, err := driver.Execute(context.Background(), segments, ExecutionOptions{Model: bedrockEmptyTurnModel, Tools: tools, Conversation: bedrockPrompt{Messages: stored}}); err != nil {
			t.Fatal(err)
		}
		expectValidConverseMessages(t, runtime.got.Messages)
		last := runtime.got.Messages[len(runtime.got.Messages)-1].Content
		if len(last) != 2 || bedrockToolResultID(t, last[0]) != "call-1" || bedrockBlockText(t, last[1]) != "continue" {
			t.Fatalf("last message = %#v", last)
		}
	})

	t.Run("non-streaming response without a message is not persisted", func(t *testing.T) {
		runtime := &fakeBedrockFilteredRuntime{}
		driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
		resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "next"}}, ExecutionOptions{Model: bedrockEmptyTurnModel, Conversation: bedrockPrompt{Messages: bedrockEmptyTurnToolTurn()[:2]}})
		if err != nil {
			t.Fatal(err)
		}
		messages := resp.Conversation.(bedrockPrompt).Messages
		expectValidConverseMessages(t, messages)
		if messages[len(messages)-1].Role != brtypes.ConversationRoleUser {
			t.Fatalf("conversation = %#v", messages)
		}
	})
}
