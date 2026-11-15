package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrockruntime "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/vertesia/llumiverse-go/common"
	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
)

const claudeTestModel = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

// fakeClaudeRuntime is a runtime that can stream InvokeModel responses, so Claude runs on the Messages API.
type fakeClaudeRuntime struct {
	invokes      []*bedrockruntime.InvokeModelInput
	streams      []*bedrockruntime.InvokeModelWithResponseStreamInput
	converses    int
	responseBody string
	tier         brtypes.ServiceTierType
	events       []string
	streamErr    error
}

func (f *fakeClaudeRuntime) Converse(context.Context, *bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	f.converses++
	return &bedrockruntime.ConverseOutput{
		Output:     &brtypes.ConverseOutputMemberMessage{Value: brtypes.Message{Role: brtypes.ConversationRoleAssistant, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "converse ok"}}}},
		StopReason: brtypes.StopReasonEndTurn,
		Usage:      &brtypes.TokenUsage{InputTokens: aws.Int32(1), OutputTokens: aws.Int32(1), TotalTokens: aws.Int32(2)},
	}, nil
}

func (f *fakeClaudeRuntime) ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (bedrockConverseStream, error) {
	return nil, errors.New("unexpected ConverseStream")
}

func (f *fakeClaudeRuntime) InvokeModel(_ context.Context, input *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	f.invokes = append(f.invokes, input)
	body := f.responseBody
	if body == "" {
		body = `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`
	}
	return &bedrockruntime.InvokeModelOutput{Body: []byte(body), ServiceTier: f.tier}, nil
}

func (f *fakeClaudeRuntime) InvokeModelWithResponseStream(_ context.Context, input *bedrockruntime.InvokeModelWithResponseStreamInput, _ ...func(*bedrockruntime.Options)) (InvokeModelStream, error) {
	f.streams = append(f.streams, input)
	events := make(chan brtypes.ResponseStream, len(f.events))
	for _, event := range f.events {
		events <- &brtypes.ResponseStreamMemberChunk{Value: brtypes.PayloadPart{Bytes: []byte(event)}}
	}
	close(events)
	return &fakeInvokeModelStream{events: events, err: f.streamErr, tier: string(f.tier)}, nil
}

type fakeInvokeModelStream struct {
	events chan brtypes.ResponseStream
	err    error
	tier   string
}

func (s *fakeInvokeModelStream) Events() <-chan brtypes.ResponseStream { return s.events }
func (s *fakeInvokeModelStream) Close() error                          { return nil }
func (s *fakeInvokeModelStream) Err() error                            { return s.err }
func (s *fakeInvokeModelStream) ServiceTier() string                   { return s.tier }

// converseOnlyRuntime hides InvokeModelWithResponseStream, as an older injected runtime would.
type converseOnlyRuntime struct{ inner *fakeClaudeRuntime }

func (r converseOnlyRuntime) Converse(ctx context.Context, input *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	return r.inner.Converse(ctx, input, optFns...)
}

func (r converseOnlyRuntime) ConverseStream(ctx context.Context, input *bedrockruntime.ConverseStreamInput, optFns ...func(*bedrockruntime.Options)) (bedrockConverseStream, error) {
	return r.inner.ConverseStream(ctx, input, optFns...)
}

func (r converseOnlyRuntime) InvokeModel(ctx context.Context, input *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	return r.inner.InvokeModel(ctx, input, optFns...)
}

func claudeTestSegments(text string) []PromptSegment {
	return []PromptSegment{{Role: PromptRoleUser, Content: text}}
}

func decodeInvokeBody(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

var legacyConverseConversation = map[string]any{
	"messages": []any{
		map[string]any{"role": "user", "content": []any{map[string]any{"text": "earlier"}}},
		map[string]any{"role": "assistant", "content": []any{map[string]any{"text": "reply"}}},
	},
}

func TestIsConverseConversation(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct {
		conversation      any
		converse, history bool
	}{
		"converse blocks":           {conversation: legacyConverseConversation, converse: true},
		"go struct shape":           {conversation: map[string]any{"Messages": []any{map[string]any{"Role": "user", "Content": []any{map[string]any{"Value": "hi"}}}}}, converse: true},
		"system blocks alone":       {conversation: map[string]any{"messages": []any{}, "system": []any{map[string]any{"text": "be brief"}}}, converse: true},
		"empty content skipped":     {conversation: map[string]any{"messages": []any{map[string]any{"role": "assistant", "content": []any{}}, legacyConverseConversation["messages"].([]any)[0]}}, converse: true},
		"in-memory converse prompt": {conversation: bedrockPrompt{}, converse: true},
		"nil":                       {},
		"no messages":               {conversation: map[string]any{"messages": []any{}}},
		"string content":            {conversation: map[string]any{"messages": []any{map[string]any{"role": "user", "content": "hi"}}}, history: true},
		"typed blocks":              {conversation: map[string]any{"messages": []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "hi"}}}}}, history: true},
		"tool change":               {conversation: map[string]any{"messages": []any{map[string]any{"role": "system", "content": []any{map[string]any{"type": "tool_addition", "tool": map[string]any{"type": "tool_reference", "name": "a"}}}}}}, history: true},
		"in-memory claude prompt":   {conversation: bedrockClaudePrompt{Messages: []internalclaude.Message{{Role: "user"}}}, history: true},
	} {
		if got := isConverseConversation(tc.conversation); got != tc.converse {
			t.Errorf("%s: isConverseConversation = %v", name, got)
		}
		if got := isMessagesAPIConversation(tc.conversation); got != tc.history {
			t.Errorf("%s: isMessagesAPIConversation = %v", name, got)
		}
	}
}

func TestBedrockClaudeRoutesNewConversationsToTheMessagesAPI(t *testing.T) {
	t.Parallel()

	runtime := &fakeClaudeRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, &fakeBedrockModels{})
	ctx := context.Background()

	fresh, err := driver.CreatePrompt(ctx, claudeTestSegments("hello"), ExecutionOptions{Model: claudeTestModel})
	if err != nil {
		t.Fatal(err)
	}
	if prompt, ok := fresh.(bedrockClaudePrompt); !ok || prompt.Messages[0].Content[0].Text != "hello" {
		t.Fatalf("fresh prompt = %#v", fresh)
	}
	// A foundation-model ARN names its model; an application inference profile does not.
	arn, _ := driver.CreatePrompt(ctx, claudeTestSegments("hello"), ExecutionOptions{Model: "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0"})
	if _, ok := arn.(bedrockClaudePrompt); !ok {
		t.Fatalf("ARN prompt = %T", arn)
	}
	profile, _ := driver.CreatePrompt(ctx, claudeTestSegments("hello"), ExecutionOptions{Model: "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"})
	legacy, _ := driver.CreatePrompt(ctx, claudeTestSegments("hello"), ExecutionOptions{Model: claudeTestModel, Conversation: legacyConverseConversation})
	other, _ := driver.CreatePrompt(ctx, claudeTestSegments("hello"), ExecutionOptions{Model: "amazon.nova-pro-v1:0"})
	older, _ := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, converseOnlyRuntime{runtime}, &fakeBedrockModels{}).
		CreatePrompt(ctx, claudeTestSegments("hello"), ExecutionOptions{Model: claudeTestModel})
	for name, prompt := range map[string]any{"profile": profile, "legacy": legacy, "other": other, "older runtime": older} {
		if _, ok := prompt.(bedrockPrompt); !ok {
			t.Errorf("%s prompt = %T", name, prompt)
		}
	}

	// A legacy conversation continues on Converse.
	resp, err := driver.Execute(ctx, claudeTestSegments("hello"), ExecutionOptions{Model: claudeTestModel, Conversation: legacyConverseConversation})
	if err != nil {
		t.Fatal(err)
	}
	if runtime.converses != 1 || len(runtime.invokes) != 0 || !isConverseConversation(resp.Conversation) {
		t.Fatalf("converses = %d, invokes = %d, conversation = %T", runtime.converses, len(runtime.invokes), resp.Conversation)
	}
}

func TestBedrockClaudeExecuteUsesInvokeModel(t *testing.T) {
	t.Parallel()

	runtime := &fakeClaudeRuntime{tier: brtypes.ServiceTierTypePriority}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, &fakeBedrockModels{})
	modelOptions := map[string]any{"max_tokens": 10_000_000, "service_tier": "priority"}
	resp, err := driver.Execute(context.Background(), claudeTestSegments("hello"), ExecutionOptions{Model: claudeTestModel, ModelOptions: modelOptions})
	if err != nil {
		t.Fatal(err)
	}
	input := runtime.invokes[0]
	if aws.ToString(input.ModelId) != claudeTestModel || input.ServiceTier != brtypes.ServiceTierTypePriority || aws.ToString(input.ContentType) != "application/json" {
		t.Fatalf("input = %#v", input)
	}
	body := decodeInvokeBody(t, input.Body)
	if body["anthropic_version"] != bedrockAnthropicVersion || body["model"] != nil || body["stream"] != nil || body["max_tokens"] != float64(64000) {
		t.Fatalf("body = %#v", body)
	}
	if modelOptions["max_tokens"] != 10_000_000 {
		t.Fatal("the caller's options changed")
	}
	if resp.Result[0].String() != "ok" || resp.FinishReason != "stop" || resp.ServiceTier != "priority" {
		t.Fatalf("completion = %#v", resp.Completion)
	}
	conversation, ok := resp.Conversation.(bedrockClaudePrompt)
	if !ok || len(conversation.Messages) != 2 || conversation.Messages[1].Role != "assistant" {
		t.Fatalf("conversation = %#v", resp.Conversation)
	}

	// A Messages API conversation cannot continue on a Converse model.
	_, err = driver.Execute(context.Background(), claudeTestSegments("more"), ExecutionOptions{Model: "amazon.nova-pro-v1:0", Conversation: resp.Conversation})
	var formatErr *common.LlumiverseError
	if !errors.As(err, &formatErr) || formatErr.Name != "ConversationFormatError" || formatErr.Retryable == nil || *formatErr.Retryable {
		t.Fatalf("err = %v", err)
	}
}

func TestBedrockClaudeSendsToolChangeBetasInTheBody(t *testing.T) {
	t.Parallel()

	runtime := &fakeClaudeRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, &fakeBedrockModels{})
	model := "global.anthropic.claude-opus-4-8-v1:0"
	tool := func(name string) ToolDefinition {
		return ToolDefinition{Name: name, InputSchema: JSONObject{"type": "object", "properties": map[string]any{}}}
	}
	first, err := driver.Execute(context.Background(), claudeTestSegments("hello"), ExecutionOptions{Model: model, Tools: []ToolDefinition{tool("lookup")}})
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(first.Conversation)
	var stored any
	_ = json.Unmarshal(data, &stored)
	if _, err := driver.Execute(context.Background(), claudeTestSegments("more"), ExecutionOptions{
		Model:        model,
		Tools:        []ToolDefinition{tool("lookup"), tool("list_items")},
		Conversation: stored,
	}); err != nil {
		t.Fatal(err)
	}
	body := decodeInvokeBody(t, runtime.invokes[1].Body)
	messages := body["messages"].([]any)
	if last := messages[len(messages)-1].(map[string]any); last["role"] != "system" {
		t.Fatalf("last message = %#v", last)
	}
	betas, _ := body["anthropic_beta"].([]any)
	if len(betas) != 1 || betas[0] != "mid-conversation-tool-changes-2026-07-01" {
		t.Fatalf("anthropic_beta = %#v", body["anthropic_beta"])
	}
}

func TestBedrockClaudeNativeStructuredOutputs(t *testing.T) {
	t.Parallel()

	schema := JSONObject{
		"type":                 "object",
		"properties":           map[string]any{"answer": map[string]any{"type": "string"}},
		"required":             []any{"answer"},
		"additionalProperties": false,
	}
	for _, tc := range []struct {
		model  string
		native bool
	}{
		{model: claudeTestModel, native: true},
		{model: "us.anthropic.claude-sonnet-4-20250514-v1:0", native: false},
	} {
		runtime := &fakeClaudeRuntime{responseBody: `{"role":"assistant","content":[{"type":"text","text":"{\"answer\":\"yes\"}"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`}
		driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, &fakeBedrockModels{})
		if _, err := driver.Execute(context.Background(), claudeTestSegments("Answer"), ExecutionOptions{Model: tc.model, ResultSchema: schema}); err != nil {
			t.Fatal(err)
		}
		body := decodeInvokeBody(t, runtime.invokes[0].Body)
		_, hasFormat := body["output_config"].(map[string]any)
		system, _ := body["system"].([]any)
		if hasFormat != tc.native || (len(system) == 0) != tc.native {
			t.Errorf("%s: body = %#v", tc.model, body)
		}
	}
}

func TestBedrockClaudeStream(t *testing.T) {
	t.Parallel()

	runtime := &fakeClaudeRuntime{tier: brtypes.ServiceTierTypeFlex, events: []string{
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":3,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
		`{"type":"message_stop","amazon-bedrock-invocationMetrics":{"inputTokenCount":3,"outputTokenCount":4}}`,
	}}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, &fakeBedrockModels{})
	stream, err := driver.Stream(context.Background(), claudeTestSegments("hello"), ExecutionOptions{Model: claudeTestModel, ModelOptions: map[string]any{"service_tier": "flex"}})
	if err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		for _, result := range chunk.Result {
			text.WriteString(result.String())
		}
	}
	completion := stream.Completion()
	if text.String() != "ok" || completion.FinishReason != "stop" || completion.ServiceTier != "flex" {
		t.Fatalf("text = %q, completion = %#v", text.String(), completion.Completion)
	}
	if input := runtime.streams[0]; input.ServiceTier != brtypes.ServiceTierTypeFlex || decodeInvokeBody(t, input.Body)["stream"] != nil {
		t.Fatalf("input = %#v", input)
	}
	conversation := completion.Conversation.(bedrockClaudePrompt)
	if len(conversation.Messages) != 2 || conversation.Messages[1].Content[0].Text != "ok" {
		t.Fatalf("conversation = %#v", conversation)
	}
}

func TestBedrockClaudeStreamKeepsTheBedrockRetryRules(t *testing.T) {
	t.Parallel()

	start := `{"type":"message_start","message":{"role":"assistant","content":[],"usage":{"input_tokens":1}}}`
	for _, tc := range []struct {
		code      string
		retryable bool
	}{
		{code: "ThrottlingException", retryable: true},
		{code: "ValidationException", retryable: false},
	} {
		runtime := &fakeClaudeRuntime{events: []string{start}, streamErr: fakeBedrockAPIError{code: tc.code, message: "failed"}}
		driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, &fakeBedrockModels{})
		stream, err := driver.Stream(context.Background(), claudeTestSegments("hello"), ExecutionOptions{Model: claudeTestModel})
		if err != nil {
			t.Fatal(err)
		}
		for err == nil {
			_, err = stream.Recv()
		}
		var llmErr *common.LlumiverseError
		if !errors.As(err, &llmErr) || llmErr.Retryable == nil || *llmErr.Retryable != tc.retryable || llmErr.Name != tc.code {
			t.Errorf("%s: err = %#v", tc.code, err)
		}
	}
}

func TestBedrockClaudeMaxTokens(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model   string
		options map[string]any
		want    int
	}{
		{model: "us.anthropic.claude-opus-4-7-v1:0", want: 127999},
		{model: "eu.anthropic.claude-opus-4-6-v1:0", want: 127999},
		{model: "global.anthropic.claude-sonnet-4-6", want: 64000},
		{model: "anthropic.claude-sonnet-4-5-20250929-v1:0", want: 64000},
		{model: "anthropic.claude-opus-4-1-20250805-v1:0", want: 32000},
		{model: "anthropic.claude-sonnet-4-20250514-v1:0", want: 64000},
		{model: "anthropic.claude-3-5-haiku-20241022-v1:0", want: 8192},
		{model: "anthropic.claude-3-haiku-20240307-v1:0", want: 4096},
		{model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0", want: 64000},
		{model: "anthropic.claude-sonnet-4-5-20250929-v1:0", options: map[string]any{"max_tokens": 1000}, want: 1000},
		{model: "anthropic.claude-opus-4-7-v1:0", options: map[string]any{"max_tokens": 128000}, want: 127999},
	} {
		if got := bedrockClaudeMaxTokens(ExecutionOptions{Model: tc.model, ModelOptions: tc.options}); got != tc.want {
			t.Errorf("%s %v: max_tokens = %d, want %d", tc.model, tc.options, got, tc.want)
		}
	}
}
