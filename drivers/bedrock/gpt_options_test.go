package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrockruntime "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// Mirrors the TS model-options.test.ts "transports effort and strict output together for %s".
func TestBedrockGPTReasoningAndStrictOutput(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}, "required": []any{"answer"}, "additionalProperties": false}
	for _, model := range []string{
		"openai.gpt-5.5",
		"us.openai.gpt-5.6-sol",
		"global.openai.gpt-6-luna",
		"us.openai.gpt-6.1-sol",
		"us.openai.gpt-daybreak-blue-5.6-sol",
		"arn:aws:bedrock:us-east-1:000000000000:inference-profile/global.openai.gpt-6-luna",
	} {
		runtime := bedrockOptionsRequest(t, model, nil, ExecutionOptions{ResultSchema: schema, ModelOptions: map[string]any{
			"effort": "high", "reasoning_effort": "low", "verbosity": "low",
			"temperature": 0.7, "top_p": 0.9, "stop_sequence": []string{"STOP"}, "max_tokens": 100,
		}})
		cfg := runtime.got.InferenceConfig
		if aws.ToInt32(cfg.MaxTokens) != 100 || aws.ToFloat32(cfg.Temperature) != 0.7 || aws.ToFloat32(cfg.TopP) != 0.9 || !reflect.DeepEqual(cfg.StopSequences, []string{"STOP"}) {
			t.Errorf("%s: inference config = %#v", model, cfg)
		}
		if got := bedrockAdditionalFieldsJSON(t, runtime); got != `{"reasoning":{"effort":"high"},"text":{"format":{"strict":true},"verbosity":"low"}}` {
			t.Errorf("%s: additional fields = %s", model, got)
		}
		if runtime.got.OutputConfig == nil {
			t.Errorf("%s: no native output config", model)
		}
	}
}

// Mirrors the TS "preserves the reasoning_effort alias value %s" and "leaves
// reasoning to provider defaults when effort is unset".
func TestBedrockGPTEffortAliasAndDefaults(t *testing.T) {
	for _, effort := range []string{"none", "max"} {
		got := bedrockFamilyFields(ExecutionOptions{Model: "us.openai.gpt-6-luna", ModelOptions: map[string]any{"reasoning_effort": effort}})
		if !reflect.DeepEqual(got, map[string]any{"reasoning": map[string]any{"effort": effort}}) {
			t.Errorf("%s: fields = %#v", effort, got)
		}
	}
	if got := bedrockFamilyFields(ExecutionOptions{Model: "us.openai.gpt-6.1-sol"}); len(got) != 0 {
		t.Fatalf("default fields = %#v", got)
	}
	for model, want := range map[string]bool{
		"openai.gpt-oss-120b-1:0": false, "openai.gpt-4.1": false, "openai.gpt-5": true,
		"openai.gpt-daybreak-blue-5.6-sol": true, "global.openai.gpt-6.1-terra": true,
	} {
		if got := bedrockIsGPT5(model); got != want {
			t.Errorf("bedrockIsGPT5(%s) = %v", model, got)
		}
	}
}

// Mirrors the TS "preserves omitted and zero GPT sampling values".
func TestBedrockGPTKeepsZeroSampling(t *testing.T) {
	if cfg := bedrockInferenceConfig("global.openai.gpt-6-luna", map[string]any{}); cfg != nil {
		t.Fatalf("omitted config = %#v", cfg)
	}
	cfg := bedrockInferenceConfig("global.openai.gpt-6-luna", map[string]any{"temperature": 0, "top_p": 0, "stop_sequence": []string{"STOP"}})
	if cfg.Temperature == nil || *cfg.Temperature != 0 || cfg.TopP == nil || *cfg.TopP != 0 || len(cfg.StopSequences) != 1 {
		t.Fatalf("zero config = %#v", cfg)
	}
}

// Mirrors the TS "Bedrock Nova extended thinking" suite.
func TestBedrockNovaEffort(t *testing.T) {
	for _, effort := range []string{"none", "low", "medium", "high"} {
		runtime := bedrockOptionsRequest(t, "us.amazon.nova-2-lite-v1:0", nil, ExecutionOptions{ModelOptions: map[string]any{
			"effort": effort, "max_tokens": 1000, "temperature": 0.7, "top_k": 12,
		}})
		reasoning := `{"maxReasoningEffort":"` + effort + `","type":"enabled"}`
		if effort == "none" {
			reasoning = `{"type":"disabled"}`
		}
		if got, want := bedrockAdditionalFieldsJSON(t, runtime), `{"inferenceConfig":{"topK":12},"reasoningConfig":`+reasoning+`}`; got != want {
			t.Errorf("%s: additional fields = %s, want %s", effort, got, want)
		}
		if cfg := runtime.got.InferenceConfig; aws.ToInt32(cfg.MaxTokens) != 1000 || aws.ToFloat32(cfg.Temperature) != 0.7 || cfg.TopP != nil {
			t.Errorf("%s: inference config = %#v", effort, cfg)
		}
	}
	// Explicit effort passes through on Nova versions without known reasoning support.
	runtime := bedrockOptionsRequest(t, "amazon.nova-pro-v1:0", nil, ExecutionOptions{ModelOptions: map[string]any{"effort": "high"}})
	if got := bedrockAdditionalFieldsJSON(t, runtime); got != `{"reasoningConfig":{"maxReasoningEffort":"high","type":"enabled"}}` {
		t.Fatalf("nova pro fields = %s", got)
	}
	runtime = bedrockOptionsRequest(t, "amazon.nova-2-lite-v1:0", nil, ExecutionOptions{ModelOptions: map[string]any{"max_tokens": 1000}})
	if runtime.got.AdditionalModelRequestFields != nil {
		t.Fatalf("default fields = %s", bedrockAdditionalFieldsJSON(t, runtime))
	}
}

// Mirrors the TS reasoning-replay.test.ts "keeps encrypted $model reasoning out
// of answers when streaming=$streaming": redacted bytes are replayed, never shown.
func TestBedrockKeepsRedactedReasoningOutOfAnswers(t *testing.T) {
	redacted := []byte{1, 2, 255}
	options := func(model string, conversation any) ExecutionOptions {
		return ExecutionOptions{Model: model, ModelOptions: map[string]any{"include_thoughts": true}, Conversation: conversation}
	}
	segments := []PromptSegment{{Role: PromptRoleUser, Content: "question"}}
	for _, model := range []string{"openai.gpt-6-sol", "openai.gpt-6.1-sol", "global.openai.gpt-6.1-terra"} {
		for _, streaming := range []bool{false, true} {
			runtime := &fakeBedrockReasoningRuntime{output: &bedrockruntime.ConverseOutput{
				Output: &brtypes.ConverseOutputMemberMessage{Value: brtypes.Message{
					Role:    brtypes.ConversationRoleAssistant,
					Content: []brtypes.ContentBlock{bedrockRedactedReasoning(redacted), &brtypes.ContentBlockMemberText{Value: `{"answer":"ok"}`}},
				}},
				StopReason: brtypes.StopReasonEndTurn,
			}}
			runtime.streamEvents = []brtypes.ConverseStreamOutput{
				bedrockReasoningDelta(0, &brtypes.ReasoningContentBlockDeltaMemberRedactedContent{Value: redacted}),
				&brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(1), Delta: &brtypes.ContentBlockDeltaMemberText{Value: `{"answer":"ok"}`}}},
				&brtypes.ConverseStreamOutputMemberMessageStop{Value: brtypes.MessageStopEvent{StopReason: brtypes.StopReasonEndTurn}},
			}
			driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
			var resp *ExecutionResponse
			if streaming {
				stream, err := driver.Stream(context.Background(), segments, options(model, nil))
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
				resp = stream.Completion()
				_ = stream.Close()
			} else {
				var err error
				if resp, err = driver.Execute(context.Background(), segments, options(model, nil)); err != nil {
					t.Fatal(err)
				}
			}
			if text := completionResultsToText(resp.Result); text != `{"answer":"ok"}` {
				t.Errorf("%s streaming=%v: result = %q", model, streaming, text)
			}
			data, err := json.Marshal(resp.Conversation)
			if err != nil {
				t.Fatal(err)
			}
			var persisted any
			if err := json.Unmarshal(data, &persisted); err != nil {
				t.Fatal(err)
			}
			if _, err := driver.Execute(context.Background(), segments, options(model, persisted)); err != nil {
				t.Fatal(err)
			}
			replay := runtime.calls[len(runtime.calls)-1].Messages[1].Content[0]
			reasoning, ok := replay.(*brtypes.ContentBlockMemberReasoningContent)
			if !ok {
				t.Fatalf("%s streaming=%v: replay = %#v", model, streaming, replay)
			}
			if got, ok := reasoning.Value.(*brtypes.ReasoningContentBlockMemberRedactedContent); !ok || !bytes.Equal(got.Value, redacted) {
				t.Errorf("%s streaming=%v: replayed reasoning = %#v", model, streaming, reasoning.Value)
			}
		}
	}
}
