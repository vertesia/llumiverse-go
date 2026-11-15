package bedrock

import (
	"context"
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// bedrockOptionsRequest runs one Converse request and returns what was sent.
func bedrockOptionsRequest(t *testing.T, model string, segments []PromptSegment, options ExecutionOptions) *fakeBedrockRuntime {
	t.Helper()
	runtime := &fakeBedrockRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	if segments == nil {
		segments = []PromptSegment{{Role: PromptRoleUser, Content: "hello"}}
	}
	options.Model = model
	if _, err := driver.Execute(context.Background(), segments, options); err != nil {
		t.Fatal(err)
	}
	return runtime
}

// Mirrors the TS model-options.test.ts "serializes supported fields for %s".
func TestBedrockFamilyAdditionalFields(t *testing.T) {
	for _, c := range []struct {
		model   string
		options map[string]any
		want    string
	}{
		{"amazon.nova-pro-v1:0", map[string]any{"top_k": 12}, `{"inferenceConfig":{"topK":12}}`},
		{"mistral.mixtral-8x7b-instruct-v0:1", map[string]any{"top_k": 12}, `{"top_k":12}`},
		{"cohere.command-text-v14", map[string]any{"top_k": 12}, `{"k":12}`},
		{"cohere.command-r-v1:0", map[string]any{"top_k": 12, "presence_penalty": 0.4, "frequency_penalty": 0.2}, `{"frequency_penalty":0.2,"k":12,"presence_penalty":0.4}`},
		{"ai21.jamba-1-5-large-v1:0", map[string]any{"presence_penalty": 0.4, "frequency_penalty": 0.2}, `{"frequency_penalty":0.2,"presence_penalty":0.4}`},
		{"ai21.j2-ultra-v1", map[string]any{"presence_penalty": 0.4, "frequency_penalty": 0.2}, `{"frequencyPenalty":{"scale":0.2},"presencePenalty":{"scale":0.4}}`},
		{"writer.palmyra-x5-v1:0", map[string]any{"seed": 7, "min_tokens": 3}, `{"min_tokens":3,"seed":7}`},
		{"openai.gpt-oss-120b-1:0", map[string]any{"reasoning_effort": "low", "top_k": 12}, `{"reasoning_effort":"low"}`},
		// Families without additional fields send none, even when the caller set top_k.
		{"mistral.mistral-large-2407-v1:0", map[string]any{"top_k": 12, "reasoning_effort": "medium"}, ""},
		{"meta.llama3-3-70b-instruct-v1:0", map[string]any{"top_k": 12}, ""},
		{"deepseek.v3-v1:0", map[string]any{"top_k": 12}, ""},
		{"amazon.titan-text-premier-v1:0", map[string]any{"top_k": 12}, ""},
	} {
		runtime := bedrockOptionsRequest(t, c.model, nil, ExecutionOptions{ModelOptions: c.options})
		if got := bedrockAdditionalFieldsJSON(t, runtime); got != c.want {
			t.Errorf("%s: additional fields = %s, want %s", c.model, got, c.want)
		}
	}
}

// Mirrors the TS Converse system handling for families without system blocks.
func TestBedrockFoldsSystemPromptForFamiliesWithoutSystemBlocks(t *testing.T) {
	segments := []PromptSegment{{Role: PromptRoleSystem, Content: "Be brief."}, {Role: PromptRoleUser, Content: "hello"}}
	for _, model := range []string{"mistral.mistral-7b-instruct-v0:2", "ai21.j2-ultra-v1", "cohere.command-text-v14"} {
		runtime := bedrockOptionsRequest(t, model, segments, ExecutionOptions{})
		want := []brtypes.Message{{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{
			&brtypes.ContentBlockMemberText{Value: "hello"},
			&brtypes.ContentBlockMemberText{Value: "Be brief."},
		}}}
		if runtime.got.System != nil || !reflect.DeepEqual(runtime.got.Messages, want) {
			t.Errorf("%s: system = %#v, messages = %#v", model, runtime.got.System, runtime.got.Messages)
		}
	}
	for _, model := range []string{"cohere.command-r-plus-v1:0", "mistral.mistral-large-2407-v1:0", "amazon.nova-pro-v1:0"} {
		runtime := bedrockOptionsRequest(t, model, segments, ExecutionOptions{})
		if len(runtime.got.System) != 1 || len(runtime.got.Messages[0].Content) != 1 {
			t.Errorf("%s: system = %#v, messages = %#v", model, runtime.got.System, runtime.got.Messages)
		}
	}
}

// Mirrors the TS sampling precedence: temperature wins over top_p, and DeepSeek
// takes neither top_p nor stop sequences.
func TestBedrockInferenceConfigSamplingPrecedence(t *testing.T) {
	sampling := map[string]any{"temperature": 0.7, "top_p": 0.9, "stop_sequence": []string{"STOP"}, "max_tokens": 100}
	cfg := bedrockInferenceConfig("mistral.mistral-large-2407-v1:0", sampling)
	if aws.ToFloat32(cfg.Temperature) != 0.7 || cfg.TopP != nil || !reflect.DeepEqual(cfg.StopSequences, []string{"STOP"}) {
		t.Fatalf("mistral config = %#v", cfg)
	}
	if cfg := bedrockInferenceConfig("mistral.mistral-large-2407-v1:0", map[string]any{"top_p": 0.9}); aws.ToFloat32(cfg.TopP) != 0.9 {
		t.Fatalf("top_p alone = %#v", cfg)
	}
	cfg = bedrockInferenceConfig("us.deepseek.r1-v1:0", map[string]any{"top_p": 0.9, "stop_sequence": []string{"STOP"}, "max_tokens": 100})
	if cfg.TopP != nil || cfg.StopSequences != nil || aws.ToInt32(cfg.MaxTokens) != 100 {
		t.Fatalf("deepseek config = %#v", cfg)
	}
	if _, ok := sampling["top_p"]; !ok {
		t.Fatal("caller options modified")
	}
}
