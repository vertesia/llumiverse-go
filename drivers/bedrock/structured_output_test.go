package bedrock

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrock "github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

var bedrockTestResultSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"object": map[string]any{"type": "string"},
		"color":  map[string]any{"type": "string"},
	},
	"required": []any{"object", "color"},
}

// bedrockStructuredRequest runs Execute for model with the test result schema
// and returns the prepared prompt and the Converse request.
func bedrockStructuredRequest(t *testing.T, model string) (bedrockPrompt, *fakeBedrockRuntime) {
	t.Helper()
	runtime := &fakeBedrockRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	options := ExecutionOptions{Model: model, ResultSchema: bedrockTestResultSchema}
	prompt, err := driver.CreatePrompt(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "hello"}}, options)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "hello"}}, options); err != nil {
		t.Fatal(err)
	}
	return prompt.(bedrockPrompt), runtime
}

func bedrockSystemMentionsSchema(prompt bedrockPrompt) bool {
	for _, block := range prompt.System {
		if text, ok := block.(*brtypes.SystemContentBlockMemberText); ok && strings.Contains(text.Value, "JSON Schema") {
			return true
		}
	}
	return false
}

// Mirrors the TS structured-output.test.ts "uses outputConfig for non-Claude models"
// and "keeps native schemas out of non-Claude prompts".
func TestBedrockNativeStructuredOutput(t *testing.T) {
	prompt, runtime := bedrockStructuredRequest(t, "google.gemma-3-12b-it")
	if len(prompt.System) != 0 {
		t.Fatalf("native schema leaked into the prompt: %#v", prompt.System)
	}
	cfg := runtime.got.OutputConfig
	if cfg == nil || cfg.TextFormat == nil || cfg.TextFormat.Type != brtypes.OutputFormatTypeJsonSchema {
		t.Fatalf("outputConfig = %#v", cfg)
	}
	structure, ok := cfg.TextFormat.Structure.(*brtypes.OutputFormatStructureMemberJsonSchema)
	if !ok || aws.ToString(structure.Value.Name) != "output" {
		t.Fatalf("structure = %#v", cfg.TextFormat.Structure)
	}
	want, _ := json.Marshal(bedrockTestResultSchema)
	if aws.ToString(structure.Value.Schema) != string(want) {
		t.Fatalf("schema = %s, want %s", aws.ToString(structure.Value.Schema), want)
	}
}

// Mirrors the TS "falls back to prompt alignment when %s rejects outputConfig" cases.
func TestBedrockPromptGuidedStructuredOutput(t *testing.T) {
	for _, model := range []string{
		"google.gemma-3-4b-it",
		"us.amazon.nova-micro-v1:0",
		"global.amazon.nova-2-lite-v1:0",
		"ai21.jamba-1-5-large-v1:0",
		"cohere.command-r-v1:0",
		"us.meta.llama3-3-70b-instruct-v1:0",
		"us.deepseek.r1-v1:0",
		"mistral.mistral-large-2402-v1:0",
		"mistral.pixtral-large-2502-v1:0",
		"us.writer.palmyra-x5-v1:0",
		"anthropic.claude-sonnet-4-6",
	} {
		prompt, runtime := bedrockStructuredRequest(t, model)
		if !bedrockSystemMentionsSchema(prompt) {
			t.Errorf("%s: schema missing from the prompt: %#v", model, prompt.System)
		}
		if runtime.got.OutputConfig != nil {
			t.Errorf("%s: outputConfig = %#v, want none", model, runtime.got.OutputConfig)
		}
	}
}

// Mirrors the TS "supplements MiniMax M2.5 outputConfig with prompt alignment".
func TestBedrockMiniMaxM25UsesBothSchemaPaths(t *testing.T) {
	prompt, runtime := bedrockStructuredRequest(t, "minimax.minimax-m2.5")
	if !bedrockSystemMentionsSchema(prompt) {
		t.Fatalf("schema missing from the prompt: %#v", prompt.System)
	}
	if runtime.got.OutputConfig == nil || runtime.got.OutputConfig.TextFormat.Type != brtypes.OutputFormatTypeJsonSchema {
		t.Fatalf("outputConfig = %#v", runtime.got.OutputConfig)
	}
}

// Mirrors the TS model-listing.test.ts discovery cases.
func TestBedrockListModelsDiscoversKnownPublishers(t *testing.T) {
	cases := []struct {
		id           string
		discoverable bool
	}{
		{"ai21.jamba-2-large-v1:0", true},
		{"amazon.nova-3-lite-v1:0", true},
		{"anthropic.claude-sonnet-6", true},
		{"cohere.command-r-next-v1:0", true},
		{"deepseek.v4", true},
		{"google.gemma-3-30b-it", true},
		{"meta.llama5-scout-instruct-v1:0", true},
		{"minimax.minimax-m3", true},
		{"mistral.mistral-large-4-instruct", true},
		{"moonshot.kimi-k3-thinking", true},
		{"moonshotai.kimi-k3", true},
		{"nvidia.nemotron-super-4-180b", true},
		{"openai.gpt-oss-200b-1:0", true},
		{"qwen.qwen4-coder", true},
		{"writer.palmyra-x6-v1:0", true},
		{"zai.glm-6", true},
		{"amazon.titan-image-generator-v3", false},
		// Every Sonic generation is a bidirectional speech stream rather than Converse.
		{"amazon.nova-sonic-v1:0", false},
		{"amazon.nova-2-sonic-v1:0", false},
		{"amazon.nova-2-5-sonic", false},
		{"xai.grok-4.7", true},
		{"amazon.nova-canvas-v2:0", true},
		{"unverified.model-1", false},
	}
	summaries := make([]bedrocktypes.FoundationModelSummary, 0, len(cases))
	for _, c := range cases {
		summaries = append(summaries, bedrockTestFoundationModel(c.id, strings.Split(c.id, ".")[0], c.id))
	}
	// Publisher display names carry punctuation, e.g. "Z.AI".
	summaries = append(summaries, bedrockTestFoundationModel("zai.glm-7", "Z.AI", "GLM 7"))
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-2"}, &fakeBedrockRuntime{}, &fakeBedrockModels{
		foundation: &bedrock.ListFoundationModelsOutput{ModelSummaries: summaries},
	})
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	listed := map[string]AIModel{}
	for _, model := range models {
		listed[model.ID] = model
	}
	for _, c := range cases {
		if _, ok := listed[c.id]; ok != c.discoverable {
			t.Errorf("%s listed = %v, want %v", c.id, ok, c.discoverable)
		}
	}
	if _, ok := listed["zai.glm-7"]; !ok {
		t.Error("Z.AI model not listed")
	}
}

// Mirrors the TS "groups inference profiles under the publisher name AWS reports for foundation models".
func TestBedrockListModelsGroupsProfilesUnderPublisherName(t *testing.T) {
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-2"}, &fakeBedrockRuntime{}, &fakeBedrockModels{
		foundation: &bedrock.ListFoundationModelsOutput{ModelSummaries: []bedrocktypes.FoundationModelSummary{
			bedrockTestFoundationModel("zai.glm-5", "Z.ai", "GLM 5"),
			bedrockTestFoundationModel("moonshotai.kimi-k2.5", "Moonshot AI", "Kimi K2.5"),
		}},
		profiles: &bedrock.ListInferenceProfilesOutput{InferenceProfileSummaries: []bedrocktypes.InferenceProfileSummary{
			{
				InferenceProfileArn:  aws.String("arn:aws:bedrock:us-east-2::inference-profile/global.zai.glm-5.3"),
				InferenceProfileId:   aws.String("global.zai.glm-5.3"),
				InferenceProfileName: aws.String("Z.AI GLM 5.3"),
			},
			{
				InferenceProfileArn:  aws.String("arn:aws:bedrock:us-east-2::inference-profile/global.moonshotai.kimi-k3"),
				InferenceProfileId:   aws.String("global.moonshotai.kimi-k3"),
				InferenceProfileName: aws.String("Moonshot AI Kimi K3"),
			},
		}},
	})
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	owners := map[string]string{}
	for _, model := range models {
		owners[model.Name] = aws.ToString(model.Owner)
	}
	want := map[string]string{"GLM 5": "Z.ai", "Z.AI GLM 5.3": "Z.ai", "Kimi K2.5": "Moonshot AI", "Moonshot AI Kimi K3": "Moonshot AI"}
	if !reflect.DeepEqual(owners, want) {
		t.Fatalf("owners = %#v", owners)
	}
}

// Mirrors the TS "uses the AWS model name without repeating the publisher".
func TestBedrockListModelsUsesAWSModelName(t *testing.T) {
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-2"}, &fakeBedrockRuntime{}, &fakeBedrockModels{
		foundation: &bedrock.ListFoundationModelsOutput{ModelSummaries: []bedrocktypes.FoundationModelSummary{
			bedrockTestFoundationModel("deepseek.v3.2", "DeepSeek", "DeepSeek V3.2"),
		}},
	})
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].Name != "DeepSeek V3.2" || aws.ToString(models[0].Owner) != "DeepSeek" {
		t.Fatalf("models = %#v", models)
	}
}

func bedrockTestFoundationModel(id string, provider string, name string) bedrocktypes.FoundationModelSummary {
	return bedrocktypes.FoundationModelSummary{
		ModelId:                    aws.String(id),
		ModelName:                  aws.String(name),
		ProviderName:               aws.String(provider),
		InferenceTypesSupported:    []bedrocktypes.InferenceType{bedrocktypes.InferenceTypeOnDemand},
		InputModalities:            []bedrocktypes.ModelModality{bedrocktypes.ModelModalityText},
		OutputModalities:           []bedrocktypes.ModelModality{bedrocktypes.ModelModalityText},
		ResponseStreamingSupported: aws.Bool(true),
	}
}
