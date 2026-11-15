package bedrock

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrock "github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
)

// Mirrors the TS model-listing.test.ts "derives custom-model capabilities from the
// reported base model identity", plus the embedding exclusions of llumiverse #579.
func TestBedrockListModelsUsesCustomBaseModelAndSkipsEmbeddings(t *testing.T) {
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-2"}, &fakeBedrockRuntime{}, &fakeBedrockModels{
		foundation: &bedrock.ListFoundationModelsOutput{ModelSummaries: []bedrocktypes.FoundationModelSummary{
			bedrockTestFoundationModel("amazon.nova-pro-v1:0", "Amazon", "Nova Pro"),
		}},
		custom: &bedrock.ListCustomModelsOutput{ModelSummaries: []bedrocktypes.CustomModelSummary{
			{
				ModelArn:      aws.String("arn:aws:bedrock:us-east-2:123456789012:custom-model/opaque-name"),
				ModelName:     aws.String("opaque-name"),
				BaseModelName: aws.String("openai.gpt-5.6-sol"),
			},
			{
				ModelArn:      aws.String("arn:aws:bedrock:us-east-2:123456789012:custom-model/opaque-embedder"),
				ModelName:     aws.String("opaque-embedder"),
				BaseModelName: aws.String("amazon.titan-embed-text-v2:0"),
			},
		}},
		profiles: &bedrock.ListInferenceProfilesOutput{InferenceProfileSummaries: []bedrocktypes.InferenceProfileSummary{
			{
				InferenceProfileArn:  aws.String("arn:aws:bedrock:us-east-2:123456789012:inference-profile/us.cohere.embed-v4:0"),
				InferenceProfileId:   aws.String("us.cohere.embed-v4:0"),
				InferenceProfileName: aws.String("US Cohere Embed v4"),
			},
		}},
	})
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 {
		t.Fatalf("models = %#v", models)
	}
	custom := models[1]
	if models[0].ID != "amazon.nova-pro-v1:0" || custom.ID != "arn:aws:bedrock:us-east-2:123456789012:custom-model/opaque-name" {
		t.Fatalf("models = %#v", models)
	}
	if inputs, outputs := *custom.InputModalities, *custom.OutputModalities; strings.Join(inputs, ",") != "text,image" || strings.Join(outputs, ",") != "text" {
		t.Fatalf("custom model modalities = %v -> %v, want the base model's", inputs, outputs)
	}
	if custom.ToolSupport == nil || !*custom.ToolSupport {
		t.Fatalf("custom model tool support = %v, want the base model's", custom.ToolSupport)
	}
}

// Mirrors the TS model-listing.test.ts "uses AWS modalities only when the catalog
// has no model information": the catalogue describes a model it knows, and the
// modalities AWS lists (SPEECH read as audio) describe an unknown one.
func TestBedrockListModelsUsesAWSModalitiesOnlyForUnknownModels(t *testing.T) {
	catalogued := bedrockTestFoundationModel("nvidia.nemotron-nano-12b-v2-vl-bf16", "NVIDIA", "Nemotron Nano 12B")
	catalogued.OutputModalities = []bedrocktypes.ModelModality{bedrocktypes.ModelModalityText, bedrocktypes.ModelModalityImage}
	future := bedrockTestFoundationModel("nvidia.future-foundation-model", "NVIDIA", "Future")
	future.InputModalities = []bedrocktypes.ModelModality{bedrocktypes.ModelModalityText, bedrocktypes.ModelModalityImage, "SPEECH"}
	claude := bedrockTestFoundationModel("anthropic.claude-sonnet-4-5-20250929-v1:0", "Anthropic", "Claude Sonnet 4.5")
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-2"}, &fakeBedrockRuntime{}, &fakeBedrockModels{
		foundation: &bedrock.ListFoundationModelsOutput{ModelSummaries: []bedrocktypes.FoundationModelSummary{catalogued, future, claude}},
	})
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, model := range models {
		tools := "unknown"
		if model.ToolSupport != nil {
			tools = fmt.Sprint(*model.ToolSupport)
		}
		got[model.ID] = strings.Join(*model.InputModalities, ",") + " -> " + strings.Join(*model.OutputModalities, ",") + " tools:" + tools
	}
	want := map[string]string{
		"nvidia.nemotron-nano-12b-v2-vl-bf16":       "text,image -> text tools:unknown",
		"nvidia.future-foundation-model":            "text,image,audio -> text tools:unknown",
		"anthropic.claude-sonnet-4-5-20250929-v1:0": "text,image -> text tools:true",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("models = %v, want %v", got, want)
	}
}
