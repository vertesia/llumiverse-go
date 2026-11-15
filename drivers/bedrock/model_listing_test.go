package bedrock

import (
	"context"
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
				BaseModelName: aws.String("amazon.nova-canvas-v1:0"),
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
	if outputs := *custom.OutputModalities; len(outputs) != 1 || outputs[0] != "image" {
		t.Fatalf("custom model outputs = %v, want the base model's image output", outputs)
	}
}

func TestBedrockModalitiesMapSpeechToAudio(t *testing.T) {
	got := bedrockModalities([]bedrocktypes.ModelModality{bedrocktypes.ModelModalityText, "SPEECH"})
	if len(got) != 2 || got[1] != "audio" {
		t.Fatalf("modalities = %v", got)
	}
}
