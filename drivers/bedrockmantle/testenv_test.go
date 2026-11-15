package bedrockmantle

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/joho/godotenv"
	"github.com/vertesia/llumiverse-go/common"
)

func TestMain(m *testing.M) {
	_ = godotenv.Load("../.env", "../../.env", "../../../.env")
	os.Exit(m.Run())
}

// TestLiveMantleExecute runs one completion on Bedrock Mantle when BEDROCK_MANTLE_MODEL and AWS_REGION are set. It
// authenticates with BEDROCK_MANTLE_API_KEY when set, else with the default AWS credentials.
func TestLiveMantleExecute(t *testing.T) {
	model, region := os.Getenv("BEDROCK_MANTLE_MODEL"), os.Getenv("AWS_REGION")
	if model == "" || region == "" {
		t.Skip("set BEDROCK_MANTLE_MODEL and AWS_REGION, and BEDROCK_MANTLE_API_KEY or AWS credentials")
	}
	driver, err := NewBedrockMantleDriver(context.Background(), BedrockMantleOptions{Region: region, APIKey: os.Getenv("BEDROCK_MANTLE_API_KEY")})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: common.PromptRoleUser, Content: "Say hello."}}, ExecutionOptions{
		Model:        model,
		ModelOptions: map[string]any{"max_tokens": 64},
	})
	if err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	for _, result := range resp.Result {
		text.WriteString(result.String())
	}
	if text.Len() == 0 {
		t.Fatalf("empty result: %+v", resp.Completion)
	}
}
