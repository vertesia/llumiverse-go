package togetherai

import (
	"context"
	"os"
	"testing"

	"github.com/joho/godotenv"
	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
)

func TestMain(m *testing.M) {
	_ = godotenv.Load("../.env", "../../.env", "../../../.env")
	os.Exit(m.Run())
}

// TestLiveTogetherAIExecute runs one completion on TogetherAI when TOGETHER_API_KEY and TOGETHER_MODEL are set.
func TestLiveTogetherAIExecute(t *testing.T) {
	key, model := os.Getenv("TOGETHER_API_KEY"), os.Getenv("TOGETHER_MODEL")
	if key == "" || model == "" {
		t.Skip("set TOGETHER_API_KEY and TOGETHER_MODEL")
	}
	driver, err := NewTogetherAIDriver(TogetherAIOptions{APIKey: key})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Say hello."}}, common.ExecutionOptions{
		Model:        model,
		ModelOptions: map[string]any{"max_tokens": 64},
	})
	if err != nil {
		t.Fatal(err)
	}
	if core.CompletionResultsToText(resp.Result) == "" {
		t.Fatalf("empty result: %+v", resp.Completion)
	}
}
