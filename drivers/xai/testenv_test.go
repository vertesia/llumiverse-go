package xai

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

// TestLiveXAIExecute runs one completion on XAI when XAI_API_KEY and XAI_MODEL are set.
func TestLiveXAIExecute(t *testing.T) {
	key, model := os.Getenv("XAI_API_KEY"), os.Getenv("XAI_MODEL")
	if key == "" || model == "" {
		t.Skip("set XAI_API_KEY and XAI_MODEL")
	}
	driver, err := NewXAIDriver(XAIOptions{APIKey: key})
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
