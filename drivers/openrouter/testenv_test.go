package openrouter

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

// TestLiveOpenRouterExecute runs one completion on OpenRouter when OPENROUTER_API_KEY and OPENROUTER_MODEL are set.
func TestLiveOpenRouterExecute(t *testing.T) {
	key, model := os.Getenv("OPENROUTER_API_KEY"), os.Getenv("OPENROUTER_MODEL")
	if key == "" || model == "" {
		t.Skip("set OPENROUTER_API_KEY and OPENROUTER_MODEL")
	}
	driver, err := NewOpenRouterDriver(OpenRouterOptions{APIKey: key})
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
