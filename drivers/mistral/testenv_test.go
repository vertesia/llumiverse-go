package mistral

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

// TestLiveMistralExecute runs one completion on Mistral when MISTRAL_API_KEY and MISTRAL_MODEL are set.
func TestLiveMistralExecute(t *testing.T) {
	key, model := os.Getenv("MISTRAL_API_KEY"), os.Getenv("MISTRAL_MODEL")
	if key == "" || model == "" {
		t.Skip("set MISTRAL_API_KEY and MISTRAL_MODEL")
	}
	driver, err := NewMistralDriver(MistralOptions{APIKey: key})
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
