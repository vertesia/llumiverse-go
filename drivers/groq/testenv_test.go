package groq

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

// TestLiveGroqExecute runs one completion on Groq when GROQ_API_KEY and GROQ_MODEL are set.
func TestLiveGroqExecute(t *testing.T) {
	key, model := os.Getenv("GROQ_API_KEY"), os.Getenv("GROQ_MODEL")
	if key == "" || model == "" {
		t.Skip("set GROQ_API_KEY and GROQ_MODEL")
	}
	driver, err := NewGroqDriver(GroqOptions{APIKey: key})
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
