package azurefoundry

import (
	"context"
	"os"
	"testing"

	"github.com/joho/godotenv"
)

func TestMain(m *testing.M) {
	_ = godotenv.Load("../.env", "../../.env", "../../../.env")
	os.Exit(m.Run())
}

// TestLiveFoundryExecute runs one completion against a real deployment when
// AZURE_FOUNDRY_ENDPOINT, AZURE_FOUNDRY_MODEL and AZURE_FOUNDRY_API_KEY or
// AZURE_FOUNDRY_TOKEN are set.
func TestLiveFoundryExecute(t *testing.T) {
	endpoint, model := os.Getenv("AZURE_FOUNDRY_ENDPOINT"), os.Getenv("AZURE_FOUNDRY_MODEL")
	apiKey, token := os.Getenv("AZURE_FOUNDRY_API_KEY"), os.Getenv("AZURE_FOUNDRY_TOKEN")
	if endpoint == "" || model == "" || (apiKey == "" && token == "") {
		t.Skip("set AZURE_FOUNDRY_ENDPOINT, AZURE_FOUNDRY_MODEL and AZURE_FOUNDRY_API_KEY or AZURE_FOUNDRY_TOKEN")
	}
	options := AzureFoundryOptions{Endpoint: endpoint, APIKey: apiKey}
	if token != "" {
		options.TokenProvider = StaticToken(token)
	}
	driver, err := NewAzureFoundryDriver(options)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Say hello."}}, ExecutionOptions{
		Model:        model,
		ModelOptions: map[string]any{"max_tokens": 64},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resultText(resp.Result) == "" {
		t.Fatalf("empty result: %+v", resp.Completion)
	}
}
