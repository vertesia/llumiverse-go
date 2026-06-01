package llumiverse_test

import (
	"context"

	"github.com/vertesia/llumiverse-go"
	"golang.org/x/oauth2"
)

func ExampleNewOpenAIDriver() {
	driver, err := llumiverse.NewOpenAIDriver(llumiverse.OpenAICompatibleOptions{
		APIKey:   "sk-test",
		Endpoint: "https://api.openai.com/v1",
	})
	if err != nil {
		panic(err)
	}

	_, _ = driver.Execute(context.Background(), []llumiverse.PromptSegment{{
		Role:    llumiverse.PromptRoleUser,
		Content: "Summarize the release notes.",
	}}, llumiverse.ExecutionOptions{
		Model: "gpt-4.1-mini",
	})
}

func ExampleNewVertexAIDriver() {
	driver, err := llumiverse.NewVertexAIDriver(llumiverse.VertexAIOptions{
		Project:     "my-project",
		Region:      "us-central1",
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "test-token"}),
	})
	if err != nil {
		panic(err)
	}

	_, _ = driver.Execute(context.Background(), []llumiverse.PromptSegment{{
		Role:    llumiverse.PromptRoleUser,
		Content: "Extract the contract parties as JSON.",
	}}, llumiverse.ExecutionOptions{
		Model:        "gemini-2.5-flash",
		ResultSchema: map[string]any{"type": "object"},
	})
}

func ExampleExecutionOptions_tools() {
	driver, err := llumiverse.NewOpenAIDriver(llumiverse.OpenAICompatibleOptions{
		APIKey:   "sk-test",
		Endpoint: "https://api.openai.com/v1",
	})
	if err != nil {
		panic(err)
	}

	_, _ = driver.Execute(context.Background(), []llumiverse.PromptSegment{{
		Role:    llumiverse.PromptRoleUser,
		Content: "What is the invoice status?",
	}}, llumiverse.ExecutionOptions{
		Model: "gpt-4.1-mini",
		Tools: []llumiverse.ToolDefinition{{
			Name:        "lookup_invoice",
			Description: "Lookup an invoice by ID.",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"id": map[string]any{"type": "string"}},
				"required":   []string{"id"},
			},
		}},
	})
}
