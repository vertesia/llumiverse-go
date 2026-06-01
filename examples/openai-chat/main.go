package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"
	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/drivers/openai"
)

func main() {
	_ = godotenv.Load()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("set OPENAI_API_KEY to run this example")
		return
	}
	baseURL := os.Getenv("OPENAI_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	driver, err := openai.NewOpenAIDriver(openai.OpenAICompatibleOptions{APIKey: apiKey, Endpoint: baseURL})
	if err != nil {
		log.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []common.PromptSegment{{
		Role:    common.PromptRoleUser,
		Content: "Write one sentence about maintainable SDK design.",
	}}, common.ExecutionOptions{Model: "gpt-4.1-mini"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Result[0].Value)
}
