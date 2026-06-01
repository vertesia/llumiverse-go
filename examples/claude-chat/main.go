package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"
	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/drivers/anthropic"
)

func main() {
	_ = godotenv.Load()

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Println("set ANTHROPIC_API_KEY to run this example")
		return
	}
	driver, err := anthropic.NewAnthropicDriver(anthropic.AnthropicOptions{APIKey: apiKey})
	if err != nil {
		log.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []common.PromptSegment{{
		Role:    common.PromptRoleUser,
		Content: "Write one sentence about Claude tool-use ergonomics.",
	}}, common.ExecutionOptions{Model: "claude-3-5-sonnet-20241022", ModelOptions: map[string]any{"max_tokens": 128}})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Result[0].Value)
}
