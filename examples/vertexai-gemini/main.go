package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"
	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/drivers/vertexai"
)

func main() {
	_ = godotenv.Load()

	project := os.Getenv("GOOGLE_PROJECT_ID")
	if project == "" {
		fmt.Println("set GOOGLE_PROJECT_ID to run this example")
		return
	}
	region := os.Getenv("GOOGLE_REGION")
	if region == "" {
		region = "us-central1"
	}
	driver, err := vertexai.NewVertexAIDriver(vertexai.VertexAIOptions{Project: project, Region: region})
	if err != nil {
		log.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []common.PromptSegment{{
		Role:    common.PromptRoleUser,
		Content: "Write one sentence about Gemini Enterprise Agent Platform integrations.",
	}}, common.ExecutionOptions{Model: "gemini-2.5-flash"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Result[0].Value)
}
