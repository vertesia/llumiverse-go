package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"
	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/drivers/bedrock"
)

func main() {
	_ = godotenv.Load()

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = "us-east-1"
	}
	driver, err := bedrock.NewBedrockDriver(context.Background(), bedrock.BedrockOptions{Region: region})
	if err != nil {
		log.Fatal(err)
	}
	result, err := driver.GenerateEmbeddings(context.Background(), common.EmbeddingsOptions{
		Inputs: []common.EmbeddingInput{{Text: "policy renewal"}},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("model=%s vectors=%d\n", result.Model, len(result.Results))
}
