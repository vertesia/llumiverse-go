// Command bedrock-mantle-chat sends one prompt to a model on Amazon Bedrock Mantle.
//
// It authenticates with a Bedrock API key, or with the default AWS credentials
// (environment, shared config, SSO, instance role) when no key is given.
//
// Usage:
//
//	go run ./examples/bedrock-mantle-chat -region us-west-2 -model openai.gpt-oss-120b [-api-key <key>] [-stream] [-list]
//
// The flags default to AWS_REGION, BEDROCK_MANTLE_MODEL and BEDROCK_MANTLE_API_KEY.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/joho/godotenv"
	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/drivers/bedrockmantle"
)

func main() {
	_ = godotenv.Load()

	region := flag.String("region", os.Getenv("AWS_REGION"), "AWS region of the Mantle endpoint")
	model := flag.String("model", os.Getenv("BEDROCK_MANTLE_MODEL"), "Mantle model ID, for example anthropic.claude-opus-4-7")
	apiKey := flag.String("api-key", os.Getenv("BEDROCK_MANTLE_API_KEY"), "Bedrock API key (default: AWS credentials)")
	prompt := flag.String("prompt", "Write one sentence about Amazon Bedrock.", "prompt to send")
	stream := flag.Bool("stream", false, "stream the answer")
	list := flag.Bool("list", false, "list the models the driver can route")
	flag.Parse()

	if *region == "" || (*model == "" && !*list) {
		fmt.Println("set -region and -model (or AWS_REGION and BEDROCK_MANTLE_MODEL) to run this example")
		return
	}
	ctx := context.Background()
	driver, err := bedrockmantle.NewBedrockMantleDriver(ctx, bedrockmantle.BedrockMantleOptions{Region: *region, APIKey: *apiKey})
	if err != nil {
		log.Fatal(err)
	}

	if *list {
		models, err := driver.ListModels(ctx, nil)
		if err != nil {
			log.Fatal(err)
		}
		for _, m := range models {
			fmt.Printf("%s\t%s\n", m.ID, m.Name)
		}
		return
	}

	segments := []common.PromptSegment{{Role: common.PromptRoleUser, Content: *prompt}}
	execution := common.ExecutionOptions{Model: *model, ModelOptions: map[string]any{"max_tokens": 256}}
	if !*stream {
		resp, err := driver.Execute(ctx, segments, execution)
		if err != nil {
			log.Fatal(err)
		}
		for _, result := range resp.Result {
			fmt.Println(result.String())
		}
		return
	}
	s, err := driver.Stream(ctx, segments, execution)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	for {
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		for _, result := range chunk.Result {
			fmt.Print(result.String())
		}
	}
	fmt.Println()
}
