// Command azurefoundry-chat sends one prompt to a Microsoft Foundry deployment.
//
// Authenticate with a resource key, or with a Microsoft Entra ID access token,
// for example one from the Azure CLI:
//
//	az account get-access-token --resource https://ai.azure.com --query accessToken -o tsv
//
// Usage:
//
//	go run ./examples/azurefoundry-chat -endpoint https://<resource>.services.ai.azure.com/api/projects/<project> \
//		-model <deployment>[::<source-model>] [-api-key <key> | -token <entra-token>] [-stream]
//
// The flags default to AZURE_FOUNDRY_ENDPOINT, AZURE_FOUNDRY_MODEL,
// AZURE_FOUNDRY_API_KEY and AZURE_FOUNDRY_TOKEN.
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
	"github.com/vertesia/llumiverse-go/drivers/azurefoundry"
)

func main() {
	_ = godotenv.Load()

	endpoint := flag.String("endpoint", os.Getenv("AZURE_FOUNDRY_ENDPOINT"), "Foundry project endpoint")
	model := flag.String("model", os.Getenv("AZURE_FOUNDRY_MODEL"), "deployment name, optionally deployment::source-model")
	apiKey := flag.String("api-key", os.Getenv("AZURE_FOUNDRY_API_KEY"), "Foundry resource key")
	token := flag.String("token", os.Getenv("AZURE_FOUNDRY_TOKEN"), "Microsoft Entra ID access token")
	prompt := flag.String("prompt", "Write one sentence about Microsoft Foundry.", "prompt to send")
	stream := flag.Bool("stream", false, "stream the answer")
	list := flag.Bool("list", false, "list the project's deployments (needs a token)")
	flag.Parse()

	if *endpoint == "" || (*apiKey == "" && *token == "") || (*model == "" && !*list) {
		fmt.Println("set -endpoint, -model and -api-key or -token (or the AZURE_FOUNDRY_* variables) to run this example")
		return
	}
	options := azurefoundry.AzureFoundryOptions{Endpoint: *endpoint, APIKey: *apiKey}
	if *token != "" {
		options.TokenProvider = azurefoundry.StaticToken(*token)
	}
	driver, err := azurefoundry.NewAzureFoundryDriver(options)
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	if *list {
		models, err := driver.ListModels(ctx, nil)
		if err != nil {
			log.Fatal(err)
		}
		for _, m := range models {
			fmt.Println(m.ID)
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
