// Command provider-chat sends one prompt to Groq, Mistral AI, Together AI, OpenRouter or xAI.
//
// Usage:
//
//	go run ./examples/provider-chat -provider groq -model llama-3.3-70b-versatile [-stream] [-list]
//
// The API key comes from GROQ_API_KEY, MISTRAL_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY or XAI_API_KEY, and
// the model defaults to GROQ_MODEL, MISTRAL_MODEL, TOGETHER_MODEL, OPENROUTER_MODEL or XAI_MODEL.
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
	"github.com/vertesia/llumiverse-go/drivers/groq"
	"github.com/vertesia/llumiverse-go/drivers/mistral"
	"github.com/vertesia/llumiverse-go/drivers/openrouter"
	"github.com/vertesia/llumiverse-go/drivers/togetherai"
	"github.com/vertesia/llumiverse-go/drivers/xai"
)

// providers maps each provider to its key and model variables and its constructor.
var providers = map[string]struct {
	keyVar, modelVar string
	create           func(key string) (common.Driver, error)
}{
	"groq": {"GROQ_API_KEY", "GROQ_MODEL", func(key string) (common.Driver, error) {
		return groq.NewGroqDriver(groq.GroqOptions{APIKey: key})
	}},
	"mistral": {"MISTRAL_API_KEY", "MISTRAL_MODEL", func(key string) (common.Driver, error) {
		return mistral.NewMistralDriver(mistral.MistralOptions{APIKey: key})
	}},
	"together": {"TOGETHER_API_KEY", "TOGETHER_MODEL", func(key string) (common.Driver, error) {
		return togetherai.NewTogetherAIDriver(togetherai.TogetherAIOptions{APIKey: key})
	}},
	"openrouter": {"OPENROUTER_API_KEY", "OPENROUTER_MODEL", func(key string) (common.Driver, error) {
		return openrouter.NewOpenRouterDriver(openrouter.OpenRouterOptions{APIKey: key})
	}},
	"xai": {"XAI_API_KEY", "XAI_MODEL", func(key string) (common.Driver, error) {
		return xai.NewXAIDriver(xai.XAIOptions{APIKey: key})
	}},
}

func main() {
	_ = godotenv.Load()

	name := flag.String("provider", "groq", "groq, mistral, together, openrouter or xai")
	model := flag.String("model", "", "model ID (default: the provider's *_MODEL variable)")
	prompt := flag.String("prompt", "Write one sentence about fast inference.", "prompt to send")
	stream := flag.Bool("stream", false, "stream the answer")
	list := flag.Bool("list", false, "list the provider's models")
	flag.Parse()

	provider, ok := providers[*name]
	if !ok {
		log.Fatalf("unknown provider %q", *name)
	}
	key := os.Getenv(provider.keyVar)
	if *model == "" {
		*model = os.Getenv(provider.modelVar)
	}
	if key == "" || (*model == "" && !*list) {
		fmt.Printf("set %s and -model (or %s) to run this example\n", provider.keyVar, provider.modelVar)
		return
	}
	driver, err := provider.create(key)
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
