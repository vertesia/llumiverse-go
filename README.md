# llumiverse-go

A universal, lightweight LLM driver for Go

llumiverse-go abstracts the connection and execution protocols of each provider
without imposing prompt templating or orchestration. You stay in control of your
application architecture.

## Why llumiverse-go?

- **No vendor lock-in.** Every provider implements the same `Driver` interface,
  so switching models or providers is a one-line change.
- **Normalized I/O.** One prompt shape, one completion shape, one streaming and
  embeddings API across all providers — provider quirks (Claude thinking,
  Bedrock Converse blocks, OpenAI Responses items, Gemini function responses)
  stay inside the drivers.
- **Lightweight & idiomatic.** Standard library `net/http` and official provider
  SDKs only; no heavyweight framework, `context.Context` throughout.
- **Stateful conversations.** Drivers return provider-shaped conversation state
  you can pass straight back to continue a multi-turn exchange, with built-in
  retention policies for large media.

## Supported platforms

| Provider | Completion | Streaming | Multimodal in | Tool calling | Structured output | Embeddings | Image generation |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| OpenAI (Responses API) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (dall-e / gpt-image) |
| OpenAI-compatible (Responses API) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Anthropic Claude (Messages API) | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| AWS Bedrock (Converse / InvokeModel) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (Nova / Titan / Cohere / Marengo) | ✅ (Titan / Nova Canvas / Stable Diffusion) |
| Gemini Enterprise Agent Platform¹ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (Imagen / Gemini) |

¹ Formerly Vertex AI. The package keeps the `vertexai` provider and package
names for compatibility, and also serves Claude partner models on the platform.

## Requirements

- Go 1.25 or newer.
- Provider credentials for whichever drivers you use (see
  [`.env.example`](.env.example)).

## Installation

```sh
go get github.com/vertesia/llumiverse-go
```

The root `llumiverse` package re-exports the shared types, constants, and driver
constructors, so a single import is enough to get started. New code can also
import the organized subpackages directly: `common` (types), `core` (driver
helpers), and `drivers/openai`, `drivers/anthropic`, `drivers/bedrock`,
`drivers/vertexai`.

## Quick start

### 1. Initialize a driver

```go
import "github.com/vertesia/llumiverse-go"

// OpenAI (first-party; uses the Responses API)
openaiDriver, _ := llumiverse.NewOpenAIDriver(llumiverse.OpenAICompatibleOptions{
    APIKey:   os.Getenv("OPENAI_API_KEY"),
    Endpoint: "https://api.openai.com/v1",
})

// Any OpenAI-compatible endpoint that exposes the Responses API
compatDriver, _ := llumiverse.NewOpenAICompatibleDriver(llumiverse.OpenAICompatibleOptions{
    APIKey:   os.Getenv("OPENROUTER_API_KEY"),
    Endpoint: "https://openrouter.ai/api/v1",
})

// Anthropic Claude (Messages API)
claudeDriver, _ := llumiverse.NewAnthropicDriver(llumiverse.AnthropicOptions{
    APIKey: os.Getenv("ANTHROPIC_API_KEY"),
})

// AWS Bedrock (uses the standard AWS credential chain / IAM role)
bedrockDriver, _ := llumiverse.NewBedrockDriver(context.Background(), llumiverse.BedrockOptions{
    Region: "us-east-1",
})

// Gemini Enterprise Agent Platform (uses Application Default Credentials)
vertexDriver, _ := llumiverse.NewVertexAIDriver(llumiverse.VertexAIOptions{
    Project: os.Getenv("GOOGLE_PROJECT_ID"),
    Region:  "us-central1",
})
```

### 2. Execute a prompt

The same prompt and options work on every driver — only the model name changes.

```go
prompt := []llumiverse.PromptSegment{
    {Role: llumiverse.PromptRoleSystem, Content: "You are a concise assistant."},
    {Role: llumiverse.PromptRoleUser, Content: "Summarize the release notes."},
}

resp, err := openaiDriver.Execute(context.Background(), prompt, llumiverse.ExecutionOptions{
    Model: "gpt-4.1-mini",
    // Provider sampling/runtime knobs go in ModelOptions:
    ModelOptions: map[string]any{
        "temperature": 0.7,
        "max_tokens":  1024,
    },
})
if err != nil {
    log.Fatal(err)
}

fmt.Println(resp.Result[0].Value)
fmt.Printf("tokens: prompt=%d result=%d\n", resp.TokenUsage.Prompt, resp.TokenUsage.Result)
```

`resp.Result` is a slice of normalized results (`ResultTypeText`,
`ResultTypeJSON`, or `ResultTypeImage`).

### 3. Stream a response

Streams are a pull-based iterator. `Recv` returns `io.EOF` when finished, and the
fully-assembled response is available afterward via `Completion()`.

```go
stream, err := openaiDriver.Stream(context.Background(), prompt, llumiverse.ExecutionOptions{
    Model: "gpt-4.1-mini",
})
if err != nil {
    log.Fatal(err)
}
defer stream.Close()

for {
    chunk, err := stream.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    for _, r := range chunk.Result {
        if r.Type == llumiverse.ResultTypeText {
            fmt.Print(r.Value)
        }
    }
}

final := stream.Completion() // *ExecutionResponse with merged text, tools, usage
```

### 4. Generate embeddings

Inputs and outputs are normalized; media models that return segmented vectors
keep them grouped per input.

```go
res, err := openaiDriver.GenerateEmbeddings(context.Background(), llumiverse.EmbeddingsOptions{
    Model:  "text-embedding-3-small",
    Inputs: []llumiverse.EmbeddingInput{{Text: "policy renewal"}},
})
if err != nil {
    log.Fatal(err)
}

vector := res.Results[0].Outputs[0].Values // []float64
fmt.Printf("dims=%d\n", len(vector))
```

### 5. List available models

```go
models, err := claudeDriver.ListModels(context.Background(), nil)
for _, m := range models {
    fmt.Printf("%s (%s) tools=%v\n", m.ID, m.Provider, m.ToolSupport)
}
```

## Advanced features

### Tool calling

Define tools with a JSON-schema input, inspect any returned `ToolUse`, run the
tool yourself, then feed the result back as a `PromptRoleTool` segment to
continue.

```go
opts := llumiverse.ExecutionOptions{
    Model: "gpt-4.1-mini",
    Tools: []llumiverse.ToolDefinition{{
        Name:        "lookup_invoice",
        Description: "Look up an invoice by ID.",
        InputSchema: map[string]any{
            "type":       "object",
            "properties": map[string]any{"id": map[string]any{"type": "string"}},
            "required":   []string{"id"},
        },
    }},
}

resp, _ := openaiDriver.Execute(ctx, prompt, opts)
for _, call := range resp.ToolUse {
    output := runTool(call.ToolName, call.ToolInput) // your code

    prompt = append(prompt, llumiverse.PromptSegment{
        Role:      llumiverse.PromptRoleTool,
        ToolUseID: call.ID,
        Content:   output,
    })
}
// Execute again with the appended tool result to get the final answer.
```

### Structured outputs

Provide a JSON schema and the driver requests structured output natively
(OpenAI/Gemini structured output, schema-in-prompt for others) and parses the
result into a `ResultTypeJSON` value.

```go
resp, _ := vertexDriver.Execute(ctx, prompt, llumiverse.ExecutionOptions{
    Model: "gemini-2.5-flash",
    ResultSchema: map[string]any{
        "type": "object",
        "properties": map[string]any{
            "parties": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
        },
        "required": []string{"parties"},
    },
})

if resp.Result[0].Type == llumiverse.ResultTypeJSON {
    data := resp.Result[0].Value.(map[string]any)
    _ = data["parties"]
}
```

### Multi-turn conversations and retention

Pass `resp.Conversation` back through `ExecutionOptions.Conversation` to continue
where you left off — the driver preserves the provider-native conversation shape.
Retention options trim stored history while keeping it valid for the next request:

- `StripImagesAfterTurns` — replace media blocks with text placeholders after N turns.
- `StripTextMaxTokens` — truncate long retained text (≈4 chars/token).
- `StripHeartbeatsAfterTurns` — drop `<heartbeat>…</heartbeat>` status blocks.

```go
first, _ := claudeDriver.Execute(ctx, prompt, llumiverse.ExecutionOptions{Model: model})

second, _ := claudeDriver.Execute(ctx, followUp, llumiverse.ExecutionOptions{
    Model:                 model,
    Conversation:          first.Conversation,
    StripImagesAfterTurns: llumiverse.BoolPtr(true),
})
```

### Multimodal inputs

Attach files to any segment via `DataSource`. `BytesDataSource` carries inline
bytes; provider-native URLs (`gs://`, `s3://`, signed HTTPS) are passed through
directly when the API can consume them.

```go
prompt := []llumiverse.PromptSegment{{
    Role:    llumiverse.PromptRoleUser,
    Content: "Describe this image.",
    Files: []llumiverse.DataSource{llumiverse.BytesDataSource{
        FileName: "chart.png", MIME: "image/png", Data: pngBytes,
    }},
}}
```

### Error handling

Provider errors are wrapped in `*LlumiverseError`, which carries the provider,
model, operation, original error, and a tri-state `Retryable` (`true`/`false`/
`nil` when unknown).

```go
if _, err := driver.Execute(ctx, prompt, opts); err != nil {
    var lerr *llumiverse.LlumiverseError
    if errors.As(err, &lerr) && lerr.Retryable != nil && *lerr.Retryable {
        // back off and retry
    }
}
```

## Package layout

```text
common/                 shared public types, options, and errors
core/                   Driver helpers, streams, prompts, and conversations
drivers/openai          OpenAI and OpenAI-compatible driver
drivers/anthropic       direct Claude driver
drivers/bedrock         AWS Bedrock driver
drivers/vertexai        Gemini Enterprise Agent Platform driver
drivers/internal/claude shared Claude prompt/stream/conversation logic
examples/               runnable usage examples
```

## Examples

Runnable programs live under [`examples/`](examples). Set the relevant
credentials (a `.env` file is loaded automatically) and run, e.g.:

```sh
go run ./examples/openai-chat
go run ./examples/claude-chat
go run ./examples/vertexai-gemini
go run ./examples/bedrock-embeddings
```

## Tests

Unit tests use fake provider clients where possible. Integration-style tests can
load credentials from a `.env` file through `github.com/joho/godotenv`; see
[`.env.example`](.env.example) for supported environment variables.

The GitHub Actions test workflow includes a `live-test` job that uses the
`build` environment and runs only on trusted push/manual dispatch events.
Configure these environment variables for cloud identity and region selection:

- `GOOGLE_PROJECT_ID`
- `GOOGLE_REGION`
- `GOOGLE_WORKLOAD_IDENTITY_PROVIDER`
- `AWS_ROLE_TO_ASSUME`
- `AWS_REGION`
- `BEDROCK_REGION`
- `AWS_ROLE_AUDIENCE` (optional; defaults to `sts.amazonaws.com`)

Configure these environment secrets for direct provider APIs:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

```sh
go test ./...
go test -cover ./...
go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest run
```

## License

Apache 2.0, matching the upstream `vertesia/llumiverse` project.
