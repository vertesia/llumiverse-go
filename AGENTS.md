# Agent guide

Guidance for AI coding agents (and humans) working in this repository. Keep this
file current when conventions change.

## What this is

`llumiverse-go` is a provider-neutral Go client for LLM completions, streaming,
model listing, image generation, and embeddings across OpenAI, Anthropic Claude,
AWS Bedrock, and the Gemini Enterprise Agent Platform (formerly Vertex AI). It is
a Go port of the TypeScript [`vertesia/llumiverse`](https://github.com/vertesia/llumiverse)
and intentionally keeps that project's public concepts: callers build
`PromptSegment`s, pick a `Driver`, pass `ExecutionOptions`, and get back a
normalized `Completion`. Training / fine-tuning APIs are out of scope.

## Commands

```sh
go build ./...        # must pass
go vet ./...          # must pass
go test ./...         # unit tests; no network required (mocks/fakes)
go test -cover ./...
gofmt -l .            # MUST be empty — CI fails on any unformatted file
go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest run
```

Run `gofmt -w` on every file you touch before finishing — the `test.yaml`
workflow enforces a clean `gofmt -l .`.

## Layout

```text
.                       root package "llumiverse": re-exports types + constructors (compat)
common/                 shared public types, options, errors (package common)
core/                   driver-contract helpers: lifecycle, streams, HTTP/SSE/JSON,
                        option accessors, conversation-retention, version parsing
drivers/openai          OpenAI + OpenAI-compatible driver (Responses API)
drivers/anthropic       thin shell over the shared Claude implementation
drivers/bedrock         AWS Bedrock driver (Converse / ConverseStream / InvokeModel)
drivers/vertexai        Gemini Enterprise Agent Platform driver (Gemini/Imagen/Claude)
drivers/internal/claude shared Claude prompt/stream/conversation logic
examples/               runnable per-provider examples
```

Most packages carry a `doc.go` package comment, an `aliases.go` that re-exports
`common`/`core` names, and (where they own retention) a `conversation.go`. The
`anthropic` package is a thin shell that delegates to `drivers/internal/claude`.

## Conventions

- **Normalize at the edges, isolate quirks in drivers.** Public shapes
  (`PromptSegment`, `Completion`, `CompletionStream`, `ExecutionOptions`,
  embeddings) stay provider-neutral. Provider-specific behavior (Claude thinking,
  Bedrock Converse blocks, OpenAI Responses items, Gemini function responses)
  lives inside the driver and never leaks into the common types.
- **Provider knobs go in `ExecutionOptions.ModelOptions` (a `map[string]any`)** —
  e.g. `temperature`, `max_tokens`, `top_p`, `effort`, `thinking_budget_tokens`,
  `cache_enabled`. Do **not** add per-knob struct fields to `ExecutionOptions`.
  Read them with the `optionInt`/`optionFloat`/`optionBool`/`optionString` helpers
  (exported from `core` as `OptionInt`, etc.).
- **Shared Claude logic lives in `drivers/internal/claude`.** The `anthropic`
  driver and the Claude-on-Vertex path both consume it via its `export.go`
  facade (`FormatPrompt`, `Payload`, `ExtractCompletion`, `SSEToChunk`, …). Don't
  duplicate Claude prompt/stream handling.
- **Conversation state is provider-shaped and round-tripped.** Drivers return it
  in `Completion.Conversation`; callers pass it back via
  `ExecutionOptions.Conversation`. Apply retention through the shared helpers
  (`StripImagesAfterTurns`, `StripTextMaxTokens`, `StripHeartbeatsAfterTurns`);
  expired media is replaced with text placeholders so stored history stays a
  valid provider request.
- **Errors** are wrapped as `*common.LlumiverseError` with provider/model/
  operation context and a **tri-state `Retryable`** (`true`/`false`/`nil` when
  unknown). Preserve that classification; don't collapse `nil` to `false`.
- **Library code never reads environment variables.** Constructors take explicit
  options/credentials. Only tests load `.env` (see Testing).
- **Comment style.** Exported symbols get a doc comment beginning with the
  symbol name. Non-obvious internal logic gets a short "what/why" comment;
  parity decisions are flagged with references to "the TS client". Don't add
  noise to trivial helpers or to the `aliases.go` re-export lists.
- **Root re-exports are for compatibility.** New code may import `common`,
  `core`, and the `drivers/*` packages directly; keep the root `llumiverse`
  package's `aliases.go` in sync if you add public types or constructors.

## Testing

- Unit tests must pass offline. Use fakes/mocks for provider clients (e.g.
  `NewBedrockDriverWithClient`, injected `*http.Client`, static `oauth2.TokenSource`).
- Live-credential, integration-style tests are opt-in: `TestMain` loads a `.env`
  via `github.com/joho/godotenv`, and tests skip when the relevant key is unset.
  See `.env.example` for variable names. Never make `go test ./...` require
  network or secrets.

## Adding a new driver

1. Create `drivers/<name>/` with a `doc.go`, the driver type implementing
   `common.Driver`, and an `aliases.go` re-exporting the `common`/`core` names it
   uses.
2. Implement `CreatePrompt`/`Execute`/`Stream` on top of `core.ExecuteWithPrompt`
   / `core.StreamWithPrompt` and the channel-stream helpers, so error wrapping
   and JSON-result normalization stay shared.
3. Map provider errors through a `formatError` that produces a
   `*common.LlumiverseError` with a sensible `Retryable`.
4. Put conversation retention in `conversation.go`; reuse `core` placeholders.
5. Add a runnable `examples/<name>-*/main.go` and unit tests with fakes.
6. If it adds a public constructor/type, re-export it from the root `aliases.go`.

## CI / .github

Workflows live in `.github/workflows/` (`test`, `lint`, `zizmor`, `release`,
`stale`) with `dependabot.yml`. **All action `uses:` are pinned to commit SHAs**
with a trailing `# vX.Y.Z` comment — keep them pinned (Dependabot bumps them;
zizmor audits the workflows). `release.yaml` is a manual `workflow_dispatch` that
validates a `vX.Y.Z` tag and cuts a GitHub Release.
