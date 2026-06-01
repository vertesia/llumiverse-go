# Client Libraries

This Go rewrite targets the first driver slice requested for `llumiverse`:

- OpenAI API: `github.com/openai/openai-go` for Responses, Images, Models, Embeddings, and streaming.
- OpenAI-compatible APIs: `github.com/openai/openai-go` Responses, Models, Embeddings, and streaming, with `option.WithBaseURL` for custom compatible endpoints.
- Claude / Anthropic API: `net/http` against the Anthropic Messages API.
- Gemini Enterprise Agent Platform (formerly Vertex AI): `golang.org/x/oauth2/google` for Application Default Credentials and `net/http` against Google Cloud AI Platform REST endpoints for Gemini and Claude partner models. The Go driver keeps the `vertexai` provider ID for compatibility.
- Bedrock: `github.com/aws/aws-sdk-go-v2/service/bedrockruntime` for `Converse`, `ConverseStream`, and `InvokeModel`; `github.com/aws/aws-sdk-go-v2/service/bedrock` for model listing.
- Test dotenv loading: `github.com/joho/godotenv`.
- Formatting and linting: `gofmt` and `golangci-lint`.
