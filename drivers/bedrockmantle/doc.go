// Package bedrockmantle implements the llumiverse driver for Amazon Bedrock
// Mantle, the Bedrock endpoint (https://bedrock-mantle.<region>.api.aws) that
// serves models on the OpenAI and Anthropic APIs instead of Converse.
//
// Each model family uses the protocol Mantle serves it with (see
// ModelProtocol): OpenAI GPT, Grok and Gemma 4 and later use the Responses API
// on /openai/v1, through the shared drivers/openai implementation; Claude uses
// the Anthropic Messages API on /anthropic, through the shared
// drivers/internal/claude implementation; GPT-OSS, Gemma 3, DeepSeek, Qwen,
// Mistral, Kimi, MiniMax, NVIDIA, Writer and GLM use Chat Completions on /v1.
// Model families without a known protocol are rejected.
//
// Requests authenticate with a Bedrock API key, or with AWS credentials: the
// OpenAI endpoints take a short-lived Bedrock bearer token derived from them
// (as @aws/bedrock-token-generator creates it) for each request, and the
// Messages API takes SigV4 signatures for the bedrock-mantle service.
package bedrockmantle
