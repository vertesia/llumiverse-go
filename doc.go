// Package llumiverse provides a provider-neutral Go client for model
// completions, streaming completions, model listing, image generation, and
// embeddings across OpenAI, Anthropic Claude, AWS Bedrock, and Gemini Enterprise
// Agent Platform (formerly Vertex AI).
//
// The package keeps the public concepts from the TypeScript llumiverse client:
// callers build prompt segments, choose a driver, pass ExecutionOptions, and
// receive normalized completion results, tool calls, token usage, and retained
// conversation state. Provider-specific details such as Claude thinking,
// Bedrock Converse content blocks, OpenAI Responses tool items, and Gemini
// function responses stay inside the drivers. New code may import the organized
// subpackages directly: common for shared types, core for driver helpers, and
// drivers/openai, drivers/anthropic, drivers/bedrock, or drivers/vertexai for
// constructors.
//
// Drivers preserve provider-shaped conversation state in Completion.Conversation.
// Pass that value back through ExecutionOptions.Conversation to continue a
// conversation. Retention options such as StripImagesAfterTurns,
// StripTextMaxTokens, and StripHeartbeatsAfterTurns mirror the TypeScript client
// and replace expired large media with text placeholders so stored history
// remains valid for later provider requests.
//
// Tests can load API keys and provider settings from a dotenv file through the
// repository test harness. Library constructors do not read environment
// variables directly; production callers should pass credentials through the
// provider-specific options.
package llumiverse
