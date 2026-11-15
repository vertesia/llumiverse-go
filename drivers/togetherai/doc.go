// Package togetherai implements the llumiverse driver for Together AI, on its Chat Completions API
// (https://api.together.ai/v1) through the shared drivers/openai protocol.
//
// Result schemas go in the prompt. Tools are sent non-strict, assistant and tool content as text, and streamed
// usage is requested. On the hybrid Qwen 3.5-3.7 models an "effort" of "none" switches reasoning off with
// reasoning.enabled. Models are listed from the serverless catalogue (/v2/supported-models); embeddings use
// /v1/embeddings.
package togetherai
