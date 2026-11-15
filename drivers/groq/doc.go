// Package groq implements the llumiverse driver for Groq, on its OpenAI-compatible Chat Completions API
// (https://api.groq.com/openai/v1) through the shared drivers/openai protocol.
//
// Result schemas go in the prompt; gpt-oss and Qwen 3.8 and later also get Groq Structured Outputs
// (response_format json_schema) when the request has no tools. ModelOptions "reasoning_format" sets Groq's
// reasoning_format, and "effort" its reasoning_effort. Groq serves no embedding models.
package groq
