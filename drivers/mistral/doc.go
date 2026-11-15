// Package mistral implements the llumiverse driver for Mistral AI, on its native Chat Completions API
// (https://api.mistral.ai/v1).
//
// Result schemas go in the prompt as a system instruction. Reasoning models answer with thinking content chunks,
// which become thoughts results unless ModelOptions "include_thoughts" is false, and stay in the conversation
// with their signatures. ModelOptions random_seed, safe_prompt, parallel_tool_calls, tool_choice, prompt_mode
// and effort (reasoning_effort) are passed through, and ExecutionOptions.PromptCacheKey as prompt_cache_key.
//
// The returned conversation holds Mistral messages in wire shape. Conversations stored by the TS client (with
// camelCase field names) and Chat Completions conversations (_is_openai_chat_completions) are read too.
package mistral
