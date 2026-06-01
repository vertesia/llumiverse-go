package claude

import (
	"context"
	"encoding/json"
)

// These aliases expose the internal Claude Messages-API representation under
// stable public names so sibling driver packages (anthropic, vertexai) can
// share the same prompt, conversation, and response handling without depending
// on the unexported types directly.

// Prompt is the Claude Messages API request shape: system blocks plus an
// ordered list of user/assistant messages, with a retained-turn counter.
type Prompt = claudePrompt

// Message is a single Claude role-tagged message containing content blocks.
type Message = claudeMessage

// Block is one Claude content block (text, image, document, tool_use, or
// tool_result); which fields are populated depends on its Type.
type Block = claudeBlock

// Source is the inline data source for an image or document block (base64 or
// text), carrying its media type.
type Source = claudeSource

// Response is the decoded Claude Messages API response: content blocks, stop
// reason, and token usage.
type Response = claudeResponse

// FormatPrompt converts provider-neutral prompt segments into the Claude
// Messages request shape, mapping roles to blocks and folding in any result
// schema. Shared by the native Anthropic and Claude-on-Vertex drivers.
func FormatPrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (Prompt, error) {
	return formatClaudePrompt(ctx, segments, options)
}

// Payload builds the Claude Messages API request body and any extra headers
// (max_tokens, thinking, tools, caching, the 128k output beta), returning the
// stream-aware payload ready to send.
func Payload(prompt Prompt, options ExecutionOptions, stream bool, headers map[string]string) (map[string]any, map[string]string, error) {
	return claudePayload(prompt, options, stream, headers)
}

// ConversationInput merges stored conversation state with the new prompt,
// re-sanitizing and merging messages so user/assistant alternation stays valid.
func ConversationInput(conversation any, prompt Prompt) Prompt {
	return claudeConversationInput(conversation, prompt)
}

// AppendResponseToConversation appends a model response as an assistant message
// to the retained conversation.
func AppendResponseToConversation(conversation Prompt, response Response) Prompt {
	return appendClaudeResponseToConversation(conversation, response)
}

// BuildStreamingConversation reconstructs the assistant turn from accumulated
// streaming text and tool-use chunks and appends it to the conversation.
func BuildStreamingConversation(conversation Prompt, completion *Completion) Prompt {
	return buildClaudeStreamingConversation(conversation, completion)
}

// ExtractCompletion normalizes a Claude response into a llumiverse Completion
// (text, tool calls, usage, finish reason, and optional thinking output).
func ExtractCompletion(response Response, options ExecutionOptions) *Completion {
	return extractClaudeCompletion(response, options)
}

// SSEToChunk converts one decoded Claude streaming event envelope into a
// normalized CompletionChunk.
func SSEToChunk(envelope map[string]json.RawMessage, options ExecutionOptions) CompletionChunk {
	return claudeSSEToChunk(envelope, options)
}

// FinalizeConversation applies the conversation-retention policy (media
// stripping, text truncation, heartbeat removal) to the stored conversation.
func FinalizeConversation(conversation Prompt, options ExecutionOptions) Prompt {
	return finalizeClaudeConversation(conversation, options)
}
