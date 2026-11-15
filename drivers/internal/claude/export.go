package claude

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/vertesia/llumiverse-go/common"
)

// These exports let the drivers that serve Claude (anthropic, vertexai, bedrock) share one implementation of the
// Messages API prompt, conversation, payload, response and stream handling; only transport and auth differ.

// Prompt is the Claude Messages prompt and stored conversation: system blocks, an ordered list of turns, and the
// conversation metadata (turn number, declared tools) stored as _llumiverse_meta.
type Prompt = claudePrompt

// Message is one Claude turn holding content blocks.
type Message = claudeMessage

// Block is one Claude content block; which fields are set depends on its Type.
type Block = claudeBlock

// Source is the inline data of an image or document block.
type Source = claudeSource

// Response is a decoded Claude Messages API response.
type Response = claudeResponse

// FormatPrompt converts prompt segments into a Claude prompt, with the result schema as prompt instructions.
func FormatPrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (Prompt, error) {
	return formatClaudePrompt(ctx, segments, options, false)
}

// FormatPromptWithNativeSchema converts prompt segments into a Claude prompt. With nativeResultSchema the result
// schema is left out of the prompt, because the provider enforces it as output_config.format (see
// UsesNativeResultSchema).
func FormatPromptWithNativeSchema(ctx context.Context, segments []PromptSegment, options ExecutionOptions, nativeResultSchema bool) (Prompt, error) {
	return formatClaudePrompt(ctx, segments, options, nativeResultSchema)
}

// FormatDebugPrompt returns a copy of a Claude prompt whose base64 image and document payloads are truncated, for
// ExecutionResponse.Prompt. The prompt itself is not modified.
func FormatDebugPrompt(prompt Prompt) Prompt {
	return formatClaudeDebugPrompt(prompt)
}

// RequestSettings are the provider settings of a Claude request.
type RequestSettings struct {
	// Provider is ProviderNameAnthropic, ProviderNameVertexAI or ProviderNameBedrock.
	Provider string
	// Operation names the operation in errors: execute or stream.
	Operation string
	// AllowNativeResultSchema lets a qualifying request on Bedrock or Vertex AI send its result schema as
	// output_config.format. False keeps the prompt path, for instance after a policy denial.
	AllowNativeResultSchema bool
}

// Request is a prepared Claude request: the conversation it continues (this turn's prompt appended and its tools
// recorded) and the Messages API body, without the model, stream or provider version fields.
type Request struct {
	// Conversation is the conversation the response is appended to.
	Conversation Prompt
	// Body is the Messages API body.
	Body map[string]any
	// Betas are the beta values the request needs.
	Betas []string
	// ServiceTier is the processing tier the transport reported, for a provider that returns it outside the
	// message usage (Bedrock sends it in a response header). It applies when the usage names no tier.
	ServiceTier string

	options  ExecutionOptions
	settings RequestSettings
}

// NewRequest prepares a Claude request for a prompt.
func NewRequest(prompt Prompt, options ExecutionOptions, settings RequestSettings) (*Request, error) {
	conversation := prepareClaudeConversation(options, prompt, settings.Provider)
	body, betas, err := buildClaudePayload(options, conversation, claudePayloadSettings(settings))
	if err != nil {
		return nil, err
	}
	return &Request{Conversation: conversation, Body: body, Betas: betas, options: options, settings: settings}, nil
}

// BetaHeader is the anthropic-beta header value: one comma-separated header, because Google Cloud reads only the
// first anthropic-beta header. It is empty when the request needs no beta.
func (r *Request) BetaHeader() string {
	return strings.Join(r.Betas, ",")
}

// Complete builds the completion of a response and the conversation to store.
func (r *Request) Complete(response Response) *Completion {
	if response.Usage.ServiceTier == "" {
		response.Usage.ServiceTier = r.ServiceTier
	}
	completion := extractClaudeCompletion(response, r.options)
	completion.Conversation = finalizeClaudeConversation(r.Conversation, claudeMessage{Role: response.Role, Content: response.Content}, r.options)
	return completion
}

// EventSource delivers the JSON payload of each Messages API stream event to yield, in order, and returns the error
// that ended the stream (nil or io.EOF at its end). It stops when yield returns an error and returns that error.
type EventSource func(yield func(data []byte) error) error

// SSEEvents reads the stream events of a Messages API SSE response.
func SSEEvents(body io.Reader) EventSource {
	return func(yield func([]byte) error) error {
		return scanSSE(io.NopCloser(body), func(event sseEvent) error {
			if event.Data == "" {
				return nil
			}
			return yield([]byte(event.Data))
		})
	}
}

// Stream returns the completion stream of a request whose response events come from source. formatError formats
// transport errors; error events in the stream, and failures after the stream started, are reported as retryable
// when the provider says so. The finalizer stores the rebuilt message in the conversation, or, when the stream
// ended early, a conversation rebuilt from the streamed text and tool calls.
func (r *Request) Stream(prompt any, source EventSource, closeFn func() error, formatError func(error) error) CompletionStream {
	ch := make(chan streamItem)
	accumulator := newClaudeStreamAccumulator(r.options)
	completion := &ExecutionResponse{Prompt: prompt}
	go func() {
		defer close(ch)
		err := source(func(data []byte) error {
			chunk, err := accumulator.event(data)
			if err != nil {
				return err
			}
			if chunk != nil && (len(chunk.Result) > 0 || len(chunk.ToolUse) > 0 || chunk.FinishReason != "" || chunk.TokenUsage != nil) {
				ch <- streamItem{Chunk: *chunk}
			}
			return nil
		})
		if err != nil && !errors.Is(err, io.EOF) {
			ch <- streamItem{Err: r.streamError(err, accumulator.started, formatError)}
		}
	}()
	return newChannelStreamWithFinalizer(ch, completion, closeFn, func(response *ExecutionResponse) {
		if accumulator.started {
			response.ServiceTier = claudeServiceTier(accumulator.message.Usage)
		}
		if response.ServiceTier == "" {
			response.ServiceTier = r.ServiceTier
		}
		if !accumulator.complete() {
			response.Conversation = buildClaudeStreamingConversation(r.Conversation, &response.Completion, r.options)
			return
		}
		final := accumulator.final()
		// A token limit or refusal can stop the response inside its last tool call: leave that call out, as
		// Execute does; the conversation keeps it.
		if n := len(final.Content); n > 0 && final.Content[n-1].Type == "tool_use" && claudeCutOffStopReasons[final.StopReason] {
			cut := final.Content[n-1].ID
			tools := response.ToolUse[:0:0]
			for _, tool := range response.ToolUse {
				if tool.ID != cut {
					tools = append(tools, tool)
				}
			}
			response.ToolUse = tools
		}
		if r.options.IncludeOriginalResponse {
			response.OriginalResponse = final
		}
		response.Conversation = finalizeClaudeConversation(r.Conversation, claudeMessage{Role: final.Role, Content: final.Content}, r.options)
	})
}

// streamError formats an error that ended a stream.
func (r *Request) streamError(err error, started bool, formatError func(error) error) error {
	errContext := LlumiverseErrorContext{Provider: Provider(r.settings.Provider), Model: r.options.Model, Operation: "stream"}
	var streamErr *claudeStreamError
	if errors.As(err, &streamErr) {
		return newLlumiverseError(fmt.Sprintf("[%s] %s", r.settings.Provider, streamErr.Error()), streamErr.retryable(), errContext, err, 0, "ClaudeStreamError")
	}
	// An error the event source already classified, such as a Bedrock exception raised mid-stream, keeps the
	// provider's retry rules.
	var classified *common.LlumiverseError
	if errors.As(err, &classified) {
		return err
	}
	if code, _ := errorStatusAndName(err); code == 0 && started && !errors.Is(err, context.Canceled) {
		// A read failure after the first event: the request reached the provider, so it can be retried.
		return newLlumiverseError(fmt.Sprintf("[%s] %s", r.settings.Provider, err.Error()), boolPtr(true), errContext, err, 0, "ClaudeStreamError")
	}
	return formatError(err)
}
