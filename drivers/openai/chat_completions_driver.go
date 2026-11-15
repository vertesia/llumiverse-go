package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
)

// OpenAIChatCompletionsOptions configures the generic Chat Completions driver.
type OpenAIChatCompletionsOptions struct {
	OpenAICompatibleOptions
	ChatCompletionsProtocolOptions
}

// OpenAIChatCompletionsDriver implements Driver for services that implement
// the OpenAI Chat Completions API (POST /chat/completions) but not Responses.
// Model listing, embeddings and connection checks are inherited from
// OpenAICompatibleDriver.
type OpenAIChatCompletionsDriver struct {
	*OpenAICompatibleDriver
	protocol ChatCompletionsProtocol
}

// NewOpenAIChatCompletionsDriver creates a driver for an OpenAI-compatible
// Chat Completions endpoint.
func NewOpenAIChatCompletionsDriver(options OpenAIChatCompletionsOptions) (*OpenAIChatCompletionsDriver, error) {
	base, err := NewOpenAICompatibleDriver(options.OpenAICompatibleOptions)
	if err != nil {
		return nil, err
	}
	protocolOptions := options.ChatCompletionsProtocolOptions
	if protocolOptions.Provider == "" {
		protocolOptions.Provider = base.Provider()
	}
	return &OpenAIChatCompletionsDriver{OpenAICompatibleDriver: base, protocol: ChatCompletionsProtocol{Options: protocolOptions}}, nil
}

// CreatePrompt builds the Chat Completions messages from prompt segments.
func (d *OpenAIChatCompletionsDriver) CreatePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (any, error) {
	return d.protocol.CreatePrompt(ctx, segments, options)
}

// FormatDebugPrompt returns a copy of the prompt with inline payloads truncated.
func (d *OpenAIChatCompletionsDriver) FormatDebugPrompt(prompt any) any {
	return FormatChatCompletionsDebugPrompt(prompt)
}

// Execute runs a single non-streaming completion.
func (d *OpenAIChatCompletionsDriver) Execute(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (*ExecutionResponse, error) {
	return executeWithPrompt(ctx, d, segments, options, func(ctx context.Context, prompt any, options ExecutionOptions) (*Completion, error) {
		return d.protocol.Execute(ctx, chatCompletionsSDKTransport{d.OpenAICompatibleDriver}, prompt, options)
	})
}

// Stream runs a streaming completion, emitting chunks as they arrive.
func (d *OpenAIChatCompletionsDriver) Stream(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (CompletionStream, error) {
	return streamWithPrompt(ctx, d, segments, options, func(ctx context.Context, prompt any, options ExecutionOptions) (CompletionStream, error) {
		return d.protocol.Stream(ctx, chatCompletionsSDKTransport{d.OpenAICompatibleDriver}, prompt, options)
	})
}

// chatCompletionsSDKTransport posts Chat Completions requests with the
// driver's SDK client, flattening extra_body into the body as the SDK does.
type chatCompletionsSDKTransport struct {
	driver *OpenAICompatibleDriver
}

func (t chatCompletionsSDKTransport) PostChatCompletion(ctx context.Context, payload map[string]any, options ExecutionOptions) (json.RawMessage, error) {
	body := sdkChatCompletionsBody(payload)
	var raw json.RawMessage
	if err := t.driver.sdk.Post(ctx, "chat/completions", body, &raw, t.driver.requestOptions(options.HTTPTimeout)...); err != nil {
		return nil, t.driver.formatError(err, options.Model, "execute")
	}
	return raw, nil
}

func (t chatCompletionsSDKTransport) PostChatCompletionStream(ctx context.Context, payload map[string]any, options ExecutionOptions) (io.ReadCloser, error) {
	body := sdkChatCompletionsBody(payload)
	body["stream_options"] = map[string]any{"include_usage": true}
	var res *http.Response
	if err := t.driver.sdk.Post(ctx, "chat/completions", body, &res, t.driver.requestOptions(options.HTTPTimeout)...); err != nil {
		return nil, t.driver.formatError(err, options.Model, "stream")
	}
	return res.Body, nil
}

func (t chatCompletionsSDKTransport) FormatChatCompletionsError(err error, model string, operation string) error {
	return t.driver.formatError(err, model, operation)
}

// sdkChatCompletionsBody flattens extra_body and keeps only the text parts of
// assistant, system, developer and tool messages, which the API accepts only
// as text.
func sdkChatCompletionsBody(payload map[string]any) map[string]any {
	body := MergeChatCompletionsExtraBody(payload)
	messages, _ := body["messages"].([]any)
	out := make([]any, len(messages))
	for i, raw := range messages {
		message, ok := raw.(map[string]any)
		if !ok {
			out[i] = raw
			continue
		}
		if parts, ok := message["content"].([]any); ok && message["role"] != "user" {
			copied := make(map[string]any, len(message))
			for key, value := range message {
				copied[key] = value
			}
			texts := make([]any, 0, len(parts))
			for _, part := range parts {
				if p, ok := part.(map[string]any); ok && p["type"] == "text" {
					texts = append(texts, p)
				}
			}
			copied["content"] = texts
			message = copied
		}
		out[i] = message
	}
	body["messages"] = out
	return body
}
