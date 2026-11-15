package bedrockmantle

import (
	"context"
	"fmt"
	"net/http"

	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
)

// claudeAPIVersion is the anthropic-version header of the Messages API on Mantle, the version the TS client pins
// for the first-party Messages API.
const claudeAPIVersion = "2023-06-01"

// claudeRequest prepares a Mantle Claude request: the shared Claude request named for Bedrock Mantle, which takes
// mid-conversation tool changes but not native result schemas (the schema stays in the prompt), its body, and the
// anthropic-version and anthropic-beta headers.
func (d *BedrockMantleDriver) claudeRequest(prompt any, options ExecutionOptions, operation string) (string, *internalclaude.Request, map[string]any, map[string]string, error) {
	claude, ok := prompt.(internalclaude.Prompt)
	if !ok {
		return "", nil, nil, nil, fmt.Errorf("bedrock mantle claude models require a Claude Messages prompt, got %T", prompt)
	}
	endpoint, err := joinEndpoint(d.anthropic, "v1", "messages")
	if err != nil {
		return "", nil, nil, nil, err
	}
	request, err := internalclaude.NewRequest(claude, options, internalclaude.RequestSettings{
		Provider:  internalclaude.ProviderNameBedrockMantle,
		Operation: operation,
	})
	if err != nil {
		return "", nil, nil, nil, err
	}
	headers := map[string]string{"anthropic-version": claudeAPIVersion}
	if beta := request.BetaHeader(); beta != "" {
		headers["anthropic-beta"] = beta
	}
	body := make(map[string]any, len(request.Body)+2)
	for key, value := range request.Body {
		body[key] = value
	}
	body["model"] = options.Model
	if operation == "stream" {
		body["stream"] = true
	}
	return endpoint, request, body, headers, nil
}

// requestClaude runs a non-streaming Claude Messages request.
func (d *BedrockMantleDriver) requestClaude(ctx context.Context, prompt any, options ExecutionOptions) (*Completion, error) {
	endpoint, request, body, headers, err := d.claudeRequest(prompt, options, "execute")
	if err != nil {
		return nil, err
	}
	var response internalclaude.Response
	if err := doJSON(ctx, d.messagesClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, body, &response); err != nil {
		return nil, d.formatError(err, options.Model, "execute", "AnthropicError")
	}
	return request.Complete(response), nil
}

// streamClaude runs a streaming Claude Messages request.
func (d *BedrockMantleDriver) streamClaude(ctx context.Context, prompt any, options ExecutionOptions) (CompletionStream, error) {
	endpoint, request, body, headers, err := d.claudeRequest(prompt, options, "stream")
	if err != nil {
		return nil, err
	}
	sse, err := postSSE(ctx, d.messagesClient(options.HTTPTimeout), endpoint, headers, body)
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream", "AnthropicError")
	}
	return request.Stream(prompt, internalclaude.SSEEvents(sse), sse.Close, func(err error) error {
		return d.formatError(err, options.Model, "stream", "AnthropicError")
	}), nil
}

// messagesClient returns the Messages API client for one request, applying per-request timeouts.
func (d *BedrockMantleDriver) messagesClient(timeout HTTPTimeoutOptions) *http.Client {
	if !hasHTTPTimeout(timeout) {
		return d.messages
	}
	return newHTTPClient(d.messages, timeout)
}
