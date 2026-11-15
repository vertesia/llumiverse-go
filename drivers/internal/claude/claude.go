package claude

import (
	"context"
	"errors"
	"fmt"
	"net/http"
)

const defaultAnthropicBaseURL = "https://api.anthropic.com"

// anthropicVersion is the pinned Messages API version. Newer versions reject temperature, top_p and top_k and change
// several request rules, so the driver stays on it until it moves deliberately.
const anthropicVersion = "2023-06-01"

// AnthropicOptions configures the native Anthropic Claude Messages API driver.
type AnthropicOptions struct {
	DriverOptions
	// APIKey is sent as the x-api-key header.
	APIKey string
	// BaseURL defaults to the official Anthropic API.
	BaseURL string
	// HTTPClient overrides the default HTTP client.
	HTTPClient *http.Client
}

// AnthropicDriver implements Driver for Anthropic Claude models.
type AnthropicDriver struct {
	options AnthropicOptions
	client  *http.Client
}

// NewAnthropicDriver creates a Claude driver backed by the Anthropic Messages API.
func NewAnthropicDriver(options AnthropicOptions) (*AnthropicDriver, error) {
	if options.APIKey == "" {
		return nil, errors.New("api key is required")
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultAnthropicBaseURL
	}
	client := options.HTTPClient
	if client == nil {
		client = newHTTPClient(nil, options.HTTPTimeout)
	} else if hasHTTPTimeout(options.HTTPTimeout) {
		client = newHTTPClient(client, options.HTTPTimeout)
	}
	return &AnthropicDriver{options: options, client: client}, nil
}

// Provider returns ProviderAnthropic.
func (d *AnthropicDriver) Provider() Provider {
	return ProviderAnthropic
}

// CreatePrompt builds the Claude Messages prompt from the given prompt segments.
func (d *AnthropicDriver) CreatePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (any, error) {
	return formatClaudePrompt(ctx, segments, options, false)
}

// FormatDebugPrompt returns the prompt with its base64 image and document payloads truncated, for
// ExecutionResponse.Prompt. The prompt being sent is not modified.
func (d *AnthropicDriver) FormatDebugPrompt(prompt any) any {
	if claude, ok := prompt.(claudePrompt); ok {
		return formatClaudeDebugPrompt(claude)
	}
	return prompt
}

// Execute runs a single non-streaming completion against POST /v1/messages and
// returns the assembled response with updated conversation state.
func (d *AnthropicDriver) Execute(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (*ExecutionResponse, error) {
	return executeWithPrompt(ctx, d, segments, options, d.requestTextCompletion)
}

// Stream runs a streaming completion against POST /v1/messages (stream=true),
// parsing the SSE event stream into CompletionChunks.
func (d *AnthropicDriver) Stream(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (CompletionStream, error) {
	stream, err := streamWithPrompt(ctx, d, segments, options, d.requestTextCompletionStream)
	if err != nil {
		return nil, err
	}
	// A stream nobody reads is cancelled after StreamStartTimeout, releasing the connection.
	return leaseCompletionStream(stream, d.options.StreamStartTimeout), nil
}

// anthropicRequest prepares a Messages API request: its body (with the model) and headers.
func (d *AnthropicDriver) anthropicRequest(prompt any, options ExecutionOptions, operation string, stream bool) (*Request, map[string]any, map[string]string, error) {
	claude, ok := prompt.(claudePrompt)
	if !ok {
		return nil, nil, nil, fmt.Errorf("anthropic: unexpected prompt type %T", prompt)
	}
	request, err := NewRequest(claude, options, RequestSettings{Provider: ProviderNameAnthropic, Operation: operation})
	if err != nil {
		return nil, nil, nil, err
	}
	body := make(map[string]any, len(request.Body)+2)
	for key, value := range request.Body {
		body[key] = value
	}
	body["model"] = options.Model
	if stream {
		body["stream"] = true
	}
	headers := d.headers()
	if beta := request.BetaHeader(); beta != "" {
		headers["anthropic-beta"] = beta
	}
	return request, body, headers, nil
}

// requestTextCompletion posts a non-streaming Messages API request and returns
// the parsed completion plus the conversation extended with the response.
func (d *AnthropicDriver) requestTextCompletion(ctx context.Context, prompt any, options ExecutionOptions) (*Completion, error) {
	endpoint, err := joinEndpoint(d.options.BaseURL, "v1", "messages")
	if err != nil {
		return nil, err
	}
	request, body, headers, err := d.anthropicRequest(prompt, options, "execute", false)
	if err != nil {
		return nil, err
	}
	var response claudeResponse
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, body, &response); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	return request.Complete(response), nil
}

// requestTextCompletionStream posts a streaming Messages API request and decodes
// its SSE events into chunks; the finalizer stores the rebuilt message.
func (d *AnthropicDriver) requestTextCompletionStream(ctx context.Context, prompt any, options ExecutionOptions) (CompletionStream, error) {
	endpoint, err := joinEndpoint(d.options.BaseURL, "v1", "messages")
	if err != nil {
		return nil, err
	}
	request, body, headers, err := d.anthropicRequest(prompt, options, "stream", true)
	if err != nil {
		return nil, err
	}
	stream, err := postSSE(ctx, d.httpClient(options.HTTPTimeout), endpoint, headers, body)
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	return request.Stream(prompt, SSEEvents(stream), stream.Close, func(err error) error {
		return d.formatError(err, options.Model, "stream")
	}), nil
}

// headers returns the base auth headers (x-api-key plus the pinned
// anthropic-version) for every request.
func (d *AnthropicDriver) headers() map[string]string {
	return map[string]string{
		"x-api-key":         d.options.APIKey,
		"anthropic-version": anthropicVersion,
	}
}

// ListModels fetches the available Claude models from GET /v1/models, marking
// them all as text models with streaming and tool support.
func (d *AnthropicDriver) ListModels(ctx context.Context, _ *ModelSearchPayload) ([]AIModel, error) {
	endpoint, err := joinEndpoint(d.options.BaseURL, "v1", "models")
	if err != nil {
		return nil, err
	}
	var response struct {
		Data []struct {
			ID          string `json:"id"`
			DisplayName string `json:"display_name"`
		} `json:"data"`
	}
	if err := doJSON(ctx, d.client, http.MethodGet, endpoint, d.headers(), nil, &response); err != nil {
		return nil, d.formatError(err, "", "listModels")
	}
	models := make([]AIModel, 0, len(response.Data))
	for _, item := range response.Data {
		name := item.DisplayName
		if name == "" {
			name = item.ID
		}
		models = append(models, AIModel{
			ID:          item.ID,
			Name:        name,
			Provider:    string(ProviderAnthropic),
			Type:        ptr(ModelType("text")),
			CanStream:   ptr(true),
			ToolSupport: ptr(true),
		})
	}
	return models, nil
}

// ValidateConnection verifies the Anthropic API key by listing models.
func (d *AnthropicDriver) ValidateConnection(ctx context.Context) error {
	_, err := d.ListModels(ctx, nil)
	return err
}

func (d *AnthropicDriver) httpClient(timeout HTTPTimeoutOptions) *http.Client {
	if !hasHTTPTimeout(timeout) {
		return d.client
	}
	return newHTTPClient(d.client, timeout)
}

// GenerateEmbeddings returns a non-retryable error because Anthropic Claude does not expose embeddings.
func (d *AnthropicDriver) GenerateEmbeddings(_ context.Context, options EmbeddingsOptions) (*EmbeddingsResult, error) {
	return nil, newLlumiverseError("Anthropic does not support embeddings", boolPtr(false), LlumiverseErrorContext{
		Provider:  ProviderAnthropic,
		Model:     options.Model,
		Operation: "embeddings",
	}, nil, 0, "UnsupportedOperation")
}

// formatError wraps an HTTP status error in a LlumiverseError with provider
// context and retryability derived from the status code and body; non-HTTP
// errors pass through unchanged.
func (d *AnthropicDriver) formatError(err error, model string, operation string) error {
	code, _ := errorStatusAndName(err)
	if code != 0 {
		return newLlumiverseError(err.Error(), retryableFromStatusAndMessage(code, err.Error()), LlumiverseErrorContext{
			Provider:  d.Provider(),
			Model:     model,
			Operation: operation,
		}, err, code, "AnthropicError")
	}
	return err
}
