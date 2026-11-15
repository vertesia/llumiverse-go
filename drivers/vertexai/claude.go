package vertexai

import (
	"context"
	"net/http"
	"strings"

	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
)

// vertexAnthropicVersion is the anthropic_version the Claude partner endpoint expects in the body.
const vertexAnthropicVersion = "vertex-2023-10-16"

// vertexClaudeStructuredOutputDenials remembers the project, region and model combinations where an organization
// policy refused Claude structured outputs, so each fails once and then uses the prompt-based result schema. It is
// shared by every driver in the process.
var vertexClaudeStructuredOutputDenials internalclaude.StructuredOutputDenials

// isClaudeModel reports whether a model ID refers to a Claude partner model, including a bare Claude ID such as
// claude-sonnet-4-5 without a publisher path.
func isClaudeModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "claude") || strings.Contains(strings.ToLower(model), "anthropic")
}

// vertexClaudeDenialKey identifies the project, region and model of a Claude request for structured-output
// denials. A locations/{region}/ prefix on the model overrides the driver region.
func (d *VertexAIDriver) vertexClaudeDenialKey(model string) string {
	region := d.options.Region
	if parts := strings.Split(model, "/"); len(parts) >= 2 && parts[0] == "locations" {
		region = parts[1]
	}
	return internalclaude.StructuredOutputDenialKey(d.options.Project, region, shortResourceName(model))
}

// vertexClaudeNativeResultSchema reports whether a Claude request sends its result schema as output_config.format:
// the request qualifies and no policy denied structured outputs for its project, region and model.
func (d *VertexAIDriver) vertexClaudeNativeResultSchema(options ExecutionOptions) bool {
	return internalclaude.UsesNativeResultSchema(withShortModel(options)) &&
		!vertexClaudeStructuredOutputDenials.Has(d.vertexClaudeDenialKey(options.Model))
}

// createVertexClaudePrompt formats a Claude prompt, leaving the result schema out when the request sends it
// natively. getClaudePayload adds the instructions back if the request later falls back to the prompt path.
func (d *VertexAIDriver) createVertexClaudePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (internalclaude.Prompt, error) {
	return internalclaude.FormatPromptWithNativeSchema(ctx, segments, options, d.vertexClaudeNativeResultSchema(options))
}

// vertexClaudeRequest prepares a Claude request for the rawPredict endpoint: Vertex selects the model from the
// resource path and rejects the model field, and takes anthropic_version in the body and the betas in one
// anthropic-beta header.
func (d *VertexAIDriver) vertexClaudeRequest(ctx context.Context, prompt internalclaude.Prompt, options ExecutionOptions, operation string, allowNative bool) (*internalclaude.Request, map[string]any, map[string]string, error) {
	request, err := internalclaude.NewRequest(prompt, withShortModel(options), internalclaude.RequestSettings{
		Provider:                internalclaude.ProviderNameVertexAI,
		Operation:               operation,
		AllowNativeResultSchema: allowNative,
	})
	if err != nil {
		return nil, nil, nil, err
	}
	body := make(map[string]any, len(request.Body)+2)
	for key, value := range request.Body {
		body[key] = value
	}
	body["anthropic_version"] = vertexAnthropicVersion
	if operation == "stream" {
		body["stream"] = true
	}
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, nil, nil, err
	}
	if beta := request.BetaHeader(); beta != "" {
		headers["anthropic-beta"] = beta
	}
	return request, body, headers, nil
}

// requestVertexClaude POSTs a Claude partner-model request via rawPredict. A structured-output policy denial is
// remembered and the request retried with the schema in the prompt.
func (d *VertexAIDriver) requestVertexClaude(ctx context.Context, prompt internalclaude.Prompt, options ExecutionOptions) (*Completion, error) {
	native := d.vertexClaudeNativeResultSchema(options)
	completion, err := d.executeVertexClaude(ctx, prompt, options, native)
	if native && internalclaude.IsStructuredOutputPolicyDenial(err) {
		vertexClaudeStructuredOutputDenials.Add(d.vertexClaudeDenialKey(options.Model))
		return d.executeVertexClaude(ctx, prompt, options, false)
	}
	return completion, err
}

// executeVertexClaude sends one rawPredict request.
func (d *VertexAIDriver) executeVertexClaude(ctx context.Context, prompt internalclaude.Prompt, options ExecutionOptions, allowNative bool) (*Completion, error) {
	endpoint, err := d.vertexEndpoint(options.Model, "rawPredict")
	if err != nil {
		return nil, err
	}
	request, body, headers, err := d.vertexClaudeRequest(ctx, prompt, options, "execute", allowNative)
	if err != nil {
		return nil, err
	}
	var response internalclaude.Response
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, body, &response); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	return request.Complete(response), nil
}

// requestVertexClaudeStream opens a Claude partner-model SSE stream via rawPredict?alt=sse and decodes the Claude
// events. A structured-output policy denial is a 400, so it arrives before the stream starts; the request is then
// retried with the schema in the prompt.
func (d *VertexAIDriver) requestVertexClaudeStream(ctx context.Context, prompt internalclaude.Prompt, options ExecutionOptions) (CompletionStream, error) {
	native := d.vertexClaudeNativeResultSchema(options)
	stream, err := d.streamVertexClaude(ctx, prompt, options, native)
	if native && internalclaude.IsStructuredOutputPolicyDenial(err) {
		vertexClaudeStructuredOutputDenials.Add(d.vertexClaudeDenialKey(options.Model))
		return d.streamVertexClaude(ctx, prompt, options, false)
	}
	return stream, err
}

// streamVertexClaude opens one rawPredict SSE stream.
func (d *VertexAIDriver) streamVertexClaude(ctx context.Context, prompt internalclaude.Prompt, options ExecutionOptions, allowNative bool) (CompletionStream, error) {
	endpoint, err := d.vertexEndpoint(options.Model, "rawPredict")
	if err != nil {
		return nil, err
	}
	if !strings.Contains(endpoint, "?") {
		endpoint += "?alt=sse"
	}
	request, body, headers, err := d.vertexClaudeRequest(ctx, prompt, options, "stream", allowNative)
	if err != nil {
		return nil, err
	}
	sse, err := postSSE(ctx, d.httpClient(options.HTTPTimeout), endpoint, headers, body)
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	return request.Stream(prompt, internalclaude.SSEEvents(sse), sse.Close, func(err error) error {
		return d.formatError(err, options.Model, "stream")
	}), nil
}
