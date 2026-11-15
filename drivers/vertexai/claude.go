package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
)

// vertexAnthropicVersion is the anthropic_version the Claude partner endpoint expects in the body.
const vertexAnthropicVersion = "vertex-2023-10-16"

// isClaudeModel reports whether a model ID refers to a Claude/Anthropic partner model.
func isClaudeModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "claude") || strings.Contains(strings.ToLower(model), "anthropic")
}

// vertexClaudePayload builds the shared Claude Messages payload for the rawPredict
// endpoint: Vertex selects the model from the resource path and rejects the
// Anthropic-native model field, and wants anthropic_version in the body.
func vertexClaudePayload(prompt internalclaude.Prompt, options ExecutionOptions, stream bool) (map[string]any, map[string]string, error) {
	payload, headers, err := internalclaude.Payload(prompt, withShortModel(options), stream, map[string]string{})
	if err != nil {
		return nil, nil, err
	}
	delete(payload, "model")
	payload["anthropic_version"] = vertexAnthropicVersion
	return payload, headers, nil
}

// requestVertexClaude POSTs a Claude partner-model request via rawPredict, reusing
// the shared Claude payload/response handling and stamping the anthropic_version.
func (d *VertexAIDriver) requestVertexClaude(ctx context.Context, prompt internalclaude.Prompt, options ExecutionOptions) (*Completion, error) {
	endpoint, err := d.vertexEndpoint(options.Model, "rawPredict")
	if err != nil {
		return nil, err
	}
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, err
	}
	conversation := internalclaude.ConversationInput(options.Conversation, prompt)
	payload, payloadHeaders, err := vertexClaudePayload(conversation, options, false)
	if err != nil {
		return nil, err
	}
	for key, value := range payloadHeaders {
		headers[key] = value
	}
	var response internalclaude.Response
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, payload, &response); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	completion := internalclaude.ExtractCompletion(response, options)
	completion.Conversation = internalclaude.FinalizeConversation(internalclaude.AppendResponseToConversation(conversation, response), options)
	return completion, nil
}

// requestVertexClaudeStream opens a Claude partner-model SSE stream via
// streamRawPredict-compatible rawPredict?alt=sse and decodes the Claude events.
func (d *VertexAIDriver) requestVertexClaudeStream(ctx context.Context, prompt internalclaude.Prompt, options ExecutionOptions) (CompletionStream, error) {
	endpoint, err := d.vertexEndpoint(options.Model, "rawPredict")
	if err != nil {
		return nil, err
	}
	if !strings.Contains(endpoint, "?") {
		endpoint += "?alt=sse"
	}
	conversation := internalclaude.ConversationInput(options.Conversation, prompt)
	payload, payloadHeaders, err := vertexClaudePayload(conversation, options, true)
	if err != nil {
		return nil, err
	}
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, err
	}
	for key, value := range payloadHeaders {
		headers[key] = value
	}
	body, err := postSSE(ctx, d.httpClient(options.HTTPTimeout), endpoint, headers, payload)
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	ch := make(chan streamItem)
	completion := &ExecutionResponse{Prompt: prompt}
	go func() {
		defer close(ch)
		err := scanSSE(body, func(event sseEvent) error {
			if event.Data == "" {
				return nil
			}
			var envelope map[string]json.RawMessage
			if err := json.Unmarshal([]byte(event.Data), &envelope); err != nil {
				return err
			}
			chunk := internalclaude.SSEToChunk(envelope, options)
			if len(chunk.Result) > 0 || len(chunk.ToolUse) > 0 || chunk.FinishReason != "" || chunk.TokenUsage != nil {
				ch <- streamItem{Chunk: chunk}
			}
			return nil
		})
		if err != nil && !errors.Is(err, io.EOF) {
			ch <- streamItem{Err: d.formatError(err, options.Model, "stream")}
		}
	}()
	return newChannelStreamWithFinalizer(ch, completion, body.Close, func(resp *ExecutionResponse) {
		resp.Conversation = internalclaude.FinalizeConversation(internalclaude.BuildStreamingConversation(conversation, &resp.Completion), options)
	}), nil
}
