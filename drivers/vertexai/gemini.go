package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
)

// requestGemini POSTs a generateContent request and converts the response into a
// Completion plus the updated conversation.
func (d *VertexAIDriver) requestGemini(ctx context.Context, prompt geminiPrompt, options ExecutionOptions) (*Completion, error) {
	prompt = geminiConversationInput(options.Conversation, prompt)
	endpoint, err := d.vertexEndpoint(options.Model, "generateContent")
	if err != nil {
		return nil, err
	}
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	var response geminiResponse
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, d.geminiPayload(prompt, options), &response); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	if len(response.Candidates) > 0 {
		assignGeminiFunctionCallIDs(response.Candidates[0].Content.Parts, collectGeminiFunctionCallIDs(prompt.Contents))
	}
	completion := geminiResponseToCompletion(response, options.IncludeOriginalResponse)
	completion.Conversation = finalizeGeminiConversation(appendGeminiResponseToConversation(prompt, response), options)
	return completion, nil
}

// requestGeminiStream opens a streamGenerateContent SSE stream, decodes each
// chunk, and rebuilds the conversation once the stream is consumed.
func (d *VertexAIDriver) requestGeminiStream(ctx context.Context, prompt geminiPrompt, options ExecutionOptions) (CompletionStream, error) {
	endpoint, err := d.vertexEndpoint(options.Model, "streamGenerateContent")
	if err != nil {
		return nil, err
	}
	geminiInput := geminiConversationInput(options.Conversation, prompt)
	payload := d.geminiPayload(geminiInput, options)
	if !strings.Contains(endpoint, "?") {
		endpoint += "?alt=sse"
	}
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	delete(payload, "model")
	body, err := postSSE(ctx, d.httpClient(options.HTTPTimeout), endpoint, headers, payload)
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	ch := make(chan streamItem)
	completion := &ExecutionResponse{Prompt: geminiInput}
	// A functionCall part always arrives complete in one chunk, so each call gets
	// its unique id on arrival and the shared stream keeps parallel calls to one
	// tool apart.
	usedCallIDs := collectGeminiFunctionCallIDs(geminiInput.Contents)
	go func() {
		defer close(ch)
		err := scanSSE(body, func(event sseEvent) error {
			if event.Data == "" {
				return nil
			}
			var response geminiResponse
			if err := json.Unmarshal([]byte(event.Data), &response); err != nil {
				return err
			}
			for _, candidate := range response.Candidates {
				assignGeminiFunctionCallIDs(candidate.Content.Parts, usedCallIDs)
			}
			chunk := geminiResponseToChunk(response)
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
		resp.Conversation = finalizeGeminiConversation(buildGeminiStreamingConversation(geminiInput, &resp.Completion), options)
	}), nil
}
