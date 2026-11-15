package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
)

// requestGemini POSTs a generateContent request and converts the response into a
// Completion plus the updated conversation.
func (d *VertexAIDriver) requestGemini(ctx context.Context, prompt geminiPrompt, options ExecutionOptions) (*Completion, error) {
	if err := validateGeminiRequest(options, "execute"); err != nil {
		return nil, err
	}
	prompt = geminiConversationInput(options.Conversation, prompt)
	endpoint, err := d.vertexEndpoint(options.Model, "generateContent")
	if err != nil {
		return nil, err
	}
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	addGeminiServiceTierHeaders(headers, options.ModelOptions)
	var response geminiResponse
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, d.geminiPayload(prompt, options), &response); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	if err := checkGeminiResponseFinishReasons(response); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	if len(response.Candidates) > 0 {
		assignGeminiFunctionCallIDs(response.Candidates[0].Content.Parts, collectGeminiFunctionCallIDs(prompt.Contents))
	}
	completion := geminiResponseToCompletion(response, options)
	completion.Conversation = finalizeGeminiConversation(appendGeminiResponseToConversation(prompt, response), options)
	return completion, nil
}

// requestGeminiStream opens a streamGenerateContent SSE stream, decodes each
// chunk, and keeps the model's native parts (thoughts and their signatures
// included) for the conversation once the stream is consumed.
func (d *VertexAIDriver) requestGeminiStream(ctx context.Context, prompt geminiPrompt, options ExecutionOptions) (CompletionStream, error) {
	if err := validateGeminiRequest(options, "stream"); err != nil {
		return nil, err
	}
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
	addGeminiServiceTierHeaders(headers, options.ModelOptions)
	delete(payload, "model")
	body, err := postSSE(ctx, d.httpClient(options.HTTPTimeout), endpoint, headers, payload)
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	ch := make(chan streamItem)
	completion := &ExecutionResponse{Prompt: geminiInput}
	includeThoughts := geminiIncludeThoughts(options.ModelOptions)
	// A functionCall part always arrives complete in one chunk, so each call gets
	// its unique id on arrival and the shared stream keeps parallel calls to one
	// tool apart.
	usedCallIDs := collectGeminiFunctionCallIDs(geminiInput.Contents)
	var (
		mu          sync.Mutex
		nativeParts []geminiPart
	)
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
			if err := checkGeminiResponseFinishReasons(response); err != nil {
				return err
			}
			for _, candidate := range response.Candidates {
				assignGeminiFunctionCallIDs(candidate.Content.Parts, usedCallIDs)
			}
			if len(response.Candidates) > 0 && geminiIsModelContent(response.Candidates[0].Content) {
				mu.Lock()
				nativeParts = appendGeminiStreamParts(nativeParts, response.Candidates[0].Content.Parts)
				mu.Unlock()
			}
			chunk := geminiResponseToChunk(response, includeThoughts)
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
		mu.Lock()
		parts := append([]geminiPart(nil), nativeParts...)
		mu.Unlock()
		resp.Conversation = finalizeGeminiConversation(appendGeminiStreamedTurn(geminiInput, parts), options)
	}), nil
}

// geminiIsModelContent reports whether a streamed candidate carries model parts.
func geminiIsModelContent(content geminiContent) bool {
	return len(content.Parts) > 0 && (content.Role == "" || content.Role == "model")
}

// appendGeminiStreamedTurn appends the model turn accumulated from a stream.
func appendGeminiStreamedTurn(prompt geminiPrompt, parts []geminiPart) geminiPrompt {
	if len(parts) > 0 {
		prompt.Contents = append(prompt.Contents, geminiContent{Role: "model", Parts: parts})
	}
	return prompt
}
