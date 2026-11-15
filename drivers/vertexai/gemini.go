package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"maps"
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
	endpoint, err := d.geminiEndpoint(options.Model, "generateContent")
	if err != nil {
		return nil, err
	}
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	addGeminiServiceTierHeaders(headers, options.ModelOptions)
	payload := d.geminiPayload(prompt, options)
	var response geminiResponse
	diagnostic, err := d.sendWithGeminiContextCache(ctx, options, payload, d.geminiLocation(options.Model), func(body map[string]any) error {
		// Continuation hops add their token to a copy, so a fallback to another
		// payload starts clean.
		body = maps.Clone(body)
		response = geminiResponse{}
		for token := ""; ; {
			var hop geminiResponse
			if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, body, &hop); err != nil {
				return err
			}
			if err := checkGeminiResponseFinishReasons(hop); err != nil {
				return err
			}
			if err := checkGeminiAudioOutput(hop); err != nil {
				return err
			}
			if token == "" {
				response = hop
			} else {
				response = mergeGeminiContinuation(response, hop)
			}
			next := geminiContinuationToken(hop)
			if next == "" || next == token {
				return nil
			}
			token = next
			body["continuationToken"] = token
		}
	})
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	if len(response.Candidates) > 0 {
		assignGeminiFunctionCallIDs(response.Candidates[0].Content.Parts, collectGeminiFunctionCallIDs(prompt.Contents))
	}
	completion := geminiResponseToCompletion(response, options)
	completion.PromptCacheDiagnostic = diagnostic
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
	endpoint, err := d.geminiEndpoint(options.Model, "streamGenerateContent")
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
	var body io.ReadCloser
	diagnostic, err := d.sendWithGeminiContextCache(ctx, options, payload, d.geminiLocation(options.Model), func(sent map[string]any) error {
		opened, err := postSSE(ctx, d.httpClient(options.HTTPTimeout), endpoint, headers, sent)
		if err != nil {
			return err
		}
		// Continuation hops repeat the payload that was accepted.
		payload, body = maps.Clone(sent), opened
		return nil
	})
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	ch := make(chan streamItem)
	completion := &ExecutionResponse{Prompt: geminiInput}
	completion.PromptCacheDiagnostic = diagnostic
	includeThoughts := geminiIncludeThoughts(options.ModelOptions)
	// A functionCall part always arrives complete in one chunk, so each call gets
	// its unique id on arrival and the shared stream keeps parallel calls to one
	// tool apart.
	usedCallIDs := collectGeminiFunctionCallIDs(geminiInput.Contents)
	var (
		mu          sync.Mutex
		nativeParts []geminiPart
		current     = body
		closed      bool
	)
	closeBody := func() error {
		mu.Lock()
		defer mu.Unlock()
		closed = true
		return current.Close()
	}
	go func() {
		defer close(ch)
		// A long decode spans several requests (hops); each resumes from the
		// continuation token the previous one ended with, and each chunk reports
		// the usage of the finished hops plus its own.
		var finished geminiUsageMetadata
		hop := geminiStreamHop{}
		for hopIndex := 0; ; hopIndex++ {
			var hopUsage *geminiUsageMetadata
			err := scanSSE(current, func(event sseEvent) error {
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
				if err := checkGeminiAudioOutput(response); err != nil {
					return err
				}
				hop.observe(&response)
				if response.UsageMetadata.TotalTokenCount > 0 {
					if hopIndex > 0 {
						response.UsageMetadata = addGeminiUsage(finished, response.UsageMetadata)
					}
					usage := response.UsageMetadata
					hopUsage = &usage
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
				return
			}
			if hopUsage != nil {
				finished = *hopUsage
			}
			token := hop.resumeToken()
			if token == "" {
				return
			}
			payload["continuationToken"] = token
			next, err := postSSE(ctx, d.httpClient(options.HTTPTimeout), endpoint, headers, payload)
			if err != nil {
				ch <- streamItem{Err: d.formatError(err, options.Model, "stream")}
				return
			}
			mu.Lock()
			if closed {
				mu.Unlock()
				_ = next.Close()
				return
			}
			_ = current.Close()
			current = next
			mu.Unlock()
			hop = geminiStreamHop{startToken: token}
		}
	}()
	return newChannelStreamWithFinalizer(ch, completion, closeBody, func(resp *ExecutionResponse) {
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
