package azurefoundry

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"sync"

	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
)

// claudeAPIVersion is the anthropic-version header of the Messages API on
// Foundry. The TS client pins it rather than taking the Anthropic SDK v1
// default, which rejects sampling parameters.
const claudeAPIVersion = "2023-06-01"

// claudeOptions are the options of the shared Claude implementation: the source
// model drives Claude's model-specific behavior (thinking, sampling limits).
func claudeOptions(r route, options ExecutionOptions) ExecutionOptions {
	options.Model = r.source
	return options
}

// claudeRequest builds the Messages payload and headers of a Foundry Claude
// request. The deployment name is the request model.
func (d *AzureFoundryDriver) claudeRequest(ctx context.Context, conversation internalclaude.Prompt, r route, options ExecutionOptions, stream bool) (string, map[string]any, map[string]string, error) {
	endpoint, err := joinEndpoint(d.anthropic, "v1", "messages")
	if err != nil {
		return "", nil, nil, err
	}
	headers, err := d.creds.keyOrBearer(ctx, headerAnthropicKey)
	if err != nil {
		return "", nil, nil, err
	}
	headers["anthropic-version"] = claudeAPIVersion
	payload, headers, err := internalclaude.Payload(conversation, claudeOptions(r, options), stream, headers)
	if err != nil {
		return "", nil, nil, err
	}
	payload["model"] = r.deployment
	return endpoint, payload, headers, nil
}

// requestClaude runs a non-streaming Claude Messages request.
func (d *AzureFoundryDriver) requestClaude(ctx context.Context, prompt internalclaude.Prompt, r route, options ExecutionOptions) (*Completion, error) {
	conversation := internalclaude.ConversationInput(options.Conversation, prompt)
	endpoint, payload, headers, err := d.claudeRequest(ctx, conversation, r, options, false)
	if err != nil {
		return nil, err
	}
	var response internalclaude.Response
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, payload, &response); err != nil {
		return nil, d.formatClaudeError(err, options.Model, "execute")
	}
	sourceOptions := claudeOptions(r, options)
	completion := internalclaude.ExtractCompletion(response, sourceOptions)
	completion.Conversation = internalclaude.FinalizeConversation(internalclaude.AppendResponseToConversation(conversation, response), sourceOptions)
	return completion, nil
}

// streamClaude runs a streaming Claude Messages request.
func (d *AzureFoundryDriver) streamClaude(ctx context.Context, prompt internalclaude.Prompt, r route, options ExecutionOptions) (CompletionStream, error) {
	conversation := internalclaude.ConversationInput(options.Conversation, prompt)
	endpoint, payload, headers, err := d.claudeRequest(ctx, conversation, r, options, true)
	if err != nil {
		return nil, err
	}
	body, err := postSSE(ctx, d.httpClient(options.HTTPTimeout), endpoint, headers, payload)
	if err != nil {
		return nil, d.formatClaudeError(err, options.Model, "stream")
	}
	sourceOptions := claudeOptions(r, options)
	ch, send, closeFn := newStreamChannel(body)
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
			chunk := internalclaude.SSEToChunk(envelope, sourceOptions)
			if hasChunkContent(chunk) && !send(streamItem{Chunk: chunk}) {
				return errStreamClosed
			}
			return nil
		})
		if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, errStreamClosed) {
			send(streamItem{Err: d.formatClaudeError(err, options.Model, "stream")})
		}
	}()
	completion := &ExecutionResponse{Prompt: prompt}
	return newChannelStreamWithFinalizer(ch, completion, closeFn, func(resp *ExecutionResponse) {
		resp.Conversation = internalclaude.FinalizeConversation(internalclaude.BuildStreamingConversation(conversation, &resp.Completion), sourceOptions)
	}), nil
}

// errStreamClosed stops a stream reader whose consumer closed the stream.
var errStreamClosed = errors.New("stream closed")

// newStreamChannel returns the chunk channel of a stream over body, a send that
// gives up once the consumer closes the stream (so the reader goroutine never
// blocks forever), and the stream's close function.
func newStreamChannel(body io.Closer) (chan streamItem, func(streamItem) bool, func() error) {
	ch := make(chan streamItem)
	done := make(chan struct{})
	var once sync.Once
	send := func(item streamItem) bool {
		select {
		case ch <- item:
			return true
		case <-done:
			return false
		}
	}
	closeFn := func() error {
		once.Do(func() { close(done) })
		return body.Close()
	}
	return ch, send, closeFn
}

// hasChunkContent reports whether a chunk carries anything worth emitting.
func hasChunkContent(chunk CompletionChunk) bool {
	return len(chunk.Result) > 0 || len(chunk.ToolUse) > 0 || chunk.FinishReason != "" || chunk.TokenUsage != nil || chunk.ServiceTier != ""
}
