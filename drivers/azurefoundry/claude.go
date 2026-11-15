package azurefoundry

import (
	"context"
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

// claudeRequest prepares a Foundry Claude request: the shared Claude request for
// the source model, its body with the deployment name as the model, and the
// auth, anthropic-version and anthropic-beta headers. Foundry is not one of the
// platforms that take mid-conversation tool changes or native result schemas,
// so it is named as itself, as in the TS client.
func (d *AzureFoundryDriver) claudeRequest(ctx context.Context, prompt internalclaude.Prompt, r route, options ExecutionOptions, operation string) (string, *internalclaude.Request, map[string]any, map[string]string, error) {
	endpoint, err := joinEndpoint(d.anthropic, "v1", "messages")
	if err != nil {
		return "", nil, nil, nil, err
	}
	request, err := internalclaude.NewRequest(prompt, claudeOptions(r, options), internalclaude.RequestSettings{
		Provider:  string(ProviderAzureFoundry),
		Operation: operation,
	})
	if err != nil {
		return "", nil, nil, nil, err
	}
	headers, err := d.creds.keyOrBearer(ctx, headerAnthropicKey)
	if err != nil {
		return "", nil, nil, nil, err
	}
	headers["anthropic-version"] = claudeAPIVersion
	if beta := request.BetaHeader(); beta != "" {
		headers["anthropic-beta"] = beta
	}
	body := make(map[string]any, len(request.Body)+2)
	for key, value := range request.Body {
		body[key] = value
	}
	body["model"] = r.deployment
	if operation == "stream" {
		body["stream"] = true
	}
	return endpoint, request, body, headers, nil
}

// requestClaude runs a non-streaming Claude Messages request.
func (d *AzureFoundryDriver) requestClaude(ctx context.Context, prompt internalclaude.Prompt, r route, options ExecutionOptions) (*Completion, error) {
	endpoint, request, body, headers, err := d.claudeRequest(ctx, prompt, r, options, "execute")
	if err != nil {
		return nil, err
	}
	var response internalclaude.Response
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, body, &response); err != nil {
		return nil, d.formatClaudeError(err, options.Model, "execute")
	}
	return request.Complete(response), nil
}

// streamClaude runs a streaming Claude Messages request.
func (d *AzureFoundryDriver) streamClaude(ctx context.Context, prompt internalclaude.Prompt, r route, options ExecutionOptions) (CompletionStream, error) {
	endpoint, request, body, headers, err := d.claudeRequest(ctx, prompt, r, options, "stream")
	if err != nil {
		return nil, err
	}
	sse, err := postSSE(ctx, d.httpClient(options.HTTPTimeout), endpoint, headers, body)
	if err != nil {
		return nil, d.formatClaudeError(err, options.Model, "stream")
	}
	return request.Stream(prompt, internalclaude.SSEEvents(sse), sse.Close, func(err error) error {
		return d.formatClaudeError(err, options.Model, "stream")
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
