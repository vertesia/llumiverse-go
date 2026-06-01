package core

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type ExecuteFunc = executeFunc
type StreamFunc = streamFunc
type SSEEvent = sseEvent

// ExecuteWithPrompt runs the shared non-streaming driver lifecycle around fn.
func ExecuteWithPrompt(ctx context.Context, driver Driver, segments []PromptSegment, options ExecutionOptions, fn ExecuteFunc) (*ExecutionResponse, error) {
	return executeWithPrompt(ctx, driver, segments, options, fn)
}

// StreamWithPrompt runs the shared streaming driver lifecycle around fn.
func StreamWithPrompt(ctx context.Context, driver Driver, segments []PromptSegment, options ExecutionOptions, fn StreamFunc) (CompletionStream, error) {
	return streamWithPrompt(ctx, driver, segments, options, fn)
}

// FormatGenericError wraps an arbitrary provider error with llumiverse context.
func FormatGenericError(err error, context LlumiverseErrorContext) error {
	return formatGenericError(err, context)
}

// ErrorStatusAndName extracts an HTTP status code and concrete error type name.
func ErrorStatusAndName(err error) (int, string) {
	return errorStatusAndName(err)
}

// RetryableFromStatusAndMessage classifies retryability from HTTP status and text.
func RetryableFromStatusAndMessage(code int, msg string) *bool {
	return retryableFromStatusAndMessage(code, msg)
}

// HasHTTPTimeout reports whether any timeout knob is configured.
func HasHTTPTimeout(options HTTPTimeoutOptions) bool {
	return hasHTTPTimeout(options)
}

// NewHTTPClient clones base and applies llumiverse HTTP timeout options.
func NewHTTPClient(base *http.Client, options HTTPTimeoutOptions) *http.Client {
	return newHTTPClient(base, options)
}

// NormalizeJSONResult upgrades parseable text results to JSON results.
func NormalizeJSONResult(completion *Completion) {
	normalizeJSONResult(completion)
}

// ReadAllString reads and closes rc, returning its body as a string.
func ReadAllString(rc io.ReadCloser) (string, error) {
	return readAllString(rc)
}

// ReadAllStringFromDataSource opens ds, reads it, and closes the stream.
func ReadAllStringFromDataSource(ctx context.Context, ds DataSource) (string, error) {
	rc, err := ds.Open(ctx)
	if err != nil {
		return "", err
	}
	return readAllString(rc)
}

// TruncateForConversation shortens retained conversation text for summaries.
func TruncateForConversation(value string, maxLen int) string {
	return truncateForConversation(value, maxLen)
}

// RemoveNilMapValues recursively drops nil and empty nested map values.
func RemoveNilMapValues(in map[string]any) map[string]any {
	return removeNilMapValues(in)
}

// NewChannelStream creates the shared accumulating CompletionStream implementation.
func NewChannelStream(ch <-chan StreamItem, completion *ExecutionResponse, closeFn func() error) CompletionStream {
	return newChannelStream(ch, completion, closeFn)
}

// NewChannelStreamWithFinalizer creates a stream with a completion finalizer hook.
func NewChannelStreamWithFinalizer(ch <-chan StreamItem, completion *ExecutionResponse, closeFn func() error, finalizer func(*ExecutionResponse)) CompletionStream {
	return newChannelStreamWithFinalizer(ch, completion, closeFn, finalizer)
}

// DoJSON executes a JSON HTTP request and decodes the response into out.
func DoJSON(ctx context.Context, client *http.Client, method string, endpoint string, headers map[string]string, body any, out any) error {
	return doJSON(ctx, client, method, endpoint, headers, body, out)
}

// JoinEndpoint appends path elements to a base URL safely.
func JoinEndpoint(base string, elems ...string) (string, error) {
	return joinEndpoint(base, elems...)
}

// DataSourceToDataURL reads ds and returns a base64 data URL.
func DataSourceToDataURL(ctx context.Context, ds DataSource) (string, error) {
	return dataSourceToDataURL(ctx, ds)
}

// DataSourceToBase64 reads ds and returns standard base64 content.
func DataSourceToBase64(ctx context.Context, ds DataSource) (string, error) {
	return dataSourceToBase64(ctx, ds)
}

// ScanSSE parses Server-Sent Events from body.
func ScanSSE(body io.ReadCloser, onEvent func(SSEEvent) error) error {
	return scanSSE(body, func(event sseEvent) error {
		return onEvent(SSEEvent(event))
	})
}

// PostSSE starts a JSON POST request that expects a Server-Sent Events response.
func PostSSE(ctx context.Context, client *http.Client, endpoint string, headers map[string]string, body any) (io.ReadCloser, error) {
	return postSSE(ctx, client, endpoint, headers, body)
}

// OptionFloat reads a numeric model option as float64.
func OptionFloat(options map[string]any, key string) *float64 {
	return optionFloat(options, key)
}

// OptionInt reads a numeric model option as int.
func OptionInt(options map[string]any, key string) *int {
	return optionInt(options, key)
}

// OptionBool reads a boolean model option, defaulting to false.
func OptionBool(options map[string]any, key string) bool {
	return optionBool(options, key)
}

// OptionStringSlice reads a string-slice model option.
func OptionStringSlice(options map[string]any, key string) []string {
	return optionStringSlice(options, key)
}

// OptionString returns the first string model option found under keys.
func OptionString(options map[string]any, keys ...string) string {
	if options == nil {
		return ""
	}
	for _, key := range keys {
		if value, ok := options[key].(string); ok {
			return value
		}
	}
	return ""
}

// SafeJSONParse decodes JSON text or returns the original value on failure.
func SafeJSONParse(value string) any {
	return safeJSONParse(value)
}

// ToString returns strings unchanged and JSON-encodes other values.
func ToString(value any) string {
	return toString(value)
}

// NormalizeEmbeddingsOptions applies shared embedding defaults.
func NormalizeEmbeddingsOptions(options EmbeddingsOptions) (EmbeddingsOptions, error) {
	return normalizeEmbeddingsOptions(options)
}

// CompletionResultsToText flattens text and JSON completion results.
func CompletionResultsToText(results []CompletionResult) string {
	return completionResultsToText(results)
}

// IsClaudeVersionGTE reports whether a Claude model name is at least major.minor.
func IsClaudeVersionGTE(model string, major int, minor int) bool {
	return isClaudeVersionGTE(model, major, minor)
}

// ClaudeSupportsAdaptiveThinking reports support for Claude adaptive thinking.
func ClaudeSupportsAdaptiveThinking(model string) bool {
	return claudeSupportsAdaptiveThinking(model)
}

// ClaudeHasSamplingParameterRestriction reports whether sampling knobs must be omitted.
func ClaudeHasSamplingParameterRestriction(model string) bool {
	return claudeHasSamplingParameterRestriction(model)
}

// ToolInputString formats a tool input as JSON or a plain string.
func ToolInputString(input any) string {
	if input == nil {
		return ""
	}
	if s, ok := input.(string); ok {
		return s
	}
	data, err := json.Marshal(input)
	if err != nil {
		return fmt.Sprint(input)
	}
	return string(data)
}

// ShortResourceName returns the last slash-delimited segment of a resource name.
func ShortResourceName(value string) string {
	value = strings.Trim(value, "/")
	if value == "" {
		return ""
	}
	parts := strings.Split(value, "/")
	return parts[len(parts)-1]
}
