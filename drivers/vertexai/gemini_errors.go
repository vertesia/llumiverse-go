package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"golang.org/x/oauth2"
)

// vertexErrorName is the error name used when a Gemini Enterprise Agent Platform
// error carries no recognizable Google status.
const vertexErrorName = "GeminiEnterpriseAgentPlatformError"

// vertexStorageHint is appended to permission denials that name Cloud Storage,
// which usually means the Vertex service agent cannot read the referenced bucket.
const vertexStorageHint = " Ask your project administrator to verify bucket permissions for the Vertex AI service agent" +
	" or configured storage principal."

var vertexStorageDenialPattern = regexp.MustCompile(`(?i)storage\.(objects|buckets)\.|cloud storage|gs://`)

// googleAPIErrorBody is the AIP-193 error envelope Google APIs return.
type googleAPIErrorBody struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

// parseGoogleAPIError extracts the message and status of an AIP-193 error from an
// HTTP status error ("HTTP 400: {...}"). Streamed errors may wrap the envelope in a
// one-element array. When the body is not an AIP-193 envelope, the raw body is the
// message and the status is empty.
func parseGoogleAPIError(err error, code int) (message string, status string) {
	raw := err.Error()
	body := strings.TrimPrefix(raw, fmt.Sprintf("HTTP %d: ", code))
	if body == raw {
		body = strings.TrimPrefix(raw, fmt.Sprintf("HTTP %d", code))
	}
	body = strings.TrimSpace(body)
	var envelope googleAPIErrorBody
	if json.Unmarshal([]byte(body), &envelope) != nil || envelope.Error.Message == "" {
		var list []googleAPIErrorBody
		if json.Unmarshal([]byte(body), &list) == nil && len(list) > 0 {
			envelope = list[0]
		}
	}
	if envelope.Error.Message != "" {
		return envelope.Error.Message, envelope.Error.Status
	}
	return body, ""
}

// geminiErrorNamePatterns recognize the error type in a Google error message, in
// the order the TS client tries them.
var geminiErrorNamePatterns = []*regexp.Regexp{
	regexp.MustCompile(`^Error code ([a-zA-Z0-9_-]+):`), // "Error code invalid_grant: message"
	regexp.MustCompile(`^([A-Z_]+):`),                   // "ERROR_NAME: message"
	regexp.MustCompile(`\[([A-Z_]+)\]`),                 // "[ERROR_NAME] message"
	regexp.MustCompile(`^(\w+Error):`),                  // "ErrorTypeError: message"
}

// extractGeminiErrorName returns the error type embedded in a Google error
// message, or "" when it has none.
func extractGeminiErrorName(message string) string {
	for _, pattern := range geminiErrorNamePatterns {
		if match := pattern.FindStringSubmatch(message); match != nil {
			return match[1]
		}
	}
	return ""
}

// geminiErrorRetryable classifies a Gemini Enterprise Agent Platform error the way
// the TS client's isGeminiErrorRetryable does (Google AIP-194): explicit auth
// failures never retry; 408, 429, 499 and 5xx do; the inline URL fetcher's
// throttle, overload and timeout rejections retry although they arrive as 400
// INVALID_ARGUMENT; transport aborts, cancellations and deadline-exceeded errors
// retry even under a 4xx status; other 4xx errors do not. Anything else is unknown
// (nil) so callers decide.
func geminiErrorRetryable(code int, message string) *bool {
	lower := strings.ToLower(message)
	if strings.Contains(lower, "invalid_grant") || strings.Contains(lower, "credential's issuer") {
		return ptr(false)
	}
	if code == 408 || code == 429 || code == 499 || (code >= 500 && code < 600) {
		return ptr(true)
	}
	// The fetcher's transient statuses share these markers, e.g.
	// URL_REJECTED-REJECTED_CLIENT_THROTTLED, -REJECTED_PROXY_THROTTLED,
	// -REJECTED_RATE_LIMITED, -REJECTED_FC_TOO_MANY_PENDING,
	// -REJECTED_NO_RPC_BUFFERS and -REJECTED_FC_TIMEOUT. Permanent rejections
	// (robots-denied, unsafe or unsupported content) lack them.
	if code == 400 && strings.Contains(message, "URL_REJECTED") {
		for _, marker := range []string{"THROTTLED", "RATE_LIMITED", "TOO_MANY_PENDING", "NO_RPC_BUFFERS", "TIMEOUT"} {
			if strings.Contains(message, marker) {
				return ptr(true)
			}
		}
	}
	if strings.Contains(lower, "aborted") || strings.Contains(lower, "cancelled") || strings.Contains(lower, "deadline") {
		return ptr(true)
	}
	if code >= 400 && code < 500 {
		return ptr(false)
	}
	if code == 0 {
		// Errors without a status (transport failures, token refresh) keep the
		// shared message-based classification for rate limits and timeouts.
		return retryableFromStatusAndMessage(0, message)
	}
	return nil
}

// formatGeminiError wraps an error from a Gemini, Imagen or embedding request in a
// LlumiverseError: typed finish-reason and inline-media errors keep their own
// classification, HTTP errors carry the Google status and message, and other
// errors (transport, OAuth token refresh) are classified by message.
func formatGeminiError(err error, model string, operation string) error {
	if err == nil {
		return nil
	}
	var llumiErr *LlumiverseError
	if errors.As(err, &llumiErr) {
		return err
	}
	context := LlumiverseErrorContext{Provider: ProviderVertexAI, Model: model, Operation: operation}
	var finishErr *geminiFinishReasonError
	if errors.As(err, &finishErr) {
		// A terminal finish reason has no HTTP status; the reason names the error.
		return newLlumiverseError(finishErr.Error(), finishErr.retryable, context, err, 0, finishErr.finishReason)
	}
	var mediaErr *geminiInlineMediaLimitError
	if errors.As(err, &mediaErr) {
		return newLlumiverseError(mediaErr.Error(), ptr(false), context, err, mediaErr.StatusCode(), "GeminiInlineMediaLimitError")
	}
	var retrieveErr *oauth2.RetrieveError
	if errors.As(err, &retrieveErr) {
		code := 0
		if retrieveErr.Response != nil {
			code = retrieveErr.Response.StatusCode
		}
		name := retrieveErr.ErrorCode
		if name == "" {
			name = vertexErrorName
		}
		return newLlumiverseError(err.Error(), geminiErrorRetryable(code, err.Error()), context, err, code, name)
	}
	code, typeName := errorStatusAndName(err)
	if code == 0 {
		return newLlumiverseError(err.Error(), geminiErrorRetryable(0, err.Error()), context, err, 0, typeName)
	}
	message, status := parseGoogleAPIError(err, code)
	name := status
	if name == "" {
		name = extractGeminiErrorName(message)
	}
	if name == "" {
		name = vertexErrorName
	}
	// Classify on the whole response so a status such as DEADLINE_EXCEEDED or
	// CANCELLED counts even when the message does not repeat it.
	formatted := newLlumiverseError(fmt.Sprintf("[%d] %s", code, message), geminiErrorRetryable(code, err.Error()), context, err, code, name)
	return withVertexStorageHint(formatted)
}

// formatPromptError classifies an error raised while building the prompt (an
// inline media cap, an unreadable file). As the TS client does, it goes through
// the model's error formatter instead of surfacing as an unclassified failure.
// Errors already classified, and the caller's own cancellation, pass through.
func (d *VertexAIDriver) formatPromptError(ctx context.Context, err error, model string, operation string) error {
	if err == nil || ctx.Err() != nil {
		return err
	}
	var llumiErr *LlumiverseError
	if errors.As(err, &llumiErr) {
		return err
	}
	return d.formatError(err, model, operation)
}

// vertexPromptDriver is the driver handed to the shared execute and stream flow:
// it classifies a failure to build the prompt (an inline media limit, an
// unreadable file) with the Vertex error formatter for that operation, before
// the shared flow would wrap it generically.
type vertexPromptDriver struct {
	*VertexAIDriver
	operation string
}

// CreatePrompt builds the prompt and classifies its error for the operation.
func (p vertexPromptDriver) CreatePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (any, error) {
	prompt, err := p.VertexAIDriver.CreatePrompt(ctx, segments, options)
	return prompt, p.formatPromptError(ctx, err, options.Model, p.operation)
}

// withVertexStorageHint appends the bucket-permission hint to a 403 that names
// Cloud Storage, as the TS driver does for every Vertex model.
func withVertexStorageHint(err *LlumiverseError) *LlumiverseError {
	if err.Code == 403 && vertexStorageDenialPattern.MatchString(err.Message) {
		err.Message += vertexStorageHint
	}
	return err
}
