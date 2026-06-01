package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"
)

// executeFunc is the provider-specific completion call invoked by
// executeWithPrompt once the prompt has been formatted.
type executeFunc func(ctx context.Context, prompt any, options ExecutionOptions) (*Completion, error)

// streamFunc is the provider-specific streaming call invoked by streamWithPrompt
// once the prompt has been formatted.
type streamFunc func(ctx context.Context, prompt any, options ExecutionOptions) (CompletionStream, error)

// executeWithPrompt is the shared non-streaming lifecycle wrapper: it validates
// options, calls the driver's CreatePrompt, invokes the provider execute call,
// wraps non-LlumiverseError failures, optionally normalizes a JSON result, and
// returns the assembled ExecutionResponse with timing.
func executeWithPrompt(ctx context.Context, driver Driver, segments []PromptSegment, options ExecutionOptions, fn executeFunc) (*ExecutionResponse, error) {
	// Keep error wrapping and JSON-result normalization shared so provider
	// drivers only own provider request/response translation.
	if options.Model == "" {
		return nil, errors.New("model is required")
	}
	start := time.Now()
	prompt, err := driver.CreatePrompt(ctx, segments, options)
	if err != nil {
		return nil, err
	}
	completion, err := fn(ctx, prompt, options)
	if err != nil {
		var llumiErr *LlumiverseError
		if errors.As(err, &llumiErr) {
			return nil, err
		}
		return nil, formatGenericError(err, LlumiverseErrorContext{
			Provider:  driver.Provider(),
			Model:     options.Model,
			Operation: "execute",
		})
	}
	if completion == nil {
		return nil, errors.New("driver returned nil completion")
	}
	if options.ResultSchema != nil && completion.Error == nil && len(completion.ToolUse) == 0 {
		normalizeJSONResult(completion)
	}
	return &ExecutionResponse{
		Completion:    *completion,
		Prompt:        prompt,
		ExecutionTime: time.Since(start),
	}, nil
}

// streamWithPrompt is the streaming counterpart of executeWithPrompt: it shares
// option validation, prompt formatting, and error wrapping, and ensures the
// stream's accumulating completion carries the formatted prompt.
func streamWithPrompt(ctx context.Context, driver Driver, segments []PromptSegment, options ExecutionOptions, fn streamFunc) (CompletionStream, error) {
	// Streaming drivers still use CreatePrompt so prompt formatting errors match
	// non-streaming execution.
	if options.Model == "" {
		return nil, errors.New("model is required")
	}
	prompt, err := driver.CreatePrompt(ctx, segments, options)
	if err != nil {
		return nil, err
	}
	stream, err := fn(ctx, prompt, options)
	if err != nil {
		var llumiErr *LlumiverseError
		if errors.As(err, &llumiErr) {
			return nil, err
		}
		return nil, formatGenericError(err, LlumiverseErrorContext{
			Provider:  driver.Provider(),
			Model:     options.Model,
			Operation: "stream",
		})
	}
	if stream == nil {
		return nil, errors.New("driver returned nil stream")
	}
	if completion := stream.Completion(); completion != nil && completion.Prompt == nil {
		completion.Prompt = prompt
	}
	return stream, nil
}

// formatGenericError wraps a provider error in a LlumiverseError, attaching the
// request context, an extracted HTTP status/type name, and a retryability
// classification derived from the status and message.
func formatGenericError(err error, context LlumiverseErrorContext) error {
	if err == nil {
		return nil
	}
	code, name := errorStatusAndName(err)
	return NewLlumiverseError(err.Error(), retryableFromStatusAndMessage(code, err.Error()), context, err, code, name)
}

// statusCoder is implemented by errors that carry an HTTP status code, such as
// httpStatusError, so the status can be recovered for retryability decisions.
type statusCoder interface {
	StatusCode() int
}

// errorStatusAndName returns the error's HTTP status code (0 if none) and its
// concrete Go type name.
func errorStatusAndName(err error) (int, string) {
	var sc statusCoder
	if errors.As(err, &sc) {
		return sc.StatusCode(), fmt.Sprintf("%T", err)
	}
	return 0, fmt.Sprintf("%T", err)
}

// retryableFromStatusAndMessage classifies whether an error is worth retrying.
func retryableFromStatusAndMessage(code int, msg string) *bool {
	// Match the TS client's retryability policy: rate limits, timeouts,
	// throttling, and 5xx responses are retryable; known 4xx client errors are
	// not; unknown status leaves Retryable nil for callers to decide.
	lower := strings.ToLower(msg)
	if strings.Contains(lower, "rate") && strings.Contains(lower, "limit") {
		return BoolPtr(true)
	}
	if strings.Contains(lower, "timeout") || strings.Contains(lower, "timed out") {
		return BoolPtr(true)
	}
	if strings.Contains(lower, "resource") && strings.Contains(lower, "exhaust") {
		return BoolPtr(true)
	}
	if strings.Contains(lower, "overload") || strings.Contains(lower, "throttl") {
		return BoolPtr(true)
	}
	if code == 408 || code == 429 || code == 529 || code >= 500 {
		return BoolPtr(true)
	}
	if code >= 400 {
		return BoolPtr(false)
	}
	return nil
}

// normalizeJSONResult upgrades the first text result that parses as valid JSON
// into a ResultTypeJSON result, used when a ResultSchema was requested but the
// provider returned the structured output as plain text.
func normalizeJSONResult(completion *Completion) {
	for i := range completion.Result {
		if completion.Result[i].Type != ResultTypeText {
			continue
		}
		text, ok := completion.Result[i].Value.(string)
		if !ok {
			continue
		}
		var parsed any
		if parseJSONResultText(text, &parsed) {
			completion.Result[i] = CompletionResult{Type: ResultTypeJSON, Value: parsed}
			return
		}
	}
}

func parseJSONResultText(text string, out *any) bool {
	for _, candidate := range jsonResultCandidates(text) {
		if json.Unmarshal([]byte(candidate), out) == nil {
			return true
		}
	}
	return false
}

func jsonResultCandidates(text string) []string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil
	}
	candidates := []string{trimmed}
	if fenced := markdownJSONFenceBody(trimmed); fenced != "" {
		candidates = append(candidates, fenced)
	}
	if bounded := boundedJSONObject(trimmed); bounded != "" {
		candidates = append(candidates, bounded)
	}
	return candidates
}

func markdownJSONFenceBody(text string) string {
	if !strings.HasPrefix(text, "```") {
		return ""
	}
	afterFence := strings.TrimPrefix(text, "```")
	if idx := strings.IndexByte(afterFence, '\n'); idx >= 0 {
		afterFence = afterFence[idx+1:]
	}
	if idx := strings.LastIndex(afterFence, "```"); idx >= 0 {
		afterFence = afterFence[:idx]
	}
	return strings.TrimSpace(afterFence)
}

func boundedJSONObject(text string) string {
	start := strings.IndexAny(text, "{[")
	if start == -1 {
		return ""
	}
	endObject := strings.LastIndexAny(text, "}]")
	if endObject <= start {
		return ""
	}
	return strings.TrimSpace(text[start : endObject+1])
}

// readAllString reads rc to completion and closes it, returning the body as a string.
func readAllString(rc io.ReadCloser) (string, error) {
	defer func() { _ = rc.Close() }()
	bytes, err := io.ReadAll(rc)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

// truncateForConversation shortens value to maxLen, appending an ellipsis when
// it is cut. A non-positive maxLen disables truncation.
func truncateForConversation(value string, maxLen int) string {
	if maxLen <= 0 || len(value) <= maxLen {
		return value
	}
	return value[:maxLen] + "..."
}

// removeNilMapValues returns a copy of in with nil entries dropped, recursing into
// nested maps and omitting any that become empty, so providers receive only set fields.
func removeNilMapValues(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for key, value := range in {
		if value == nil {
			continue
		}
		if nested, ok := value.(map[string]any); ok {
			value = removeNilMapValues(nested)
			if len(value.(map[string]any)) == 0 {
				continue
			}
		}
		out[key] = value
	}
	return out
}

// safeJSONParse decodes a JSON string, returning the original string on parse
// failure and an empty object for empty input.
func safeJSONParse(value string) any {
	if value == "" {
		return map[string]any{}
	}
	var parsed any
	if json.Unmarshal([]byte(value), &parsed) == nil {
		return parsed
	}
	return value
}

// completionResultsToText flattens text and JSON completion results into a single
// string for retained conversation history.
func completionResultsToText(results []CompletionResult) string {
	var b strings.Builder
	for _, result := range results {
		switch result.Type {
		case ResultTypeText:
			b.WriteString(toString(result.Value))
		case ResultTypeJSON:
			if s, ok := result.Value.(string); ok {
				b.WriteString(s)
			} else {
				data, _ := json.Marshal(result.Value)
				b.Write(data)
			}
		}
	}
	return b.String()
}
