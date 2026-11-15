package vertexai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"golang.org/x/oauth2"
)

// fakeStatusError mimics the core HTTP status error ("HTTP <code>: <body>").
type fakeStatusError struct {
	code int
	body string
}

func (e *fakeStatusError) Error() string   { return fmt.Sprintf("HTTP %d: %s", e.code, e.body) }
func (e *fakeStatusError) StatusCode() int { return e.code }

// googleError builds an AIP-193 error body the way Vertex returns it.
func googleError(code int, status string, message string) error {
	return &fakeStatusError{code: code, body: fmt.Sprintf(`{"error":{"code":%d,"message":%q,"status":%q}}`, code, message, status)}
}

// plainGoogleError mirrors the TS tests, whose errors carry only a status and a message.
func plainGoogleError(code int, message string) error {
	return &fakeStatusError{code: code, body: fmt.Sprintf(`{"error":{"code":%d,"message":%q}}`, code, message)}
}

func asLlumiverseError(t *testing.T, err error) *LlumiverseError {
	t.Helper()
	var llumiErr *LlumiverseError
	if !errors.As(err, &llumiErr) {
		t.Fatalf("error = %#v, want *LlumiverseError", err)
	}
	return llumiErr
}

func retryableString(v *bool) string {
	if v == nil {
		return "unknown"
	}
	return fmt.Sprint(*v)
}

// TestGeminiErrorClassification mirrors gemini-error-handling.test.ts.
func TestGeminiErrorClassification(t *testing.T) {
	t.Parallel()

	urlFetcher := func(status string) string {
		return "Cannot fetch content from the provided URL. Please ensure the URL is valid and accessible by Vertex AI. " +
			"Vertex AI respects robots.txt rules, so confirm the URL is allowed to be crawled. Status: " + status
	}
	cases := []struct {
		name      string
		err       error
		code      int
		retryable string
		errName   string
	}{
		{"invalid argument", plainGoogleError(400, "INVALID_ARGUMENT: Invalid value for temperature. Must be between 0 and 2."), 400, "false", "INVALID_ARGUMENT"},
		{"unauthenticated", plainGoogleError(401, "UNAUTHENTICATED: Request had invalid authentication credentials."), 401, "false", "UNAUTHENTICATED"},
		{"permission denied", plainGoogleError(403, "PERMISSION_DENIED: The caller does not have permission to execute the specified operation."), 403, "false", "PERMISSION_DENIED"},
		{"not found", plainGoogleError(404, "NOT_FOUND: Requested entity was not found."), 404, "false", "NOT_FOUND"},
		{"499 aborted", plainGoogleError(499, "This operation was aborted"), 499, "true", vertexErrorName},
		{"499 cancelled", plainGoogleError(499, "Client Closed Request: The operation was cancelled."), 499, "true", vertexErrorName},
		{"deadline under 400", plainGoogleError(400, "DEADLINE_EXCEEDED: Deadline expired before operation could complete"), 400, "true", "DEADLINE_EXCEEDED"},
		{"resource exhausted", plainGoogleError(429, "RESOURCE_EXHAUSTED: Quota exceeded for quota metric"), 429, "true", "RESOURCE_EXHAUSTED"},
		{"internal", plainGoogleError(500, "INTERNAL: Internal server error"), 500, "true", "INTERNAL"},
		{"bad gateway", plainGoogleError(502, "Bad gateway"), 502, "true", vertexErrorName},
		{"unavailable", plainGoogleError(503, "UNAVAILABLE: The service is currently unavailable."), 503, "true", "UNAVAILABLE"},
		{"gateway timeout", plainGoogleError(504, "DEADLINE_EXCEEDED: Request timeout"), 504, "true", "DEADLINE_EXCEEDED"},
		{"request timeout", plainGoogleError(408, "Request timeout"), 408, "true", vertexErrorName},
		{"bracket name", plainGoogleError(400, "[INVALID_ARGUMENT] Invalid parameter"), 400, "false", "INVALID_ARGUMENT"},
		{"error suffix name", plainGoogleError(400, "ValidationError: Invalid input"), 400, "false", "ValidationError"},
		{"no extractable name", plainGoogleError(500, "Something went wrong"), 500, "true", vertexErrorName},
		{"google status wins", googleError(429, "RESOURCE_EXHAUSTED", "Quota exceeded"), 429, "true", "RESOURCE_EXHAUSTED"},
		{"cancelled status", googleError(499, "CANCELLED", "The request was dropped"), 499, "true", "CANCELLED"},
		{"client throttled", plainGoogleError(400, urlFetcher("URL_REJECTED-REJECTED_CLIENT_THROTTLED")), 400, "true", vertexErrorName},
		{"proxy throttled", plainGoogleError(400, urlFetcher("URL_REJECTED-REJECTED_PROXY_THROTTLED")), 400, "true", vertexErrorName},
		{"rate limited", plainGoogleError(400, urlFetcher("URL_REJECTED-REJECTED_RATE_LIMITED")), 400, "true", vertexErrorName},
		{"too many pending", plainGoogleError(400, urlFetcher("URL_REJECTED-REJECTED_FC_TOO_MANY_PENDING")), 400, "true", vertexErrorName},
		{"no rpc buffers", plainGoogleError(400, urlFetcher("URL_REJECTED-REJECTED_NO_RPC_BUFFERS")), 400, "true", vertexErrorName},
		{"fetcher timeout", plainGoogleError(400, urlFetcher("URL_REJECTED-REJECTED_FC_TIMEOUT")), 400, "true", vertexErrorName},
		{"robots denied", plainGoogleError(400, urlFetcher("URL_REJECTED-REJECTED_ROBOTED")), 400, "false", vertexErrorName},
		{"invalid_grant over 503", plainGoogleError(503, "Error code invalid_grant: Error connecting to the given credential's issuer."), 503, "false", "invalid_grant"},
		{"invalid_grant 400", plainGoogleError(400, "Error code invalid_grant: Error connecting to the given credential's issuer."), 400, "false", "invalid_grant"},
		{"other 4xx", plainGoogleError(405, "Method not allowed"), 405, "false", vertexErrorName},
		{"other 5xx", plainGoogleError(599, "odd"), 599, "true", vertexErrorName},
		{"non-JSON body", &fakeStatusError{code: 503, body: "overloaded"}, 503, "true", vertexErrorName},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := asLlumiverseError(t, formatGeminiError(tc.err, "gemini-2.5-flash", "execute"))
			if got.Code != tc.code || retryableString(got.Retryable) != tc.retryable || got.Name != tc.errName {
				t.Fatalf("code=%d retryable=%s name=%q, want %d %s %q", got.Code, retryableString(got.Retryable), got.Name, tc.code, tc.retryable, tc.errName)
			}
			if got.Context.Provider != ProviderVertexAI || got.Context.Model != "gemini-2.5-flash" || got.Context.Operation != "execute" {
				t.Fatalf("context = %#v", got.Context)
			}
			if !strings.HasPrefix(got.Message, fmt.Sprintf("[vertexai] [%d] ", tc.code)) {
				t.Fatalf("message = %q", got.Message)
			}
			if !errors.Is(got, tc.err) {
				t.Fatal("original error not preserved")
			}
		})
	}
}

func TestGeminiErrorClassificationWithoutStatus(t *testing.T) {
	t.Parallel()

	cases := []struct {
		err       error
		retryable string
	}{
		{errors.New("Regular error"), "unknown"},
		{errors.New("request aborted by transport"), "true"},
		{fmt.Errorf("post: %w", context.DeadlineExceeded), "true"},
		{errors.New("invalid_grant: Error connecting to the given credential's issuer."), "false"},
		{errors.New("rate limit reached"), "true"},
		// A caller's own cancellation is not a provider failure.
		{context.Canceled, "unknown"},
	}
	for _, tc := range cases {
		got := asLlumiverseError(t, formatGeminiError(tc.err, "gemini-2.5-flash", "execute"))
		if got.Code != 0 || retryableString(got.Retryable) != tc.retryable {
			t.Fatalf("%v: code=%d retryable=%s, want %s", tc.err, got.Code, retryableString(got.Retryable), tc.retryable)
		}
	}
}

// TestGeminiInvalidGrantTokenErrorIsNotRetryable covers a token refresh that
// fails with invalid_grant before any Gemini request is sent.
func TestGeminiInvalidGrantTokenErrorIsNotRetryable(t *testing.T) {
	t.Parallel()

	tokenErr := &oauth2.RetrieveError{
		Response:  &http.Response{StatusCode: http.StatusBadRequest},
		Body:      []byte(`{"error":"invalid_grant","error_description":"Error connecting to the given credential's issuer."}`),
		ErrorCode: "invalid_grant",
	}
	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     "http://127.0.0.1:1",
		TokenSource: failingTokenSource{err: tokenErr},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{Model: "gemini-2.5-flash"})
	got := asLlumiverseError(t, err)
	if retryableString(got.Retryable) != "false" || got.Code != http.StatusBadRequest || got.Name != "invalid_grant" {
		t.Fatalf("error = %#v", got)
	}
}

type failingTokenSource struct{ err error }

func (s failingTokenSource) Token() (*oauth2.Token, error) { return nil, s.err }

// TestVertexStorageDenialAddsBucketHint mirrors the TS storage-permission clarification.
func TestVertexStorageDenialAddsBucketHint(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"error":{"code":403,"message":"Permission 'storage.objects.get' denied on resource gs://bucket/file.pdf","status":"PERMISSION_DENIED"}}`))
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, model := range []string{"gemini-2.5-flash", "claude-sonnet-4-5"} {
		_, err = driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{Model: model})
		got := asLlumiverseError(t, err)
		if got.Code != http.StatusForbidden || retryableString(got.Retryable) != "false" || !strings.HasSuffix(got.Message, vertexStorageHint) {
			t.Fatalf("%s: error = %#v", model, got)
		}
	}
	plain := asLlumiverseError(t, formatGeminiError(plainGoogleError(403, "PERMISSION_DENIED: no access to the model"), "gemini-2.5-flash", "execute"))
	if strings.Contains(plain.Message, vertexStorageHint) {
		t.Fatalf("non-storage denial got the bucket hint: %q", plain.Message)
	}
}
