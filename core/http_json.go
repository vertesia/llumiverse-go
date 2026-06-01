package core

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"path"
	"strings"
)

// httpStatusError represents a non-2xx HTTP response, carrying the status code
// and response body. It implements statusCoder so the retryability classifier
// can recover the code from a wrapped error.
type httpStatusError struct {
	statusCode int
	body       string
}

func (e *httpStatusError) Error() string {
	if e.body == "" {
		return fmt.Sprintf("HTTP %d", e.statusCode)
	}
	return fmt.Sprintf("HTTP %d: %s", e.statusCode, e.body)
}

func (e *httpStatusError) StatusCode() int { return e.statusCode }

// doJSON performs a JSON request/response round trip: it marshals body (when
// non-nil), sets the supplied headers, executes the request, returns an
// httpStatusError for non-2xx responses, and unmarshals the response body into
// out when both are present.
func doJSON(ctx context.Context, client *http.Client, method string, endpoint string, headers map[string]string, body any, out any) error {
	var reqBody io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reqBody = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, endpoint, reqBody)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	res, err := client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = res.Body.Close() }()
	resBody, err := io.ReadAll(res.Body)
	if err != nil {
		return err
	}
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return &httpStatusError{statusCode: res.StatusCode, body: string(resBody)}
	}
	if out == nil || len(resBody) == 0 {
		return nil
	}
	return json.Unmarshal(resBody, out)
}

// joinEndpoint appends path elements to a base URL, preserving the base's scheme
// and host while cleaning the joined path.
func joinEndpoint(base string, elems ...string) (string, error) {
	u, err := url.Parse(strings.TrimRight(base, "/") + "/")
	if err != nil {
		return "", err
	}
	pieces := []string{u.Path}
	pieces = append(pieces, elems...)
	u.Path = path.Join(pieces...)
	return u.String(), nil
}

// dataSourceToDataURL reads a DataSource fully and encodes it as a base64 data:
// URL, resolving the MIME type from the source, then its file extension, and
// finally falling back to application/octet-stream.
func dataSourceToDataURL(ctx context.Context, ds DataSource) (string, error) {
	rc, err := ds.Open(ctx)
	if err != nil {
		return "", err
	}
	defer func() { _ = rc.Close() }()
	bytes, err := io.ReadAll(rc)
	if err != nil {
		return "", err
	}
	mimeType := ds.MIMEType()
	if mimeType == "" {
		mimeType = mime.TypeByExtension(path.Ext(ds.Name()))
	}
	if mimeType == "" {
		mimeType = "application/octet-stream"
	}
	return "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(bytes), nil
}

// dataSourceToBase64 reads a DataSource fully and returns its standard base64
// encoding without any data-URL prefix.
func dataSourceToBase64(ctx context.Context, ds DataSource) (string, error) {
	rc, err := ds.Open(ctx)
	if err != nil {
		return "", err
	}
	defer func() { _ = rc.Close() }()
	bytes, err := io.ReadAll(rc)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(bytes), nil
}

// sseEvent is one parsed Server-Sent Events message, with its optional event
// type and the joined data payload.
type sseEvent struct {
	Type string
	Data string
}

// scanSSE parses a Server-Sent Events stream from body, invoking onEvent for each
// event delimited by a blank line. It joins multiple data: lines with newlines,
// ignores comment lines, and closes body when done.
func scanSSE(body io.ReadCloser, onEvent func(sseEvent) error) error {
	defer func() { _ = body.Close() }()
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var eventType string
	var dataLines []string
	flush := func() error {
		if len(dataLines) == 0 {
			eventType = ""
			return nil
		}
		event := sseEvent{Type: eventType, Data: strings.Join(dataLines, "\n")}
		eventType = ""
		dataLines = nil
		return onEvent(event)
	}
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		field, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			eventType = value
		case "data":
			dataLines = append(dataLines, value)
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return flush()
}

// postSSE issues a JSON POST that requests an event stream and returns the
// response body for scanSSE to consume. Non-2xx responses are returned as an
// httpStatusError and the body is closed.
func postSSE(ctx context.Context, client *http.Client, endpoint string, headers map[string]string, body any) (io.ReadCloser, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(encoded))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		defer func() { _ = res.Body.Close() }()
		resBody, readErr := io.ReadAll(res.Body)
		if readErr != nil {
			return nil, errors.Join(&httpStatusError{statusCode: res.StatusCode}, readErr)
		}
		return nil, &httpStatusError{statusCode: res.StatusCode, body: string(resBody)}
	}
	return res.Body, nil
}

// optionFloat reads key from a provider option map as a *float64, coercing the
// common numeric types (including json.Number), or nil when absent or not numeric.
func optionFloat(options map[string]any, key string) *float64 {
	if options == nil {
		return nil
	}
	switch v := options[key].(type) {
	case float64:
		return &v
	case float32:
		out := float64(v)
		return &out
	case int:
		out := float64(v)
		return &out
	case int64:
		out := float64(v)
		return &out
	case json.Number:
		if out, err := v.Float64(); err == nil {
			return &out
		}
	}
	return nil
}

// optionInt reads key from a provider option map as a *int, coercing the common
// numeric types (including json.Number), or nil when absent or not numeric.
func optionInt(options map[string]any, key string) *int {
	if options == nil {
		return nil
	}
	switch v := options[key].(type) {
	case int:
		return &v
	case int64:
		out := int(v)
		return &out
	case float64:
		out := int(v)
		return &out
	case json.Number:
		if i, err := v.Int64(); err == nil {
			out := int(i)
			return &out
		}
	}
	return nil
}

// optionBool reads key from a provider option map as a bool, returning false when
// absent or not a bool.
func optionBool(options map[string]any, key string) bool {
	if options == nil {
		return false
	}
	v, ok := options[key].(bool)
	return ok && v
}

// optionStringSlice reads key from a provider option map as a []string, accepting
// either a native []string or a []any of strings, or nil when absent or unusable.
func optionStringSlice(options map[string]any, key string) []string {
	if options == nil {
		return nil
	}
	switch v := options[key].(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}
