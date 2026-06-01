package core

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestDoJSONSuccessAndStatusError(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/ok":
			if r.Method != http.MethodPost {
				t.Fatalf("method = %s", r.Method)
			}
			if r.Header.Get("X-Test") != "yes" || r.Header.Get("Content-Type") != "application/json" {
				t.Fatalf("headers = %#v", r.Header)
			}
			var got map[string]string
			if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
				t.Fatal(err)
			}
			if got["in"] != "ok" {
				t.Fatalf("body = %#v", got)
			}
			_, _ = w.Write([]byte(`{"out":"yes"}`))
		case "/fail":
			http.Error(w, "overloaded", http.StatusServiceUnavailable)
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer server.Close()

	var out map[string]string
	if err := DoJSON(context.Background(), server.Client(), http.MethodPost, server.URL+"/ok", map[string]string{"X-Test": "yes"}, map[string]string{"in": "ok"}, &out); err != nil {
		t.Fatal(err)
	}
	if out["out"] != "yes" {
		t.Fatalf("out = %#v", out)
	}
	err := DoJSON(context.Background(), server.Client(), http.MethodGet, server.URL+"/fail", nil, nil, nil)
	if err == nil {
		t.Fatal("expected status error")
	}
	var status *httpStatusError
	if !errors.As(err, &status) || status.StatusCode() != http.StatusServiceUnavailable || !strings.Contains(status.Error(), "overloaded") {
		t.Fatalf("status err = %#v", err)
	}
	code, _ := ErrorStatusAndName(err)
	if code != http.StatusServiceUnavailable {
		t.Fatalf("code = %d", code)
	}
	if got := (&httpStatusError{statusCode: http.StatusTeapot}).Error(); got != "HTTP 418" {
		t.Fatalf("empty status error = %q", got)
	}
	if err := DoJSON(context.Background(), server.Client(), http.MethodPost, server.URL+"/ok", nil, func() {}, nil); err == nil {
		t.Fatal("expected JSON marshal error")
	}
}

func TestEndpointAndDataSourceHelpers(t *testing.T) {
	t.Parallel()

	endpoint, err := JoinEndpoint("https://example.com/base/", "projects", "p1", "models/gemini:generateContent")
	if err != nil {
		t.Fatal(err)
	}
	if endpoint != "https://example.com/base/projects/p1/models/gemini:generateContent" {
		t.Fatalf("endpoint = %q", endpoint)
	}
	ds := BytesDataSource{FileName: "image.png", Data: []byte("img")}
	dataURL, err := DataSourceToDataURL(context.Background(), ds)
	if err != nil {
		t.Fatal(err)
	}
	if dataURL != "data:image/png;base64,aW1n" {
		t.Fatalf("dataURL = %q", dataURL)
	}
	encoded, err := DataSourceToBase64(context.Background(), ds)
	if err != nil {
		t.Fatal(err)
	}
	if encoded != "aW1n" {
		t.Fatalf("base64 = %q", encoded)
	}
	text, err := ReadAllStringFromDataSource(context.Background(), BytesDataSource{Data: []byte("hello")})
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello" {
		t.Fatalf("text = %q", text)
	}
	text, err = ReadAllString(io.NopCloser(strings.NewReader("direct")))
	if err != nil {
		t.Fatal(err)
	}
	if text != "direct" {
		t.Fatalf("ReadAllString = %q", text)
	}
	fallbackURL, err := DataSourceToDataURL(context.Background(), BytesDataSource{Data: []byte("x")})
	if err != nil {
		t.Fatal(err)
	}
	if fallbackURL != "data:application/octet-stream;base64,eA==" {
		t.Fatalf("fallback dataURL = %q", fallbackURL)
	}
	byExtension, err := DataSourceToDataURL(context.Background(), BytesDataSource{FileName: "image.webp", Data: []byte("x")})
	if err != nil {
		t.Fatal(err)
	}
	if byExtension != "data:image/webp;base64,eA==" {
		t.Fatalf("extension dataURL = %q", byExtension)
	}
	if _, err := DataSourceToBase64(context.Background(), failingDataSource{err: errors.New("open failed")}); err == nil {
		t.Fatal("expected data source open error")
	}
}

func TestScanSSEAndPostSSE(t *testing.T) {
	t.Parallel()

	var events []SSEEvent
	err := ScanSSE(io.NopCloser(strings.NewReader(": ignored\nevent: delta\ndata: one\ndata: two\n\ndata: tail")), func(event SSEEvent) error {
		events = append(events, event)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 || events[0].Type != "delta" || events[0].Data != "one\ntwo" || events[1].Data != "tail" {
		t.Fatalf("events = %#v", events)
	}
	expectedErr := errors.New("stop")
	err = ScanSSE(io.NopCloser(strings.NewReader("data: one\n\n")), func(SSEEvent) error {
		return expectedErr
	})
	if !errors.Is(err, expectedErr) {
		t.Fatalf("callback err = %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") != "text/event-stream" {
			t.Fatalf("Accept = %q", r.Header.Get("Accept"))
		}
		if r.URL.Path == "/bad" {
			http.Error(w, "bad", http.StatusBadGateway)
			return
		}
		_, _ = w.Write([]byte("data: ok\n\n"))
	}))
	defer server.Close()
	body, err := PostSSE(context.Background(), server.Client(), server.URL+"/sse", nil, map[string]string{"hello": "world"})
	if err != nil {
		t.Fatal(err)
	}
	got, err := io.ReadAll(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	if string(got) != "data: ok\n\n" {
		t.Fatalf("SSE body = %q", string(got))
	}
	if _, err := PostSSE(context.Background(), server.Client(), server.URL+"/bad", nil, map[string]string{}); err == nil {
		t.Fatal("expected PostSSE status error")
	}
}

func TestOptionAndValueHelpers(t *testing.T) {
	t.Parallel()

	options := map[string]any{
		"float":         json.Number("1.5"),
		"float32":       float32(2.5),
		"float64":       3.5,
		"floatBad":      json.Number("not-a-number"),
		"int":           7.9,
		"intNative":     8,
		"int64":         int64(9),
		"intJSON":       json.Number("10"),
		"intBad":        json.Number("10.2"),
		"bool":          true,
		"strings":       []any{"a", 2, "b"},
		"nativeStrings": []string{"x", "y"},
		"alt":           "value",
		"notString":     123,
	}
	if v := OptionFloat(options, "float"); v == nil || *v != 1.5 {
		t.Fatalf("OptionFloat = %#v", v)
	}
	if v := OptionFloat(options, "float32"); v == nil || *v != 2.5 {
		t.Fatalf("OptionFloat float32 = %#v", v)
	}
	if v := OptionFloat(options, "float64"); v == nil || *v != 3.5 {
		t.Fatalf("OptionFloat float64 = %#v", v)
	}
	if v := OptionFloat(map[string]any{"i": 4}, "i"); v == nil || *v != 4 {
		t.Fatalf("OptionFloat int = %#v", v)
	}
	if v := OptionFloat(map[string]any{"i": int64(5)}, "i"); v == nil || *v != 5 {
		t.Fatalf("OptionFloat int64 = %#v", v)
	}
	if v := OptionFloat(options, "floatBad"); v != nil {
		t.Fatalf("OptionFloat bad = %#v", v)
	}
	if v := OptionFloat(nil, "float"); v != nil {
		t.Fatalf("OptionFloat nil = %#v", v)
	}
	if v := OptionInt(options, "int"); v == nil || *v != 7 {
		t.Fatalf("OptionInt = %#v", v)
	}
	if v := OptionInt(options, "intNative"); v == nil || *v != 8 {
		t.Fatalf("OptionInt native = %#v", v)
	}
	if v := OptionInt(options, "int64"); v == nil || *v != 9 {
		t.Fatalf("OptionInt int64 = %#v", v)
	}
	if v := OptionInt(options, "intJSON"); v == nil || *v != 10 {
		t.Fatalf("OptionInt json = %#v", v)
	}
	if v := OptionInt(options, "intBad"); v != nil {
		t.Fatalf("OptionInt bad = %#v", v)
	}
	if v := OptionInt(nil, "int"); v != nil {
		t.Fatalf("OptionInt nil = %#v", v)
	}
	if !OptionBool(options, "bool") || OptionBool(options, "missing") {
		t.Fatal("OptionBool returned unexpected value")
	}
	if got := OptionStringSlice(options, "strings"); !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("OptionStringSlice = %#v", got)
	}
	if got := OptionStringSlice(options, "nativeStrings"); !reflect.DeepEqual(got, []string{"x", "y"}) {
		t.Fatalf("OptionStringSlice native = %#v", got)
	}
	if got := OptionString(options, "missing", "alt"); got != "value" {
		t.Fatalf("OptionString = %q", got)
	}
	if got := OptionString(nil, "alt"); got != "" {
		t.Fatalf("OptionString nil = %q", got)
	}
	if got := OptionString(options, "notString"); got != "" {
		t.Fatalf("OptionString non-string = %q", got)
	}
	if got := SafeJSONParse(`{"x":1}`); got.(map[string]any)["x"] != float64(1) {
		t.Fatalf("SafeJSONParse object = %#v", got)
	}
	if got := SafeJSONParse("not json"); got != "not json" {
		t.Fatalf("SafeJSONParse string = %#v", got)
	}
	if got := ToString(map[string]any{"x": 1}); got != `{"x":1}` {
		t.Fatalf("ToString = %q", got)
	}
	if got := ToolInputString(map[string]any{"x": 1}); got != `{"x":1}` {
		t.Fatalf("ToolInputString = %q", got)
	}
	if got := ToolInputString(nil); got != "" {
		t.Fatalf("ToolInputString nil = %q", got)
	}
	if got := ToolInputString("raw"); got != "raw" {
		t.Fatalf("ToolInputString string = %q", got)
	}
	if got := ShortResourceName("/projects/p/locations/us/models/gemini"); got != "gemini" {
		t.Fatalf("ShortResourceName = %q", got)
	}
	if got := ShortResourceName("///"); got != "" {
		t.Fatalf("ShortResourceName empty = %q", got)
	}
}

func TestHTTPTimeoutClientBuilder(t *testing.T) {
	t.Parallel()

	base := &http.Client{Transport: http.DefaultTransport}
	if got := NewHTTPClient(base, HTTPTimeoutOptions{}); got != base {
		t.Fatal("NewHTTPClient without timeout should reuse the base client")
	}
	client := NewHTTPClient(base, HTTPTimeoutOptions{
		HeadersTimeout:   2 * time.Second,
		BodyTimeout:      3 * time.Second,
		ConnectTimeout:   4 * time.Second,
		KeepAliveTimeout: 5 * time.Second,
	})
	if client == base {
		t.Fatal("NewHTTPClient should clone when timeouts are set")
	}
	if client.Timeout != 3*time.Second {
		t.Fatalf("client timeout = %s", client.Timeout)
	}
	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport = %T", client.Transport)
	}
	if transport.ResponseHeaderTimeout != 2*time.Second || transport.IdleConnTimeout != 5*time.Second || transport.DialContext == nil {
		t.Fatalf("transport timeouts = %#v", transport)
	}
	if !HasHTTPTimeout(HTTPTimeoutOptions{BodyTimeout: time.Second}) || HasHTTPTimeout(HTTPTimeoutOptions{}) {
		t.Fatal("HasHTTPTimeout returned unexpected result")
	}
	baseWithCustomTransport := &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("unused")
	})}
	custom := NewHTTPClient(baseWithCustomTransport, HTTPTimeoutOptions{HeadersTimeout: time.Second})
	if custom == baseWithCustomTransport || custom.Transport == nil {
		t.Fatalf("custom transport should be reused: %#v", custom.Transport)
	}
	if _, ok := custom.Transport.(roundTripperFunc); !ok {
		t.Fatalf("custom transport type = %T", custom.Transport)
	}
}

type failingDataSource struct {
	err error
}

func (d failingDataSource) Name() string { return "broken.bin" }

func (d failingDataSource) MIMEType() string { return "application/octet-stream" }

func (d failingDataSource) Open(context.Context) (io.ReadCloser, error) { return nil, d.err }

func (d failingDataSource) URL(context.Context) (string, error) { return "", d.err }

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
