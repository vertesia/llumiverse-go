// Package fakehttp is a recording HTTP server for the offline driver tests: it answers each path with a canned
// response and keeps the requests it received.
package fakehttp

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

// Request is a request the server received.
type Request struct {
	Method string
	Path   string
	Query  string
	Header http.Header
	Body   map[string]any
}

// Response is a canned answer. A Body starting with "data:" is sent as an event stream.
type Response struct {
	Status int
	Body   string
}

// Server records requests and answers them from Routes, keyed by "/path?query", "METHOD /path" or "/path".
type Server struct {
	*httptest.Server
	mu       sync.Mutex
	requests []Request
	Routes   map[string]Response
}

// New starts a server answering from routes; it is closed when the test ends.
func New(t *testing.T, routes map[string]Response) *Server {
	t.Helper()
	s := &Server{Routes: routes}
	s.Server = httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(s.Close)
	return s
}

func (s *Server) serve(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	request := Request{Method: r.Method, Path: r.URL.Path, Query: r.URL.RawQuery, Header: r.Header.Clone()}
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &request.Body)
	}
	s.mu.Lock()
	s.requests = append(s.requests, request)
	var response Response
	ok := false
	keys := []string{r.Method + " " + r.URL.Path, r.URL.Path}
	if r.URL.RawQuery != "" {
		keys = append([]string{r.URL.Path + "?" + r.URL.RawQuery}, keys...)
	}
	for _, key := range keys {
		if response, ok = s.Routes[key]; ok {
			break
		}
	}
	s.mu.Unlock()
	if !ok {
		http.Error(w, `{"error":{"message":"no route"}}`, http.StatusNotFound)
		return
	}
	status := response.Status
	if status == 0 {
		status = http.StatusOK
	}
	if strings.HasPrefix(response.Body, "data:") {
		w.Header().Set("Content-Type", "text/event-stream")
	} else {
		w.Header().Set("Content-Type", "application/json")
	}
	w.WriteHeader(status)
	_, _ = io.WriteString(w, response.Body)
}

// Requests returns the requests received so far.
func (s *Server) Requests() []Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Request(nil), s.requests...)
}

// Last returns the last request received to path; the test fails when there is none.
func (s *Server) Last(t *testing.T, path string) Request {
	t.Helper()
	requests := s.Requests()
	for i := len(requests) - 1; i >= 0; i-- {
		if requests[i].Path == path {
			return requests[i]
		}
	}
	t.Fatalf("no request to %s; got %+v", path, requests)
	return Request{}
}

// SSE joins JSON events into an event stream body ending with [DONE].
func SSE(events ...string) string {
	var out strings.Builder
	for _, event := range events {
		out.WriteString("data: ")
		out.WriteString(event)
		out.WriteString("\n\n")
	}
	out.WriteString("data: [DONE]\n\n")
	return out.String()
}

// Drain reads a stream to its end and returns its chunks and final completion.
func Drain(t *testing.T, stream common.CompletionStream) ([]common.CompletionChunk, *common.ExecutionResponse) {
	t.Helper()
	defer func() { _ = stream.Close() }()
	var chunks []common.CompletionChunk
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return chunks, stream.Completion()
		}
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		chunks = append(chunks, chunk)
	}
}
