package common

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
)

func TestBytesDataSource(t *testing.T) {
	t.Parallel()

	ds := BytesDataSource{
		FileName: "notes.txt",
		MIME:     "text/plain",
		Data:     []byte("hello"),
		URI:      "gs://bucket/notes.txt",
	}
	if ds.Name() != "notes.txt" {
		t.Fatalf("Name() = %q", ds.Name())
	}
	if ds.MIMEType() != "text/plain" {
		t.Fatalf("MIMEType() = %q", ds.MIMEType())
	}
	rc, err := ds.Open(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	body, err := io.ReadAll(rc)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != "hello" {
		t.Fatalf("body = %q", string(body))
	}
	if err := rc.Close(); err != nil {
		t.Fatal(err)
	}
	uri, err := ds.URL(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if uri != "gs://bucket/notes.txt" {
		t.Fatalf("URL() = %q", uri)
	}
	if _, err := (BytesDataSource{}).URL(context.Background()); err == nil {
		t.Fatal("URL() without URI returned nil error")
	}
}

func TestLlumiverseError(t *testing.T) {
	t.Parallel()

	original := errors.New("provider failed")
	retryable := BoolPtr(true)
	err := NewLlumiverseError("provider failed", retryable, LlumiverseErrorContext{
		Provider:  ProviderOpenAI,
		Model:     "gpt-test",
		Operation: "execute",
	}, original, 429, "")
	if err.Error() != "[openai] provider failed" {
		t.Fatalf("Error() = %q", err.Error())
	}
	if err.Name != "LlumiverseError" {
		t.Fatalf("Name = %q", err.Name)
	}
	if !*err.Retryable {
		t.Fatalf("Retryable = %#v", err.Retryable)
	}
	if !errors.Is(err, original) {
		t.Fatal("errors.Is did not find original error")
	}
	prefixed := NewLlumiverseError("[openai] already prefixed", nil, LlumiverseErrorContext{Provider: ProviderOpenAI}, nil, 0, "ProviderError")
	if strings.Count(prefixed.Error(), "[openai]") != 1 {
		t.Fatalf("prefixed Error() = %q", prefixed.Error())
	}
	if prefixed.Name != "ProviderError" {
		t.Fatalf("prefixed Name = %q", prefixed.Name)
	}
	var nilErr *LlumiverseError
	if nilErr.Error() != "" {
		t.Fatalf("nil Error() = %q", nilErr.Error())
	}
	if nilErr.Unwrap() != nil {
		t.Fatal("nil Unwrap() returned non-nil")
	}
	if *BoolPtr(false) {
		t.Fatal("BoolPtr(false) returned true")
	}
}
