package core

import (
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

// countingCloser counts Close calls on a channel stream's provider resource.
type countingCloser struct{ calls atomic.Int32 }

func (c *countingCloser) close() error {
	c.calls.Add(1)
	return nil
}

// Mirrors the TS client's lifecycle tests for an unconsumed stream: the lease releases it.
func TestLeaseCompletionStreamClosesAnUnconsumedStream(t *testing.T) {
	t.Parallel()

	closer := &countingCloser{}
	ch := make(chan StreamItem)
	stream := LeaseCompletionStream(NewChannelStream(ch, &ExecutionResponse{}, closer.close), 10*time.Millisecond)
	deadline := time.Now().Add(2 * time.Second)
	for closer.calls.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if closer.calls.Load() != 1 {
		t.Fatalf("provider stream closed %d times, want 1", closer.calls.Load())
	}
	if _, err := stream.Recv(); !errors.Is(err, ErrCompletionStreamNotConsumed) {
		t.Fatalf("Recv after expiry = %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if closer.calls.Load() != 1 {
		t.Fatalf("provider stream closed %d times after Close, want 1", closer.calls.Load())
	}
}

func TestLeaseCompletionStreamLeavesAConsumedStreamOpen(t *testing.T) {
	t.Parallel()

	closer := &countingCloser{}
	ch := make(chan StreamItem, 1)
	ch <- StreamItem{Chunk: CompletionChunk{Result: []CompletionResult{NewTextResult("first")}}}
	stream := LeaseCompletionStream(NewChannelStream(ch, &ExecutionResponse{}, closer.close), 20*time.Millisecond)
	if _, err := stream.Recv(); err != nil {
		t.Fatal(err)
	}
	time.Sleep(60 * time.Millisecond)
	if closer.calls.Load() != 0 {
		t.Fatal("lease closed a stream that was being read")
	}
	close(ch)
	if err := drainStream(t, stream); err != nil {
		t.Fatal(err)
	}
	if got := stream.Completion().Result[0].String(); got != "first" {
		t.Fatalf("completion = %q", got)
	}
	if err := stream.Close(); err != nil || closer.calls.Load() != 1 {
		t.Fatalf("Close = %v, calls %d", err, closer.calls.Load())
	}
}

func TestLeaseCompletionStreamWithoutTimeoutReturnsTheStream(t *testing.T) {
	t.Parallel()

	stream := NewChannelStream(make(chan StreamItem), &ExecutionResponse{}, nil)
	if LeaseCompletionStream(stream, 0) != stream {
		t.Fatal("a zero timeout must not wrap the stream")
	}
}

// Closing a stream mid-flight must not leave the producing goroutine blocked on a send.
func TestChannelStreamCloseReleasesAnInFlightProducer(t *testing.T) {
	t.Parallel()

	ch := make(chan StreamItem)
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer close(ch)
		for range 5 {
			ch <- StreamItem{Chunk: CompletionChunk{Result: []CompletionResult{NewTextResult("x")}}}
		}
	}()
	stream := NewChannelStream(ch, &ExecutionResponse{}, nil)
	if _, err := stream.Recv(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("producer goroutine still blocked after Close")
	}
}
