package core

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// ErrCompletionStreamNotConsumed is returned by Recv on a stream that LeaseCompletionStream closed
// because nobody started reading it within its start timeout.
var ErrCompletionStreamNotConsumed = errors.New("completion stream was not consumed")

// LeaseCompletionStream bounds how long a returned stream may stay unconsumed, the TS client's
// CompletionStreamLease: when Recv has not been called within timeout, the stream is closed so
// the provider connection and the goroutine reading it are released, and a later Recv returns
// ErrCompletionStreamNotConsumed. Once reading starts the timeout no longer applies; the
// transport timeouts and the request context bound the rest. Drivers wrap the stream they return
// from Stream with DriverOptions.StreamStartTimeout. A non-positive timeout returns stream as is.
func LeaseCompletionStream(stream CompletionStream, timeout time.Duration) CompletionStream {
	if stream == nil || timeout <= 0 {
		return stream
	}
	lease := &leasedStream{inner: stream, timeout: timeout}
	lease.timer = time.AfterFunc(timeout, lease.expire)
	return lease
}

// leasedStream is the CompletionStream LeaseCompletionStream returns.
type leasedStream struct {
	inner   CompletionStream
	timeout time.Duration
	timer   *time.Timer
	mu      sync.Mutex
	started bool // Recv or Close was called: the lease no longer applies
	expired bool // the lease closed the stream before it was read
}

// expire closes the stream when nobody has started reading it.
func (s *leasedStream) expire() {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return
	}
	s.expired = true
	s.mu.Unlock()
	_ = s.inner.Close() // best effort: the caller sees ErrCompletionStreamNotConsumed instead
}

// start ends the lease and reports whether it had already expired.
func (s *leasedStream) start() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.started {
		s.started = true
		s.timer.Stop()
	}
	return s.expired
}

// Recv returns the next chunk, or ErrCompletionStreamNotConsumed when the lease expired first.
func (s *leasedStream) Recv() (CompletionChunk, error) {
	if s.start() {
		return CompletionChunk{}, fmt.Errorf("%w within %s", ErrCompletionStreamNotConsumed, s.timeout)
	}
	return s.inner.Recv()
}

// Close ends the lease and closes the underlying stream, unless the lease already closed it.
func (s *leasedStream) Close() error {
	if s.start() {
		return nil
	}
	return s.inner.Close()
}

// Completion returns the underlying stream's accumulating completion.
func (s *leasedStream) Completion() *ExecutionResponse {
	return s.inner.Completion()
}
