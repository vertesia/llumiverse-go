package core

import (
	"encoding/json"
	"io"
	"maps"
	"strconv"
	"sync"
	"time"
)

// channelStream is the shared CompletionStream implementation: provider drivers
// push CompletionChunk deltas down a channel and this type accumulates them into
// the running ExecutionResponse exposed by Completion(). It tracks streamed tool
// calls in arrival order so partial-JSON argument deltas can be stitched once the
// stream finishes.
type channelStream struct {
	ch         <-chan StreamItem
	closeFn    func() error
	completion *ExecutionResponse
	once       sync.Once                         // guards Close so closeFn runs at most once
	finalOnce  sync.Once                         // guards finalizeCompletion so it runs at most once
	finalizer  func(*ExecutionResponse)          // optional driver hook run after finalization
	start      time.Time                         // stream start, used to compute ExecutionTime
	tools      map[string]*streamToolAccumulator // streamed tool calls keyed by tool ID
	toolOrder  []string                          // tool IDs in first-seen order, to keep output stable
	lastToolID string                            // ID of the tool whose deltas are currently arriving
	nextToolID int                               // counter for synthesizing IDs when a provider omits them
	errContext LlumiverseErrorContext            // provider and model for errors raised while finalizing
	finalErr   error                             // error Recv returns instead of io.EOF, set by finalizeCompletion
}

// streamToolAccumulator gathers the deltas for a single streamed tool call.
type streamToolAccumulator struct {
	tool ToolUseChunk
}

// StreamItem is one value carried on the stream channel: either a chunk or an error.
type StreamItem struct {
	Chunk CompletionChunk
	Err   error
}

// newChannelStream wraps a chunk channel as a CompletionStream that accumulates
// into completion. closeFn releases the underlying provider stream on Close.
func newChannelStream(ch <-chan StreamItem, completion *ExecutionResponse, closeFn func() error) CompletionStream {
	return &channelStream{ch: ch, completion: completion, closeFn: closeFn, start: time.Now()}
}

// newChannelStreamWithFinalizer is like newChannelStream but runs finalizer once
// the stream is fully consumed, letting a driver post-process the accumulated
// ExecutionResponse (for example, normalizing the JSON result).
func newChannelStreamWithFinalizer(ch <-chan StreamItem, completion *ExecutionResponse, closeFn func() error, finalizer func(*ExecutionResponse)) CompletionStream {
	return &channelStream{ch: ch, completion: completion, closeFn: closeFn, finalizer: finalizer, start: time.Now()}
}

// Recv returns the next chunk, applying it to the accumulated completion, or
// io.EOF once the stream is exhausted (at which point the completion is finalized).
// When finalizing finds a streamed tool call it cannot return, Recv returns that
// error instead of io.EOF.
func (s *channelStream) Recv() (CompletionChunk, error) {
	item, ok := <-s.ch
	if !ok {
		s.finalizeCompletion()
		if s.finalErr != nil {
			return CompletionChunk{}, s.finalErr
		}
		return CompletionChunk{}, io.EOF
	}
	if item.Err == nil {
		s.applyChunk(item.Chunk)
	}
	return item.Chunk, item.Err
}

// Close finalizes the completion and releases the underlying provider stream.
// It is safe to call more than once, and while the stream is still in flight:
// the chunks nobody will read are drained so the driver goroutine producing
// them is never left blocked on a send.
func (s *channelStream) Close() error {
	var err error
	s.once.Do(func() {
		s.finalizeCompletion()
		if s.closeFn != nil {
			err = s.closeFn()
		}
		if s.ch != nil {
			// Closing the provider stream makes the producer fail and close the
			// channel; until then, discard what it still sends.
			go func() {
				for item := range s.ch {
					_ = item
				}
			}()
		}
	})
	return err
}

// Completion returns the accumulating ExecutionResponse. It is fully populated
// only after the stream has been consumed and finalized.
func (s *channelStream) Completion() *ExecutionResponse {
	return s.completion
}

// applyChunk merges a single chunk's result, tool, usage, finish-reason, and service-tier
// deltas into the running completion and increments the chunk count.
func (s *channelStream) applyChunk(chunk CompletionChunk) {
	// Each provider emits different delta shapes; this is the normalization
	// point that accumulates chunks into the final ExecutionResponse.
	if s.completion == nil {
		return
	}
	if len(chunk.Result) > 0 {
		s.completion.Result = mergeStreamResults(s.completion.Result, chunk.Result)
	}
	if len(chunk.ToolUse) > 0 {
		s.mergeStreamTools(chunk.ToolUse)
	}
	if chunk.TokenUsage != nil {
		s.completion.TokenUsage = mergeStreamUsage(s.completion.TokenUsage, chunk.TokenUsage)
	}
	if chunk.FinishReason != "" {
		s.completion.FinishReason = chunk.FinishReason
	}
	if chunk.ServiceTier != "" {
		s.completion.ServiceTier = chunk.ServiceTier
	}
	s.completion.Chunks++
}

// mergeStreamTools accumulates streamed tool-call deltas into per-ID
// accumulators, synthesizing an ID when the provider omits one and reusing the
// last seen ID for continuation deltas.
func (s *channelStream) mergeStreamTools(tools []ToolUseChunk) {
	if s.tools == nil {
		s.tools = map[string]*streamToolAccumulator{}
	}
	for _, tool := range tools {
		id := tool.ID
		if id == "" {
			id = s.lastToolID
		}
		if id == "" {
			s.nextToolID++
			id = "tool_" + strconv.Itoa(s.nextToolID)
		}
		acc := s.tools[id]
		if acc == nil {
			acc = &streamToolAccumulator{tool: ToolUseChunk{ID: id}}
			s.tools[id] = acc
			s.toolOrder = append(s.toolOrder, id)
		}
		s.lastToolID = id
		tool.ID = id
		accumulateToolUseChunk(&acc.tool, tool)
	}
}

// finalizeCompletion is run once when the stream ends: it sets ExecutionTime,
// reconciles the token-usage total, then finalizes the accumulated tool calls
// with FinalizeStreamingToolUse. A tool call with malformed arguments is left
// out of the completion, and unless the response was cut off it is reported as
// the stream's final error. Finally any finalizer hook runs.
func (s *channelStream) finalizeCompletion() {
	s.finalOnce.Do(func() {
		if s.completion == nil {
			return
		}
		s.completion.ExecutionTime = time.Since(s.start)
		if usage := s.completion.TokenUsage; usage != nil {
			if sum := Count(usage.Prompt) + Count(usage.Result); Count(usage.Total) < sum {
				usage.Total = &sum
			}
		}
		if len(s.toolOrder) > 0 {
			chunks := make([]ToolUseChunk, 0, len(s.toolOrder))
			for _, id := range s.toolOrder {
				chunks = append(chunks, s.tools[id].tool)
			}
			s.completion.ToolUse, s.finalErr = finalizeStreamingToolUse(chunks, s.completion.FinishReason, s.errContext)
		}
		if s.finalizer != nil {
			s.finalizer(s.completion)
		}
	})
}

// mergeStreamResults appends src result deltas onto dst, coalescing consecutive
// same-type items: text and thoughts values are concatenated and JSON object
// values are merged key-by-key, so streamed fragments collapse into a single
// result item. Media results (image, video, audio) keep their boundaries.
func mergeStreamResults(dst []CompletionResult, src []CompletionResult) []CompletionResult {
	for _, result := range src {
		if len(dst) == 0 {
			dst = append(dst, result)
			continue
		}
		last := &dst[len(dst)-1]
		kind := result.Type()
		if last.Type() == kind && kind == ResultTypeText {
			*last = NewTextResult(last.String() + result.String())
			continue
		}
		// Thoughts stay a separate typed result, never merged into the answer
		// text, but their own fragments collapse like text.
		if last.Type() == kind && kind == ResultTypeThoughts {
			*last = NewThoughtsResult(last.String() + result.String())
			continue
		}
		if last.Type() == kind && kind == ResultTypeJSON {
			existing, errExisting := last.AsJSONResult()
			incoming, errIncoming := result.AsJSONResult()
			if errExisting == nil && errIncoming == nil {
				existingObject, okExisting := existing.Value.(map[string]any)
				incomingObject, okIncoming := incoming.Value.(map[string]any)
				if okExisting && okIncoming {
					maps.Copy(existingObject, incomingObject)
					*last = NewJSONResult(existingObject)
					continue
				}
			}
		}
		dst = append(dst, result)
	}
	return dst
}

// mergeStreamUsage folds a usage delta into the running total. Prompt/Result/Total
// are taken as the running maximum (providers may resend cumulative counts), while
// the cache-related fields are overwritten when the delta reports a nonzero value.
func mergeStreamUsage(dst *ExecutionTokenUsage, src *ExecutionTokenUsage) *ExecutionTokenUsage {
	if src == nil {
		return dst
	}
	if dst == nil {
		copy := *src
		return &copy
	}
	dst.Prompt = maxCount(dst.Prompt, src.Prompt)
	dst.Result = maxCount(dst.Result, src.Result)
	dst.Total = maxCount(dst.Total, src.Total)
	for _, field := range []struct{ dst, src **float64 }{
		{&dst.PromptCached, &src.PromptCached},
		{&dst.PromptCacheWrite, &src.PromptCacheWrite},
		{&dst.PromptCacheWrite1H, &src.PromptCacheWrite1H},
		{&dst.PromptNew, &src.PromptNew},
		{&dst.ResultImage, &src.ResultImage},
		{&dst.ProviderCostUsd, &src.ProviderCostUsd},
	} {
		if Count(*field.src) != 0 {
			*field.dst = *field.src
		}
	}
	return dst
}

// maxCount returns the larger of two optional counts, keeping unset when both are unset.
func maxCount(a, b *float64) *float64 {
	if a == nil {
		return b
	}
	if b == nil || *a >= *b {
		return a
	}
	return b
}

// toString returns a string value unchanged and JSON-encodes anything else.
func toString(value any) string {
	if value == nil {
		return ""
	}
	if s, ok := value.(string); ok {
		return s
	}
	data, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	return string(data)
}
