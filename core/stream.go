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
}

// streamToolAccumulator gathers the deltas for a single streamed tool call.
// When arguments arrive as partial JSON strings they are appended to stringInput
// (usesStringBuf) and parsed once the stream completes.
type streamToolAccumulator struct {
	tool          ToolUse
	stringInput   string
	usesStringBuf bool
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
func (s *channelStream) Recv() (CompletionChunk, error) {
	item, ok := <-s.ch
	if !ok {
		s.finalizeCompletion()
		return CompletionChunk{}, io.EOF
	}
	if item.Err == nil {
		s.applyChunk(item.Chunk)
	}
	return item.Chunk, item.Err
}

// Close finalizes the completion and releases the underlying provider stream.
// It is safe to call more than once.
func (s *channelStream) Close() error {
	var err error
	s.once.Do(func() {
		s.finalizeCompletion()
		if s.closeFn != nil {
			err = s.closeFn()
		}
	})
	return err
}

// Completion returns the accumulating ExecutionResponse. It is fully populated
// only after the stream has been consumed and finalized.
func (s *channelStream) Completion() *ExecutionResponse {
	return s.completion
}

// applyChunk merges a single chunk's result, tool, usage, and finish-reason
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
	s.completion.Chunks++
}

// mergeStreamTools accumulates streamed tool-call deltas into per-ID
// accumulators, synthesizing an ID when the provider omits one and reusing the
// last seen ID for continuation deltas. String argument fragments are appended
// for later parsing; object arguments are merged key-by-key.
func (s *channelStream) mergeStreamTools(tools []ToolUse) {
	// Tool-call arguments often arrive as partial JSON strings. Preserve order
	// by tool ID and stitch string deltas before exposing the final tool list.
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
			acc = &streamToolAccumulator{tool: ToolUse{ID: id}}
			s.tools[id] = acc
			s.toolOrder = append(s.toolOrder, id)
		}
		s.lastToolID = id
		if tool.ToolName != "" {
			acc.tool.ToolName = tool.ToolName
		}
		if tool.ThoughtSignature != "" {
			acc.tool.ThoughtSignature = tool.ThoughtSignature
		}
		if tool.ToolInput == nil {
			continue
		}
		if value, ok := tool.ToolInput.(string); ok {
			acc.stringInput += value
			acc.usesStringBuf = true
			acc.tool.ToolInput = acc.stringInput
			continue
		}
		if existing, ok := acc.tool.ToolInput.(map[string]any); ok {
			if incoming, ok := tool.ToolInput.(map[string]any); ok {
				maps.Copy(existing, incoming)
				acc.tool.ToolInput = existing
				continue
			}
		}
		acc.tool.ToolInput = tool.ToolInput
	}
}

// finalizeCompletion is run once when the stream ends: it sets ExecutionTime,
// reconciles the token-usage total, then parses each accumulated tool's stitched
// string arguments into JSON. A tool whose buffered arguments fail to parse is
// marked truncated, and such tools are dropped when FinishReason=="length"
// (the model was cut off mid tool call). Finally any finalizer hook runs.
func (s *channelStream) finalizeCompletion() {
	s.finalOnce.Do(func() {
		if s.completion == nil {
			return
		}
		s.completion.ExecutionTime = time.Since(s.start)
		if s.completion.TokenUsage != nil && s.completion.TokenUsage.Total < s.completion.TokenUsage.Prompt+s.completion.TokenUsage.Result {
			s.completion.TokenUsage.Total = s.completion.TokenUsage.Prompt + s.completion.TokenUsage.Result
		}
		if len(s.toolOrder) == 0 {
			if s.finalizer != nil {
				s.finalizer(s.completion)
			}
			return
		}
		truncated := map[string]bool{}
		tools := make([]ToolUse, 0, len(s.toolOrder))
		for _, id := range s.toolOrder {
			acc := s.tools[id]
			if acc == nil {
				continue
			}
			tool := acc.tool
			if acc.usesStringBuf {
				var parsed any
				if err := json.Unmarshal([]byte(acc.stringInput), &parsed); err == nil {
					tool.ToolInput = parsed
				} else {
					tool.ToolInput = map[string]any{}
					truncated[id] = true
				}
			}
			if s.completion.FinishReason == "length" && truncated[id] {
				continue
			}
			tools = append(tools, tool)
		}
		s.completion.ToolUse = tools
		if s.finalizer != nil {
			s.finalizer(s.completion)
		}
	})
}

// mergeStreamResults appends src result deltas onto dst, coalescing consecutive
// same-type items: text values are concatenated and JSON object values are
// merged key-by-key, so streamed fragments collapse into a single result item.
func mergeStreamResults(dst []CompletionResult, src []CompletionResult) []CompletionResult {
	for _, result := range src {
		if len(dst) == 0 {
			dst = append(dst, result)
			continue
		}
		last := &dst[len(dst)-1]
		if last.Type == result.Type && (result.Type == ResultTypeText || result.Type == ResultTypeJSON) {
			if result.Type == ResultTypeText {
				last.Value = toString(last.Value) + toString(result.Value)
				continue
			}
			if existing, ok := last.Value.(map[string]any); ok {
				if incoming, ok := result.Value.(map[string]any); ok {
					maps.Copy(existing, incoming)
					last.Value = existing
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
	dst.Prompt = max(dst.Prompt, src.Prompt)
	dst.Result = max(dst.Result, src.Result)
	dst.Total = max(dst.Total, src.Total)
	if src.PromptCached != 0 {
		dst.PromptCached = src.PromptCached
	}
	if src.PromptCacheWrite != 0 {
		dst.PromptCacheWrite = src.PromptCacheWrite
	}
	if src.PromptNew != 0 {
		dst.PromptNew = src.PromptNew
	}
	return dst
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
