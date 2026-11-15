package core

import (
	"bytes"
	"encoding/base64"
	"errors"
	"io"
	"testing"
)

func TestChannelStreamAccumulatesChunksToolsUsageAndFinalizer(t *testing.T) {
	t.Parallel()

	ch := make(chan StreamItem, 8)
	ch <- StreamItem{Chunk: CompletionChunk{
		Result:     []CompletionResult{NewTextResult("hello ")},
		TokenUsage: &ExecutionTokenUsage{Prompt: Tokens(10), Result: Tokens(1)},
	}}
	ch <- StreamItem{Chunk: CompletionChunk{
		Result:     []CompletionResult{NewTextResult("world")},
		TokenUsage: &ExecutionTokenUsage{Prompt: Tokens(8), Result: Tokens(3), Total: Tokens(11), PromptCached: Tokens(2)},
	}}
	ch <- StreamItem{Chunk: CompletionChunk{Result: []CompletionResult{NewJSONResult(map[string]any{"a": 1})}}}
	ch <- StreamItem{Chunk: CompletionChunk{Result: []CompletionResult{NewJSONResult(map[string]any{"b": 2})}}}
	ch <- StreamItem{Chunk: CompletionChunk{ToolUse: []ToolUseChunk{{ID: "tool_1", ToolName: "lookup", ToolInput: `{"q"`}}}}
	ch <- StreamItem{Chunk: CompletionChunk{ToolUse: []ToolUseChunk{{ToolInput: `:"tokyo"}`}}}}
	ch <- StreamItem{Chunk: CompletionChunk{ToolUse: []ToolUseChunk{{ID: "tool_2", ToolName: "merge", ToolInput: map[string]any{"x": 1}}}}}
	ch <- StreamItem{Chunk: CompletionChunk{ToolUse: []ToolUseChunk{{ID: "tool_2", ToolInput: map[string]any{"y": 2}}}, FinishReason: "tool_use"}}
	close(ch)

	var closeCalls, finalizerCalls int
	completion := &ExecutionResponse{}
	stream := NewChannelStreamWithFinalizer(ch, completion, func() error {
		closeCalls++
		return nil
	}, func(resp *ExecutionResponse) {
		finalizerCalls++
		if resp.FinishReason != "tool_use" {
			t.Fatalf("finalizer saw FinishReason = %q", resp.FinishReason)
		}
	})

	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if completion.Chunks != 8 {
		t.Fatalf("Chunks = %d", completion.Chunks)
	}
	if len(completion.Result) != 2 || completion.Result[0].Type() != ResultTypeText || completion.Result[0].String() != "hello world" {
		t.Fatalf("Result = %#v", completion.Result)
	}
	merged, err := completion.Result[1].AsJSONResult()
	if err != nil {
		t.Fatal(err)
	}
	// JSON values come back decoded from JSON, so numbers are float64.
	jsonResult := merged.Value.(map[string]any)
	if jsonResult["a"] != 1.0 || jsonResult["b"] != 2.0 {
		t.Fatalf("JSON result = %#v", jsonResult)
	}
	usage := completion.TokenUsage
	if usage == nil || Count(usage.Prompt) != 10 || Count(usage.Result) != 3 || Count(usage.Total) != 13 || Count(usage.PromptCached) != 2 {
		t.Fatalf("TokenUsage = %#v", completion.TokenUsage)
	}
	if len(completion.ToolUse) != 2 {
		t.Fatalf("ToolUse = %#v", completion.ToolUse)
	}
	firstInput := *completion.ToolUse[0].ToolInput
	if completion.ToolUse[0].ID != "tool_1" || firstInput["q"] != "tokyo" {
		t.Fatalf("first tool = %#v", completion.ToolUse[0])
	}
	secondInput := *completion.ToolUse[1].ToolInput
	if secondInput["x"] != 1 || secondInput["y"] != 2 {
		t.Fatalf("second tool = %#v", completion.ToolUse[1])
	}
	if finalizerCalls != 1 {
		t.Fatalf("finalizer calls = %d", finalizerCalls)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if closeCalls != 1 {
		t.Fatalf("close calls = %d", closeCalls)
	}
	if finalizerCalls != 1 {
		t.Fatalf("finalizer calls after Close = %d", finalizerCalls)
	}
}

func TestChannelStreamDropsTruncatedToolOnLength(t *testing.T) {
	t.Parallel()

	ch := make(chan StreamItem, 1)
	ch <- StreamItem{Chunk: CompletionChunk{
		ToolUse:      []ToolUseChunk{{ID: "tool_1", ToolName: "lookup", ToolInput: `{"q"`}},
		FinishReason: "length",
	}}
	close(ch)
	stream := NewChannelStream(ch, &ExecutionResponse{}, nil)
	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(stream.Completion().ToolUse) != 0 {
		t.Fatalf("ToolUse = %#v", stream.Completion().ToolUse)
	}
}

func TestChannelStreamReturnsChunkErrorWithoutApplyingChunk(t *testing.T) {
	t.Parallel()

	ch := make(chan StreamItem, 1)
	ch <- StreamItem{
		Chunk: CompletionChunk{Result: []CompletionResult{NewTextResult("ignored")}},
		Err:   errors.New("stream failed"),
	}
	close(ch)
	stream := newChannelStream(ch, &ExecutionResponse{}, nil)
	chunk, err := stream.Recv()
	if err == nil || err.Error() != "stream failed" {
		t.Fatalf("err = %v", err)
	}
	if len(chunk.Result) != 1 {
		t.Fatalf("chunk = %#v", chunk)
	}
	if len(stream.Completion().Result) != 0 {
		t.Fatalf("completion result = %#v", stream.Completion().Result)
	}
}

func TestChannelStreamHandlesNilCompletion(t *testing.T) {
	t.Parallel()

	ch := make(chan StreamItem, 1)
	ch <- StreamItem{Chunk: CompletionChunk{
		Result:       []CompletionResult{NewTextResult("ignored")},
		TokenUsage:   &ExecutionTokenUsage{Prompt: Tokens(1)},
		FinishReason: "stop",
		ToolUse:      []ToolUseChunk{{ID: "tool_1", ToolInput: "{}"}},
	}}
	close(ch)
	stream := NewChannelStream(ch, nil, nil)
	if _, err := stream.Recv(); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("expected EOF, got %v", err)
	}
	if stream.Completion() != nil {
		t.Fatalf("completion = %#v", stream.Completion())
	}
}

// drainStream reads stream to the end and returns the first error other than io.EOF.
func drainStream(t *testing.T, stream CompletionStream) error {
	t.Helper()
	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

// streamToolChunks runs fragments through a channel stream, one chunk per fragment, with
// finishReason on the last chunk, and returns the finished stream.
func streamToolChunks(t *testing.T, finishReason string, fragments ...ToolUseChunk) (CompletionStream, error) {
	t.Helper()
	ch := make(chan StreamItem, len(fragments))
	for i, fragment := range fragments {
		chunk := CompletionChunk{ToolUse: []ToolUseChunk{fragment}}
		if i == len(fragments)-1 {
			chunk.FinishReason = finishReason
		}
		ch <- StreamItem{Chunk: chunk}
	}
	close(ch)
	stream := NewChannelStream(ch, &ExecutionResponse{}, nil)
	return stream, drainStream(t, stream)
}

// Mirrors the TS client's "streaming tool_use thought_signature reassembly" tests.
func TestChannelStreamReassemblesThoughtSignature(t *testing.T) {
	t.Parallel()

	raw := make([]byte, 9000)
	for i := range raw {
		raw[i] = byte((i*37 + 11) % 256)
	}
	signature := base64.StdEncoding.EncodeToString(raw)
	args := `{"query":"weather in Paris"}`

	t.Run("split across chunks", func(t *testing.T) {
		const pieces = 5
		sigSize := (len(signature) + pieces - 1) / pieces
		argSize := (len(args) + pieces - 1) / pieces
		var fragments []ToolUseChunk
		for i := range pieces {
			fragment := ToolUseChunk{
				ID:               "get_weather",
				ThoughtSignature: signature[min(i*sigSize, len(signature)):min((i+1)*sigSize, len(signature))],
				ToolInput:        args[min(i*argSize, len(args)):min((i+1)*argSize, len(args))],
			}
			if i == 0 {
				fragment.ToolName = "get_weather"
			}
			fragments = append(fragments, fragment)
		}
		stream, err := streamToolChunks(t, "tool_use", fragments...)
		if err != nil {
			t.Fatal(err)
		}
		tools := stream.Completion().ToolUse
		if len(tools) != 1 || tools[0].ThoughtSignature == nil || *tools[0].ThoughtSignature != signature {
			t.Fatalf("signature not reassembled: %#v", tools)
		}
		decoded, err := base64.StdEncoding.DecodeString(*tools[0].ThoughtSignature)
		if err != nil || !bytes.Equal(decoded, raw) {
			t.Fatalf("signature does not decode to the original bytes: %v", err)
		}
		if (*tools[0].ToolInput)["query"] != "weather in Paris" {
			t.Fatalf("arguments = %#v", tools[0].ToolInput)
		}
	})

	t.Run("only on a later chunk", func(t *testing.T) {
		stream, err := streamToolChunks(t, "tool_use",
			ToolUseChunk{ID: "plan", ToolName: "plan", ToolInput: `{"steps":`},
			ToolUseChunk{ID: "plan", ToolInput: `[]}`, ThoughtSignature: signature},
		)
		if err != nil {
			t.Fatal(err)
		}
		if tools := stream.Completion().ToolUse; len(tools) != 1 || *tools[0].ThoughtSignature != signature {
			t.Fatalf("tools = %#v", tools)
		}
	})

	t.Run("whole in a single chunk", func(t *testing.T) {
		stream, err := streamToolChunks(t, "tool_use", ToolUseChunk{ID: "plan", ToolName: "plan", ToolInput: `{}`, ThoughtSignature: signature})
		if err != nil {
			t.Fatal(err)
		}
		if tools := stream.Completion().ToolUse; len(tools) != 1 || *tools[0].ThoughtSignature != signature {
			t.Fatalf("tools = %#v", tools)
		}
	})
}
