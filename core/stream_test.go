package core

import (
	"errors"
	"io"
	"testing"
)

func TestChannelStreamAccumulatesChunksToolsUsageAndFinalizer(t *testing.T) {
	t.Parallel()

	ch := make(chan StreamItem, 8)
	ch <- StreamItem{Chunk: CompletionChunk{
		Result:     []CompletionResult{{Type: ResultTypeText, Value: "hello "}},
		TokenUsage: &ExecutionTokenUsage{Prompt: 10, Result: 1},
	}}
	ch <- StreamItem{Chunk: CompletionChunk{
		Result:     []CompletionResult{{Type: ResultTypeText, Value: "world"}},
		TokenUsage: &ExecutionTokenUsage{Prompt: 8, Result: 3, Total: 11, PromptCached: 2},
	}}
	ch <- StreamItem{Chunk: CompletionChunk{Result: []CompletionResult{{Type: ResultTypeJSON, Value: map[string]any{"a": 1}}}}}
	ch <- StreamItem{Chunk: CompletionChunk{Result: []CompletionResult{{Type: ResultTypeJSON, Value: map[string]any{"b": 2}}}}}
	ch <- StreamItem{Chunk: CompletionChunk{ToolUse: []ToolUse{{ID: "tool_1", ToolName: "lookup", ToolInput: `{"q"`}}}}
	ch <- StreamItem{Chunk: CompletionChunk{ToolUse: []ToolUse{{ToolInput: `:"tokyo"}`}}}}
	ch <- StreamItem{Chunk: CompletionChunk{ToolUse: []ToolUse{{ID: "tool_2", ToolName: "merge", ToolInput: map[string]any{"x": 1}}}}}
	ch <- StreamItem{Chunk: CompletionChunk{ToolUse: []ToolUse{{ID: "tool_2", ToolInput: map[string]any{"y": 2}}}, FinishReason: "tool_use"}}
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
	if len(completion.Result) != 2 || completion.Result[0].Value != "hello world" {
		t.Fatalf("Result = %#v", completion.Result)
	}
	jsonResult := completion.Result[1].Value.(map[string]any)
	if jsonResult["a"] != 1 || jsonResult["b"] != 2 {
		t.Fatalf("JSON result = %#v", jsonResult)
	}
	if completion.TokenUsage == nil || completion.TokenUsage.Prompt != 10 || completion.TokenUsage.Result != 3 || completion.TokenUsage.Total != 13 || completion.TokenUsage.PromptCached != 2 {
		t.Fatalf("TokenUsage = %#v", completion.TokenUsage)
	}
	if len(completion.ToolUse) != 2 {
		t.Fatalf("ToolUse = %#v", completion.ToolUse)
	}
	firstInput := completion.ToolUse[0].ToolInput.(map[string]any)
	if completion.ToolUse[0].ID != "tool_1" || firstInput["q"] != "tokyo" {
		t.Fatalf("first tool = %#v", completion.ToolUse[0])
	}
	secondInput := completion.ToolUse[1].ToolInput.(map[string]any)
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
		ToolUse:      []ToolUse{{ID: "tool_1", ToolName: "lookup", ToolInput: `{"q"`}},
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
		Chunk: CompletionChunk{Result: []CompletionResult{{Type: ResultTypeText, Value: "ignored"}}},
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
		Result:       []CompletionResult{{Type: ResultTypeText, Value: "ignored"}},
		TokenUsage:   &ExecutionTokenUsage{Prompt: 1},
		FinishReason: "stop",
		ToolUse:      []ToolUse{{ID: "tool_1", ToolInput: "{}"}},
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
