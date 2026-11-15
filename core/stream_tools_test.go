package core

import (
	"context"
	"errors"
	"io"
	"reflect"
	"testing"
)

var streamToolTestContext = LlumiverseErrorContext{Provider: ProviderBedrock, Model: "anthropic.claude-sonnet-4-5"}

// Mirrors the TS client's "streaming tool_use argument finalization" tests.
func TestFinalizeStreamingToolUse(t *testing.T) {
	t.Parallel()

	t.Run("parses complete streamed JSON arguments", func(t *testing.T) {
		tools, err := FinalizeStreamingToolUse([]ToolUseChunk{
			{ID: "write", ToolName: "write_artifact", ToolInput: `{"name":"report","type":"text"}`},
		}, "tool_use", streamToolTestContext)
		if err != nil {
			t.Fatal(err)
		}
		want := []ToolUse{{ID: "write", ToolName: "write_artifact", ToolInput: &JSONObject{"name": "report", "type": "text"}}}
		if !reflect.DeepEqual(tools, want) {
			t.Fatalf("tools = %#v", tools)
		}
	})

	t.Run("rejects malformed arguments when the stream was not cut off", func(t *testing.T) {
		_, err := FinalizeStreamingToolUse([]ToolUseChunk{
			{ID: "write", ToolName: "write_artifact", ToolInput: `{`},
		}, "", streamToolTestContext)
		var llumiErr *LlumiverseError
		if !errors.As(err, &llumiErr) {
			t.Fatalf("err = %v", err)
		}
		want := LlumiverseErrorContext{Provider: ProviderBedrock, Model: "anthropic.claude-sonnet-4-5", Operation: "stream"}
		if llumiErr.Context != want {
			t.Fatalf("context = %#v", llumiErr.Context)
		}
	})

	t.Run("drops malformed arguments on a length stop", func(t *testing.T) {
		tools, err := FinalizeStreamingToolUse([]ToolUseChunk{
			{ID: "write", ToolName: "write_artifact", ToolInput: `{"name":`},
		}, "length", streamToolTestContext)
		if err != nil || tools != nil {
			t.Fatalf("tools = %#v, err = %v", tools, err)
		}
	})
}

// streamThroughLifecycle streams fixed chunks through the shared lifecycle, so errors raised
// while finalizing carry the provider and model.
func streamThroughLifecycle(t *testing.T, chunks ...CompletionChunk) CompletionStream {
	t.Helper()
	driver := &fakeDriver{provider: ProviderVertexAI, prompt: "prompt"}
	stream, err := StreamWithPrompt(context.Background(), driver, nil, ExecutionOptions{Model: "test-model"}, func(context.Context, any, ExecutionOptions) (CompletionStream, error) {
		ch := make(chan StreamItem, len(chunks))
		for _, chunk := range chunks {
			ch <- StreamItem{Chunk: chunk}
		}
		close(ch)
		return NewChannelStream(ch, &ExecutionResponse{}, nil), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return stream
}

func TestChannelStreamReportsMalformedToolArguments(t *testing.T) {
	t.Parallel()

	stream := streamThroughLifecycle(t,
		CompletionChunk{ToolUse: []ToolUseChunk{{ID: "search", ToolName: "search", ToolInput: `{"query":"expenses"}`}}},
		CompletionChunk{ToolUse: []ToolUseChunk{{ID: "write", ToolName: "write_artifact", ToolInput: `{"name"`}}, FinishReason: "tool_use"},
	)
	err := drainStream(t, stream)
	var llumiErr *LlumiverseError
	if !errors.As(err, &llumiErr) || errors.Is(err, io.EOF) {
		t.Fatalf("drain err = %v", err)
	}
	if llumiErr.Context.Provider != ProviderVertexAI || llumiErr.Context.Model != "test-model" || llumiErr.Context.Operation != "stream" {
		t.Fatalf("context = %#v", llumiErr.Context)
	}
	// The malformed call is never returned with invented arguments; the complete one is kept.
	tools := stream.Completion().ToolUse
	if len(tools) != 1 || tools[0].ID != "search" {
		t.Fatalf("tools = %#v", tools)
	}
}
