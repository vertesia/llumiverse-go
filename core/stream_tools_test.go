package core

import (
	"context"
	"errors"
	"io"
	"reflect"
	"strings"
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

	t.Run("does not let an empty placeholder erase accumulated argument bytes", func(t *testing.T) {
		var tool ToolUseChunk
		for _, fragment := range []ToolUseChunk{
			{ID: "write", ToolName: "write_artifact", ToolInput: `{"name"`},
			{ID: "write", ToolInput: map[string]any{}},
			{ID: "write", ToolInput: ""},
			{ID: "write", ToolInput: `:"report"}`},
		} {
			accumulateToolUseChunk(&tool, fragment)
		}
		if tool.ToolInput != `{"name":"report"}` {
			t.Fatalf("ToolInput = %#v", tool.ToolInput)
		}
	})

	t.Run("normalizes empty streamed arguments for parameterless tools", func(t *testing.T) {
		tools, err := FinalizeStreamingToolUse([]ToolUseChunk{
			{ID: "inspect", ToolName: "inspect_status", ToolInput: ""},
		}, "tool_use", streamToolTestContext)
		want := []ToolUse{{ID: "inspect", ToolName: "inspect_status", ToolInput: &JSONObject{}}}
		if err != nil || !reflect.DeepEqual(tools, want) {
			t.Fatalf("tools = %#v, err = %v", tools, err)
		}
	})

	t.Run("rejects malformed arguments with a stable non-retryable identity", func(t *testing.T) {
		_, err := FinalizeStreamingToolUse([]ToolUseChunk{
			{ID: "write", ToolName: "write_artifact", ToolInput: `{`},
		}, "", streamToolTestContext)
		var llumiErr *LlumiverseError
		if !errors.As(err, &llumiErr) || !IsMalformedStreamingToolArguments(err) {
			t.Fatalf("err = %v", err)
		}
		if llumiErr.Name != MalformedStreamingToolArgumentsErrorName {
			t.Fatalf("Name = %q", llumiErr.Name)
		}
		if !strings.Contains(llumiErr.Message, `streamed tool "write_artifact" (finish_reason=unknown, argument_chars=1)`) {
			t.Fatalf("Message = %q", llumiErr.Message)
		}
		if llumiErr.Retryable == nil || *llumiErr.Retryable {
			t.Fatalf("Retryable = %#v", llumiErr.Retryable)
		}
		want := LlumiverseErrorContext{Provider: ProviderBedrock, Model: "anthropic.claude-sonnet-4-5", Operation: "stream"}
		if llumiErr.Context != want {
			t.Fatalf("context = %#v", llumiErr.Context)
		}
	})

	t.Run("drops empty arguments on a length stop", func(t *testing.T) {
		tools, err := FinalizeStreamingToolUse([]ToolUseChunk{
			{ID: "write", ToolName: "write_artifact", ToolInput: ""},
		}, "length", streamToolTestContext)
		if err != nil || tools != nil {
			t.Fatalf("tools = %#v, err = %v", tools, err)
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
