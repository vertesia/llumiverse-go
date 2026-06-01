package core

import (
	"context"
	"errors"
	"io"
	"reflect"
	"testing"
)

type fakeDriver struct {
	provider Provider
	prompt   any
	err      error
}

func (d *fakeDriver) Provider() Provider { return d.provider }

func (d *fakeDriver) CreatePrompt(context.Context, []PromptSegment, ExecutionOptions) (any, error) {
	if d.err != nil {
		return nil, d.err
	}
	return d.prompt, nil
}

func (d *fakeDriver) Execute(context.Context, []PromptSegment, ExecutionOptions) (*ExecutionResponse, error) {
	panic("not used")
}

func (d *fakeDriver) Stream(context.Context, []PromptSegment, ExecutionOptions) (CompletionStream, error) {
	panic("not used")
}

func (d *fakeDriver) ListModels(context.Context, *ModelSearchPayload) ([]AIModel, error) {
	panic("not used")
}

func (d *fakeDriver) ValidateConnection(context.Context) error {
	panic("not used")
}

func (d *fakeDriver) GenerateEmbeddings(context.Context, EmbeddingsOptions) (*EmbeddingsResult, error) {
	panic("not used")
}

func TestExecuteWithPromptNormalizesJSONAndSetsMetadata(t *testing.T) {
	t.Parallel()

	driver := &fakeDriver{provider: ProviderOpenAI, prompt: map[string]any{"prompt": true}}
	resp, err := ExecuteWithPrompt(context.Background(), driver, nil, ExecutionOptions{
		Model:        "gpt-test",
		ResultSchema: map[string]any{"type": "object"},
	}, func(_ context.Context, prompt any, options ExecutionOptions) (*Completion, error) {
		if !reflect.DeepEqual(prompt, driver.prompt) {
			t.Fatalf("prompt = %#v", prompt)
		}
		if options.Model != "gpt-test" {
			t.Fatalf("model = %q", options.Model)
		}
		return &Completion{Result: []CompletionResult{{Type: ResultTypeText, Value: `{"ok":true}`}}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(resp.Prompt, driver.prompt) {
		t.Fatalf("Prompt = %#v", resp.Prompt)
	}
	if len(resp.Result) != 1 || resp.Result[0].Type != ResultTypeJSON {
		t.Fatalf("Result = %#v", resp.Result)
	}
	parsed := resp.Result[0].Value.(map[string]any)
	if parsed["ok"] != true {
		t.Fatalf("parsed = %#v", parsed)
	}
	if resp.ExecutionTime <= 0 {
		t.Fatalf("ExecutionTime = %v", resp.ExecutionTime)
	}
}

func TestExecuteWithPromptErrorPaths(t *testing.T) {
	t.Parallel()

	driver := &fakeDriver{provider: ProviderBedrock, prompt: "prompt"}
	if _, err := ExecuteWithPrompt(context.Background(), driver, nil, ExecutionOptions{}, nil); err == nil || err.Error() != "model is required" {
		t.Fatalf("missing model err = %v", err)
	}
	if _, err := ExecuteWithPrompt(context.Background(), &fakeDriver{err: errors.New("bad prompt")}, nil, ExecutionOptions{Model: "m"}, nil); err == nil || err.Error() != "bad prompt" {
		t.Fatalf("prompt err = %v", err)
	}
	if _, err := ExecuteWithPrompt(context.Background(), driver, nil, ExecutionOptions{Model: "m"}, func(context.Context, any, ExecutionOptions) (*Completion, error) {
		return nil, errors.New("rate limit exceeded")
	}); err == nil {
		t.Fatal("expected wrapped provider error")
	} else {
		var llumiErr *LlumiverseError
		if !errors.As(err, &llumiErr) {
			t.Fatalf("err is not LlumiverseError: %T", err)
		}
		if llumiErr.Context.Provider != ProviderBedrock || llumiErr.Context.Operation != "execute" || llumiErr.Retryable == nil || !*llumiErr.Retryable {
			t.Fatalf("wrapped error = %#v", llumiErr)
		}
	}
	original := NewLlumiverseError("known", BoolPtr(false), LlumiverseErrorContext{Provider: ProviderOpenAI}, nil, 400, "Known")
	if _, err := ExecuteWithPrompt(context.Background(), driver, nil, ExecutionOptions{Model: "m"}, func(context.Context, any, ExecutionOptions) (*Completion, error) {
		return nil, original
	}); !errors.Is(err, original) {
		t.Fatalf("expected original error, got %v", err)
	}
	if _, err := ExecuteWithPrompt(context.Background(), driver, nil, ExecutionOptions{Model: "m"}, func(context.Context, any, ExecutionOptions) (*Completion, error) {
		return nil, nil
	}); err == nil || err.Error() != "driver returned nil completion" {
		t.Fatalf("nil completion err = %v", err)
	}
}

func TestStreamWithPromptSetsPromptAndWrapsErrors(t *testing.T) {
	t.Parallel()

	driver := &fakeDriver{provider: ProviderVertexAI, prompt: "provider prompt"}
	ch := make(chan StreamItem)
	close(ch)
	stream, err := StreamWithPrompt(context.Background(), driver, nil, ExecutionOptions{Model: "gemini-test"}, func(context.Context, any, ExecutionOptions) (CompletionStream, error) {
		return NewChannelStream(ch, &ExecutionResponse{}, nil), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stream.Completion().Prompt != "provider prompt" {
		t.Fatalf("Prompt = %#v", stream.Completion().Prompt)
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("Recv err = %v", err)
	}
	if _, err := StreamWithPrompt(context.Background(), driver, nil, ExecutionOptions{}, nil); err == nil || err.Error() != "model is required" {
		t.Fatalf("missing model err = %v", err)
	}
	if _, err := StreamWithPrompt(context.Background(), driver, nil, ExecutionOptions{Model: "m"}, func(context.Context, any, ExecutionOptions) (CompletionStream, error) {
		return nil, errors.New("timeout waiting for stream")
	}); err == nil {
		t.Fatal("expected wrapped stream error")
	} else {
		var llumiErr *LlumiverseError
		if !errors.As(err, &llumiErr) || llumiErr.Context.Operation != "stream" || llumiErr.Retryable == nil || !*llumiErr.Retryable {
			t.Fatalf("wrapped stream err = %#v", err)
		}
	}
	if _, err := StreamWithPrompt(context.Background(), driver, nil, ExecutionOptions{Model: "m"}, func(context.Context, any, ExecutionOptions) (CompletionStream, error) {
		return nil, nil
	}); err == nil || err.Error() != "driver returned nil stream" {
		t.Fatalf("nil stream err = %v", err)
	}
}
