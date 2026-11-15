package core

import (
	"context"
	"strings"
	"testing"
)

func TestTruncateBinaryForDebug(t *testing.T) {
	t.Parallel()

	short := strings.Repeat("a", binaryDebugEdgeChars*2+len(BinaryTruncatedMarker))
	if got := TruncateBinaryForDebug(short); got != short {
		t.Fatalf("short value changed: %q", got)
	}
	long := "0123456789" + strings.Repeat("x", 100) + "abcdefghij"
	if got, want := TruncateBinaryForDebug(long), "0123456789"+BinaryTruncatedMarker+"abcdefghij"; got != want {
		t.Fatalf("TruncateBinaryForDebug = %q, want %q", got, want)
	}
	url := "data:image/png;base64," + long
	if got, want := TruncateDataURLForDebug(url), "data:image/png;base64,0123456789"+BinaryTruncatedMarker+"abcdefghij"; got != want {
		t.Fatalf("TruncateDataURLForDebug = %q, want %q", got, want)
	}
	if got := TruncateDataURLForDebug("https://example.com/" + long); got != "https://example.com/"+long {
		t.Fatalf("non-data URL changed: %q", got)
	}
}

type debugPromptDriver struct{ *fakeDriver }

func (debugPromptDriver) FormatDebugPrompt(prompt any) any { return "debug:" + prompt.(string) }

func TestExecuteReturnsDebugPrompt(t *testing.T) {
	t.Parallel()

	var sent any
	driver := debugPromptDriver{&fakeDriver{provider: ProviderOpenAI, prompt: "prompt"}}
	response, err := executeWithPrompt(context.Background(), driver, nil, ExecutionOptions{Model: "m"}, func(_ context.Context, prompt any, _ ExecutionOptions) (*Completion, error) {
		sent = prompt
		return &Completion{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if sent != "prompt" || response.Prompt != "debug:prompt" {
		t.Fatalf("sent %v, response prompt %v", sent, response.Prompt)
	}
}
