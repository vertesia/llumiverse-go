package bedrock

import (
	"context"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/vertesia/llumiverse-go/common"
)

var bedrockToolChoiceTools = []ToolDefinition{{Name: "write_artifact", InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}}}

func bedrockToolChoiceRequest(t *testing.T, model string, tools []ToolDefinition, modelOptions map[string]any) (*fakeBedrockRuntime, error) {
	t.Helper()
	runtime := &fakeBedrockRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	_, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Act now."}}, ExecutionOptions{
		Model:        model,
		Tools:        tools,
		ModelOptions: modelOptions,
	})
	return runtime, err
}

func bedrockAdditionalFieldsJSON(t *testing.T, runtime *fakeBedrockRuntime) string {
	t.Helper()
	if runtime.got.AdditionalModelRequestFields == nil {
		return ""
	}
	data, err := runtime.got.AdditionalModelRequestFields.MarshalSmithyDocument()
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// Mirrors the TS streaming-tool-use.test.ts "BedrockDriver private tool-choice mapping" suite.
func TestBedrockToolChoice(t *testing.T) {
	t.Run("named tool on Nova", func(t *testing.T) {
		runtime, err := bedrockToolChoiceRequest(t, "us.amazon.nova-pro-v1:0", bedrockToolChoiceTools, map[string]any{"tool_choice": "required", "required_tool_name": "write_artifact"})
		if err != nil {
			t.Fatal(err)
		}
		choice, ok := runtime.got.ToolConfig.ToolChoice.(*brtypes.ToolChoiceMemberTool)
		if !ok || aws.ToString(choice.Value.Name) != "write_artifact" {
			t.Fatalf("tool choice = %#v", runtime.got.ToolConfig.ToolChoice)
		}
	})
	t.Run("any tool on Nova", func(t *testing.T) {
		runtime, err := bedrockToolChoiceRequest(t, "us.amazon.nova-pro-v1:0", bedrockToolChoiceTools, map[string]any{"tool_choice": "any"})
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := runtime.got.ToolConfig.ToolChoice.(*brtypes.ToolChoiceMemberAny); !ok {
			t.Fatalf("tool choice = %#v", runtime.got.ToolConfig.ToolChoice)
		}
	})
	t.Run("none keeps tools in auto mode", func(t *testing.T) {
		runtime, err := bedrockToolChoiceRequest(t, "anthropic.claude-sonnet-4-6", bedrockToolChoiceTools, map[string]any{"tool_choice": "none"})
		if err != nil {
			t.Fatal(err)
		}
		if len(runtime.got.ToolConfig.Tools) != 1 || runtime.got.ToolConfig.ToolChoice != nil {
			t.Fatalf("tool config = %#v", runtime.got.ToolConfig)
		}
	})
	t.Run("forced Claude turn disables thinking", func(t *testing.T) {
		runtime, err := bedrockToolChoiceRequest(t, "anthropic.claude-sonnet-4-6", bedrockToolChoiceTools, map[string]any{"effort": "medium", "tool_choice": "required", "required_tool_name": "write_artifact"})
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := runtime.got.ToolConfig.ToolChoice.(*brtypes.ToolChoiceMemberTool); !ok {
			t.Fatalf("tool choice = %#v", runtime.got.ToolConfig.ToolChoice)
		}
		if got := bedrockAdditionalFieldsJSON(t, runtime); got != `{"reasoning_config":{"type":"disabled"}}` {
			t.Fatalf("additional fields = %s", got)
		}
	})
	t.Run("adaptive-only Claude keeps thinking and automatic choice", func(t *testing.T) {
		runtime, err := bedrockToolChoiceRequest(t, "anthropic.claude-fable-5", bedrockToolChoiceTools, map[string]any{"effort": "medium", "tool_choice": "required", "required_tool_name": "write_artifact"})
		if err != nil {
			t.Fatal(err)
		}
		if runtime.got.ToolConfig.ToolChoice != nil {
			t.Fatalf("tool choice = %#v", runtime.got.ToolConfig.ToolChoice)
		}
		// Thinking is left as the caller configured it: the forced turn sends the
		// same fields as an unforced one.
		unforced, err := bedrockToolChoiceRequest(t, "anthropic.claude-fable-5", bedrockToolChoiceTools, map[string]any{"effort": "medium"})
		if err != nil {
			t.Fatal(err)
		}
		if got, want := bedrockAdditionalFieldsJSON(t, runtime), bedrockAdditionalFieldsJSON(t, unforced); got != want {
			t.Fatalf("additional fields = %s, want %s", got, want)
		}
	})
	for name, c := range map[string]struct {
		model string
		tools []ToolDefinition
	}{
		"no tools":           {"anthropic.claude-sonnet-4-6", nil},
		"unsupported family": {"meta.llama3-3-70b-instruct-v1:0", bedrockToolChoiceTools},
	} {
		t.Run("rejects forced turn with "+name, func(t *testing.T) {
			runtime, err := bedrockToolChoiceRequest(t, c.model, c.tools, map[string]any{"tool_choice": "required", "required_tool_name": "write_artifact"})
			var llumiErr *common.LlumiverseError
			if !errors.As(err, &llumiErr) || llumiErr.Name != toolChoiceConfigurationErrorName || llumiErr.Code != 400 || llumiErr.Retryable == nil || *llumiErr.Retryable {
				t.Fatalf("error = %#v", err)
			}
			if runtime.got != nil {
				t.Fatal("request was sent")
			}
		})
	}
}
