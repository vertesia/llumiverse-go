package vertexai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

// TestGeminiNamedToolChoice mirrors the tool_choice cases of gemini-orphaned-tool-result.test.ts.
func TestGeminiNamedToolChoice(t *testing.T) {
	t.Parallel()

	tools := []ToolDefinition{{Name: "a", InputSchema: map[string]any{"type": "object"}}, {Name: "b", InputSchema: map[string]any{"type": "object"}}}
	payload := (&VertexAIDriver{}).geminiPayload(geminiPrompt{}, ExecutionOptions{Model: "gemini-2.5-flash", Tools: tools, ModelOptions: map[string]any{"tool_choice": "required", "required_tool_name": "a"}})
	config := payload["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)
	if !reflect.DeepEqual(config, map[string]any{"mode": "ANY", "allowedFunctionNames": []string{"a"}}) {
		t.Fatalf("required tool config = %#v", config)
	}
	for choice, mode := range map[string]string{"none": "NONE", "any": "ANY", "auto": "AUTO", "": "AUTO"} {
		got := geminiFunctionCallingConfig(map[string]any{"tool_choice": choice})
		if got["mode"] != mode {
			t.Fatalf("tool_choice %q mode = %v, want %s", choice, got["mode"], mode)
		}
	}

	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("a request without tools was sent")
	}))
	defer server.Close()
	driver := newTestVertexDriver(t, server.URL)
	for _, modelOptions := range []map[string]any{{"tool_choice": "required"}, {"tool_choice": "any"}, {"required_tool_name": "a"}} {
		_, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{Model: "gemini-2.5-flash", ModelOptions: modelOptions})
		got := asLlumiverseError(t, err)
		if got.Code != 400 || retryableString(got.Retryable) != "false" || got.Name != "ToolChoiceConfigurationError" {
			t.Fatalf("%v: error = %#v", modelOptions, got)
		}
	}
}

// TestGeminiRefusesNonArrayConversation mirrors conversation-shape.test.ts for Gemini.
func TestGeminiRefusesNonArrayConversation(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("a request with an unusable conversation was sent")
	}))
	defer server.Close()
	driver := newTestVertexDriver(t, server.URL)
	for _, conversation := range []any{"not a conversation", 42, map[string]any{"messages": []any{}}} {
		_, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{Model: "gemini-2.5-flash", Conversation: conversation})
		got := asLlumiverseError(t, err)
		if got.Code != 400 || retryableString(got.Retryable) != "false" || got.Name != "InvalidConversationError" || got.Context.Operation != "stream" {
			t.Fatalf("%#v: error = %#v", conversation, got)
		}
	}
	wrapper := map[string]any{
		"_arrayConversation": []any{map[string]any{"role": "user", "parts": []any{map[string]any{"text": "earlier"}}}},
		"_llumiverse_system": map[string]any{"role": "user", "parts": []any{map[string]any{"text": "system"}}},
	}
	// Stored shapes stay accepted, including the TS client's wrapper.
	for _, conversation := range []any{
		nil,
		[]any{map[string]any{"role": "user", "parts": []any{map[string]any{"text": "earlier"}}}},
		map[string]any{"contents": []any{}},
		wrapper,
	} {
		if err := validateGeminiRequest(ExecutionOptions{Model: "gemini-2.5-flash", Conversation: conversation}, "execute"); err != nil {
			t.Fatalf("%#v: %v", conversation, err)
		}
	}
	stored, _ := geminiStoredConversation(wrapper)
	if len(stored.Contents) != 1 || stored.System == nil || stored.System.Parts[0].Text != "system" {
		t.Fatalf("TS wrapper = %#v", stored)
	}
}
