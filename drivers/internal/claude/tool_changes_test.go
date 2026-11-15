package claude

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

const toolChangesModel = "claude-sonnet-5-5"

// namedTool is a tool definition with a description.
func namedTool(name string) ToolDefinition {
	return ToolDefinition{Name: name, Description: ptr(name + " tool"), InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}}
}

// toolTurnResult is the outcome of one request in a tool-change test.
type toolTurnResult struct {
	body         map[string]any
	betas        string
	conversation any
}

// toolTurn sends one request with these tools on the stored conversation; the model answers with a call to callName.
func toolTurn(t *testing.T, prompt claudePrompt, tools []ToolDefinition, conversation any, callID, callName, provider, model string) toolTurnResult {
	t.Helper()
	request, err := NewRequest(prompt, ExecutionOptions{Model: model, Tools: tools, Conversation: conversation}, RequestSettings{Provider: provider})
	if err != nil {
		t.Fatal(err)
	}
	completion := request.Complete(claudeResponse{Role: "assistant", Content: []claudeBlock{toolUse(callID, callName)}, StopReason: "tool_use"})
	return toolTurnResult{body: jsonValue(t, request.Body).(map[string]any), betas: request.BetaHeader(), conversation: jsonValue(t, completion.Conversation)}
}

// resultPrompt answers a tool call.
func resultPrompt(id string) claudePrompt {
	return claudePrompt{Messages: []claudeMessage{{Role: "user", Content: []claudeBlock{toolResult(id, "done")}}}}
}

// lastMessage is the last message of a body.
func lastMessage(body map[string]any) map[string]any {
	messages := body["messages"].([]any)
	return messages[len(messages)-1].(map[string]any)
}

// change is a decoded tool-change system message.
func change(kind, name string) map[string]any {
	return map[string]any{"role": "system", "content": []any{map[string]any{"type": kind, "tool": map[string]any{"type": "tool_reference", "name": name}}}}
}

func TestClaudeToolChangesDeclareThenAppend(t *testing.T) {
	t.Parallel()

	t1 := toolTurn(t, userPrompt("Audit the project."), []ToolDefinition{namedTool("learn_skill")}, nil, "c1", "learn_skill", ProviderNameAnthropic, toolChangesModel)
	if strings.Contains(t1.betas, claudeToolChangesBeta) {
		t.Fatalf("the first declaration needs no beta: %q", t1.betas)
	}
	t2 := toolTurn(t, resultPrompt("c1"), []ToolDefinition{namedTool("learn_skill"), namedTool("list_items")}, t1.conversation, "c2", "list_items", ProviderNameAnthropic, toolChangesModel)
	tools := t2.body["tools"].([]any)
	if !reflect.DeepEqual(tools[0], t1.body["tools"].([]any)[0]) || tools[1].(map[string]any)["defer_loading"] != true {
		t.Fatalf("tools = %#v", tools)
	}
	if !reflect.DeepEqual(lastMessage(t2.body), change("tool_addition", "list_items")) {
		t.Fatalf("last message = %#v", lastMessage(t2.body))
	}
	if !strings.Contains(t2.betas, claudeToolChangesBeta) {
		t.Fatalf("betas = %q", t2.betas)
	}
	t3 := toolTurn(t, resultPrompt("c2"), []ToolDefinition{namedTool("learn_skill"), namedTool("list_items")}, t2.conversation, "c3", "list_items", ProviderNameAnthropic, toolChangesModel)
	if !reflect.DeepEqual(t3.body["tools"], t2.body["tools"]) {
		t.Fatal("tools changed without a tool change")
	}
	systems := 0
	for _, message := range t3.body["messages"].([]any) {
		if message.(map[string]any)["role"] == "system" {
			systems++
		}
	}
	if systems != 1 {
		t.Fatalf("system messages = %d", systems)
	}
	for _, pair := range [][2]toolTurnResult{{t1, t2}, {t2, t3}} {
		previous := stripCacheControl(pair[0].body["messages"]).([]any)
		current := stripCacheControl(pair[1].body["messages"]).([]any)
		if !reflect.DeepEqual(current[:len(previous)], previous) {
			t.Fatal("a request does not replay the previous one as a prefix")
		}
	}
}

func TestClaudeToolChangesWithdrawAndReoffer(t *testing.T) {
	t.Parallel()

	t1 := toolTurn(t, userPrompt("go"), []ToolDefinition{namedTool("a"), namedTool("b")}, nil, "c1", "a", ProviderNameVertexAI, toolChangesModel)
	t2 := toolTurn(t, resultPrompt("c1"), []ToolDefinition{namedTool("a")}, t1.conversation, "c2", "a", ProviderNameVertexAI, toolChangesModel)
	if len(t2.body["tools"].([]any)) != 2 || !reflect.DeepEqual(lastMessage(t2.body), change("tool_removal", "b")) {
		t.Fatalf("withdrawn = %#v", t2.body)
	}
	t3 := toolTurn(t, resultPrompt("c2"), []ToolDefinition{namedTool("a"), namedTool("b")}, t2.conversation, "c3", "b", ProviderNameVertexAI, toolChangesModel)
	if len(t3.body["tools"].([]any)) != 2 || !reflect.DeepEqual(lastMessage(t3.body), change("tool_addition", "b")) {
		t.Fatalf("re-offered = %#v", t3.body)
	}
	// A request without tools keeps the declared tools, with tool_choice none.
	final := toolTurn(t, resultPrompt("c3"), nil, t3.conversation, "c4", "a", ProviderNameVertexAI, toolChangesModel)
	if len(final.body["tools"].([]any)) != 2 || !reflect.DeepEqual(final.body["tool_choice"], map[string]any{"type": "none"}) {
		t.Fatalf("final = %#v", final.body)
	}
	if data, _ := json.Marshal(final.body["messages"]); !strings.Contains(string(data), `"tool_use"`) {
		t.Fatal("tool calls must stay tool blocks while tools are declared")
	}
}

func TestClaudeToolChangesReplaceAndCompare(t *testing.T) {
	t.Parallel()

	v1 := ToolDefinition{Name: "mcp_search", Description: ptr("v1"), InputSchema: map[string]any{"type": "object"}}
	v2 := ToolDefinition{Name: "mcp_search", Description: ptr("v2"), InputSchema: map[string]any{"type": "object"}}
	declared := applyClaudeToolChanges(userPrompt("hi"), []ToolDefinition{v1})
	declared.Messages = append(declared.Messages, claudeMessage{Role: "user", Content: []claudeBlock{text("again")}})
	next := applyClaudeToolChanges(declared, []ToolDefinition{v2})
	state := next.toolState()
	if len(state.Declared) != 1 || state.Declared[0].Definition["description"] != "v2" || state.Declared[0].Deferred {
		t.Fatalf("declared = %#v", state.Declared)
	}
	for _, message := range next.Messages {
		if message.Role == "system" {
			t.Fatal("a replaced definition is not a tool change")
		}
	}
	if !sameClaudeToolDefinition(map[string]any{"name": "a", "input_schema": map[string]any{"type": "object", "x": 1}}, map[string]any{"input_schema": map[string]any{"x": 1.0, "type": "object"}, "name": "a"}) {
		t.Fatal("key order and number types do not make a definition different")
	}
	twice := applyClaudeToolChanges(userPrompt("hi"), []ToolDefinition{namedTool("a"), namedTool("a")})
	if len(twice.toolState().Declared) != 1 {
		t.Fatal("a tool listed twice is declared once")
	}
}

func TestClaudeToolChangesUnsupportedModelSendsToolsTheOldWay(t *testing.T) {
	t.Parallel()

	t1 := toolTurn(t, userPrompt("go"), []ToolDefinition{namedTool("a")}, nil, "c1", "a", ProviderNameBedrock, toolChangesModel)
	t2 := toolTurn(t, resultPrompt("c1"), []ToolDefinition{namedTool("a"), namedTool("b")}, t1.conversation, "c2", "b", ProviderNameBedrock, toolChangesModel)
	legacy := toolTurn(t, resultPrompt("c2"), []ToolDefinition{namedTool("b")}, t2.conversation, "c3", "b", ProviderNameAnthropic, "claude-sonnet-5")
	tools := legacy.body["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["name"] != "b" {
		t.Fatalf("tools = %#v", tools)
	}
	for _, message := range legacy.body["messages"].([]any) {
		if message.(map[string]any)["role"] == "system" {
			t.Fatal("tool changes must be stripped")
		}
	}
	if strings.Contains(legacy.betas, claudeToolChangesBeta) {
		t.Fatalf("betas = %q", legacy.betas)
	}
	if decodeClaudeConversation(legacy.conversation).toolState() != nil {
		t.Fatal("the tool state is cleared")
	}
	// Back on a supporting model, the tools are declared again, loaded up front.
	again := toolTurn(t, resultPrompt("c3"), []ToolDefinition{namedTool("b"), namedTool("c")}, legacy.conversation, "c4", "c", ProviderNameAnthropic, toolChangesModel)
	for _, tool := range again.body["tools"].([]any) {
		if tool.(map[string]any)["defer_loading"] != nil {
			t.Fatalf("redeclared tools load up front: %#v", again.body["tools"])
		}
	}
}

func TestClaudeToolChangesPlacement(t *testing.T) {
	t.Parallel()

	tools := []ToolDefinition{namedTool("a"), namedTool("b")}
	roles := func(prompt claudePrompt) []string {
		body, _ := payloadFor(t, ExecutionOptions{Model: toolChangesModel, Tools: tools}, prompt, ProviderNameAnthropic)
		var out []string
		for _, message := range body["messages"].([]any) {
			out = append(out, message.(map[string]any)["role"].(string))
		}
		return out
	}

	declared := applyClaudeToolChanges(userPrompt("start"), []ToolDefinition{namedTool("a")})
	afterAssistant := declared
	afterAssistant.Messages = append(append([]claudeMessage(nil), declared.Messages...), claudeMessage{Role: "assistant", Content: []claudeBlock{text("ok")}})
	next := applyClaudeToolChanges(afterAssistant, tools)
	body, _ := payloadFor(t, ExecutionOptions{Model: toolChangesModel, Tools: tools}, next, ProviderNameAnthropic)
	messages := body["messages"].([]any)
	if !reflect.DeepEqual(messages[len(messages)-2], map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "Continue."}}}) ||
		!reflect.DeepEqual(messages[len(messages)-1], change("tool_addition", "b")) {
		t.Fatalf("continuation before the change = %#v", messages)
	}

	stored := declared
	stored.Messages = append(append([]claudeMessage(nil), declared.Messages...),
		claudeMessage{Role: "assistant", Content: []claudeBlock{text("no tool call")}},
		claudeMessage{Role: "user", Content: []claudeBlock{toolResult("gone", "x")}},
		claudeMessage{Role: "system", Content: []claudeBlock{{Type: "tool_addition", Tool: &claudeToolReference{Type: "tool_reference", Name: "b"}}}},
		claudeMessage{Role: "assistant", Content: []claudeBlock{text("done")}},
		claudeMessage{Role: "user", Content: []claudeBlock{text("next")}},
	)
	if got := roles(stored); !reflect.DeepEqual(got[1:], []string{"assistant", "user", "system", "assistant", "user"}) {
		t.Fatalf("roles = %v", got)
	}

	// Changes that cancel out while pending leave nothing to announce.
	pending := declared
	pending.Messages = append(append([]claudeMessage(nil), declared.Messages...),
		claudeMessage{Role: "system", Content: []claudeBlock{{Type: "tool_addition", Tool: &claudeToolReference{Type: "tool_reference", Name: "b"}}}},
		claudeMessage{Role: "system", Content: []claudeBlock{{Type: "tool_removal", Tool: &claudeToolReference{Type: "tool_reference", Name: "b"}}}},
		claudeMessage{Role: "assistant", Content: []claudeBlock{text("ok")}},
		claudeMessage{Role: "user", Content: []claudeBlock{text("more")}},
	)
	if got := roles(pending); !reflect.DeepEqual(got, []string{"user", "assistant", "user"}) {
		t.Fatalf("roles = %v", got)
	}

	rebuilt := claudePrompt{Messages: []claudeMessage{
		{Role: "user", Content: []claudeBlock{text("hi")}},
		{Role: "system", Content: []claudeBlock{{Type: "tool_addition", Tool: &claudeToolReference{Type: "tool_reference", Name: "gone"}}}},
		{Role: "assistant", Content: []claudeBlock{text("ok")}},
	}}
	if got := applyClaudeToolChanges(rebuilt, []ToolDefinition{namedTool("a")}); containsClaudeToolChanges(got.Messages) {
		t.Fatal("tool changes without stored state are dropped")
	}
}

func TestClaudeToolChangesKeepCacheBreakpointsOffSystemMessages(t *testing.T) {
	t.Parallel()

	t1 := toolTurn(t, userPrompt("go"), []ToolDefinition{namedTool("a")}, nil, "c1", "a", ProviderNameAnthropic, toolChangesModel)
	request, err := NewRequest(resultPrompt("c1"), ExecutionOptions{Model: toolChangesModel, PromptCacheKey: "agent-x", Tools: []ToolDefinition{namedTool("a"), namedTool("b")}, Conversation: t1.conversation}, RequestSettings{Provider: ProviderNameAnthropic})
	if err != nil {
		t.Fatal(err)
	}
	body := jsonValue(t, request.Body).(map[string]any)
	last := lastMessage(body)
	if last["role"] != "system" {
		t.Fatalf("last = %#v", last)
	}
	if data, _ := json.Marshal(last); strings.Contains(string(data), "cache_control") {
		t.Fatalf("a tool-change message carries a breakpoint: %s", data)
	}
	tools := body["tools"].([]any)
	if tools[0].(map[string]any)["cache_control"] == nil || tools[1].(map[string]any)["cache_control"] != nil {
		t.Fatalf("the last tool loaded up front is marked, never a deferred one: %#v", tools)
	}
}

func TestClaudeToolChangeSupport(t *testing.T) {
	t.Parallel()

	cases := map[string]bool{
		"claude-opus-4-8":                true,
		"claude-opus-4-7":                false,
		"claude-fable-5":                 true,
		"claude-mythos-5-1":              true,
		"claude-mythos-preview":          false,
		"claude-sonnet-5-5":              true,
		"claude-sonnet-5":                false,
		"claude-haiku-5-5":               true,
		"us.anthropic.claude-sonnet-5-5": true,
		"gpt-5":                          false,
	}
	for model, want := range cases {
		if got := canUseClaudeToolChanges(model, ProviderNameBedrock); got != want {
			t.Fatalf("%s: %v, want %v", model, got, want)
		}
	}
	if canUseClaudeToolChanges("claude-sonnet-5-5", "azure_foundry_unknown") {
		t.Fatal("unknown providers send tools the old way")
	}
}
