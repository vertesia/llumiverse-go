package claude

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

// userPrompt is a one-turn prompt.
func userPrompt(text string) claudePrompt {
	return claudePrompt{Messages: []claudeMessage{{Role: "user", Content: []claudeBlock{{Type: "text", Text: text}}}}}
}

// jsonValue round-trips a value through JSON, as the request is sent.
func jsonValue(t *testing.T, value any) any {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	var decoded any
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	return decoded
}

// payloadFor builds the JSON body of a request on an unprepared prompt.
func payloadFor(t *testing.T, options ExecutionOptions, prompt claudePrompt, provider string) (map[string]any, []string) {
	t.Helper()
	body, betas, err := buildClaudePayload(options, prompt, claudePayloadSettings{Provider: provider, Operation: "execute", AllowNativeResultSchema: true})
	if err != nil {
		t.Fatal(err)
	}
	return jsonValue(t, body).(map[string]any), betas
}

func TestResolveClaudeThinking(t *testing.T) {
	t.Parallel()

	adaptive := func(display string) map[string]any { return map[string]any{"type": "adaptive", "display": display} }
	bound := func(display string) map[string]any {
		return map[string]any{"type": "adaptive", "display": display, "block_binding": map[string]any{"prefix_mismatch_behavior": "drop_block"}}
	}
	cases := []struct {
		name     string
		model    string
		options  map[string]any
		thinking map[string]any
		effort   any
		betas    []string
	}{
		{"between tools ignores display and budget", "claude-sonnet-5-5", map[string]any{"thinking_mode": "between_tools", "effort": "low", "include_thoughts": true, "thinking_budget_tokens": 8000}, map[string]any{"type": "between_tools"}, "low", nil},
		{"Sonnet 5.5 defaults to bound adaptive thinking", "claude-sonnet-5-5", nil, bound("omitted"), nil, []string{claudeThinkingBindingBeta}},
		{"explicit adaptive mode", "claude-sonnet-5-5", map[string]any{"thinking_mode": "adaptive", "include_thoughts": true}, bound("summarized"), nil, []string{claudeThinkingBindingBeta}},
		{"Sonnet 4.6 effort selects adaptive", "claude-sonnet-4-6", map[string]any{"effort": "medium", "thinking_budget_tokens": 12000}, adaptive("omitted"), "medium", nil},
		{"Sonnet 4.6 budget keeps extended thinking", "claude-sonnet-4-6", map[string]any{"thinking_budget_tokens": 12000}, map[string]any{"type": "enabled", "budget_tokens": 12000}, nil, nil},
		{"3.7 budget", "claude-3-7-sonnet", map[string]any{"thinking_budget_tokens": 8000}, map[string]any{"type": "enabled", "budget_tokens": 8000}, nil, nil},
		{"3.7 effort leaves thinking disabled", "claude-3-7-sonnet", map[string]any{"effort": "medium"}, map[string]any{"type": "disabled"}, "medium", nil},
		{"Opus 5.5 always-on thinking ignores a budget", "claude-opus-5-5", map[string]any{"include_thoughts": true, "thinking_budget_tokens": 12000}, bound("summarized"), nil, []string{claudeThinkingBindingBeta}},
		{"Fable 5.1 default display", "claude-fable-5-1", map[string]any{"include_thoughts": true}, bound("summarized"), nil, []string{claudeThinkingBindingBeta}},
		{"Opus 5 default adaptive, unbound", "claude-opus-5", nil, adaptive("omitted"), nil, nil},
		{"Haiku 5.5 default adaptive", "anthropic.claude-haiku-5-5", map[string]any{"thinking_budget_tokens": 8000}, bound("omitted"), nil, []string{claudeThinkingBindingBeta}},
		{"Haiku 6 default adaptive", "claude-haiku-6", nil, bound("omitted"), nil, []string{claudeThinkingBindingBeta}},
		{"Haiku 5.4 keeps the provider default", "claude-haiku-5-4", map[string]any{"include_thoughts": true}, nil, nil, nil},
		{"Haiku 4.5 budget", "claude-haiku-4-5", map[string]any{"thinking_budget_tokens": 8000}, map[string]any{"type": "enabled", "budget_tokens": 8000}, nil, nil},
		{"Haiku 4.5 default disabled", "claude-haiku-4-5", nil, map[string]any{"type": "disabled"}, nil, nil},
		{"Opus 4.7 drops a legacy budget", "claude-opus-4-7", map[string]any{"thinking_budget_tokens": 12000}, nil, nil, nil},
		{"Mythos preview is unbound", "claude-mythos-preview", map[string]any{"effort": "medium"}, adaptive("omitted"), "medium", nil},
		{"region-prefixed Opus 5.5 is bound", "us.anthropic.claude-opus-5-5", map[string]any{"effort": "medium"}, bound("omitted"), "medium", []string{claudeThinkingBindingBeta}},
		{"pre-3.7 has no thinking", "claude-3-5-sonnet", map[string]any{"thinking_budget_tokens": 8000}, nil, nil, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := resolveClaudeThinking(tc.model, tc.options)
			if !reflect.DeepEqual(jsonValue(t, got.Thinking), jsonValue(t, tc.thinking)) {
				t.Fatalf("thinking = %#v, want %#v", got.Thinking, tc.thinking)
			}
			var effort any
			if got.OutputConfig != nil {
				effort = got.OutputConfig["effort"]
			}
			if effort != tc.effort {
				t.Fatalf("effort = %v, want %v", effort, tc.effort)
			}
			if !reflect.DeepEqual(got.Betas, tc.betas) {
				t.Fatalf("betas = %v, want %v", got.Betas, tc.betas)
			}
		})
	}
}

func TestClaudePayloadSamplingAndStops(t *testing.T) {
	t.Parallel()

	options := map[string]any{"temperature": 0.4, "top_p": 0.8, "top_k": 40, "stop_sequence": []any{"END"}}
	body, _ := payloadFor(t, ExecutionOptions{Model: "claude-sonnet-4-5", ModelOptions: options}, userPrompt("hi"), ProviderNameAnthropic)
	if body["temperature"] != 0.4 || body["top_p"] != nil || body["top_k"] != float64(40) {
		t.Fatalf("temperature must win over top_p: %#v", body)
	}
	if !reflect.DeepEqual(body["stop_sequences"], []any{"END"}) {
		t.Fatalf("stop_sequences = %#v", body["stop_sequences"])
	}
	if body["max_tokens"] != float64(64000) {
		t.Fatalf("default max_tokens = %v", body["max_tokens"])
	}
	restricted, _ := payloadFor(t, ExecutionOptions{Model: "claude-opus-4-7", ModelOptions: options}, userPrompt("hi"), ProviderNameAnthropic)
	if restricted["temperature"] != nil || restricted["top_p"] != nil || restricted["top_k"] != nil {
		t.Fatalf("4.7+ rejects sampling parameters: %#v", restricted)
	}
	if restricted["max_tokens"] != float64(128000) {
		t.Fatalf("Opus 4.7 max_tokens = %v", restricted["max_tokens"])
	}
	topP, _ := payloadFor(t, ExecutionOptions{Model: "claude-sonnet-4-5", ModelOptions: map[string]any{"top_p": 0.8}}, userPrompt("hi"), ProviderNameAnthropic)
	if topP["top_p"] != 0.8 || topP["temperature"] != nil {
		t.Fatalf("top_p alone = %#v", topP)
	}
	// No caller value means no default: the provider decides.
	plain, _ := payloadFor(t, ExecutionOptions{Model: "claude-sonnet-4-5"}, userPrompt("hi"), ProviderNameAnthropic)
	if _, ok := plain["temperature"]; ok {
		t.Fatalf("a default temperature was sent: %#v", plain)
	}
}

func TestClaudePayloadOutput128KBeta(t *testing.T) {
	t.Parallel()

	_, betas := payloadFor(t, ExecutionOptions{Model: "claude-3-7-sonnet-20250219", ModelOptions: map[string]any{"max_tokens": 100000}}, userPrompt("hi"), ProviderNameAnthropic)
	if !reflect.DeepEqual(betas, []string{claudeOutput128KBeta}) {
		t.Fatalf("betas = %v", betas)
	}
	_, betas = payloadFor(t, ExecutionOptions{Model: "claude-3-7-sonnet-20250219", ModelOptions: map[string]any{"max_tokens": 64000}}, userPrompt("hi"), ProviderNameAnthropic)
	if len(betas) != 0 {
		t.Fatalf("64K needs no beta: %v", betas)
	}
	body, _ := payloadFor(t, ExecutionOptions{Model: "claude-3-7-sonnet-20250219", ModelOptions: map[string]any{"thinking_budget_tokens": 50000}}, userPrompt("hi"), ProviderNameAnthropic)
	if body["max_tokens"] != float64(128000) {
		t.Fatalf("3.7 with a large budget uses its full limit: %v", body["max_tokens"])
	}
}

func TestClaudePayloadFastMode(t *testing.T) {
	t.Parallel()

	cases := []struct {
		model, provider string
		beta            bool
	}{
		{"claude-opus-5", ProviderNameAnthropic, true},
		{"claude-opus-5-5", ProviderNameAnthropic, false},
		{"claude-fable-5-2", ProviderNameVertexAI, false},
		{"claude-mythos-5-2", ProviderNameVertexAI, false},
		{"claude-opus-4-8", ProviderNameAnthropic, true},
		{"some-unknown-model", ProviderNameAnthropic, true},
	}
	for _, tc := range cases {
		body, betas := payloadFor(t, ExecutionOptions{Model: tc.model, ModelOptions: map[string]any{"speed": "fast"}}, userPrompt("hi"), tc.provider)
		if body["speed"] != "fast" {
			t.Fatalf("%s: speed = %v", tc.model, body["speed"])
		}
		if got := strings.Contains(strings.Join(betas, ","), claudeFastModeBeta); got != tc.beta {
			t.Fatalf("%s: fast mode beta = %v, want %v (%v)", tc.model, got, tc.beta, betas)
		}
		standard, betas := payloadFor(t, ExecutionOptions{Model: tc.model, ModelOptions: map[string]any{"speed": "standard"}}, userPrompt("hi"), tc.provider)
		if _, ok := standard["speed"]; ok || strings.Contains(strings.Join(betas, ","), claudeFastModeBeta) {
			t.Fatalf("%s: standard speed must be left out: %#v %v", tc.model, standard, betas)
		}
	}
	if claudeFastModeStatus("claude-opus-4-8") != "" || claudeFastModeStatus("claude-opus-5") != "research_preview" || claudeFastModeStatus("claude-fable-5-2") != "available" {
		t.Fatal("fast mode status")
	}
	if !SupportsVertexAIFastMode("claude-fable-5-2") || !SupportsVertexAIFastMode("claude-mythos-6") || SupportsVertexAIFastMode("claude-opus-5-5") || SupportsVertexAIFastMode("claude-fable-5-1") {
		t.Fatal("Vertex AI fast mode")
	}
}

func TestClaudePayloadToolChoice(t *testing.T) {
	t.Parallel()

	tools := []ToolDefinition{{Name: "write_artifact", InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}}}
	forced := func(model string, extra map[string]any) ExecutionOptions {
		options := map[string]any{"tool_choice": "required", "required_tool_name": "write_artifact", "parallel_tool_calls": false}
		for key, value := range extra {
			options[key] = value
		}
		return ExecutionOptions{Model: model, Tools: tools, ModelOptions: options}
	}

	body, _ := payloadFor(t, forced("claude-sonnet-4-5", nil), userPrompt("Act now."), ProviderNameAnthropic)
	if !reflect.DeepEqual(body["tool_choice"], map[string]any{"type": "tool", "name": "write_artifact", "disable_parallel_tool_use": true}) {
		t.Fatalf("named tool choice = %#v", body["tool_choice"])
	}
	parallel := forced("claude-sonnet-4-5", map[string]any{"parallel_tool_calls": true})
	body, _ = payloadFor(t, parallel, userPrompt("Act now."), ProviderNameAnthropic)
	if body["tool_choice"].(map[string]any)["disable_parallel_tool_use"] != false {
		t.Fatalf("parallel calls allowed = %#v", body["tool_choice"])
	}
	anyChoice := ExecutionOptions{Model: "claude-sonnet-4-5", Tools: tools, ModelOptions: map[string]any{"tool_choice": "any"}}
	body, _ = payloadFor(t, anyChoice, userPrompt("Act now."), ProviderNameAnthropic)
	if !reflect.DeepEqual(body["tool_choice"], map[string]any{"type": "any", "disable_parallel_tool_use": false}) {
		t.Fatalf("any = %#v", body["tool_choice"])
	}
	for _, model := range []string{"claude-opus-5-5", "global.anthropic.claude-sonnet-5-5", "claude-opus-6", "claude-fable-5-2"} {
		body, _ = payloadFor(t, forced(model, nil), userPrompt("Act now."), ProviderNameAnthropic)
		if !reflect.DeepEqual(body["tool_choice"], map[string]any{"type": "auto", "disable_parallel_tool_use": true}) {
			t.Fatalf("%s falls back to auto: %#v", model, body["tool_choice"])
		}
	}
	// A supported forced choice disables manual thinking, which the API rejects alongside it.
	body, _ = payloadFor(t, forced("claude-sonnet-4-5", map[string]any{"thinking_budget_tokens": 2048, "effort": "high"}), userPrompt("Act now."), ProviderNameAnthropic)
	if !reflect.DeepEqual(body["thinking"], map[string]any{"type": "disabled"}) || body["output_config"] != nil {
		t.Fatalf("forced tool with manual thinking = %#v / %#v", body["thinking"], body["output_config"])
	}

	for name, options := range map[string]ExecutionOptions{
		"no tools":       {Model: "claude-sonnet-4-5", ModelOptions: map[string]any{"tool_choice": "required"}},
		"mythos preview": forced("claude-mythos-preview", nil),
	} {
		_, _, err := buildClaudePayload(options, userPrompt("Act now."), claudePayloadSettings{Provider: ProviderNameAnthropic, Operation: "stream"})
		var llumiErr *common.LlumiverseError
		if !errors.As(err, &llumiErr) || llumiErr.Code != 400 || llumiErr.Name != "ToolChoiceConfigurationError" || llumiErr.Retryable == nil || *llumiErr.Retryable || llumiErr.Context.Operation != "stream" {
			t.Fatalf("%s: error = %#v", name, err)
		}
	}

	body, _ = payloadFor(t, ExecutionOptions{Model: "claude-sonnet-4-5", Tools: tools, ModelOptions: map[string]any{"tool_choice": "none"}}, userPrompt("hi"), ProviderNameAnthropic)
	if !reflect.DeepEqual(body["tool_choice"], map[string]any{"type": "none"}) {
		t.Fatalf("none = %#v", body["tool_choice"])
	}
	body, _ = payloadFor(t, ExecutionOptions{Model: "claude-sonnet-4-5", Tools: tools}, userPrompt("hi"), ProviderNameAnthropic)
	if _, ok := body["tool_choice"]; ok {
		t.Fatalf("an omitted tool choice stays omitted: %#v", body["tool_choice"])
	}
	tool := body["tools"].([]any)[0].(map[string]any)
	if _, ok := tool["description"]; ok {
		t.Fatalf("an omitted description stays omitted: %#v", tool)
	}
	if _, _, err := buildClaudePayload(ExecutionOptions{Model: "claude-sonnet-4-5", Tools: []ToolDefinition{{Name: "bad", InputSchema: map[string]any{"type": "string"}}}}, userPrompt("hi"), claudePayloadSettings{}); err == nil {
		t.Fatal("expected an input_schema error")
	}
}

func TestClaudePayloadConvertsToolBlocksWithoutTools(t *testing.T) {
	t.Parallel()

	prompt := claudePrompt{Messages: []claudeMessage{
		{Role: "user", Content: []claudeBlock{{Type: "text", Text: "look it up"}}},
		{Role: "assistant", Content: []claudeBlock{{Type: "tool_use", ID: "t1", Name: "lookup", Input: map[string]any{"q": "kyoto"}}}},
		{Role: "user", Content: []claudeBlock{{Type: "tool_result", ToolUseID: "t1", Content: []claudeBlock{{Type: "text", Text: "sunny"}}}, {Type: "text", Text: "thanks"}}},
	}}
	body, _ := payloadFor(t, ExecutionOptions{Model: "claude-sonnet-4-5"}, prompt, ProviderNameAnthropic)
	messages := body["messages"].([]any)
	call := messages[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	result := messages[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if call["text"] != `[Tool call: lookup({"q":"kyoto"})]` || result["text"] != "[Tool result: sunny]" {
		t.Fatalf("tool blocks as text = %#v / %#v", call, result)
	}
}

func TestClaudePayloadContinuesAfterTrailingAssistantTurn(t *testing.T) {
	t.Parallel()

	prompt := claudePrompt{Messages: []claudeMessage{
		{Role: "user", Content: []claudeBlock{{Type: "text", Text: "go"}}},
		{Role: "assistant", Content: []claudeBlock{{Type: "tool_use", ID: "t1", Name: "lookup", Input: map[string]any{}}}},
	}}
	tools := []ToolDefinition{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}}
	body, _ := payloadFor(t, ExecutionOptions{Model: "claude-sonnet-4-6", Tools: tools}, prompt, ProviderNameAnthropic)
	messages := body["messages"].([]any)
	last := messages[len(messages)-1].(map[string]any)
	content := last["content"].([]any)
	if last["role"] != "user" || len(content) != 2 || content[0].(map[string]any)["type"] != "tool_result" || content[1].(map[string]any)["text"] != "Continue." {
		t.Fatalf("continuation turn = %#v", last)
	}
	if !strings.Contains(content[0].(map[string]any)["content"].(string), `before "lookup" could execute`) {
		t.Fatalf("interrupted result = %#v", content[0])
	}
	// Older models keep a trailing assistant turn: it can be an intentional prefill.
	body, _ = payloadFor(t, ExecutionOptions{Model: "claude-sonnet-4-5"}, claudePrompt{Messages: []claudeMessage{
		{Role: "user", Content: []claudeBlock{{Type: "text", Text: "answer in JSON"}}},
		{Role: "assistant", Content: []claudeBlock{{Type: "text", Text: "{"}}},
	}}, ProviderNameAnthropic)
	if messages := body["messages"].([]any); messages[len(messages)-1].(map[string]any)["role"] != "assistant" {
		t.Fatalf("prefill = %#v", messages)
	}
}

func TestClaudePayloadCompaction(t *testing.T) {
	t.Parallel()

	body, betas := payloadFor(t, ExecutionOptions{Model: "claude-opus-4-8", Compaction: &ConversationCompactionOptions{TriggerTokens: 100000, Instructions: "Keep the decisions."}}, userPrompt("hi"), ProviderNameAnthropic)
	want := map[string]any{"edits": []any{map[string]any{"type": "compact_20260112", "trigger": map[string]any{"type": "input_tokens", "value": float64(100000)}, "instructions": "Keep the decisions."}}}
	if !reflect.DeepEqual(body["context_management"], want) || !reflect.DeepEqual(betas, []string{claudeCompactionBeta}) {
		t.Fatalf("context_management = %#v, betas = %v", body["context_management"], betas)
	}
	body, _ = payloadFor(t, ExecutionOptions{Model: "claude-opus-4-8", Compaction: &ConversationCompactionOptions{}}, userPrompt("hi"), ProviderNameAnthropic)
	if !reflect.DeepEqual(body["context_management"], map[string]any{"edits": []any{map[string]any{"type": "compact_20260112"}}}) {
		t.Fatalf("defaults = %#v", body["context_management"])
	}
	body, betas = payloadFor(t, ExecutionOptions{
		Model:               "claude-opus-4-8",
		Compaction:          &ConversationCompactionOptions{TriggerTokens: 100000},
		CompactConversation: &CompactConversationOptions{Instructions: "Summarize."},
	}, userPrompt("hi"), ProviderNameAnthropic)
	if !reflect.DeepEqual(body["compaction"], map[string]any{"type": "summarize", "instructions": "Summarize."}) || body["context_management"] != nil {
		t.Fatalf("on-demand compaction = %#v", body)
	}
	if !reflect.DeepEqual(betas, []string{claudeOnDemandCompactionBeta}) {
		t.Fatalf("betas = %v", betas)
	}
	body, betas = payloadFor(t, ExecutionOptions{Model: "claude-opus-4-8"}, userPrompt("hi"), ProviderNameAnthropic)
	if body["compaction"] != nil || body["context_management"] != nil || len(betas) != 0 {
		t.Fatalf("no compaction = %#v %v", body, betas)
	}
}

func TestClaudePayloadNativeResultSchema(t *testing.T) {
	t.Parallel()

	schema := map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}, "required": []any{"answer"}, "additionalProperties": false}
	options := ExecutionOptions{Model: "claude-sonnet-4-5", ResultSchema: schema, ModelOptions: map[string]any{"effort": "high"}}
	for _, provider := range []string{ProviderNameBedrock, ProviderNameVertexAI} {
		body, _ := payloadFor(t, options, userPrompt("hi"), provider)
		outputConfig := body["output_config"].(map[string]any)
		format := outputConfig["format"].(map[string]any)
		if outputConfig["effort"] != "high" || format["type"] != "json_schema" || !reflect.DeepEqual(format["schema"], jsonValue(t, schema)) {
			t.Fatalf("%s: output_config = %#v", provider, outputConfig)
		}
		if body["system"] != nil {
			t.Fatalf("%s: native schemas need no instructions: %#v", provider, body["system"])
		}
	}
	// The Anthropic API keeps the prompt path, and its prompt already carries the instructions.
	body, _ := payloadFor(t, options, userPrompt("hi"), ProviderNameAnthropic)
	if _, ok := body["output_config"].(map[string]any)["format"]; ok {
		t.Fatalf("anthropic output_config = %#v", body["output_config"])
	}
	// A refused or prefilled request adds the instructions the prompt left out.
	for name, test := range map[string]struct {
		prompt claudePrompt
		allow  bool
	}{
		"denied":    {userPrompt("hi"), false},
		"prefilled": {claudePrompt{Messages: []claudeMessage{{Role: "user", Content: []claudeBlock{{Type: "text", Text: "hi"}}}, {Role: "assistant", Content: []claudeBlock{{Type: "text", Text: "{"}}}}}, true},
	} {
		raw, _, err := buildClaudePayload(ExecutionOptions{Model: "claude-sonnet-4-5", ResultSchema: schema}, test.prompt, claudePayloadSettings{Provider: ProviderNameBedrock, AllowNativeResultSchema: test.allow})
		if err != nil {
			t.Fatal(err)
		}
		body := jsonValue(t, raw).(map[string]any)
		system := body["system"].([]any)
		if body["output_config"] != nil || len(system) != 1 || !strings.HasPrefix(system[0].(map[string]any)["text"].(string), claudeJSONSchemaInstructionPrefix) {
			t.Fatalf("%s: prompt path = %#v", name, body)
		}
	}
	// Instructions already in the prompt are not added twice.
	prompt := userPrompt("hi")
	prompt.System = []claudeBlock{{Type: "text", Text: claudeSchemaInstructions(ExecutionOptions{ResultSchema: schema})}}
	raw, _, _ := buildClaudePayload(ExecutionOptions{Model: "claude-sonnet-4-5", ResultSchema: schema}, prompt, claudePayloadSettings{Provider: ProviderNameVertexAI})
	if system := jsonValue(t, raw).(map[string]any)["system"].([]any); len(system) != 1 {
		t.Fatalf("duplicate instructions = %#v", system)
	}
	// Tools keep the prompt path.
	withTools := options
	withTools.Tools = []ToolDefinition{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}}
	if UsesNativeResultSchema(withTools) {
		t.Fatal("a request with tools must keep the prompt path")
	}
}

func TestClaudePayloadDoesNotModifyTheConversation(t *testing.T) {
	t.Parallel()

	conversation := claudePrompt{
		System: []claudeBlock{{Type: "text", Text: "system"}},
		Messages: []claudeMessage{
			{Role: "user", Content: []claudeBlock{{Type: "text", Text: "q1"}}},
			{Role: "assistant", Content: []claudeBlock{{Type: "tool_use", ID: "t1", Name: "lookup", Input: map[string]any{}}}},
			{Role: "user", Content: []claudeBlock{{Type: "text", Text: "q2"}}},
		},
	}
	before, _ := json.Marshal(conversation)
	_, _, err := buildClaudePayload(ExecutionOptions{Model: "claude-sonnet-4-6", PromptCacheKey: "agent-run", Tools: []ToolDefinition{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}}}, conversation, claudePayloadSettings{Provider: ProviderNameAnthropic})
	if err != nil {
		t.Fatal(err)
	}
	if after, _ := json.Marshal(conversation); string(after) != string(before) {
		t.Fatalf("conversation changed:\n%s\n%s", before, after)
	}
}
