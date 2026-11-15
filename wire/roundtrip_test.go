package wire

import (
	"encoding/json"
	"reflect"
	"testing"
)

// Each case is JSON the TypeScript llumiverse produces; decoding it and encoding it again must give
// back the same JSON, field for field.
func TestRoundTripMatchesTypeScriptJSON(t *testing.T) {
	cases := []struct {
		name string
		into any
		json string
	}{
		{"prompt segment", &PromptSegment{}, `{"role":"tool","content":"42","tool_use_id":"call_1","thought_signature":"sig","files":[{"name":"a.png","mime_type":"image/png"}]}`},
		{"tool use with input", &ToolUse{}, `{"id":"call_1","tool_name":"lookup","tool_input":{"q":"x","n":2,"nested":{"ok":true}}}`},
		{"tool use with null input", &ToolUse{}, `{"id":"call_1","tool_name":"lookup","tool_input":null}`},
		{"tool definition", &ToolDefinition{}, `{"name":"lookup","description":"Find","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}`},
		{"token usage", &ExecutionTokenUsage{}, `{"prompt":1200,"result":34,"total":1234,"prompt_cached":1000,"prompt_new":200,"provider_cost_usd":0.0012}`},
		{"text result", &CompletionResult{}, `{"type":"text","value":"hello"}`},
		{"json result", &CompletionResult{}, `{"type":"json","value":{"a":[1,2,{"b":null}]}}`},
		{"audio result", &CompletionResult{}, `{"type":"audio","value":"gs://b/s.wav","mime_type":"audio/wav","sample_rate":24000,"channels":1}`},
		{"model options", &ModelOptions{}, `{"_option_id":"anthropic-claude","max_tokens":1024,"thinking_budget_tokens":2048}`},
		{"embeddings", &EmbeddingsResult{}, `{"model":"m","results":[{"outputs":[{"values":[0.1,0.2],"modality":"text"}],"input_tokens":3}],"usage":{"input_tokens":3}}`},
		{"result schema keeps keywords it does not declare", &StatelessExecutionOptions{}, `{"model":"m","result_schema":{"type":"object","properties":{"kind":{"type":"string","enum":["a","b"]},"n":{"type":"number","minimum":0},"v":{"oneOf":[{"type":"string"},{"type":"integer"}]}},"required":["kind"],"additionalProperties":false}}`},
		{"model", &AIModel{}, `{"id":"claude","name":"Claude","provider":"anthropic","type":"text","can_stream":true,"tags":["chat"]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := json.Unmarshal([]byte(tc.json), tc.into); err != nil {
				t.Fatalf("decode: %v", err)
			}
			out, err := json.Marshal(tc.into)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			assertSameJSON(t, tc.json, string(out))
		})
	}
}

func TestCompletionResultHelpers(t *testing.T) {
	text := NewTextResult("hi")
	if text.Type() != "text" || text.String() != "hi" {
		t.Fatalf("text result = %q %q", text.Type(), text.String())
	}
	data := NewJSONResult(map[string]any{"a": 1})
	if data.Type() != "json" || data.String() != `{"a":1}` {
		t.Fatalf("json result = %q %q", data.Type(), data.String())
	}
	value, err := data.ValueByDiscriminator()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := value.(JSONResult); !ok {
		t.Fatalf("ValueByDiscriminator = %T, want JSONResult", value)
	}
	encoded, err := json.Marshal([]CompletionResult{text, NewImageResult("gs://b/i.png")})
	if err != nil {
		t.Fatal(err)
	}
	assertSameJSON(t, `[{"type":"text","value":"hi"},{"type":"image","value":"gs://b/i.png"}]`, string(encoded))
}

func TestModelOptionsHelpers(t *testing.T) {
	options, err := ModelOptionsFromMap(map[string]any{"_option_id": "openai-text", "temperature": 0.2})
	if err != nil {
		t.Fatal(err)
	}
	if options.OptionID() != "openai-text" {
		t.Fatalf("OptionID = %q", options.OptionID())
	}
	typed, err := options.AsOpenAiTextOptions()
	if err != nil {
		t.Fatal(err)
	}
	if typed.Temperature == nil || *typed.Temperature != 0.2 {
		t.Fatalf("temperature = %v", typed.Temperature)
	}
	back, err := options.Map()
	if err != nil || back["temperature"] != 0.2 {
		t.Fatalf("Map = %v, %v", back, err)
	}
	if (ModelOptions{}).OptionID() != "" {
		t.Fatal("empty options have no option id")
	}
}

func assertSameJSON(t *testing.T, want, got string) {
	t.Helper()
	var w, g any
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("want is not JSON: %v", err)
	}
	if err := json.Unmarshal([]byte(got), &g); err != nil {
		t.Fatalf("got is not JSON: %v", err)
	}
	if !reflect.DeepEqual(w, g) {
		t.Fatalf("JSON differs\nwant %s\n got %s", want, got)
	}
}
