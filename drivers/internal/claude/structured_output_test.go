package claude

import (
	"errors"
	"fmt"
	"testing"
)

// structuredSchema is a schema within the supported subset, with extra keys merged in (nil deletes a key).
func structuredSchema(extra map[string]any) map[string]any {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"tags": map[string]any{"type": "array", "items": map[string]any{"type": "string", "enum": []any{"a", "b"}}, "minItems": float64(1)},
			"address": map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"city": map[string]any{"type": "string"}},
				"required":             []any{"city"},
				"additionalProperties": false,
			},
		},
		"required":             []any{"name", "tags", "address"},
		"additionalProperties": false,
	}
	for key, value := range extra {
		if value == nil {
			delete(schema, key)
		} else {
			schema[key] = value
		}
	}
	return schema
}

// prop is a properties map with one property.
func prop(name string, schema map[string]any) map[string]any {
	return map[string]any{name: schema}
}

func TestClaudeSupportsNativeResultSchema(t *testing.T) {
	t.Parallel()

	for model, want := range map[string]bool{
		"us.anthropic.claude-sonnet-4-5-20250929-v1:0": true,
		"anthropic.claude-haiku-4-5-20251001-v1:0":     true,
		"global.anthropic.claude-opus-4-6-v1":          true,
		"us.anthropic.claude-fable-5-20260115-v1:0":    true,
		"claude-sonnet-4-5@20250929":                   true,
		"anthropic.claude-sonnet-4-20250514-v1:0":      false,
		"anthropic.claude-3-7-sonnet-20250219-v1:0":    false,
	} {
		if got := claudeSupportsNativeResultSchema(model); got != want {
			t.Fatalf("%s: %v, want %v", model, got, want)
		}
	}
}

func TestClaudeStructuredOutputSchema(t *testing.T) {
	t.Parallel()

	if !isClaudeStructuredOutputSchema(structuredSchema(nil)) {
		t.Fatal("the base schema qualifies")
	}
	place := map[string]any{"type": "object", "properties": prop("city", map[string]any{"type": "string"}), "required": []any{"city"}, "additionalProperties": false}
	if !isClaudeStructuredOutputSchema(structuredSchema(map[string]any{"properties": prop("home", map[string]any{"$ref": "#/$defs/Place"}), "required": []any{"home"}, "$defs": map[string]any{"Place": place}})) {
		t.Fatal("internal non-recursive $refs qualify")
	}
	rejected := map[string]map[string]any{
		"a non-object root":                    {"type": "array"},
		"no additionalProperties false":        {"additionalProperties": nil},
		"an open nested object":                {"properties": prop("a", map[string]any{"type": "object", "properties": prop("b", map[string]any{"type": "string"})})},
		"a numeric bound":                      {"properties": prop("n", map[string]any{"type": "number", "minimum": 0})},
		"a string length bound":                {"properties": prop("s", map[string]any{"type": "string", "maxLength": 10})},
		"minItems above 1":                     {"properties": prop("l", map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "minItems": float64(2)})},
		"a non-primitive enum value":           {"properties": prop("e", map[string]any{"enum": []any{map[string]any{"a": 1}}})},
		"oneOf":                                {"properties": prop("o", map[string]any{"oneOf": []any{map[string]any{"type": "string"}}})},
		"maxItems":                             {"properties": prop("l", map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "maxItems": 3})},
		"maxProperties":                        {"maxProperties": 3},
		"a backreference":                      {"properties": prop("p", map[string]any{"type": "string", "pattern": `^(a)\1$`})},
		"a lookahead":                          {"properties": prop("p", map[string]any{"type": "string", "pattern": `^(?=a).*$`})},
		"a word boundary":                      {"properties": prop("p", map[string]any{"type": "string", "pattern": `\bword\b`})},
		"an allOf $ref":                        {"properties": prop("a", map[string]any{"allOf": []any{map[string]any{"$ref": "#/$defs/Place"}}}), "$defs": map[string]any{"Place": place}},
		"an external $ref":                     {"properties": prop("r", map[string]any{"$ref": "https://example.com/schema.json"})},
		"a $ref to the root":                   {"properties": prop("self", map[string]any{"$ref": "#"})},
		"a $ref to a missing definition":       {"properties": prop("r", map[string]any{"$ref": "#/$defs/Missing"})},
		"recursive definitions":                {"properties": prop("node", map[string]any{"$ref": "#/$defs/Node"}), "$defs": map[string]any{"Node": map[string]any{"type": "object", "properties": prop("child", map[string]any{"$ref": "#/$defs/Node"}), "additionalProperties": false}}},
		"an unsupported string format":         {"properties": prop("v", map[string]any{"type": "string", "format": "uri-reference"})},
		"a union-typed object without closure": {"properties": prop("u", map[string]any{"type": []any{"object", "null"}})},
	}
	for name, extra := range rejected {
		if isClaudeStructuredOutputSchema(structuredSchema(extra)) {
			t.Fatalf("%s must be rejected", name)
		}
	}
	for _, format := range []string{"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"} {
		if !isClaudeStructuredOutputSchema(structuredSchema(map[string]any{"properties": prop("v", map[string]any{"type": "string", "format": format})})) {
			t.Fatalf("format %s is supported", format)
		}
	}
	properties := func(n int, schema map[string]any) map[string]any {
		out := map[string]any{}
		for i := 0; i < n; i++ {
			out[fmt.Sprintf("p%d", i)] = schema
		}
		return out
	}
	names := func(n int) []any {
		out := make([]any, n)
		for i := range out {
			out[i] = fmt.Sprintf("p%d", i)
		}
		return out
	}
	str := map[string]any{"type": "string"}
	if !isClaudeStructuredOutputSchema(structuredSchema(map[string]any{"properties": properties(12, str), "required": []any{}})) ||
		isClaudeStructuredOutputSchema(structuredSchema(map[string]any{"properties": properties(13, str), "required": []any{}})) {
		t.Fatal("at most 12 optional properties")
	}
	union := map[string]any{"type": []any{"string", "null"}}
	if !isClaudeStructuredOutputSchema(structuredSchema(map[string]any{"properties": properties(16, union), "required": names(16)})) ||
		isClaudeStructuredOutputSchema(structuredSchema(map[string]any{"properties": properties(17, union), "required": names(17)})) {
		t.Fatal("at most 16 union properties")
	}
}

func TestClaudeEndsWithPrefill(t *testing.T) {
	t.Parallel()

	user := claudeMessage{Role: "user", Content: []claudeBlock{text("q")}}
	assistant := claudeMessage{Role: "assistant", Content: []claudeBlock{text("{")}}
	system := claudeMessage{Role: "system", Content: []claudeBlock{{Type: "tool_addition", Tool: &claudeToolReference{Name: "a"}}}}
	if endsWithClaudePrefill([]claudeMessage{user}) || !endsWithClaudePrefill([]claudeMessage{user, assistant}) || !endsWithClaudePrefill([]claudeMessage{user, assistant, system}) || endsWithClaudePrefill(nil) {
		t.Fatal("prefill detection")
	}
}

// statusError is an HTTP error with a status code.
type statusError struct {
	code    int
	message string
}

func (e statusError) Error() string   { return fmt.Sprintf("HTTP %d: %s", e.code, e.message) }
func (e statusError) StatusCode() int { return e.code }

func TestClaudeStructuredOutputPolicyDenial(t *testing.T) {
	t.Parallel()

	denial := statusError{400, `[{"error":{"code":400,"message":"Organization Policy constraint constraints/vertexai.allowedPartnerModelFeatures violated for a disallowed feature structured_outputs","status":"FAILED_PRECONDITION"}}]`}
	if !IsStructuredOutputPolicyDenial(denial) || !IsStructuredOutputPolicyDenial(fmt.Errorf("wrapped: %w", denial)) {
		t.Fatal("the policy denial is matched")
	}
	for name, err := range map[string]error{
		"another 400":             statusError{400, "max_tokens: too large"},
		"another feature":         statusError{400, "FAILED_PRECONDITION allowedPartnerModelFeatures web_search"},
		"another status":          statusError{403, "FAILED_PRECONDITION allowedPartnerModelFeatures structured_outputs"},
		"an error without status": errors.New("FAILED_PRECONDITION allowedPartnerModelFeatures structured_outputs"),
		"nil":                     nil,
	} {
		if IsStructuredOutputPolicyDenial(err) {
			t.Fatalf("%s must not match", name)
		}
	}

	var denials StructuredOutputDenials
	first := StructuredOutputDenialKey("p", "global", "m0")
	denials.Add(first)
	for i := 1; i <= maxRememberedStructuredOutputDenials; i++ {
		denials.Add(StructuredOutputDenialKey("p", "global", fmt.Sprintf("m%d", i)))
	}
	if denials.Has(first) || !denials.Has(StructuredOutputDenialKey("p", "global", "m256")) {
		t.Fatal("the oldest denial is dropped first")
	}
	denials.Add(StructuredOutputDenialKey("p", "global", "m1"))
	denials.Add(StructuredOutputDenialKey("p", "global", "new"))
	if !denials.Has(StructuredOutputDenialKey("p", "global", "m1")) || denials.Has(StructuredOutputDenialKey("p", "global", "m2")) {
		t.Fatal("adding a key again makes it the newest")
	}
}
