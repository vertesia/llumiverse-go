package openai

import (
	"reflect"
	"testing"
)

// Mirrors the TS formatOpenAISchema tests: schemas inside the strict subset are
// sent strict unchanged (bar defaults); schemas outside it are sent non-strict
// instead of being narrowed.
func TestFormatOpenAISchema(t *testing.T) {
	t.Parallel()

	conforming := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{"anyOf": []any{
				map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}}, "required": []any{"name"}, "additionalProperties": false},
				map[string]any{"type": "null"},
			}},
			"email": map[string]any{"type": "string", "format": "email", "default": "a@b.c"},
			"unit":  map[string]any{"type": []any{"string", "null"}},
		},
		"required":             []any{"value", "email", "unit"},
		"additionalProperties": false,
	}
	schema, strict := formatOpenAISchema(conforming)
	if !strict {
		t.Fatal("conforming schema is not strict")
	}
	email := schema["properties"].(map[string]any)["email"].(map[string]any)
	if _, ok := email["default"]; ok || email["format"] != "email" {
		t.Fatalf("email = %#v", email)
	}
	if !reflect.DeepEqual(schema["properties"].(map[string]any)["value"], conforming["properties"].(map[string]any)["value"]) {
		t.Fatalf("anyOf changed: %#v", schema["properties"])
	}
	if _, ok := conforming["properties"].(map[string]any)["email"].(map[string]any)["default"]; !ok {
		t.Fatal("caller schema was modified")
	}

	optional := map[string]any{
		"type":       "object",
		"properties": map[string]any{"a": map[string]any{"type": "string"}, "b": map[string]any{"type": "string"}},
		"required":   []any{"a"},
	}
	schema, strict = formatOpenAISchema(optional)
	if strict || !reflect.DeepEqual(schema, optional) {
		t.Fatalf("optional fields: strict %v, schema %#v", strict, schema)
	}

	for name, candidate := range map[string]map[string]any{
		"oneOf":  {"oneOf": []any{map[string]any{"type": "string"}}},
		"format": {"type": "string", "format": "uri"},
		"empty":  {"type": "object", "additionalProperties": false},
	} {
		if _, strict := formatOpenAISchema(candidate); strict {
			t.Errorf("%s schema is strict", name)
		}
	}

	limited := limitedSchemaFormat(map[string]any{
		"properties": map[string]any{
			"doc":  map[string]any{"description": "a document", "format": "document"},
			"note": map[string]any{"description": "a note", "default": "x"},
		},
	})
	props := limited["properties"].(map[string]any)
	if limited["type"] != nil || props["doc"].(map[string]any)["type"] != "object" || props["note"].(map[string]any)["type"] != "string" || props["note"].(map[string]any)["default"] != nil {
		t.Fatalf("limited = %#v", limited)
	}
}
