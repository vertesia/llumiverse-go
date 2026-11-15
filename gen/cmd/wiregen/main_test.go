package main

import (
	"encoding/json"
	"reflect"
	"testing"
)

func downgraded(t *testing.T, schema string) map[string]any {
	t.Helper()
	var node map[string]any
	if err := json.Unmarshal([]byte(schema), &node); err != nil {
		t.Fatal(err)
	}
	return downgrade(node).(map[string]any)
}

func assertJSON(t *testing.T, got any, want string) {
	t.Helper()
	var w any
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatal(err)
	}
	g, _ := json.Marshal(got)
	var gn any
	_ = json.Unmarshal(g, &gn)
	if !reflect.DeepEqual(gn, w) {
		t.Errorf("got  %s\nwant %s", g, want)
	}
}

func TestDowngradeRewritesNestedSubschemas(t *testing.T) {
	got := downgraded(t, `{
		"type": "object",
		"properties": {
			"kind": {"const": "text"},
			"count": {"type": "number", "exclusiveMinimum": 0},
			"input": {"anyOf": [{"$ref": "#/$defs/Input"}, {"type": "null"}]},
			"tags": {"type": "object", "additionalProperties": {"type": "number"}, "propertyNames": {"type": "string"}},
			"list": {"type": "array", "items": {"oneOf": [{"const": 1}, {"type": "number"}]}}
		}
	}`)
	assertJSON(t, got, `{
		"type": "object",
		"properties": {
			"kind": {"enum": ["text"]},
			"count": {"type": "number", "format": "double", "minimum": 0, "exclusiveMinimum": true},
			"input": {"nullable": true, "allOf": [{"$ref": "#/components/schemas/Input"}]},
			"tags": {"type": "object", "additionalProperties": {"type": "number", "format": "double"}},
			"list": {"type": "array", "items": {"oneOf": [{"enum": [1]}, {"type": "number", "format": "double"}]}}
		}
	}`)
}

// Field names are data: a field named after a keyword must keep its name and its own schema.
func TestDowngradeKeepsFieldsNamedAfterKeywords(t *testing.T) {
	got := downgraded(t, `{
		"type": "object",
		"properties": {
			"const": {"type": "string"},
			"propertyNames": {"type": "string", "minLength": 1},
			"type": {"type": "string"},
			"$ref": {"type": "string"},
			"anyOf": {"type": "string"}
		},
		"required": ["const", "propertyNames"]
	}`)
	assertJSON(t, got, `{
		"type": "object",
		"properties": {
			"const": {"type": "string"},
			"propertyNames": {"type": "string", "minLength": 1},
			"type": {"type": "string"},
			"$ref": {"type": "string"},
			"anyOf": {"type": "string"}
		},
		"required": ["const", "propertyNames"]
	}`)
}

// Values of data keywords are not schemas, even when they look like one.
func TestDowngradeLeavesDataKeywordsAlone(t *testing.T) {
	got := downgraded(t, `{
		"type": "object",
		"default": {"const": 1, "propertyNames": {}},
		"enum": [{"const": 2}],
		"examples": [{"type": "number"}]
	}`)
	assertJSON(t, got, `{
		"type": "object",
		"default": {"const": 1, "propertyNames": {}},
		"enum": [{"const": 2}],
		"examples": [{"type": "number"}]
	}`)
}

func TestDowngradeRewritesDiscriminatorMapping(t *testing.T) {
	got := downgraded(t, `{
		"oneOf": [{"$ref": "#/$defs/A"}],
		"discriminator": {"propertyName": "type", "mapping": {"a": "#/$defs/A"}}
	}`)
	assertJSON(t, got, `{
		"oneOf": [{"$ref": "#/components/schemas/A"}],
		"discriminator": {"propertyName": "type", "mapping": {"a": "#/components/schemas/A"}}
	}`)
}
