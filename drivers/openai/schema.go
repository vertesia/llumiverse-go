package openai

import (
	"errors"
	"slices"
)

// openAIStrictStringFormats are the string formats OpenAI strict mode accepts.
var openAIStrictStringFormats = map[string]bool{
	"date-time": true, "time": true, "date": true, "duration": true, "email": true,
	"hostname": true, "ipv4": true, "ipv6": true, "uuid": true,
}

// formatOpenAISchema prepares a JSON schema for OpenAI structured output or
// tool parameters. A schema already inside the strict subset is sent strict
// (with defaults removed); any other schema is sent non-strict through
// limitedSchemaFormat, rather than narrowed by making optional fields required
// or forbidding extra keys behind the caller's back.
func formatOpenAISchema(schema map[string]any) (map[string]any, bool) {
	if err := assertOpenAIStrictSchemaContract(schema); err == nil {
		strict := cloneJSONValue(schema).(map[string]any)
		toStrictJSONSchema(strict)
		return strict, true
	}
	return limitedSchemaFormat(schema), false
}

// assertOpenAIStrictSchemaContract fails when a schema uses what strict mode
// rejects: unsupported formats, oneOf, or objects that are not closed
// (additionalProperties false) with every property required.
func assertOpenAIStrictSchemaContract(schema map[string]any) error {
	if format, ok := schema["format"]; ok {
		if name, _ := format.(string); !openAIStrictStringFormats[name] {
			return errors.New("unsupported OpenAI strict schema format")
		}
	}
	if _, ok := schema["oneOf"]; ok {
		return errors.New("OpenAI strict schemas do not support oneOf")
	}
	properties, hasProperties := schema["properties"].(map[string]any)
	if hasSchemaType(schema, "object") || hasProperties {
		required := stringList(schema["required"])
		if schema["additionalProperties"] != false || len(properties) == 0 || len(required) != len(properties) {
			return errors.New("schema changes object optionality or additional properties in OpenAI strict mode")
		}
		for _, name := range required {
			if _, ok := properties[name]; !ok {
				return errors.New("schema requires a property it does not define")
			}
		}
	}
	var err error
	forEachSchemaChild(schema, func(child map[string]any) {
		if err == nil {
			err = assertOpenAIStrictSchemaContract(child)
		}
	})
	return err
}

// toStrictJSONSchema applies the parts of the OpenAI SDK strict transform a
// conforming schema still needs: a single allOf is merged into its parent, and
// defaults (which strict mode rejects) are removed.
func toStrictJSONSchema(schema map[string]any) {
	if allOf, ok := schema["allOf"].([]any); ok && len(allOf) == 1 {
		if only, ok := allOf[0].(map[string]any); ok {
			delete(schema, "allOf")
			for key, value := range only {
				schema[key] = value
			}
		}
	}
	delete(schema, "default")
	forEachSchemaChild(schema, toStrictJSONSchema)
}

// limitedSchemaFormat is the non-strict form: defaults are removed, and a
// described property without a type gets one (object for documents and editor
// hints, string otherwise), since OpenAI requires a type.
func limitedSchemaFormat(schema map[string]any) map[string]any {
	out := make(map[string]any, len(schema))
	for key, value := range schema {
		if key != "default" {
			out[key] = value
		}
	}
	if _, hasType := out["type"]; !hasType && out["description"] != nil {
		if out["format"] == "document" || out["editor"] != nil {
			out["type"] = "object"
		} else {
			out["type"] = "string"
		}
	}
	mapChildren := func(key string) {
		if children, ok := out[key].(map[string]any); ok {
			next := make(map[string]any, len(children))
			for name, raw := range children {
				if child, ok := raw.(map[string]any); ok {
					next[name] = limitedSchemaFormat(child)
				} else {
					next[name] = raw
				}
			}
			out[key] = next
		}
	}
	mapChildren("properties")
	mapChildren("$defs")
	if variants, ok := out["anyOf"].([]any); ok {
		next := make([]any, len(variants))
		for i, raw := range variants {
			if variant, ok := raw.(map[string]any); ok {
				next[i] = limitedSchemaFormat(variant)
			} else {
				next[i] = raw
			}
		}
		out["anyOf"] = next
	}
	if items, ok := out["items"].(map[string]any); ok {
		out["items"] = limitedSchemaFormat(items)
	}
	return out
}

// forEachSchemaChild visits the subschemas of a schema.
func forEachSchemaChild(schema map[string]any, visit func(map[string]any)) {
	for _, key := range []string{"properties", "$defs", "definitions"} {
		if children, ok := schema[key].(map[string]any); ok {
			for _, raw := range children {
				if child, ok := raw.(map[string]any); ok {
					visit(child)
				}
			}
		}
	}
	for _, key := range []string{"items", "additionalProperties", "not", "contains"} {
		if child, ok := schema[key].(map[string]any); ok {
			visit(child)
		}
	}
	for _, key := range []string{"anyOf", "allOf", "oneOf", "prefixItems", "items"} {
		if children, ok := schema[key].([]any); ok {
			for _, raw := range children {
				if child, ok := raw.(map[string]any); ok {
					visit(child)
				}
			}
		}
	}
}

func hasSchemaType(schema map[string]any, name string) bool {
	switch value := schema["type"].(type) {
	case string:
		return value == name
	case []any:
		return slices.Contains(value, any(name))
	case []string:
		return slices.Contains(value, name)
	}
	return false
}

func stringList(value any) []string {
	switch list := value.(type) {
	case []string:
		return list
	case []any:
		out := make([]string, 0, len(list))
		for _, item := range list {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}
