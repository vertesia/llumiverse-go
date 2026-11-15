package claude

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
	"sync"
)

// Native Claude structured outputs (output_config.format) on Amazon Bedrock and Vertex AI, as in the TS client's
// claude-structured-output.ts. Both platforms accept only a subset of JSON Schema and answer 400 for anything
// else, so a request uses the native format only when the model and the schema both qualify; every other request
// keeps the schema instructions in the prompt.

// claudeNativeResultSchemaProviders receive the result schema as output_config.format when the request qualifies.
var claudeNativeResultSchemaProviders = map[string]bool{ProviderNameBedrock: true, ProviderNameVertexAI: true}

const (
	// claudeMaxOptionalProperties is the platforms' limit on optional properties, counted across nested objects.
	claudeMaxOptionalProperties = 12
	// claudeMaxUnionProperties is the platforms' limit on union-typed properties (type arrays or anyOf).
	claudeMaxUnionProperties = 16
)

// claudeUnsupportedSchemaKeywords are keywords the platforms reject for Claude structured outputs.
var claudeUnsupportedSchemaKeywords = []string{
	"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength",
	"maxItems", "uniqueItems", "contains", "minContains", "maxContains", "additionalItems",
	"minProperties", "maxProperties", "propertyNames", "dependentRequired", "dependencies",
	"oneOf", "not", "if", "then", "else", "patternProperties", "dependentSchemas", "unevaluatedProperties", "prefixItems",
}

// claudeSupportedStringFormats are the only string formats the platforms accept.
var claudeSupportedStringFormats = map[string]bool{
	"date-time": true, "time": true, "date": true, "duration": true, "email": true,
	"hostname": true, "uri": true, "ipv4": true, "ipv6": true, "uuid": true,
}

// claudeUnsupportedPattern matches regex features the platforms reject: backreferences, lookaround and word
// boundaries.
var claudeUnsupportedPattern = regexp.MustCompile(`\\[1-9]|\\k<|\(\?<?[=!]|\\[bB]`)

// UsesNativeResultSchema reports whether a Claude request on Bedrock or Vertex AI sends its result schema as
// output_config.format instead of prompt text: a result schema, no tools, Claude 4.5+, and a schema within the
// supported subset. Requests with tools keep the prompt path.
func UsesNativeResultSchema(options ExecutionOptions) bool {
	return options.ResultSchema != nil && len(options.Tools) == 0 &&
		claudeSupportsNativeResultSchema(options.Model) && isClaudeStructuredOutputSchema(options.ResultSchema)
}

// claudeSchemaStats counts the properties the platforms limit.
type claudeSchemaStats struct {
	optional, union int
}

// isClaudeStructuredOutputSchema reports whether a JSON schema stays within what the platforms accept: an object
// root, additionalProperties false on every object, none of the unsupported keywords, minItems 0 or 1, supported
// string formats, primitive enum values, only internal non-recursive $refs, and the property limits.
func isClaudeStructuredOutputSchema(schema map[string]any) bool {
	if schema == nil || schema["type"] != "object" {
		return false
	}
	definitions := map[string]any{}
	for _, container := range []string{"$defs", "definitions"} {
		if node, ok := schema[container].(map[string]any); ok {
			for name, definition := range node {
				definitions["#/"+container+"/"+name] = definition
			}
		}
	}
	if hasRecursiveClaudeSchemaReference(schema, definitions) {
		return false
	}
	var stats claudeSchemaStats
	return checkClaudeSchemaNode(schema, &stats) &&
		stats.optional <= claudeMaxOptionalProperties && stats.union <= claudeMaxUnionProperties
}

// jsonNumberEquals reports whether a decoded JSON value is the number n.
func jsonNumberEquals(value any, n float64) bool {
	switch v := value.(type) {
	case float64:
		return v == n
	case int:
		return float64(v) == n
	case int64:
		return float64(v) == n
	default:
		return fmt.Sprint(value) == fmt.Sprint(n)
	}
}

// checkClaudeSchemaNode checks one schema node and its children.
func checkClaudeSchemaNode(value any, stats *claudeSchemaStats) bool {
	node, ok := value.(map[string]any)
	if !ok {
		return false
	}
	for _, keyword := range claudeUnsupportedSchemaKeywords {
		if _, ok := node[keyword]; ok {
			return false
		}
	}
	if minItems, ok := node["minItems"]; ok && !jsonNumberEquals(minItems, 0) && !jsonNumberEquals(minItems, 1) {
		return false
	}
	if enum, ok := node["enum"]; ok {
		values, ok := enum.([]any)
		if !ok {
			return false
		}
		for _, item := range values {
			switch item.(type) {
			case nil, string, bool, float64, int, int64:
			default:
				return false
			}
		}
	}
	if ref, ok := node["$ref"].(string); ok && !strings.HasPrefix(ref, "#/") {
		return false
	}
	if format, ok := node["format"]; ok {
		name, _ := format.(string)
		if !claudeSupportedStringFormats[name] {
			return false
		}
	}
	if pattern, ok := node["pattern"]; ok {
		text, isString := pattern.(string)
		if !isString || claudeUnsupportedPattern.MatchString(text) {
			return false
		}
	}
	if allOf, ok := node["allOf"].([]any); ok {
		for _, branch := range allOf {
			if branchNode, ok := branch.(map[string]any); ok {
				if _, hasRef := branchNode["$ref"]; hasRef {
					return false
				}
			}
		}
	}

	isObject := node["type"] == "object"
	if types, ok := node["type"].([]any); ok && slices.Contains(types, any("object")) {
		isObject = true
	}
	if _, hasProperties := node["properties"]; isObject || hasProperties {
		if additional, ok := node["additionalProperties"].(bool); !ok || additional {
			return false
		}
		properties, _ := node["properties"].(map[string]any)
		required := map[string]bool{}
		if names, ok := node["required"].([]any); ok {
			for _, name := range names {
				if s, ok := name.(string); ok {
					required[s] = true
				}
			}
		}
		for name, property := range properties {
			if !required[name] {
				stats.optional++
			}
			if propertyNode, ok := property.(map[string]any); ok {
				_, isArray := propertyNode["type"].([]any)
				_, hasAnyOf := propertyNode["anyOf"]
				if isArray || hasAnyOf {
					stats.union++
				}
			}
			if !checkClaudeSchemaNode(property, stats) {
				return false
			}
		}
	}
	if items, ok := node["items"]; ok && !checkClaudeSchemaNode(items, stats) {
		return false
	}
	for _, keyword := range []string{"anyOf", "allOf"} {
		if value, ok := node[keyword]; ok {
			branches, ok := value.([]any)
			if !ok {
				return false
			}
			for _, branch := range branches {
				if !checkClaudeSchemaNode(branch, stats) {
					return false
				}
			}
		}
	}
	for _, container := range []string{"$defs", "definitions"} {
		if value, ok := node[container]; ok {
			definitions, ok := value.(map[string]any)
			if !ok {
				return false
			}
			for _, definition := range definitions {
				if !checkClaudeSchemaNode(definition, stats) {
					return false
				}
			}
		}
	}
	return true
}

// collectClaudeSchemaReferences collects every $ref string under a node.
func collectClaudeSchemaReferences(value any, references map[string]bool) map[string]bool {
	switch node := value.(type) {
	case []any:
		for _, item := range node {
			collectClaudeSchemaReferences(item, references)
		}
	case map[string]any:
		for key, child := range node {
			if ref, ok := child.(string); ok && key == "$ref" {
				references[ref] = true
			} else {
				collectClaudeSchemaReferences(child, references)
			}
		}
	}
	return references
}

// hasRecursiveClaudeSchemaReference reports a reference to the root, to an unknown target, or a cycle between
// definitions.
func hasRecursiveClaudeSchemaReference(schema map[string]any, definitions map[string]any) bool {
	for ref := range collectClaudeSchemaReferences(schema, map[string]bool{}) {
		if _, known := definitions[ref]; ref == "#" || (strings.HasPrefix(ref, "#/") && !known) {
			return true
		}
	}
	visiting := map[string]bool{}
	done := map[string]bool{}
	var visit func(ref string) bool
	visit = func(ref string) bool {
		if done[ref] {
			return false
		}
		if visiting[ref] {
			return true
		}
		visiting[ref] = true
		for next := range collectClaudeSchemaReferences(definitions[ref], map[string]bool{}) {
			if next == "#" || visit(next) {
				return true
			}
		}
		delete(visiting, ref)
		done[ref] = true
		return false
	}
	for ref := range definitions {
		if visit(ref) {
			return true
		}
	}
	return false
}

// endsWithClaudePrefill reports whether a request ends with an assistant turn, which Claude continues as a
// prefill. Structured outputs reject a prefill, so such a request keeps the schema instructions in the prompt.
func endsWithClaudePrefill(messages []claudeMessage) bool {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "system" {
			return messages[i].Role == "assistant"
		}
	}
	return false
}

// IsStructuredOutputPolicyDenial reports whether an error is Vertex AI refusing structured outputs because an
// organization policy (constraints/vertexai.allowedPartnerModelFeatures) does not allow the structured_outputs
// feature for the model: a 400 FAILED_PRECONDITION. Any other 400 is a different problem and is not matched.
func IsStructuredOutputPolicyDenial(err error) bool {
	if err == nil {
		return false
	}
	code, _ := errorStatusAndName(err)
	if code != 400 {
		return false
	}
	text := err.Error()
	return strings.Contains(text, "FAILED_PRECONDITION") &&
		strings.Contains(text, "allowedPartnerModelFeatures") &&
		strings.Contains(text, "structured_outputs")
}

// maxRememberedStructuredOutputDenials bounds StructuredOutputDenials; the oldest entry is dropped first.
const maxRememberedStructuredOutputDenials = 256

// StructuredOutputDenials remembers the project, region and model combinations whose structured outputs a policy
// denied, so later requests in the process go straight to the prompt path instead of failing once each. It is
// safe for concurrent use; the zero value is ready.
type StructuredOutputDenials struct {
	mu    sync.Mutex
	order []string
	set   map[string]bool
}

// StructuredOutputDenialKey is the key of a project, region and model combination.
func StructuredOutputDenialKey(project, region, model string) string {
	return project + "|" + region + "|" + model
}

// Has reports whether the combination was denied.
func (d *StructuredOutputDenials) Has(key string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.set[key]
}

// Add records a denied combination, dropping the oldest one past the limit.
func (d *StructuredOutputDenials) Add(key string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.set == nil {
		d.set = map[string]bool{}
	}
	if d.set[key] {
		d.order = slices.DeleteFunc(d.order, func(existing string) bool { return existing == key })
	}
	d.set[key] = true
	d.order = append(d.order, key)
	if len(d.order) > maxRememberedStructuredOutputDenials {
		delete(d.set, d.order[0])
		d.order = d.order[1:]
	}
}
