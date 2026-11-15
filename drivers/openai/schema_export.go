package openai

// StructuredOutputSchema prepares a JSON schema for OpenAI structured output, as
// the Responses API path does: a schema inside the strict subset is returned
// strict (defaults removed) with true; any other schema gets the limited
// non-strict transform and false. It is exported for drivers that speak
// another OpenAI protocol on the same models (Chat Completions on Microsoft
// Foundry).
func StructuredOutputSchema(schema map[string]any) (map[string]any, bool) {
	return formatOpenAISchema(schema)
}

// LimitedSchema returns the non-strict schema transform: defaults dropped,
// described properties given a type, and a top-level object type ensured. It
// suits OpenAI-compatible endpoints that do not implement strict mode.
func LimitedSchema(schema map[string]any) map[string]any {
	top := make(map[string]any, len(schema)+1)
	for key, value := range schema {
		top[key] = value
	}
	if _, ok := top["type"]; !ok {
		top["type"] = "object"
	}
	return limitedSchemaFormat(top)
}
