package openai

// StructuredOutputSchema prepares a JSON schema for OpenAI structured output, as
// the Responses API path does: the strict-mode transform and true when the
// schema allows it, else the limited non-strict transform and false. It is
// exported for drivers that speak another OpenAI protocol on the same models
// (Chat Completions on Microsoft Foundry).
func StructuredOutputSchema(schema map[string]any) (map[string]any, bool) {
	return openAISchemaForResponses(schema)
}

// LimitedSchema returns the non-strict schema transform: defaults dropped and a
// top-level object type ensured. It suits OpenAI-compatible endpoints that do
// not implement strict mode.
func LimitedSchema(schema map[string]any) map[string]any {
	return openAILimitedSchema(schema)
}
