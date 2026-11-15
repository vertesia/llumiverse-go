// Package capability is the model capability catalogue: what a model takes and produces, whether it calls tools,
// its context and output limits and the reasoning efforts it accepts, by provider.
//
// Rules are written by family and version rather than as model lists, so a newly released model inherits the
// newest known behaviour of its family. ResolveProfile is the entry point: it reduces a model ID to its source
// model (dropping paths, region and publisher prefixes, Azure Foundry deployment names and OpenRouter routing
// suffixes), applies the family rules, then what the provider serves differently. ForModel and SupportsToolUse
// answer the common questions, and ResolveListingMetadata fills AIModel's modalities and tool support in model
// listings, falling back to the provider's own metadata for models no rule knows.
//
// The provider tables (AnthropicCapabilities, OpenAICapabilities, VertexAICapabilities, AzureFoundryCapabilities,
// BedrockCapabilities, MistralModelKnowledge) and limits (ContextWindowSize, MaxOutputTokens, GeminiModelLimits,
// VertexAIModelLimits, AzureFoundryModelLimits, BedrockModelKnowledge) are exported for callers that need one
// directly. Limits of 0 are unknown.
//
// The catalogue follows the TypeScript llumiverse catalogue (common/src/capability and model-directory.ts).
package capability
