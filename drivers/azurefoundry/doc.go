// Package azurefoundry implements llumiverse drivers for Microsoft Foundry
// (formerly Azure AI Foundry) and Azure OpenAI.
//
// AzureFoundryDriver routes each deployment to the protocol its model speaks:
// OpenAI models use the Responses API (through the shared drivers/openai
// implementation), Claude uses the Anthropic Messages API on Foundry (through
// the shared drivers/internal/claude implementation), and every other model
// (Llama, DeepSeek, Mistral, Grok, Phi, ...) uses Chat Completions on the
// Foundry OpenAI v1 endpoint. Image models use the Images API and embedding
// models the embeddings API of the resource endpoint.
//
// Model IDs are deployment names, optionally qualified with the source model as
// "deployment::source-model" (the form ListModels returns). The source model
// selects the protocol by model family; a bare deployment name is resolved
// through the Foundry project API, or from AzureFoundryOptions.SourceModel.
//
// AzureOpenAIDriver targets the version-less Azure OpenAI v1 API
// (<resource>/openai/v1) with the same shared OpenAI implementation.
//
// Both drivers authenticate with a resource key or a Microsoft Entra ID token.
// Tokens come from a caller-supplied TokenProvider, so the library takes no
// dependency on the Azure SDK and never reads the environment; the scope it asks
// for follows the endpoint's cloud (public or Azure Government).
package azurefoundry
