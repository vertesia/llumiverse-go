package llumiverse

import (
	"context"

	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
	anthropicdriver "github.com/vertesia/llumiverse-go/drivers/anthropic"
	azuredriver "github.com/vertesia/llumiverse-go/drivers/azurefoundry"
	bedrockdriver "github.com/vertesia/llumiverse-go/drivers/bedrock"
	openaidriver "github.com/vertesia/llumiverse-go/drivers/openai"
	vertexdriver "github.com/vertesia/llumiverse-go/drivers/vertexai"
)

type Provider = common.Provider
type PromptRole = common.PromptRole
type PromptSegment = common.PromptSegment
type DataSource = common.DataSource
type BytesDataSource = common.BytesDataSource
type Driver = common.Driver
type HTTPTimeoutOptions = common.HTTPTimeoutOptions
type DriverOptions = common.DriverOptions
type ExecutionOptions = common.ExecutionOptions
type ToolDefinition = common.ToolDefinition
type ToolUse = common.ToolUse
type ToolUseChunk = common.ToolUseChunk
type JSONObject = common.JSONObject
type ModelType = common.ModelType
type CompletionResult = common.CompletionResult
type ExecutionTokenUsage = common.ExecutionTokenUsage
type PromptCacheMode = common.PromptCacheMode
type PromptCacheDiagnostic = common.PromptCacheDiagnostic
type ToolUsage = common.ToolUsage
type ToolTokenUsage = common.ToolTokenUsage
type AudioResult = common.AudioResult
type ConversationCompactionOptions = common.ConversationCompactionOptions
type CompactConversationOptions = common.CompactConversationOptions
type ResultValidationError = common.ResultValidationError
type Completion = common.Completion
type ExecutionResponse = common.ExecutionResponse
type CompletionChunk = common.CompletionChunk
type CompletionStream = common.CompletionStream
type AIModel = common.AIModel
type ModelSearchPayload = common.ModelSearchPayload
type EmbeddingTaskType = common.EmbeddingTaskType
type EmbeddingInput = common.EmbeddingInput
type EmbeddingsOptions = common.EmbeddingsOptions
type EmbeddingsResult = common.EmbeddingsResult
type EmbeddingResultItem = common.EmbeddingResultItem
type EmbeddingOutput = common.EmbeddingOutput
type EmbeddingsTokenUsage = common.EmbeddingsTokenUsage
type LlumiverseErrorContext = common.LlumiverseErrorContext
type LlumiverseError = common.LlumiverseError

type EmbeddingBatchModality = core.EmbeddingBatchModality
type EmbeddingBatchState = core.EmbeddingBatchState
type EmbeddingBatchCapability = core.EmbeddingBatchCapability
type EmbeddingBatchArtifactStore = core.EmbeddingBatchArtifactStore
type EmbeddingBatchJob = core.EmbeddingBatchJob
type EmbeddingBatchJobOptions = core.EmbeddingBatchJobOptions
type EmbeddingBatchCreateOptions = core.EmbeddingBatchCreateOptions
type EmbeddingBatchRowOptions = core.EmbeddingBatchRowOptions
type ParsedEmbeddingBatchResult = core.ParsedEmbeddingBatchResult
type EmbeddingBatchAdapter = core.EmbeddingBatchAdapter

type OpenAICompatibleOptions = openaidriver.OpenAICompatibleOptions
type OpenAICompatibleDriver = openaidriver.OpenAICompatibleDriver
type OpenAIChatCompletionsOptions = openaidriver.OpenAIChatCompletionsOptions
type OpenAIChatCompletionsDriver = openaidriver.OpenAIChatCompletionsDriver
type ChatCompletionsProtocolOptions = openaidriver.ChatCompletionsProtocolOptions
type AnthropicOptions = anthropicdriver.AnthropicOptions
type AnthropicDriver = anthropicdriver.AnthropicDriver
type BedrockOptions = bedrockdriver.BedrockOptions
type BedrockDriver = bedrockdriver.BedrockDriver
type BedrockRuntimeAPI = bedrockdriver.RuntimeAPI
type BedrockModelAPI = bedrockdriver.ModelAPI
type BedrockConverseStream = bedrockdriver.ConverseStream
type VertexAIOptions = vertexdriver.VertexAIOptions
type VertexAIDriver = vertexdriver.VertexAIDriver
type AzureFoundryOptions = azuredriver.AzureFoundryOptions
type AzureFoundryDriver = azuredriver.AzureFoundryDriver
type AzureOpenAIOptions = azuredriver.AzureOpenAIOptions
type AzureOpenAIDriver = azuredriver.AzureOpenAIDriver
type AzureTokenProvider = azuredriver.TokenProvider

const (
	ProviderOpenAI           = common.ProviderOpenAI
	ProviderOpenAICompatible = common.ProviderOpenAICompatible
	ProviderAnthropic        = common.ProviderAnthropic
	ProviderVertexAI         = common.ProviderVertexAI
	ProviderBedrock          = common.ProviderBedrock
	ProviderAzureFoundry     = common.ProviderAzureFoundry
	ProviderAzureOpenAI      = common.ProviderAzureOpenAI

	PromptRoleSafety    = common.PromptRoleSafety
	PromptRoleSystem    = common.PromptRoleSystem
	PromptRoleUser      = common.PromptRoleUser
	PromptRoleAssistant = common.PromptRoleAssistant
	PromptRoleNegative  = common.PromptRoleNegative
	PromptRoleMask      = common.PromptRoleMask
	PromptRoleTool      = common.PromptRoleTool

	ResultTypeText     = common.ResultTypeText
	ResultTypeThoughts = common.ResultTypeThoughts
	ResultTypeJSON     = common.ResultTypeJSON
	ResultTypeImage    = common.ResultTypeImage
	ResultTypeVideo    = common.ResultTypeVideo
	ResultTypeAudio    = common.ResultTypeAudio

	EmbeddingTaskQuery    = common.EmbeddingTaskQuery
	EmbeddingTaskDocument = common.EmbeddingTaskDocument

	EmbeddingInputText  = core.EmbeddingInputText
	EmbeddingInputImage = core.EmbeddingInputImage
	EmbeddingInputVideo = core.EmbeddingInputVideo
	EmbeddingInputAudio = core.EmbeddingInputAudio

	PromptCacheModeAuto     = common.PromptCacheModeAuto
	PromptCacheModeOff      = common.PromptCacheModeOff
	PromptCacheModeRequired = common.PromptCacheModeRequired
)

// Result constructors, re-exported from package common.
var (
	NewTextResult     = common.NewTextResult
	NewThoughtsResult = common.NewThoughtsResult
	NewJSONResult     = common.NewJSONResult
	NewImageResult    = common.NewImageResult
	NewVideoResult    = common.NewVideoResult
	NewAudioResult    = common.NewAudioResult
)

// Result validation, re-exported from package core.
const (
	ResultValidationCodeJSON   = core.ResultValidationCodeJSON
	ResultValidationCodeSchema = core.ResultValidationCodeSchema
)

// ValidateResult checks completion results against a JSON Schema; see core.ValidateResult. Execute and Stream
// already apply it when ExecutionOptions.ResultSchema is set.
func ValidateResult(results []CompletionResult, schema map[string]any) ([]CompletionResult, *ResultValidationError) {
	return core.ValidateResult(results, schema)
}

// Ptr returns a pointer to v, for the optional fields of the generated types.
func Ptr[T any](v T) *T {
	return &v
}

// NewOpenAIDriver creates the first-party OpenAI driver using openai-go.
func NewOpenAIDriver(options OpenAICompatibleOptions) (*OpenAICompatibleDriver, error) {
	return openaidriver.NewOpenAIDriver(options)
}

// NewOpenAICompatibleDriver creates a driver for OpenAI-compatible endpoints.
func NewOpenAICompatibleDriver(options OpenAICompatibleOptions) (*OpenAICompatibleDriver, error) {
	return openaidriver.NewOpenAICompatibleDriver(options)
}

// NewOpenAIChatCompletionsDriver creates a driver for OpenAI-compatible Chat Completions endpoints.
func NewOpenAIChatCompletionsDriver(options OpenAIChatCompletionsOptions) (*OpenAIChatCompletionsDriver, error) {
	return openaidriver.NewOpenAIChatCompletionsDriver(options)
}

// NewAnthropicDriver creates the direct Anthropic Claude driver.
func NewAnthropicDriver(options AnthropicOptions) (*AnthropicDriver, error) {
	return anthropicdriver.NewAnthropicDriver(options)
}

// NewBedrockDriver creates AWS Bedrock runtime and model clients.
func NewBedrockDriver(ctx context.Context, options BedrockOptions) (*BedrockDriver, error) {
	return bedrockdriver.NewBedrockDriver(ctx, options)
}

// NewBedrockDriverWithClient injects Bedrock clients for tests or custom wiring.
func NewBedrockDriverWithClient(options BedrockOptions, runtime BedrockRuntimeAPI, models BedrockModelAPI) *BedrockDriver {
	return bedrockdriver.NewBedrockDriverWithClient(options, runtime, models)
}

// NewVertexAIDriver creates a Gemini Enterprise Agent Platform driver.
func NewVertexAIDriver(options VertexAIOptions) (*VertexAIDriver, error) {
	return vertexdriver.NewVertexAIDriver(options)
}

// NewAzureFoundryDriver creates a Microsoft Foundry driver.
func NewAzureFoundryDriver(options AzureFoundryOptions) (*AzureFoundryDriver, error) {
	return azuredriver.NewAzureFoundryDriver(options)
}

// NewAzureOpenAIDriver creates an Azure OpenAI driver on the v1 API.
func NewAzureOpenAIDriver(options AzureOpenAIOptions) (*AzureOpenAIDriver, error) {
	return azuredriver.NewAzureOpenAIDriver(options)
}

// BoolPtr returns a pointer to v for option fields that distinguish false from unset.
func BoolPtr(v bool) *bool {
	return common.BoolPtr(v)
}

// NewLlumiverseError creates a normalized provider error.
func NewLlumiverseError(message string, retryable *bool, context LlumiverseErrorContext, original error, code int, name string) *LlumiverseError {
	return common.NewLlumiverseError(message, retryable, context, original, code, name)
}
