package llumiverse

import (
	"context"

	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
	anthropicdriver "github.com/vertesia/llumiverse-go/drivers/anthropic"
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
type ResultType = common.ResultType
type CompletionResult = common.CompletionResult
type ExecutionTokenUsage = common.ExecutionTokenUsage
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

type OpenAICompatibleOptions = openaidriver.OpenAICompatibleOptions
type OpenAICompatibleDriver = openaidriver.OpenAICompatibleDriver
type AnthropicOptions = anthropicdriver.AnthropicOptions
type AnthropicDriver = anthropicdriver.AnthropicDriver
type BedrockOptions = bedrockdriver.BedrockOptions
type BedrockDriver = bedrockdriver.BedrockDriver
type BedrockRuntimeAPI = bedrockdriver.RuntimeAPI
type BedrockModelAPI = bedrockdriver.ModelAPI
type BedrockConverseStream = bedrockdriver.ConverseStream
type VertexAIOptions = vertexdriver.VertexAIOptions
type VertexAIDriver = vertexdriver.VertexAIDriver

const (
	ProviderOpenAI           = common.ProviderOpenAI
	ProviderOpenAICompatible = common.ProviderOpenAICompatible
	ProviderAnthropic        = common.ProviderAnthropic
	ProviderVertexAI         = common.ProviderVertexAI
	ProviderBedrock          = common.ProviderBedrock

	PromptRoleSafety    = common.PromptRoleSafety
	PromptRoleSystem    = common.PromptRoleSystem
	PromptRoleUser      = common.PromptRoleUser
	PromptRoleAssistant = common.PromptRoleAssistant
	PromptRoleNegative  = common.PromptRoleNegative
	PromptRoleMask      = common.PromptRoleMask
	PromptRoleTool      = common.PromptRoleTool

	ResultTypeText  = common.ResultTypeText
	ResultTypeJSON  = common.ResultTypeJSON
	ResultTypeImage = common.ResultTypeImage

	EmbeddingTaskQuery    = common.EmbeddingTaskQuery
	EmbeddingTaskDocument = common.EmbeddingTaskDocument

	EmbeddingInputText  = core.EmbeddingInputText
	EmbeddingInputImage = core.EmbeddingInputImage
	EmbeddingInputVideo = core.EmbeddingInputVideo
	EmbeddingInputAudio = core.EmbeddingInputAudio
)

// NewOpenAIDriver creates the first-party OpenAI driver using openai-go.
func NewOpenAIDriver(options OpenAICompatibleOptions) (*OpenAICompatibleDriver, error) {
	return openaidriver.NewOpenAIDriver(options)
}

// NewOpenAICompatibleDriver creates a driver for OpenAI-compatible endpoints.
func NewOpenAICompatibleDriver(options OpenAICompatibleOptions) (*OpenAICompatibleDriver, error) {
	return openaidriver.NewOpenAICompatibleDriver(options)
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

// BoolPtr returns a pointer to v for option fields that distinguish false from unset.
func BoolPtr(v bool) *bool {
	return common.BoolPtr(v)
}

// NewLlumiverseError creates a normalized provider error.
func NewLlumiverseError(message string, retryable *bool, context LlumiverseErrorContext, original error, code int, name string) *LlumiverseError {
	return common.NewLlumiverseError(message, retryable, context, original, code, name)
}
