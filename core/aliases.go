package core

import "github.com/vertesia/llumiverse-go/common"

type Provider = common.Provider
type PromptRole = common.PromptRole
type PromptSegment = common.PromptSegment
type DataSource = common.DataSource
type BytesDataSource = common.BytesDataSource
type Driver = common.Driver
type DriverOptions = common.DriverOptions
type HTTPTimeoutOptions = common.HTTPTimeoutOptions
type ExecutionOptions = common.ExecutionOptions
type ToolDefinition = common.ToolDefinition
type ToolUse = common.ToolUse
type ToolUseChunk = common.ToolUseChunk
type CompletionResult = common.CompletionResult
type JSONObject = common.JSONObject
type ModelType = common.ModelType
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

	ResultTypeText     = common.ResultTypeText
	ResultTypeThoughts = common.ResultTypeThoughts
	ResultTypeJSON     = common.ResultTypeJSON
	ResultTypeImage    = common.ResultTypeImage
	ResultTypeVideo    = common.ResultTypeVideo
	ResultTypeAudio    = common.ResultTypeAudio

	EmbeddingTaskQuery    = common.EmbeddingTaskQuery
	EmbeddingTaskDocument = common.EmbeddingTaskDocument
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

// Tokens returns a token count as the *float64 the usage types hold.
func Tokens(n int) *float64 { return common.Tokens(n) }

// Count returns the value of an optional count, or 0 when it is unset.
func Count(v *float64) float64 { return common.Count(v) }

// ToolInputObject converts a decoded tool input into the JSONObject ToolUse holds.
func ToolInputObject(input any) *JSONObject { return common.ToolInputObject(input) }

// Ptr returns a pointer to v, for the optional fields of the generated types.
func Ptr[T any](v T) *T {
	return &v
}

// BoolPtr returns a pointer to v for option fields that distinguish false from unset.
func BoolPtr(v bool) *bool {
	return common.BoolPtr(v)
}

// NewLlumiverseError creates a normalized provider error.
func NewLlumiverseError(message string, retryable *bool, context LlumiverseErrorContext, original error, code int, name string) *LlumiverseError {
	return common.NewLlumiverseError(message, retryable, context, original, code, name)
}
