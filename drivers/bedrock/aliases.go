package bedrock

import (
	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
)

type Provider = common.Provider
type DriverOptions = common.DriverOptions
type HTTPTimeoutOptions = common.HTTPTimeoutOptions
type ExecutionOptions = common.ExecutionOptions
type PromptSegment = common.PromptSegment
type DataSource = common.DataSource
type BytesDataSource = common.BytesDataSource
type Completion = common.Completion
type ExecutionResponse = common.ExecutionResponse
type CompletionStream = common.CompletionStream
type CompletionChunk = common.CompletionChunk
type CompletionResult = common.CompletionResult
type ToolDefinition = common.ToolDefinition
type ToolUse = common.ToolUse
type ToolUseChunk = common.ToolUseChunk
type JSONObject = common.JSONObject
type ModelType = common.ModelType
type EmbeddingOutputModality = common.EmbeddingOutputModality
type ExecutionTokenUsage = common.ExecutionTokenUsage
type ResultValidationError = common.ResultValidationError
type EmbeddingTaskType = common.EmbeddingTaskType
type EmbeddingInput = common.EmbeddingInput
type EmbeddingsOptions = common.EmbeddingsOptions
type EmbeddingsResult = common.EmbeddingsResult
type EmbeddingResultItem = common.EmbeddingResultItem
type EmbeddingOutput = common.EmbeddingOutput
type EmbeddingsTokenUsage = common.EmbeddingsTokenUsage
type AIModel = common.AIModel
type ModelSearchPayload = common.ModelSearchPayload
type LlumiverseErrorContext = common.LlumiverseErrorContext
type streamItem = core.StreamItem

const (
	ProviderBedrock = common.ProviderBedrock

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

	EmbeddingTaskQuery    = common.EmbeddingTaskQuery
	EmbeddingTaskDocument = common.EmbeddingTaskDocument

	embeddingInputText  = core.EmbeddingInputText
	embeddingInputImage = core.EmbeddingInputImage
	embeddingInputVideo = core.EmbeddingInputVideo
	embeddingInputAudio = core.EmbeddingInputAudio
)

var (
	executeWithPrompt                     = core.ExecuteWithPrompt
	streamWithPrompt                      = core.StreamWithPrompt
	newChannelStreamWithFinalizer         = core.NewChannelStreamWithFinalizer
	dataSourceToBase64                    = core.DataSourceToBase64
	readAllString                         = core.ReadAllString
	truncateForConversation               = core.TruncateForConversation
	removeNilMapValues                    = core.RemoveNilMapValues
	normalizeEmbeddingsOptions            = core.NormalizeEmbeddingsOptions
	completionResultsToText               = core.CompletionResultsToText
	toolInputString                       = core.ToolInputString
	optionFloat                           = core.OptionFloat
	optionInt                             = core.OptionInt
	optionBool                            = core.OptionBool
	optionString                          = core.OptionString
	optionStringSlice                     = core.OptionStringSlice
	hasHTTPTimeout                        = core.HasHTTPTimeout
	newHTTPClient                         = core.NewHTTPClient
	isClaudeVersionGTE                    = core.IsClaudeVersionGTE
	claudeSupportsAdaptiveThinking        = core.ClaudeSupportsAdaptiveThinking
	claudeHasSamplingParameterRestriction = core.ClaudeHasSamplingParameterRestriction
	retryableFromStatusAndMessage         = core.RetryableFromStatusAndMessage
	newTextResult                         = core.NewTextResult
	newImageResult                        = core.NewImageResult
	toolInputObject                       = core.ToolInputObject
	tokens                                = core.Tokens
	count                                 = core.Count
	boolPtr                               = common.BoolPtr
	newLlumiverseError                    = common.NewLlumiverseError
)

// ptr returns a pointer to v, for the optional fields of the generated types.
func ptr[T any](v T) *T {
	return core.Ptr(v)
}
