package vertexai

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
type ResultType = common.ResultType
type ToolDefinition = common.ToolDefinition
type ToolUse = common.ToolUse
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
type sseEvent = core.SSEEvent

const (
	ProviderVertexAI = common.ProviderVertexAI

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

	embeddingInputText  = core.EmbeddingInputText
	embeddingInputImage = core.EmbeddingInputImage
	embeddingInputVideo = core.EmbeddingInputVideo
	embeddingInputAudio = core.EmbeddingInputAudio
)

var (
	executeWithPrompt             = core.ExecuteWithPrompt
	streamWithPrompt              = core.StreamWithPrompt
	doJSON                        = core.DoJSON
	joinEndpoint                  = core.JoinEndpoint
	postSSE                       = core.PostSSE
	scanSSE                       = core.ScanSSE
	newChannelStreamWithFinalizer = core.NewChannelStreamWithFinalizer
	dataSourceToBase64            = core.DataSourceToBase64
	truncateForConversation       = core.TruncateForConversation
	removeNilMapValues            = core.RemoveNilMapValues
	normalizeEmbeddingsOptions    = core.NormalizeEmbeddingsOptions
	completionResultsToText       = core.CompletionResultsToText
	toolInputString               = core.ToolInputString
	optionFloat                   = core.OptionFloat
	optionInt                     = core.OptionInt
	optionBool                    = core.OptionBool
	optionString                  = core.OptionString
	optionStringSlice             = core.OptionStringSlice
	hasHTTPTimeout                = core.HasHTTPTimeout
	newHTTPClient                 = core.NewHTTPClient
	retryableFromStatusAndMessage = core.RetryableFromStatusAndMessage
	errorStatusAndName            = core.ErrorStatusAndName
	shortResourceName             = core.ShortResourceName
	newLlumiverseError            = common.NewLlumiverseError
)
