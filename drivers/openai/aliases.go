package openai

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
	ProviderOpenAI           = common.ProviderOpenAI
	ProviderOpenAICompatible = common.ProviderOpenAICompatible

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

	embeddingInputText = core.EmbeddingInputText
)

var (
	executeWithPrompt                 = core.ExecuteWithPrompt
	streamWithPrompt                  = core.StreamWithPrompt
	newChannelStreamWithFinalizer     = core.NewChannelStreamWithFinalizer
	dataSourceToDataURL               = core.DataSourceToDataURL
	normalizeEmbeddingsOptions        = core.NormalizeEmbeddingsOptions
	shouldStripConversationMedia      = core.ShouldStripConversationMedia
	shouldStripConversationHeartbeats = core.ShouldStripConversationHeartbeats
	optionFloat                       = core.OptionFloat
	optionInt                         = core.OptionInt
	hasHTTPTimeout                    = core.HasHTTPTimeout
	newHTTPClient                     = core.NewHTTPClient
	retryableFromStatusAndMessage     = core.RetryableFromStatusAndMessage
	truncateForConversation           = core.TruncateForConversation
	shortResourceName                 = core.ShortResourceName
	toString                          = core.ToString
	boolPtr                           = common.BoolPtr
	newLlumiverseError                = common.NewLlumiverseError
)
