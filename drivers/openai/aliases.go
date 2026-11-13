package openai

import (
	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
	"github.com/vertesia/llumiverse-go/wire"
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

	modelTypeText               = wire.ModelTypeText
	embeddingOutputModalityText = wire.EmbeddingOutputModalityText
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
	newTextResult                     = core.NewTextResult
	newImageResult                    = core.NewImageResult
	toolInputObject                   = core.ToolInputObject
	tokens                            = core.Tokens
	count                             = core.Count
	boolPtr                           = common.BoolPtr
	newLlumiverseError                = common.NewLlumiverseError
)

// ptr returns a pointer to v, for the optional fields of the generated types.
func ptr[T any](v T) *T { return core.Ptr(v) }
