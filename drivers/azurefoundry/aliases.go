package azurefoundry

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
type Completion = common.Completion
type ExecutionResponse = common.ExecutionResponse
type CompletionStream = common.CompletionStream
type CompletionChunk = common.CompletionChunk
type CompletionResult = common.CompletionResult
type ToolDefinition = common.ToolDefinition
type ToolUse = common.ToolUse
type ToolUseChunk = common.ToolUseChunk
type ExecutionTokenUsage = common.ExecutionTokenUsage
type EmbeddingsOptions = common.EmbeddingsOptions
type EmbeddingsResult = common.EmbeddingsResult
type EmbeddingResultItem = common.EmbeddingResultItem
type EmbeddingOutput = common.EmbeddingOutput
type EmbeddingsTokenUsage = common.EmbeddingsTokenUsage
type AIModel = common.AIModel
type ModelSearchPayload = common.ModelSearchPayload
type LlumiverseError = common.LlumiverseError
type LlumiverseErrorContext = common.LlumiverseErrorContext
type embeddingOutputModality = wire.EmbeddingOutputModality
type streamItem = core.StreamItem
type sseEvent = core.SSEEvent

const (
	ProviderAzureFoundry = common.ProviderAzureFoundry
	ProviderAzureOpenAI  = common.ProviderAzureOpenAI

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

	embeddingInputText  = core.EmbeddingInputText
	embeddingInputImage = core.EmbeddingInputImage

	modelTypeText  = wire.ModelTypeText
	modelTypeImage = wire.ModelTypeImage

	embeddingOutputModalityText  = wire.EmbeddingOutputModalityText
	embeddingOutputModalityImage = wire.EmbeddingOutputModalityImage

	conversationImagePlaceholder = core.ConversationImagePlaceholder
)

var (
	executeWithPrompt                 = core.ExecuteWithPrompt
	streamWithPrompt                  = core.StreamWithPrompt
	newChannelStreamWithFinalizer     = core.NewChannelStreamWithFinalizer
	doJSON                            = core.DoJSON
	postSSE                           = core.PostSSE
	scanSSE                           = core.ScanSSE
	joinEndpoint                      = core.JoinEndpoint
	dataSourceToDataURL               = core.DataSourceToDataURL
	dataSourceToBase64                = core.DataSourceToBase64
	normalizeEmbeddingsOptions        = core.NormalizeEmbeddingsOptions
	shouldStripConversationMedia      = core.ShouldStripConversationMedia
	shouldStripConversationHeartbeats = core.ShouldStripConversationHeartbeats
	processConversationText           = core.ProcessConversationText
	truncateDataURLForDebug           = core.TruncateDataURLForDebug
	optionFloat                       = core.OptionFloat
	optionInt                         = core.OptionInt
	optionString                      = core.OptionString
	optionStringSlice                 = core.OptionStringSlice
	hasHTTPTimeout                    = core.HasHTTPTimeout
	newHTTPClient                     = core.NewHTTPClient
	errorStatusAndName                = core.ErrorStatusAndName
	retryableFromStatusAndMessage     = core.RetryableFromStatusAndMessage
	newTextResult                     = core.NewTextResult
	newThoughtsResult                 = core.NewThoughtsResult
	toolInputObject                   = core.ToolInputObject
	isCutOffFinishReason              = core.IsCutOffFinishReason
	isJSONObjectText                  = core.IsJSONObjectText
	leaseCompletionStream             = core.LeaseCompletionStream
	tokens                            = core.Tokens
	boolPtr                           = common.BoolPtr
	newLlumiverseError                = common.NewLlumiverseError
)

// ptr returns a pointer to v, for the optional fields of the generated types.
func ptr[T any](v T) *T { return core.Ptr(v) }
