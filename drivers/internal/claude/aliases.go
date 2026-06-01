package claude

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
type EmbeddingsOptions = common.EmbeddingsOptions
type EmbeddingsResult = common.EmbeddingsResult
type AIModel = common.AIModel
type ModelSearchPayload = common.ModelSearchPayload
type LlumiverseErrorContext = common.LlumiverseErrorContext
type StreamItem = core.StreamItem
type streamItem = core.StreamItem
type sseEvent = core.SSEEvent

const (
	ProviderAnthropic = common.ProviderAnthropic

	PromptRoleSafety    = common.PromptRoleSafety
	PromptRoleSystem    = common.PromptRoleSystem
	PromptRoleUser      = common.PromptRoleUser
	PromptRoleAssistant = common.PromptRoleAssistant
	PromptRoleNegative  = common.PromptRoleNegative
	PromptRoleMask      = common.PromptRoleMask
	PromptRoleTool      = common.PromptRoleTool

	ResultTypeText = common.ResultTypeText
)

var (
	executeWithPrompt              = core.ExecuteWithPrompt
	streamWithPrompt               = core.StreamWithPrompt
	doJSON                         = core.DoJSON
	joinEndpoint                   = core.JoinEndpoint
	postSSE                        = core.PostSSE
	scanSSE                        = core.ScanSSE
	newChannelStreamWithFinalizer  = core.NewChannelStreamWithFinalizer
	dataSourceToBase64             = core.DataSourceToBase64
	readAllStringFromDataSource    = core.ReadAllStringFromDataSource
	truncateForConversation        = core.TruncateForConversation
	safeJSONParse                  = core.SafeJSONParse
	completionResultsToText        = core.CompletionResultsToText
	optionFloat                    = core.OptionFloat
	optionInt                      = core.OptionInt
	optionBool                     = core.OptionBool
	optionString                   = core.OptionString
	optionStringSlice              = core.OptionStringSlice
	hasHTTPTimeout                 = core.HasHTTPTimeout
	newHTTPClient                  = core.NewHTTPClient
	isClaudeVersionGTE             = core.IsClaudeVersionGTE
	claudeSupportsAdaptiveThinking = core.ClaudeSupportsAdaptiveThinking
	retryableFromStatusAndMessage  = core.RetryableFromStatusAndMessage
	errorStatusAndName             = core.ErrorStatusAndName
	boolPtr                        = common.BoolPtr
	newLlumiverseError             = common.NewLlumiverseError
)
