package bedrockmantle

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
type EmbeddingsOptions = common.EmbeddingsOptions
type EmbeddingsResult = common.EmbeddingsResult
type AIModel = common.AIModel
type ModelSearchPayload = common.ModelSearchPayload
type LlumiverseError = common.LlumiverseError
type LlumiverseErrorContext = common.LlumiverseErrorContext

const (
	ProviderBedrockMantle = common.ProviderBedrockMantle

	modelTypeText = wire.ModelTypeText
)

var (
	executeWithPrompt             = core.ExecuteWithPrompt
	streamWithPrompt              = core.StreamWithPrompt
	doJSON                        = core.DoJSON
	postSSE                       = core.PostSSE
	joinEndpoint                  = core.JoinEndpoint
	hasHTTPTimeout                = core.HasHTTPTimeout
	newHTTPClient                 = core.NewHTTPClient
	errorStatusAndName            = core.ErrorStatusAndName
	retryableFromStatusAndMessage = core.RetryableFromStatusAndMessage
	leaseCompletionStream         = core.LeaseCompletionStream
	boolPtr                       = common.BoolPtr
	newLlumiverseError            = common.NewLlumiverseError
)

// ptr returns a pointer to v, for the optional fields of the generated types.
func ptr[T any](v T) *T { return core.Ptr(v) }
