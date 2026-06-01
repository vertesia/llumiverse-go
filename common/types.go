package common

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"
)

// Provider identifies a model-hosting backend supported by the driver layer.
type Provider string

const (
	ProviderOpenAI           Provider = "openai"
	ProviderOpenAICompatible Provider = "openai_compatible"
	ProviderAnthropic        Provider = "anthropic"
	ProviderVertexAI         Provider = "vertexai"
	ProviderBedrock          Provider = "bedrock"
)

// PromptRole is the provider-neutral role assigned to a prompt segment.
type PromptRole string

const (
	PromptRoleSafety    PromptRole = "safety"
	PromptRoleSystem    PromptRole = "system"
	PromptRoleUser      PromptRole = "user"
	PromptRoleAssistant PromptRole = "assistant"
	PromptRoleNegative  PromptRole = "negative"
	PromptRoleMask      PromptRole = "mask"
	PromptRoleTool      PromptRole = "tool"
)

// PromptSegment is the normalized input unit consumed by all drivers.
type PromptSegment struct {
	// Role controls how the segment is mapped into each provider's prompt format.
	Role PromptRole
	// Content is the text content for the segment.
	Content string
	// ToolUseID links a tool result segment back to the provider tool call.
	ToolUseID string
	// ThoughtSignature carries provider-specific signed reasoning metadata. Gemini
	// thinking models require it on the matching function response.
	ThoughtSignature string
	// Files are binary or text attachments associated with the segment.
	Files []DataSource
}

// DataSource abstracts binary inputs so drivers can choose URL passthrough or
// inline bytes. Provider-native URLs such as gs://, s3://, and signed HTTPS
// storage URLs are preferred when the target API can consume them directly.
type DataSource interface {
	// Name returns a display filename for providers that require one.
	Name() string
	// MIMEType returns the source media type, such as image/png or application/pdf.
	MIMEType() string
	// Open returns a readable stream for inline upload paths.
	Open(ctx context.Context) (io.ReadCloser, error)
	// URL returns an optional provider-accessible URI, such as gs:// or s3://.
	URL(ctx context.Context) (string, error)
}

// BytesDataSource is an in-memory DataSource used by tests and simple callers.
type BytesDataSource struct {
	// FileName is returned by Name.
	FileName string
	// MIME is returned by MIMEType.
	MIME string
	// Data is returned by Open.
	Data []byte
	// URI is optional; when present, drivers may pass it directly to providers.
	URI string
}

// Name returns the configured file name.
func (d BytesDataSource) Name() string { return d.FileName }

// MIMEType returns the configured MIME type.
func (d BytesDataSource) MIMEType() string { return d.MIME }

// Open returns a reader over the in-memory data.
func (d BytesDataSource) Open(context.Context) (io.ReadCloser, error) {
	return io.NopCloser(bytes.NewReader(d.Data)), nil
}

// URL returns the optional provider-accessible URI.
func (d BytesDataSource) URL(context.Context) (string, error) {
	if d.URI == "" {
		return "", errors.New("data source has no URL")
	}
	return d.URI, nil
}

// Driver is the provider-neutral contract for completions, streaming, models, and embeddings.
type Driver interface {
	// Provider returns the stable provider identifier for this driver.
	Provider() Provider
	// CreatePrompt converts normalized prompt segments into the provider-native prompt shape.
	CreatePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (any, error)
	// Execute sends one non-streaming completion request.
	Execute(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (*ExecutionResponse, error)
	// Stream sends one streaming completion request and returns a pull-based stream.
	Stream(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (CompletionStream, error)
	// ListModels returns normalized provider model metadata.
	ListModels(ctx context.Context, params *ModelSearchPayload) ([]AIModel, error)
	// ValidateConnection checks that the driver can authenticate with its provider.
	ValidateConnection(ctx context.Context) error
	// GenerateEmbeddings embeds text or media inputs with the configured provider model.
	GenerateEmbeddings(ctx context.Context, options EmbeddingsOptions) (*EmbeddingsResult, error)
}

// HTTPTimeoutOptions mirrors the timeout knobs exposed by the TypeScript client.
type HTTPTimeoutOptions struct {
	// HeadersTimeout limits waiting for response headers.
	HeadersTimeout time.Duration
	// BodyTimeout limits reading the response body.
	BodyTimeout time.Duration
	// ConnectTimeout limits establishing a new network connection.
	ConnectTimeout time.Duration
	// KeepAliveTimeout limits idle connection reuse.
	KeepAliveTimeout time.Duration
}

// DriverOptions contains options shared by provider constructors.
type DriverOptions struct {
	// HTTPTimeout supplies constructor-level transport timeout defaults.
	HTTPTimeout HTTPTimeoutOptions
}

// ExecutionOptions contains per-request options common to completion drivers.
type ExecutionOptions struct {
	// Model is the provider model ID or resource name to invoke.
	Model string
	// ResultSchema requests structured JSON output using the provider's native
	// schema feature when available.
	ResultSchema map[string]any
	// ModelOptions carries provider-specific options such as temperature,
	// max_tokens, stop_sequence, effort, cache controls, or image settings.
	ModelOptions map[string]any
	// IncludeOriginalResponse includes the raw provider response in Completion.
	IncludeOriginalResponse bool
	// Tools declares callable functions the model may request.
	Tools []ToolDefinition
	// Conversation is provider-shaped state returned from a previous call.
	Conversation any
	// Labels are provider metadata labels where supported.
	Labels map[string]string
	// HTTPTimeout overrides transport timeouts for this request when supported.
	HTTPTimeout HTTPTimeoutOptions
	// StripImagesAfterTurns controls when media is removed from retained conversations.
	StripImagesAfterTurns *int
	// StripTextMaxTokens approximates retained-history text truncation.
	StripTextMaxTokens int
	// StripHeartbeatsAfterTurns controls removal of <heartbeat>...</heartbeat> blocks.
	StripHeartbeatsAfterTurns *int
}

// ToolDefinition describes a callable tool in the provider-neutral schema.
type ToolDefinition struct {
	// Name is the provider-visible function name.
	Name string
	// Description tells the model when and how to use the tool.
	Description string
	// InputSchema is a JSON Schema object describing the tool input.
	InputSchema map[string]any
}

// ToolUse is a tool call returned by a model.
type ToolUse struct {
	// ID links the model's tool call to the later tool-result prompt segment.
	ID string
	// ToolName is the requested tool name.
	ToolName string
	// ToolInput is the parsed input object, or a partial JSON string for some stream chunks.
	ToolInput any
	// ThoughtSignature carries provider-specific signed reasoning metadata.
	ThoughtSignature string
}

// ResultType identifies the shape of a completion result item.
type ResultType string

const (
	ResultTypeText  ResultType = "text"
	ResultTypeJSON  ResultType = "json"
	ResultTypeImage ResultType = "image"
)

// CompletionResult is one text, JSON, or image output from a completion.
type CompletionResult struct {
	// Type identifies whether Value is text, parsed JSON, or an image URL/data URL.
	Type ResultType
	// Value contains the provider-normalized result payload.
	Value any
}

// ExecutionTokenUsage stores provider token accounting normalized across APIs.
type ExecutionTokenUsage struct {
	// Prompt is the total normalized prompt-token count.
	Prompt int
	// Result is the output-token count.
	Result int
	// Total is the provider-reported total token count when available.
	Total int
	// PromptCached is the number of prompt tokens read from provider cache.
	PromptCached int
	// PromptCacheWrite is the number of prompt tokens written to provider cache.
	PromptCacheWrite int
	// PromptNew is the number of uncached prompt tokens.
	PromptNew int
}

// ResultValidationError is set when JSON-schema result validation fails.
type ResultValidationError struct {
	// Code is a stable validation error code.
	Code string
	// Message describes the validation failure.
	Message string
	// Data contains the raw completion data that failed validation.
	Data []CompletionResult
}

// Completion is the normalized response body returned by completion and stream APIs.
type Completion struct {
	// Result contains text, JSON, or image outputs.
	Result []CompletionResult
	// TokenUsage contains normalized token accounting when providers report it.
	TokenUsage *ExecutionTokenUsage
	// ToolUse contains tool calls requested by the model.
	ToolUse []ToolUse
	// FinishReason is normalized across providers, for example stop, length, or tool_use.
	FinishReason string
	// Error contains structured-output validation errors.
	Error *ResultValidationError
	// OriginalResponse contains the raw provider response when requested.
	OriginalResponse any
	// Conversation is provider-shaped state that can be passed to ExecutionOptions.
	Conversation any
}

// ExecutionResponse adds request metadata around a completed model call.
type ExecutionResponse struct {
	Completion
	// Prompt is the provider-native request prompt generated from PromptSegment values.
	Prompt any
	// ExecutionTime is the elapsed time for non-streaming execution.
	ExecutionTime time.Duration
	// Chunks is the number of stream chunks consumed when available.
	Chunks int
}

// CompletionChunk is one incremental item from a streaming response.
type CompletionChunk struct {
	// Result contains incremental text, JSON, or image outputs.
	Result []CompletionResult
	// TokenUsage contains incremental or final token accounting when providers report it.
	TokenUsage *ExecutionTokenUsage
	// FinishReason is set on terminal chunks when providers report it.
	FinishReason string
	// ToolUse contains incremental tool-call data.
	ToolUse []ToolUse
}

// CompletionStream exposes provider streaming as a pull-based iterator.
type CompletionStream interface {
	// Recv returns the next chunk or io.EOF when the stream is complete.
	Recv() (CompletionChunk, error)
	// Close releases the provider stream.
	Close() error
	// Completion returns the accumulated response after stream consumption.
	Completion() *ExecutionResponse
}

// AIModel is a normalized model listing entry.
type AIModel struct {
	// ID is the provider model ID or resource name.
	ID string
	// Name is a human-readable display name.
	Name string
	// Provider identifies the driver that returned this model.
	Provider Provider
	// Description is provider-supplied model text when available.
	Description string
	// Version is provider-supplied model version metadata when available.
	Version string
	// Type classifies the model, such as text, image, or embedding.
	Type string
	// Tags are normalized or provider-supplied model labels.
	Tags []string
	// Owner is the model publisher or account owner.
	Owner string
	// Status is provider-supplied lifecycle status when available.
	Status string
	// CanStream reports whether the model can stream completion output.
	CanStream bool
	// IsCustom reports whether the model is customer-created or fine-tuned.
	IsCustom bool
	// IsMultimodal reports whether the model accepts or produces multiple modalities.
	IsMultimodal bool
	// InputModalities lists supported input modalities when providers expose them.
	InputModalities []string
	// OutputModalities lists supported output modalities when providers expose them.
	OutputModalities []string
	// ToolSupport reports whether tool/function calling is supported.
	ToolSupport bool
}

// ModelSearchPayload filters provider model listings.
type ModelSearchPayload struct {
	// Text filters model IDs and names by substring where supported.
	Text string
	// Type filters by normalized model type.
	Type string
	// Tags filters by model tags where supported.
	Tags []string
	// Owner filters by publisher or account owner where supported.
	Owner string
}

// EmbeddingTaskType gives providers retrieval intent for text embeddings. Each
// driver maps it to the provider's vocabulary, for example Gemini
// RETRIEVAL_QUERY/RETRIEVAL_DOCUMENT, Cohere search_query/search_document, or
// Nova GENERIC_RETRIEVAL/GENERIC_INDEX.
type EmbeddingTaskType string

const (
	EmbeddingTaskQuery    EmbeddingTaskType = "query"
	EmbeddingTaskDocument EmbeddingTaskType = "document"
)

// EmbeddingInput is one text, image, video, or audio item to embed. Media inputs
// may produce one vector or multiple segmented vectors depending on the provider
// model, so the normalized result keeps outputs grouped by original input.
type EmbeddingInput struct {
	// Type is text, image, video, or audio. Empty defaults to text.
	Type string
	// Text is used for text embedding inputs.
	Text string
	// Source is used for image, video, audio, or document-backed inputs.
	Source DataSource
	// TaskType gives retrieval intent for text embeddings.
	TaskType EmbeddingTaskType
	// Title is used by retrieval-document embedding models that accept title hints.
	Title string
	// StartSec, LengthSec, and IntervalSec configure segmented media embeddings.
	StartSec    *float64
	LengthSec   *float64
	IntervalSec *float64
	// UseFixedLength and MinClipSec are TwelveLabs Marengo video options.
	UseFixedLength *bool
	MinClipSec     *float64
	// EmbeddingOption selects Marengo views such as visual-text, visual-image, or audio.
	// Bedrock Marengo accepts at most one option per request; submit multiple
	// inputs when more than one view is needed.
	EmbeddingOption []string
}

// EmbeddingsOptions contains a batch of embedding inputs and model-level options.
type EmbeddingsOptions struct {
	// Inputs is the ordered batch of embedding inputs.
	Inputs []EmbeddingInput
	// Model is the provider embedding model ID. Drivers choose a default when empty.
	Model string
	// TaskType supplies a default retrieval intent for text inputs.
	TaskType EmbeddingTaskType
	// Dimensions requests an output dimensionality when supported by the model.
	Dimensions int
	// HTTPTimeout overrides transport timeouts for this embedding request when supported.
	HTTPTimeout HTTPTimeoutOptions
}

// EmbeddingsResult preserves one result item per input.
type EmbeddingsResult struct {
	// Results has one item per input, preserving input order.
	Results []EmbeddingResultItem
	// Model is the embedding model that was invoked.
	Model string
	// Usage contains aggregate token accounting when providers report it.
	Usage *EmbeddingsTokenUsage
}

// EmbeddingResultItem contains all vectors generated for a single input.
type EmbeddingResultItem struct {
	// Outputs contains one or more vectors generated for this input.
	Outputs []EmbeddingOutput
	// InputTokens is the provider-reported token count for this input.
	InputTokens int
}

// EmbeddingOutput is a single vector, optionally tied to a media segment. Text
// and image models usually return one output; segmented video/audio models may
// return several outputs for a single input.
type EmbeddingOutput struct {
	// Values is the embedding vector.
	Values []float64
	// Modality is the normalized modality for this vector.
	Modality string
	// StartSec is the segment start time for segmented media embeddings.
	StartSec *float64
	// EndSec is the segment end time for segmented media embeddings.
	EndSec *float64
	// EmbeddingOption is the provider-specific Marengo view when reported.
	EmbeddingOption string
}

// EmbeddingsTokenUsage stores aggregate embedding usage when providers report it.
type EmbeddingsTokenUsage struct {
	// InputTokens is the total input token count.
	InputTokens int
	// InputTextTokens is the text-token portion when providers report modality detail.
	InputTextTokens int
	// InputImageTokens is the image-token portion when providers report modality detail.
	InputImageTokens int
}

// LlumiverseErrorContext identifies where a provider error occurred.
type LlumiverseErrorContext struct {
	// Provider identifies the driver that produced the error.
	Provider Provider
	// Model is the model associated with the failed operation.
	Model string
	// Operation is the driver operation, such as execute, stream, or embeddings.
	Operation string
}

// LlumiverseError wraps provider errors with retryability and operation context.
// Retryable is intentionally tri-state: true and false mean the driver could
// classify the failure, nil means retryability is unknown.
type LlumiverseError struct {
	// Message is the normalized error message.
	Message string
	// Code is the provider HTTP or API status code when available.
	Code int
	// Name is the normalized error name.
	Name string
	// Retryable classifies retryability, or nil when unknown.
	Retryable *bool
	// Context identifies the provider, model, and operation.
	Context LlumiverseErrorContext
	// OriginalError keeps the underlying provider error for errors.As/errors.Is.
	OriginalError error
}

// Error returns the normalized error message.
func (e *LlumiverseError) Error() string {
	if e == nil {
		return ""
	}
	return e.Message
}

// Unwrap returns the original provider error.
func (e *LlumiverseError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.OriginalError
}

// BoolPtr returns a pointer to v for option fields that distinguish false from unset.
func BoolPtr(v bool) *bool {
	return &v
}

// NewLlumiverseError creates a normalized provider error with retryability,
// operation context, and access to the original provider error.
func NewLlumiverseError(message string, retryable *bool, context LlumiverseErrorContext, original error, code int, name string) *LlumiverseError {
	prefix := fmt.Sprintf("[%s] ", context.Provider)
	if !strings.HasPrefix(message, prefix) {
		message = prefix + message
	}
	if name == "" {
		name = "LlumiverseError"
	}
	return &LlumiverseError{
		Message:       message,
		Code:          code,
		Name:          name,
		Retryable:     retryable,
		Context:       context,
		OriginalError: original,
	}
}
