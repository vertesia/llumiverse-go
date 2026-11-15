package common

import "github.com/vertesia/llumiverse-go/wire"

// The types below are the JSON shapes the TypeScript llumiverse defines with Zod, generated into
// package wire from the shared JSON Schema document. They are aliases, so a value encodes to the same
// JSON as in the TypeScript client. See AGENTS.md, "Generated wire types".

type (
	// ToolDefinition describes a callable tool in the provider-neutral schema.
	ToolDefinition = wire.ToolDefinition
	// ToolUse is a tool call returned by a model.
	ToolUse = wire.ToolUse
	// CompletionResult is one typed piece of model output: text, thoughts, JSON, image, video or audio.
	CompletionResult = wire.CompletionResult
	// ExecutionTokenUsage reports token accounting for one completion.
	ExecutionTokenUsage = wire.ExecutionTokenUsage
	// AIModel is normalized model metadata returned by ListModels.
	AIModel = wire.AIModel
	// ModelSearchPayload filters ListModels results.
	ModelSearchPayload = wire.ModelSearchPayload
	// ModelType is the kind of model in AIModel.Type.
	ModelType = wire.ModelType
	// AIModelStatus is the availability in AIModel.Status.
	AIModelStatus = wire.AIModelStatus
	// EmbeddingTaskType tells retrieval-tuned embedding models how the vector will be used.
	EmbeddingTaskType = wire.EmbeddingTaskType
	// EmbeddingsResult is the normalized embeddings response.
	EmbeddingsResult = wire.EmbeddingsResult
	// EmbeddingResultItem holds the vectors for one input.
	EmbeddingResultItem = wire.EmbeddingResultItem
	// EmbeddingOutput is one vector, with its modality and clip window for media.
	EmbeddingOutput = wire.EmbeddingOutput
	// EmbeddingOutputModality is the modality in EmbeddingOutput.Modality.
	EmbeddingOutputModality = wire.EmbeddingOutputModality
	// EmbeddingsTokenUsage reports token accounting for one embeddings request.
	EmbeddingsTokenUsage = wire.EmbeddingsTokenUsage
	// JSONObject is a JSON object value.
	JSONObject = wire.JSONObject
)

// Result constructors, re-exported from package wire.
var (
	NewTextResult     = wire.NewTextResult
	NewThoughtsResult = wire.NewThoughtsResult
	NewJSONResult     = wire.NewJSONResult
	NewImageResult    = wire.NewImageResult
	NewVideoResult    = wire.NewVideoResult
	NewAudioResult    = wire.NewAudioResult
)

// The values CompletionResult.Type returns.
const (
	ResultTypeText     = "text"
	ResultTypeThoughts = "thoughts"
	ResultTypeJSON     = "json"
	ResultTypeImage    = "image"
	ResultTypeVideo    = "video"
	ResultTypeAudio    = "audio"
)

// EmbeddingTaskType values.
const (
	EmbeddingTaskQuery    = wire.EmbeddingTaskTypeQuery
	EmbeddingTaskDocument = wire.EmbeddingTaskTypeDocument
)

// Ptr returns a pointer to v, for the optional fields of the generated types.
func Ptr[T any](v T) *T {
	return &v
}

// Tokens returns a token count as the *float64 the usage types hold (they are JSON numbers).
func Tokens(n int) *float64 {
	v := float64(n)
	return &v
}

// Count returns the value of an optional count, or 0 when it is unset.
func Count(v *float64) float64 {
	if v == nil {
		return 0
	}
	return *v
}

// ToolInputObject converts a decoded tool input into the JSONObject ToolUse holds. An object is
// kept; nil stays nil (no input); anything else is not an object, so it becomes an empty one, as
// the stream accumulator already does for unparseable arguments.
func ToolInputObject(input any) *JSONObject {
	switch v := input.(type) {
	case nil:
		return nil
	case JSONObject:
		return &v
	case map[string]any:
		object := JSONObject(v)
		return &object
	default:
		return &JSONObject{}
	}
}
