package wire

import (
	"encoding/json"
	"fmt"
)

// Ptr returns a pointer to v, for the optional fields of the generated types.
func Ptr[T any](v T) *T {
	return &v
}

// NewTextResult returns a CompletionResult holding text.
func NewTextResult(value string) CompletionResult {
	return mustResult(TextResult{Type: TextResultTypeText, Value: value})
}

// NewThoughtsResult returns a CompletionResult holding a reasoning summary.
func NewThoughtsResult(value string) CompletionResult {
	return mustResult(ThoughtsResult{Type: ThoughtsResultTypeThoughts, Value: value})
}

// NewJSONResult returns a CompletionResult holding a JSON value.
func NewJSONResult(value JSONValue) CompletionResult {
	return mustResult(JSONResult{Type: JSONResultTypeJSON, Value: value})
}

// NewImageResult returns a CompletionResult holding an image URI or data URL.
func NewImageResult(value string) CompletionResult {
	return mustResult(ImageResult{Type: ImageResultTypeImage, Value: value})
}

// NewVideoResult returns a CompletionResult holding a video URI.
func NewVideoResult(value string) CompletionResult {
	return mustResult(VideoResult{Type: VideoResultTypeVideo, Value: value})
}

// NewAudioResult returns a CompletionResult holding an audio result.
func NewAudioResult(result AudioResult) CompletionResult {
	result.Type = AudioResultTypeAudio
	return mustResult(result)
}

// mustResult encodes a result variant. The variants are plain structs of JSON-safe fields, except
// JSONResult.Value, which the caller supplies; a value encoding/json rejects (a channel, a func) is a
// programming error, as it would be for json.Marshal on the whole completion.
func mustResult(variant any) CompletionResult {
	raw, err := json.Marshal(variant)
	if err != nil {
		panic(fmt.Sprintf("wire: encode completion result: %v", err))
	}
	return CompletionResult{union: raw}
}

// Type returns the result's discriminator ("text", "json", ...), or "" when it cannot be read.
func (t CompletionResult) Type() string {
	discriminator, err := t.Discriminator()
	if err != nil {
		return ""
	}
	return discriminator
}

// String returns the text of a text or thoughts result, the JSON encoding of a JSON result, and the
// value (a URI) of a media result, mirroring how the TypeScript client renders a result as text.
func (t CompletionResult) String() string {
	switch t.Type() {
	case "json":
		result, err := t.AsJSONResult()
		if err != nil {
			return ""
		}
		raw, err := json.Marshal(result.Value)
		if err != nil {
			return ""
		}
		return string(raw)
	default:
		var value struct {
			Value string `json:"value"`
		}
		if err := json.Unmarshal(t.union, &value); err != nil {
			return ""
		}
		return value.Value
	}
}

// OptionID returns the `_option_id` that names the option family, or "" when the options do not
// carry one. ModelOptions has no discriminator because `_option_id` is optional in every family.
func (t ModelOptions) OptionID() string {
	var id struct {
		OptionID string `json:"_option_id"`
	}
	if err := json.Unmarshal(t.union, &id); err != nil {
		return ""
	}
	return id.OptionID
}

// Map decodes the options into a generic map, the form drivers read option values from.
func (t ModelOptions) Map() (map[string]any, error) {
	if len(t.union) == 0 {
		return nil, nil
	}
	var options map[string]any
	if err := json.Unmarshal(t.union, &options); err != nil {
		return nil, fmt.Errorf("decode model options: %w", err)
	}
	return options, nil
}

// ModelOptionsFromMap encodes a generic option map as ModelOptions.
func ModelOptionsFromMap(options map[string]any) (ModelOptions, error) {
	raw, err := json.Marshal(options)
	if err != nil {
		return ModelOptions{}, fmt.Errorf("encode model options: %w", err)
	}
	return ModelOptions{union: raw}, nil
}
