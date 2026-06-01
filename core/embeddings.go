package core

import "errors"

// Provider-neutral embedding input modalities, used as the default and to drive
// task-type handling in normalizeEmbeddingsOptions.
const (
	embeddingInputText  = "text"
	embeddingInputImage = "image"
	embeddingInputVideo = "video"
	embeddingInputAudio = "audio"
)

const (
	EmbeddingInputText  = embeddingInputText
	EmbeddingInputImage = embeddingInputImage
	EmbeddingInputVideo = embeddingInputVideo
	EmbeddingInputAudio = embeddingInputAudio
)

// normalizeEmbeddingsOptions applies the shared TS-client defaults before
// drivers fan out into provider-specific embedding schemas.
func normalizeEmbeddingsOptions(options EmbeddingsOptions) (EmbeddingsOptions, error) {
	if len(options.Inputs) == 0 {
		return EmbeddingsOptions{}, errors.New("embedding inputs must contain at least one input")
	}
	normalized := options
	normalized.Inputs = make([]EmbeddingInput, len(options.Inputs))
	for i, input := range options.Inputs {
		if input.Type == "" {
			input.Type = embeddingInputText
		}
		if input.Type == embeddingInputText && input.TaskType == "" {
			input.TaskType = options.TaskType
		}
		normalized.Inputs[i] = input
	}
	return normalized, nil
}
