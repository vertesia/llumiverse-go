package vertexai

import (
	"github.com/vertesia/llumiverse-go/core"
	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
)

// FormatDebugPrompt returns the prompt to report as ExecutionResponse.Prompt:
// Gemini, Imagen and Claude prompts with their inline base64 media truncated, as
// the TS client's formatDebugPrompt does. The prompt being sent is never
// modified. Any other prompt type is returned unchanged.
func (d *VertexAIDriver) FormatDebugPrompt(prompt any) any {
	switch value := prompt.(type) {
	case geminiPrompt:
		return formatGeminiDebugPrompt(value)
	case vertexImagenPrompt:
		return formatImagenDebugPrompt(value)
	case internalclaude.Prompt:
		return internalclaude.FormatDebugPrompt(value)
	default:
		return prompt
	}
}

// formatGeminiDebugPrompt copies a Gemini prompt with every inline media payload
// truncated, including attachments carried inside function responses.
func formatGeminiDebugPrompt(prompt geminiPrompt) geminiPrompt {
	out := prompt
	out.Contents = make([]geminiContent, len(prompt.Contents))
	for i, content := range prompt.Contents {
		out.Contents[i] = formatGeminiContentForDebug(content)
	}
	if prompt.System != nil {
		system := formatGeminiContentForDebug(*prompt.System)
		out.System = &system
	}
	return out
}

func formatGeminiContentForDebug(content geminiContent) geminiContent {
	out := content
	out.Parts = make([]geminiPart, len(content.Parts))
	for i, part := range content.Parts {
		out.Parts[i] = formatGeminiPartForDebug(part)
	}
	return out
}

func formatGeminiPartForDebug(part geminiPart) geminiPart {
	if part.InlineData != nil && part.InlineData.Data != "" {
		inline := *part.InlineData
		inline.Data = core.TruncateBinaryForDebug(inline.Data)
		part.InlineData = &inline
	}
	if part.FunctionResponse != nil && len(part.FunctionResponse.Parts) > 0 {
		response := *part.FunctionResponse
		response.Parts = make([]geminiFunctionMediaPart, len(part.FunctionResponse.Parts))
		for i, media := range part.FunctionResponse.Parts {
			if media.InlineData != nil && media.InlineData.Data != "" {
				inline := *media.InlineData
				inline.Data = core.TruncateBinaryForDebug(inline.Data)
				media.InlineData = &inline
			}
			response.Parts[i] = media
		}
		part.FunctionResponse = &response
	}
	return part
}

// formatImagenDebugPrompt copies an Imagen prompt with the base64 bytes of each
// reference image truncated.
func formatImagenDebugPrompt(prompt vertexImagenPrompt) vertexImagenPrompt {
	if len(prompt.ReferenceImages) == 0 {
		return prompt
	}
	out := prompt
	out.ReferenceImages = make([]map[string]any, len(prompt.ReferenceImages))
	for i, reference := range prompt.ReferenceImages {
		image, ok := reference["referenceImage"].(map[string]any)
		encoded, _ := image["bytesBase64Encoded"].(string)
		if !ok || encoded == "" {
			out.ReferenceImages[i] = reference
			continue
		}
		copied := make(map[string]any, len(reference))
		for key, value := range reference {
			copied[key] = value
		}
		copiedImage := make(map[string]any, len(image))
		for key, value := range image {
			copiedImage[key] = value
		}
		copiedImage["bytesBase64Encoded"] = core.TruncateBinaryForDebug(encoded)
		copied["referenceImage"] = copiedImage
		out.ReferenceImages[i] = copied
	}
	return out
}
