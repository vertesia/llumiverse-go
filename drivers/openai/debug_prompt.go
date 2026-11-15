package openai

import "github.com/vertesia/llumiverse-go/core"

// FormatDebugPrompt returns a copy of a prompt built by this driver with the base64
// payloads of inline images, files and audio truncated, for ExecutionResponse.Prompt.
// The prompt being sent is not modified. It mirrors the TS client's formatOpenAIDebugPrompt.
func (d *OpenAICompatibleDriver) FormatDebugPrompt(prompt any) any {
	return formatOpenAIDebugPrompt(prompt)
}

// formatOpenAIDebugPrompt truncates the binary payloads of a Responses or Chat
// Completions prompt; other values are returned unchanged.
func formatOpenAIDebugPrompt(prompt any) any {
	switch value := prompt.(type) {
	case openAIResponsesPrompt:
		value.Items = truncateDebugItems(value.Items)
		return value
	case *openAIResponsesPrompt:
		if value == nil {
			return value
		}
		copied := *value
		copied.Items = truncateDebugItems(copied.Items)
		return &copied
	}
	return prompt
}

// truncateDebugItems deep-copies items with their data URLs truncated.
func truncateDebugItems(items []responseItem) []responseItem {
	out := make([]responseItem, len(items))
	for i, item := range items {
		out[i] = truncateDebugValue(cloneJSONValue(item)).(map[string]any)
	}
	return out
}

// truncateDebugValue truncates, in a deep copy, the data URLs carried by image,
// file and audio parts in either API's shape.
func truncateDebugValue(v any) any {
	switch value := v.(type) {
	case map[string]any:
		for key, inner := range value {
			switch key {
			case "image_url", "file_data", "url":
				if s, ok := inner.(string); ok {
					value[key] = core.TruncateDataURLForDebug(s)
					continue
				}
			case "data":
				// Chat Completions input_audio carries bare base64 rather than a data URL.
				if s, ok := inner.(string); ok && value["format"] != nil {
					value[key] = core.TruncateBinaryForDebug(s)
					continue
				}
			}
			value[key] = truncateDebugValue(inner)
		}
		return value
	case []any:
		for i, inner := range value {
			value[i] = truncateDebugValue(inner)
		}
		return value
	}
	return v
}
