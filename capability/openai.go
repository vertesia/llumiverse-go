package capability

import "strings"

// openAITools returns OpenAI capabilities, where tools work while streaming whenever they work at all.
func openAITools(input, output Modalities, tool bool) Capabilities {
	return tools(input, output, tool, tool)
}

var openAIModelCapabilities = map[string]Capabilities{
	"chatgpt-4o-latest": openAITools(only("text", "image"), only("text"), true),
}

// openAIFamilyCapabilities is matched by longest prefix: the lowest common denominator of each family.
var openAIFamilyCapabilities = map[string]Capabilities{
	"gpt-audio":     openAITools(only("text", "audio"), only("text", "audio"), false),
	"gpt":           openAITools(only("text", "image"), only("text"), true),
	"gpt-3.5":       openAITools(only("text"), only("text"), false),
	"gpt-4":         openAITools(only("text"), only("text"), true),
	"gpt-4-turbo":   openAITools(only("text", "image"), only("text"), true),
	"gpt-4o":        openAITools(only("text", "image"), only("text"), true),
	"gpt-4.1":       openAITools(only("text", "image"), only("text"), true),
	"gpt-4.5":       openAITools(only("text", "image"), only("text"), true),
	"gpt-5":         openAITools(only("text", "image"), only("text"), true),
	"gpt-image":     openAITools(only("text", "image"), only("image"), false),
	"chatgpt-image": openAITools(only("text", "image"), only("image"), false),
	"dall-e":        openAITools(only("text"), only("image"), false),
	"gpt-oss":       openAITools(only("text"), only("text"), true),
	"o":             openAITools(only("text", "image"), only("text"), true),
	"o1-mini":       openAITools(only("text", "image"), only("text"), false),
	// o3-mini is text-only; o4-mini reintroduced image input.
	"o3-mini":         openAITools(only("text"), only("text"), true),
	"o1-preview":      openAITools(only("text"), only("text"), false),
	"omni-moderation": openAITools(only("text", "image"), only("text"), false),
	"sora":            openAITools(only("text", "image", "video"), only("video", "audio"), false),
	"text-embedding":  openAITools(only("text"), only("embed"), false),
	"text-moderation": openAITools(only("text"), only("text"), false),
	"whisper":         openAITools(only("audio"), only("text"), false),
}

// containsAny reports whether model contains any of the patterns.
func containsAny(model string, patterns ...string) bool {
	for _, pattern := range patterns {
		if strings.Contains(model, pattern) {
			return true
		}
	}
	return false
}

// inferModalities guesses modalities from words in an unknown model ID, leaving the rest unknown.
func inferModalities(model string, imageWords ...string) (input, output Modalities) {
	input = Modalities{
		Text:  flagOrNil(strings.Contains(model, "text")),
		Image: flagOrNil(containsAny(model, imageWords...)),
		Video: flagOrNil(strings.Contains(model, "video")),
		Audio: flagOrNil(strings.Contains(model, "audio")),
		Embed: no,
	}
	output = Modalities{
		Text:  flagOrNil(strings.Contains(model, "text")),
		Image: flagOrNil(strings.Contains(model, "image")),
		Video: flagOrNil(strings.Contains(model, "video")),
		Audio: flagOrNil(strings.Contains(model, "audio")),
		Embed: flagOrNil(strings.Contains(model, "embed")),
	}
	return input, output
}

// OpenAICapabilities returns the capabilities of an OpenAI model: an exact record, the longest matching family,
// or a guess from the words in the ID.
func OpenAICapabilities(model string) Capabilities {
	return openAICapabilities(model).clone()
}

func openAICapabilities(model string) Capabilities {
	normalized := strings.ToLower(model)
	if record, ok := openAIModelCapabilities[normalized]; ok {
		return record
	}
	if family, ok := longestPrefix(openAIFamilyCapabilities, normalized); ok {
		return family
	}
	input, output := inferModalities(normalized, "image")
	tool := flagOrNil(containsAny(normalized, "tool", "gpt", "gpt-5", "o1", "o3", "o4"))
	return Capabilities{Input: input, Output: output, ToolSupport: tool, ToolSupportStreaming: tool}
}
