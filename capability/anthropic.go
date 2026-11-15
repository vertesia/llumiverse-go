package capability

import "strings"

// claudeTextImage is the Claude default: text and image input, text output, tools while streaming.
var claudeTextImage = tools(only("text", "image"), only("text"), true, true)

// anthropicModelCapabilities holds exact Claude model exceptions.
var anthropicModelCapabilities = map[string]Capabilities{
	// Claude 3.5 Haiku takes no image input.
	"claude-3-5-haiku-20241022": tools(only("text"), only("text"), true, true),
}

// anthropicFamilyCapabilities is matched by longest prefix.
var anthropicFamilyCapabilities = map[string]Capabilities{
	"claude-3-5-haiku": tools(only("text"), only("text"), true, true),
	"claude-3-haiku":   claudeTextImage,
	"claude-3-sonnet":  claudeTextImage,
	"claude-3-opus":    claudeTextImage,
	"claude-3-5":       claudeTextImage,
	"claude-3-7":       claudeTextImage,
	"claude-3":         claudeTextImage,
	"claude-4":         claudeTextImage,
	"claude-":          claudeTextImage,
}

// AnthropicCapabilities returns the capabilities of a Claude model on the Anthropic API.
func AnthropicCapabilities(model string) Capabilities {
	return anthropicCapabilities(model).clone()
}

func anthropicCapabilities(model string) Capabilities {
	lower := strings.ToLower(model)
	if record, ok := anthropicModelCapabilities[lower]; ok {
		return record
	}
	if family, ok := longestPrefix(anthropicFamilyCapabilities, lower); ok {
		return family
	}
	return claudeTextImage
}
