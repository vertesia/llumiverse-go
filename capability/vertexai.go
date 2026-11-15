package capability

import (
	"regexp"
	"strings"
)

// vertexTools returns Vertex AI capabilities, which record tool support but not streamed tool support.
func vertexTools(input, output Modalities, tool bool) Capabilities {
	return Capabilities{Input: input, Output: output, ToolSupport: flag(tool)}
}

// vertexModelCapabilities holds exact Vertex AI records, keyed by the last path segment.
var vertexModelCapabilities = map[string]Capabilities{
	"mistral-small-2503":        vertexTools(some("text", "image"), some("text"), true),
	"mistral-medium-3":          vertexTools(some("text", "image"), some("text"), true),
	"codestral-2":               vertexTools(Modalities{Text: yes, Image: no}, some("text"), false),
	"gemini-2.0-flash-lite-001": vertexTools(only("text", "image", "video", "audio"), only("text"), false),
	"gemma-4-26b-a4b-it-maas":   vertexTools(only("text", "image"), only("text"), true),
	// https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/deepseek/deepseek-v41-flash
	"deepseek-v4.1-flash-maas": vertexTools(only("text", "image"), only("text"), true),
}

// vertexFamilyCapabilities is matched by longest prefix. Family defaults describe the latest known behaviour, so
// newer models inherit them until a model record says otherwise.
var vertexFamilyCapabilities = map[string]Capabilities{
	"gemini":                      vertexTools(only("text", "image", "video", "audio"), only("text"), true),
	"gemini-nano-banana":          vertexTools(only("text", "image", "video"), only("text", "image"), false),
	"gemini-3-pro-image":          vertexTools(only("text", "image"), only("text", "image"), false),
	"gemini-3.1-flash-lite-image": vertexTools(only("text", "image", "video"), only("text", "image"), false),
	"gemini-transcribe":           vertexTools(only("audio"), only("text"), false),
	"gemini-3.1-flash-image":      vertexTools(only("text", "image", "video"), only("text", "image"), false),
	"gemini-2.5-flash-image":      vertexTools(only("text", "image"), only("text", "image"), false),
	"gemini-3.0-pro-image":        vertexTools(only("text", "image"), only("text", "image"), false),
	"gemini-3.0-flash-image":      vertexTools(only("text", "image"), only("text", "image"), false),
	"imagen":                      vertexTools(only("text", "image"), only("image"), false),
	"imagen-3.0":                  vertexTools(only("text"), only("image"), false),
	"imagen-3.0-capability":       vertexTools(only("text", "image"), only("image"), false),
	"imagen-4.0":                  vertexTools(only("text"), only("image"), false),
	"imagen-4.0-capability":       vertexTools(only("text", "image"), only("image"), false),
	"claude":                      vertexTools(only("text", "image"), only("text"), true),
	"llama":                       vertexTools(only("text"), only("text"), true),
	"llama-2":                     vertexTools(only("text"), only("text"), false),
	"llama-3.2":                   vertexTools(only("text", "image"), only("text"), true),
	"llama-4":                     vertexTools(only("text", "image"), only("text"), true),
	"grok-3":                      vertexTools(only("text"), only("text"), true),
	"grok":                        vertexTools(only("text", "image"), only("text"), true),
	"deepseek":                    vertexTools(only("text"), only("text"), true),
	"gemma":                       vertexTools(only("text"), only("text"), false),
	"glm":                         vertexTools(only("text"), only("text"), true),
	"gpt-oss":                     vertexTools(only("text"), only("text"), true),
	"kimi":                        vertexTools(only("text"), only("text"), true),
	"minimax":                     vertexTools(only("text"), only("text"), true),
	"qwen":                        vertexTools(only("text"), only("text"), true),
}

var (
	ttsPattern                  = regexp.MustCompile(`(?:^|[-_.])tts(?:[-_.]|$)`)
	geminiTranscribePattern     = regexp.MustCompile(`^gemini-\d+(?:\.\d+)?-transcribe(?:-|$)`)
	geminiProImagePattern       = regexp.MustCompile(`^gemini-\d+(?:\.\d+)?-pro-image(?:-|$)`)
	geminiFlashLiteImagePattern = regexp.MustCompile(`^gemini-\d+(?:\.\d+)?-flash-lite-image(?:-|$)`)
)

// VertexAICapabilities returns the capabilities of a model on Vertex AI, from the last segment of its path.
func VertexAICapabilities(model string) Capabilities {
	return vertexAICapabilities(model).clone()
}

func vertexAICapabilities(model string) Capabilities {
	normalized := lastSegment(strings.ToLower(model))
	if ttsPattern.MatchString(normalized) {
		return vertexTools(only("text"), only("audio"), false)
	}
	if strings.Contains(normalized, "transcribe") {
		return vertexTools(only("audio"), only("text"), false)
	}
	if record, ok := vertexModelCapabilities[normalized]; ok {
		return record
	}
	// Later generations inherit the latest known Vertex AI records, including hosted-model differences.
	switch {
	case familyVersionGTE(normalized, "gemma-", 4, 0):
		return vertexModelCapabilities["gemma-4-26b-a4b-it-maas"]
	case strings.HasSuffix(normalized, "-flash-maas") && familyVersionGTE(normalized, "deepseek-v", 4, 1):
		return vertexModelCapabilities["deepseek-v4.1-flash-maas"]
	case familyVersionGTE(normalized, "mistral-small-", 2503, 0):
		return vertexModelCapabilities["mistral-small-2503"]
	case familyVersionGTE(normalized, "mistral-medium-", 3, 0):
		return vertexModelCapabilities["mistral-medium-3"]
	case familyVersionGTE(normalized, "codestral-", 2, 0):
		return vertexModelCapabilities["codestral-2"]
	case familyVersionGTE(normalized, "llama-", 4, 0) || familyVersionGTE(normalized, "llama", 4, 0):
		return vertexFamilyCapabilities["llama-4"]
	// Dedicated Gemini variants must not inherit the chat model's modalities or tools.
	case geminiTranscribePattern.MatchString(normalized):
		return vertexFamilyCapabilities["gemini-transcribe"]
	case geminiProImagePattern.MatchString(normalized) && familyVersionGTE(normalized, "gemini-", 3, 0):
		return vertexFamilyCapabilities["gemini-3-pro-image"]
	case geminiFlashLiteImagePattern.MatchString(normalized) && familyVersionGTE(normalized, "gemini-", 3, 1):
		return vertexFamilyCapabilities["gemini-3.1-flash-lite-image"]
	}
	if family, ok := longestPrefix(vertexFamilyCapabilities, normalized); ok {
		return family
	}
	input, output := inferModalities(normalized, "image")
	tool := flagOrNil(containsAny(normalized, "tool", "sonnet", "opus", "gemini", "claude", "grok", "deepseek",
		"gemma", "glm", "gpt-oss", "kimi", "llama", "minimax", "qwen"))
	return Capabilities{Input: input, Output: output, ToolSupport: tool}
}
