package bedrockmantle

import (
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// Protocol is the wire protocol Bedrock Mantle serves a model with.
type Protocol string

const (
	// ProtocolResponses is the OpenAI Responses API on the /openai/v1 endpoint.
	ProtocolResponses Protocol = "responses"
	// ProtocolChatCompletions is the OpenAI Chat Completions API on the /v1 endpoint.
	ProtocolChatCompletions Protocol = "chat_completions"
	// ProtocolMessages is the Anthropic Messages API on the /anthropic endpoint.
	ProtocolMessages Protocol = "messages"
)

// chatCompletionsPublishers are the publishers whose Mantle models speak Chat Completions.
var chatCompletionsPublishers = map[string]bool{
	"deepseek":   true,
	"minimax":    true,
	"mistral":    true,
	"moonshot":   true,
	"moonshotai": true,
	"nvidia":     true,
	"qwen":       true,
	"writer":     true,
	"zai":        true,
}

// publisherNames are the display names of the publishers.
var publisherNames = map[string]string{
	"anthropic":  "Anthropic",
	"deepseek":   "DeepSeek",
	"google":     "Google",
	"minimax":    "MiniMax",
	"mistral":    "Mistral AI",
	"moonshot":   "Moonshot AI",
	"moonshotai": "Moonshot AI",
	"nvidia":     "NVIDIA",
	"openai":     "OpenAI",
	"qwen":       "Qwen",
	"writer":     "Writer",
	"xai":        "xAI",
	"zai":        "Z.AI",
}

var geoPrefix = regexp.MustCompile(`^(?:[a-z]{2}(?:-gov)?|apac|global)\.`)

// normalizeModelID lowercases a model ID and drops an ARN or path prefix and a cross-region geography prefix
// ("us.", "eu.", "global.").
func normalizeModelID(model string) string {
	normalized := strings.ToLower(model)
	if slash := strings.LastIndex(normalized, "/"); slash >= 0 {
		normalized = normalized[slash+1:]
	}
	return geoPrefix.ReplaceAllString(normalized, "")
}

func publisherOf(normalized string) string {
	publisher, _, _ := strings.Cut(normalized, ".")
	return publisher
}

// ModelProtocol returns the protocol Bedrock Mantle serves a model with, and false for a model it does not route.
// The rules are an allow-list by publisher and family, as in the TS client: a new version of a known family follows
// its family, and a new family stays unrouted until its endpoint is known.
func ModelProtocol(model string) (Protocol, bool) {
	normalized := normalizeModelID(model)
	switch publisher := publisherOf(normalized); {
	case publisher == "anthropic":
		if strings.Contains(normalized, ".claude-") {
			return ProtocolMessages, true
		}
	case publisher == "openai":
		// GPT-OSS runs on Converse in Bedrock Runtime, but the Mantle Responses route is unreliable for it:
		// the whole family, safeguard variants included, uses Chat Completions.
		if strings.Contains(normalized, ".gpt-oss") {
			return ProtocolChatCompletions, true
		}
		if strings.Contains(normalized, ".gpt-") {
			return ProtocolResponses, true
		}
	case publisher == "xai":
		if strings.Contains(normalized, ".grok-") {
			return ProtocolResponses, true
		}
	case publisher == "google":
		// Gemma 3 uses Chat Completions; Gemma 4 and later moved to the OpenAI-compatible endpoint with Responses.
		if strings.Contains(normalized, ".gemma-") {
			switch {
			case familyVersionAtLeast(normalized, "google.gemma-", 4):
				return ProtocolResponses, true
			case familyVersionAtLeast(normalized, "google.gemma-", 3):
				return ProtocolChatCompletions, true
			}
		}
	case chatCompletionsPublishers[publisher]:
		return ProtocolChatCompletions, true
	}
	return "", false
}

// familyVersionAtLeast reports whether the major version that follows family in model is at least major.
func familyVersionAtLeast(model string, family string, major int) bool {
	index := strings.Index(model, family)
	if index < 0 {
		return false
	}
	rest := model[index+len(family):]
	end := 0
	for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
		end++
	}
	if end == 0 {
		return false
	}
	version, err := strconv.Atoi(rest[:end])
	return err == nil && version >= major
}

// modelOwner returns the display name of a model's publisher.
func modelOwner(model string) string {
	publisher := publisherOf(normalizeModelID(model))
	if name, ok := publisherNames[publisher]; ok {
		return name
	}
	return publisher
}

var versionOnlyName = regexp.MustCompile(`(?i)^[a-z]?\d`)

// modelDisplayName names a Mantle model as the TS client does: the ID after its publisher, with "GPT-" and "Grok "
// spelled out; an ID that holds only a version after the publisher (deepseek.v3.1) keeps the publisher.
func modelDisplayName(model string, owner string) string {
	_, name, found := strings.Cut(model, ".")
	if !found {
		name = model
	}
	switch {
	case strings.HasPrefix(model, "openai."):
		return replacePrefixFold(name, "gpt-", "GPT-")
	case strings.HasPrefix(model, "xai."):
		return replacePrefixFold(name, "grok-", "Grok ")
	case versionOnlyName.MatchString(name):
		runes := []rune(name)
		runes[0] = unicode.ToUpper(runes[0])
		return owner + " " + string(runes)
	}
	return name
}

func replacePrefixFold(value, prefix, replacement string) string {
	if len(value) >= len(prefix) && strings.EqualFold(value[:len(prefix)], prefix) {
		return replacement + value[len(prefix):]
	}
	return value
}
