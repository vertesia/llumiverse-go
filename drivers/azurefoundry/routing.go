package azurefoundry

import (
	"net/url"
	"regexp"
	"strconv"
	"strings"
)

// parseModelID splits a "deployment::source-model" ID. A bare deployment name
// is its own source model, as in the TS client.
func parseModelID(id string) (deployment string, source string) {
	parts := strings.Split(id, "::")
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return id, id
}

// isCompositeModelID reports whether id names its source model.
func isCompositeModelID(id string) bool {
	return strings.Contains(id, "::")
}

// trimTrailingSlashes removes trailing slashes with a scan rather than a regex.
func trimTrailingSlashes(value string) string {
	end := len(value)
	for end > 0 && value[end-1] == '/' {
		end--
	}
	return value[:end]
}

var (
	repeatedSlashes   = regexp.MustCompile(`/{2,}`)
	projectOpenAIPath = regexp.MustCompile(`/api/projects/[^/]+/openai/v1/?$`)
)

// resourceInferenceURL maps the project's OpenAI v1 URL
// (<resource>/api/projects/<project>/openai/v1) to the resource's
// (<resource>/openai/v1). Images, embeddings and key-authenticated inference use
// the resource endpoint rather than the project proxy. Repeated slashes, left by
// a project URL configured with a trailing slash, are collapsed first.
func resourceInferenceURL(base string) (string, error) {
	u, err := url.Parse(base)
	if err != nil {
		return "", err
	}
	u.Path = projectOpenAIPath.ReplaceAllString(repeatedSlashes.ReplaceAllString(u.Path, "/"), "/openai/v1")
	u.RawPath = ""
	return u.String(), nil
}

// anthropicBaseURL returns the resource's Claude Messages base URL
// (<resource>/anthropic), whatever the path of the project endpoint.
func anthropicBaseURL(endpoint string) (string, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", err
	}
	u.Path = "/anthropic"
	u.RawPath = ""
	u.RawQuery = ""
	u.Fragment = ""
	return u.String(), nil
}

// protocol is the inference API a Foundry deployment speaks.
type protocol string

const (
	protocolResponses protocol = "responses"
	protocolChat      protocol = "chat_completions"
	protocolMessages  protocol = "messages"
)

// Model families the routing needs, resolved from the source model as the TS
// client's model directory does. Routing follows the family, never a list of
// model IDs, so new versions route without a release.
const (
	familyEmbedding     = "embedding"
	familyModeration    = "moderation"
	familyImage         = "image"
	familyRealtime      = "realtime"
	familyTranscription = "transcription"
	familySpeech        = "speech"
	familyAudio         = "audio"
	familyVideo         = "video"
	familyGemini        = "gemini"
	familyClaude        = "claude"
	familyGPT           = "gpt"
	familyOther         = "other"
)

var (
	embeddingFamilyPattern = regexp.MustCompile(`(^|[-_.:])(?:embed|embedding|vector)(?:[-_.:]|$)`)
	moderationPattern      = regexp.MustCompile(`(?:prompt-guard|moderation|safeguard)`)
	realtimePattern        = regexp.MustCompile(`(?:realtime|live|native-audio)`)
	transcriptionPattern   = regexp.MustCompile(`(?:whisper|transcribe)`)
	speechPattern          = regexp.MustCompile(`(?:^|[-_.])tts(?:[-_.]|$)`)
	audioPattern           = regexp.MustCompile(`(?:^|[-_.])audio(?:[-_.]|$)`)
	oSeriesPattern         = regexp.MustCompile(`^o\d+(?:[-_.]|$)`)
	sourcePrefixPattern    = regexp.MustCompile(`^(?:global|us|eu|apac)\.`)
	publisherPrefixPattern = regexp.MustCompile(`^(?:openai|anthropic|google|xai)\.`)
	codexPattern           = regexp.MustCompile(`(?:^|[-_.:/])codex(?:[-_.:/]|$)`)
	dedicatedEndpointName  = regexp.MustCompile(`(?:^|[-_.:/])(?:embed|embeddings?|flux|whisper|transcribe|tts|realtime|moderation|sora)(?:[-_.:/]|$)`)
)

// normalizeSourceModel lowercases a model ID and keeps the source half of a
// composite ID, without its path, region or publisher prefix.
func normalizeSourceModel(model string) string {
	normalized := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(model)), "~")
	if parts := strings.Split(normalized, "::"); len(parts) > 1 {
		normalized = parts[len(parts)-1]
	}
	if slash := strings.LastIndex(normalized, "/"); slash >= 0 {
		normalized = normalized[slash+1:]
	}
	normalized = sourcePrefixPattern.ReplaceAllString(normalized, "")
	return publisherPrefixPattern.ReplaceAllString(normalized, "")
}

// modelFamily returns the family of a source model and whether OpenAI publishes it.
func modelFamily(model string) (family string, openAISource bool) {
	normalized := normalizeSourceModel(model)
	switch {
	case embeddingFamilyPattern.MatchString(normalized):
		return familyEmbedding, false
	case moderationPattern.MatchString(normalized):
		return familyModeration, false
	case strings.Contains(normalized, "grok") && strings.Contains(normalized, "image"):
		return familyImage, false
	case strings.Contains(normalized, "gpt-image") || strings.Contains(normalized, "dall-e") || strings.Contains(normalized, "imagen-"):
		return familyImage, !strings.Contains(normalized, "imagen")
	case realtimePattern.MatchString(normalized):
		return familyRealtime, !strings.Contains(normalized, "gemini")
	case transcriptionPattern.MatchString(normalized):
		return familyTranscription, true
	case speechPattern.MatchString(normalized):
		return familySpeech, true
	case audioPattern.MatchString(normalized):
		return familyAudio, true
	case strings.Contains(normalized, "sora"):
		return familyVideo, true
	case strings.Contains(normalized, "gemini"):
		return familyGemini, false
	case strings.Contains(normalized, "claude"):
		return familyClaude, false
	case strings.Contains(normalized, "gpt") || oSeriesPattern.MatchString(normalized):
		return familyGPT, true
	}
	return familyOther, false
}

// isClaudeModel reports whether a source model is a Claude model.
func isClaudeModel(model string) bool {
	family, _ := modelFamily(model)
	return family == familyClaude
}

// isImageModel reports whether a source model generates images.
func isImageModel(model string) bool {
	family, _ := modelFamily(model)
	return family == familyImage
}

// inferProtocol returns the protocol of a deployment whose publisher is unknown,
// as the publisher-based routing would choose it: Claude uses Messages, OpenAI
// models other than gpt-oss (and Codex models) use the Responses API, and every
// other model uses Chat Completions.
func inferProtocol(source string) protocol {
	family, openAISource := modelFamily(source)
	if family == familyClaude {
		return protocolMessages
	}
	normalized := strings.ToLower(source)
	if codexPattern.MatchString(normalized) {
		return protocolResponses
	}
	if openAISource && !strings.Contains(normalized, "gpt-oss") {
		return protocolResponses
	}
	return protocolChat
}

// protocolForPublisher maps a deployment's model publisher to its protocol.
func protocolForPublisher(publisher string) protocol {
	switch strings.ToLower(publisher) {
	case "anthropic":
		return protocolMessages
	case "openai":
		return protocolResponses
	default:
		return protocolChat
	}
}

// isOpenAIAudioModel reports whether an OpenAI source model is an audio task
// (speech, transcription or chat audio) that the listing keeps.
func isOpenAIAudioModel(model string) bool {
	family, _ := modelFamily(model)
	return family == familySpeech || family == familyTranscription || family == familyAudio
}

// isStandardInferenceDeployment reports whether ListModels returns a
// deployment: Foundry chat, Responses, Claude, OpenAI image and OpenAI audio
// deployments, but not those that use a dedicated endpoint contract
// (embeddings, FLUX, transcription, speech, realtime, video, moderation).
func isStandardInferenceDeployment(deployment foundryDeployment) bool {
	publisher := strings.ToLower(deployment.ModelPublisher)
	if publisher == "openai" && isOpenAIAudioModel(deployment.ModelName) {
		return true
	}
	if publisher == "anthropic" {
		return true
	}
	family, _ := modelFamily(deployment.ModelName)
	source := strings.ToLower(deployment.ModelName)
	if strings.Contains(source, "dall-e") {
		return false
	}
	if family == familyImage && publisher == "openai" {
		return true
	}
	switch family {
	case familyEmbedding, familyImage, familyTranscription, familySpeech, familyRealtime, familyVideo, familyModeration:
		return false
	}
	if dedicatedEndpointName.MatchString(source) {
		return false
	}
	// Foundry capability values are strings. Only an explicit false hides a
	// deployment; omitted and future values stay visible so the listing never
	// becomes an allow-list.
	flag, known := parseCapabilityFlag(deployment.Capabilities["chat_completion"])
	return !known || flag
}

// parseCapabilityFlag reads a Foundry capability value ("true", "1", "yes", ...).
func parseCapabilityFlag(value any) (flag bool, known bool) {
	switch v := value.(type) {
	case bool:
		return v, true
	case string:
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "true", "1", "yes":
			return true, true
		case "false", "0", "no":
			return false, true
		}
	}
	return false, false
}

var deepSeekVersionPattern = regexp.MustCompile(`deepseek-v(\d+)(?:\.(\d+))?`)

// deepSeekVersionGTE reports whether a DeepSeek model is at least major.minor.
func deepSeekVersionGTE(model string, major int, minor int) bool {
	match := deepSeekVersionPattern.FindStringSubmatch(strings.ToLower(model))
	if match == nil {
		return false
	}
	gotMajor, _ := strconv.Atoi(match[1])
	gotMinor := 0
	if match[2] != "" {
		gotMinor, _ = strconv.Atoi(match[2])
	}
	return gotMajor > major || (gotMajor == major && gotMinor >= minor)
}
