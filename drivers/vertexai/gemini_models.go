package vertexai

import (
	"strings"
)

// isGlobalOnlyGeminiImageModel reports Nano Banana, which is served only from the
// global location.
func isGlobalOnlyGeminiImageModel(model string) bool {
	return strings.Contains(model, "gemini-nano-banana")
}

// isGeminiWithoutRegionalEndpoints reports Gemini models without single-region
// endpoints: Nano Banana (global only), and Gemini 4+ chat models (global and the
// us/eu multi-regions). Dedicated speech, transcription, live and image variants
// keep their own endpoints. Listing and request routing both use this rule.
func isGeminiWithoutRegionalEndpoints(model string) bool {
	if isGlobalOnlyGeminiImageModel(model) {
		return true
	}
	if !isGeminiVersionGTE(model, "4.0") {
		return false
	}
	for _, marker := range []string{"tts", "transcribe", "live", "native-audio", "image"} {
		if strings.Contains(model, marker) {
			return false
		}
	}
	return true
}

// resolveGeminiRegion picks the location of a Gemini request. Nano Banana always
// goes to global; a Gemini 4+ chat model without a caller-named location goes to
// global, while a named multi-region location is kept. "" keeps the driver region.
func resolveGeminiRegion(model string, explicitRegion string) string {
	if isGlobalOnlyGeminiImageModel(model) {
		return "global"
	}
	if explicitRegion == "" && isGeminiWithoutRegionalEndpoints(model) {
		return "global"
	}
	return explicitRegion
}

// splitVertexLocation separates a "locations/{region}/..." prefix from a model ID.
func splitVertexLocation(model string) (region string, rest string) {
	if !strings.HasPrefix(model, "locations/") {
		return "", model
	}
	parts := strings.Split(model, "/")
	if len(parts) < 3 {
		return "", model
	}
	return parts[1], strings.Join(parts[2:], "/")
}

// geminiEndpoint builds a Gemini generateContent URL. Gemini uses the v1beta1
// API, which is the only version that returns long-decode continuation tokens,
// in the location resolveGeminiRegion picks.
func (d *VertexAIDriver) geminiEndpoint(model string, method string) (string, error) {
	explicitRegion, name := splitVertexLocation(model)
	region := resolveGeminiRegion(name, explicitRegion)
	if region == "" {
		region = d.options.Region
	}
	modelPath := name
	if !strings.Contains(modelPath, "publishers/") {
		modelPath = "publishers/google/models/" + shortResourceName(modelPath)
	}
	base := d.options.BaseURL
	if base == "" {
		base = strings.TrimSuffix(vertexDefaultBaseURL(region), "/v1") + "/v1beta1"
	}
	return joinEndpoint(base, "projects", d.options.Project, "locations", region, modelPath+":"+method)
}

// vertexGoogleListingExclusions are Google publisher models the driver does not
// list: retired families, embedding models (listed through embeddings), and
// special-purpose APIs (Live, robotics, computer use) even when they advertise
// generateContent.
var vertexGoogleListingExclusions = []string{
	"gemini-pro",
	"gemini-ultra",
	"imagen-product-recontext-preview",
	"embedding",
	"embed",
	"-live",
	"live-",
	"transcribe-live",
	"-robotics",
	"native-audio",
	"robotics",
	"computer-use-preview",
}

// vertexPublisherFamilies are the model families listed per publisher: the ones
// this driver has a request path for.
var vertexPublisherFamilies = map[string][]string{
	"google":    {"gemini", "imagen"},
	"anthropic": {"claude"},
}

// isListedVertexPublisherModel reports whether a publisher model is listed: it
// belongs to a supported family and is not excluded.
func isListedVertexPublisherModel(publisher string, name string) bool {
	if publisher == "google" {
		for _, excluded := range vertexGoogleListingExclusions {
			if strings.Contains(name, excluded) {
				return false
			}
		}
	}
	for _, family := range vertexPublisherFamilies[publisher] {
		if strings.Contains(name, family) {
			return true
		}
	}
	return false
}

// vertexAdditionalPublisherModels are listed although the publisher listing does
// not return them.
var vertexAdditionalPublisherModels = map[string][]string{
	"google": append([]string{"imagen-3.0-fast-generate-001"}, geminiOmniVideoModels...),
}

// hasVertexGlobalListing reports Google models also listed under the global
// location: Gemini 2.5 and later.
func hasVertexGlobalListing(publisher string, name string) bool {
	return publisher == "google" && strings.Contains(name, "gemini") && isGeminiVersionGTE(name, "2.5")
}
