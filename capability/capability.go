package capability

import (
	"slices"
	"strings"

	"github.com/vertesia/llumiverse-go/common"
)

// ForModel returns a model's capabilities on a provider. Speech output is reported only where file synthesis is
// implemented, and video output never, since no driver returns video through the inference path.
func ForModel(model string, provider common.Provider) Capabilities {
	// Vertex AI model names may carry a locations/<location>/ prefix.
	if strings.HasPrefix(model, "locations/") {
		if parts := strings.Split(model, "/"); len(parts) >= 3 {
			model = strings.Join(parts[2:], "/")
		}
	}
	if provider == common.ProviderVertexAI && IsGeminiOmniVideoModel(lastSegment(strings.ToLower(model))) {
		return tools(only("text", "image", "video"), only("text", "video"), false, false).clone()
	}
	profile := ResolveProfile(model, provider)
	caps := profile.Capabilities
	speech := profile.Family == "speech" && slices.Contains([]common.Provider{common.ProviderOpenAI,
		common.ProviderAzureFoundry, common.ProviderOpenAICompatible, common.ProviderVertexAI}, provider)
	audio := profile.Family == "audio" &&
		(provider == common.ProviderOpenAI || provider == common.ProviderOpenAICompatible)
	caps.Output.Audio = boolPtr(speech || audio)
	caps.Output.Video = boolPtr(false)
	return caps
}

// SupportsToolUse reports whether a model calls tools on a provider. When streaming, an unknown streamed tool
// support means the same as tool support; only an explicit false turns tools off while streaming.
func SupportsToolUse(model string, provider common.Provider, streaming bool) bool {
	caps := ForModel(model, provider)
	if !streaming {
		return isTrue(caps.ToolSupport)
	}
	return isTrue(orDefault(caps.ToolSupportStreaming, caps.ToolSupport))
}

// dedicatedFamilies are served by dedicated endpoints rather than the standard inference path.
var dedicatedFamilies = []string{"embedding", "moderation", "image", "transcription", "speech", "realtime", "audio", "video"}

// IsDedicatedInferenceModel reports models that run on a dedicated endpoint (embeddings, moderation, image,
// transcription, speech, realtime, audio or video) rather than the standard inference path.
func IsDedicatedInferenceModel(model string, provider common.Provider) bool {
	return slices.Contains(dedicatedFamilies, ResolveProfile(model, provider).Family)
}

// Listing is a model as a provider's model API lists it.
type Listing struct {
	ID               string
	Type             string
	InputModalities  []string
	OutputModalities []string
}

// IsEmbeddingModel reports embedding models, which are not run through the standard inference path: by listed type,
// by ID, by listed modalities, then by the catalogue.
func IsEmbeddingModel(model Listing, provider common.Provider) bool {
	if strings.EqualFold(model.Type, "embedding") || IsDirectoryEmbedding(model.ID) {
		return true
	}
	for _, modality := range slices.Concat(model.InputModalities, model.OutputModalities) {
		switch strings.ToLower(modality) {
		case "embed", "embedding", "vectors":
			return true
		}
	}
	return isTrue(ForModel(model.ID, provider).Output.Embed)
}

// ListingMetadata is the catalogue's view of a listed model, for AIModel's modality and tool fields.
type ListingMetadata struct {
	InputModalities  []string
	OutputModalities []string
	// ToolSupport is nil when the catalogue does not know the model.
	ToolSupport *bool
}

// ResolveListingMetadata returns a listed model's modalities and tool support. Catalogue rules are curated and win as
// a complete set; the modalities the provider lists (runtimeInput, runtimeOutput, nil when not listed) are used only
// for models no rule knows, so incomplete provider data cannot erase known capabilities. Unknown models without
// provider modalities default to text in and out.
func ResolveListingMetadata(model string, provider common.Provider, runtimeInput, runtimeOutput []string) ListingMetadata {
	profile := ResolveProfile(model, provider)
	if profile.Family != "generic" || profile.ContextWindow != 0 || profile.MaxOutputTokens != 0 ||
		profile.ReasoningEffortLevels != nil || profile.Capabilities.ToolSupport != nil ||
		profile.Capabilities.ToolSupportStreaming != nil {
		caps := ForModel(model, provider)
		return ListingMetadata{
			InputModalities:  caps.Input.List(),
			OutputModalities: caps.Output.List(),
			ToolSupport:      caps.ToolSupport,
		}
	}
	return ListingMetadata{
		InputModalities:  runtimeModalities(runtimeInput),
		OutputModalities: runtimeModalities(runtimeOutput),
	}
}

// runtimeModalities normalizes provider modality names to the catalogue's, defaulting to text.
func runtimeModalities(modalities []string) []string {
	if modalities == nil {
		return []string{"text"}
	}
	normalized := make([]string, 0, len(modalities))
	for _, modality := range modalities {
		switch modality = strings.ToLower(modality); modality {
		case "embedding", "vectors":
			normalized = append(normalized, "embed")
		case "speech":
			normalized = append(normalized, "audio")
		default:
			normalized = append(normalized, modality)
		}
	}
	return normalized
}

// Apply sets a listed model's InputModalities, OutputModalities, IsMultimodal (image input) and ToolSupport from
// the capabilities.
func (c Capabilities) Apply(model *common.AIModel) {
	input, output := c.Input.List(), c.Output.List()
	model.InputModalities = &input
	model.OutputModalities = &output
	model.IsMultimodal = boolPtr(isTrue(c.Input.Image))
	if c.ToolSupport != nil {
		model.ToolSupport = boolPtr(*c.ToolSupport)
	}
}

// Apply sets a listed model's InputModalities, OutputModalities and, when known, ToolSupport.
func (m ListingMetadata) Apply(model *common.AIModel) {
	input, output := slices.Clone(m.InputModalities), slices.Clone(m.OutputModalities)
	model.InputModalities = &input
	model.OutputModalities = &output
	if m.ToolSupport != nil {
		model.ToolSupport = boolPtr(*m.ToolSupport)
	}
}
