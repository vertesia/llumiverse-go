package capability

// Modalities says which content kinds a model takes or produces. A nil field is unknown, which listings treat as
// unsupported; catalogue rules distinguish it from an explicit false when they fill defaults.
type Modalities struct {
	Text  *bool
	Image *bool
	Video *bool
	Audio *bool
	// Embed is set on output modalities only.
	Embed *bool
}

// List returns the supported modalities in the order text, image, video, audio, embed, as AIModel's
// input_modalities and output_modalities hold them.
func (m Modalities) List() []string {
	list := []string{}
	for _, entry := range []struct {
		name string
		set  *bool
	}{{"text", m.Text}, {"image", m.Image}, {"video", m.Video}, {"audio", m.Audio}, {"embed", m.Embed}} {
		if entry.set != nil && *entry.set {
			list = append(list, entry.name)
		}
	}
	return list
}

// anyTrue reports whether any modality is explicitly supported.
func (m Modalities) anyTrue() bool {
	return len(m.List()) > 0
}

// merge returns m with every field that override sets replaced.
func (m Modalities) merge(override Modalities) Modalities {
	pick := func(base, next *bool) *bool {
		if next != nil {
			return next
		}
		return base
	}
	return Modalities{
		Text:  pick(m.Text, override.Text),
		Image: pick(m.Image, override.Image),
		Video: pick(m.Video, override.Video),
		Audio: pick(m.Audio, override.Audio),
		Embed: pick(m.Embed, override.Embed),
	}
}

// Capabilities describes a model's modalities and tool use.
type Capabilities struct {
	Input  Modalities
	Output Modalities
	// ToolSupport reports tool use; nil is unknown.
	ToolSupport *bool
	// ToolSupportStreaming reports tool use while streaming; nil means the same as ToolSupport.
	ToolSupportStreaming *bool
}

// Profile is what the catalogue knows about a model on a provider.
type Profile struct {
	// ModelID is the model ID as given.
	ModelID string
	// CanonicalID is the source model the ID names, without region, publisher, path, deployment and routing
	// prefixes and suffixes.
	CanonicalID string
	// Family is the model family the catalogue rules apply to: gpt, claude, gemini, grok, mistral, llama, qwen,
	// deepseek, gemma, kimi, minimax, glm, nemotron, muse, mimo or step for inference models; embedding,
	// moderation, image, transcription, speech, realtime, audio or video for dedicated endpoints; generic when no
	// rule matches.
	Family string
	// SourceProvider is the model maker when the catalogue knows it (openai, anthropic, google, xai, ...).
	SourceProvider string
	Capabilities   Capabilities
	// ContextWindow is the input-plus-output token window, or 0 when no published limit is known.
	ContextWindow int
	// MaxOutputTokens is the output token limit, or 0 when no published limit is known.
	MaxOutputTokens int
	// ReasoningEffortLevels lists the reasoning effort values the model takes on the provider, or nil.
	ReasoningEffortLevels []string
}

var (
	yes = boolPtr(true)
	no  = boolPtr(false)
)

func boolPtr(v bool) *bool { return &v }

// flag returns yes or no.
func flag(v bool) *bool {
	if v {
		return yes
	}
	return no
}

// flagOrNil returns yes, or nil when v is false: TypeScript's `value || undefined`.
func flagOrNil(v bool) *bool {
	if v {
		return yes
	}
	return nil
}

// isTrue reports whether a tri-state flag is set to true.
func isTrue(v *bool) bool { return v != nil && *v }

// orDefault returns v, or def when v is unknown.
func orDefault(v *bool, def *bool) *bool {
	if v != nil {
		return v
	}
	return def
}

// only returns modalities with the named kinds true and every other kind explicitly false.
func only(kinds ...string) Modalities {
	m := Modalities{Text: no, Image: no, Video: no, Audio: no, Embed: no}
	for _, kind := range kinds {
		m = m.merge(some(kind))
	}
	return m
}

// some returns modalities with the named kinds true and every other kind unknown.
func some(kinds ...string) Modalities {
	var m Modalities
	for _, kind := range kinds {
		switch kind {
		case "text":
			m.Text = yes
		case "image":
			m.Image = yes
		case "video":
			m.Video = yes
		case "audio":
			m.Audio = yes
		case "embed":
			m.Embed = yes
		default:
			panic("capability: unknown modality " + kind)
		}
	}
	return m
}

// tools returns capabilities with tool use and streamed tool use set.
func tools(input, output Modalities, tool, streaming bool) Capabilities {
	return Capabilities{Input: input, Output: output, ToolSupport: flag(tool), ToolSupportStreaming: flag(streaming)}
}

// textOnly is a text-in, text-out model with the given tool support.
func textOnly(tool, streaming bool) Capabilities {
	return tools(only("text"), only("text"), tool, streaming)
}

// longestPrefix returns the longest key of table that model starts with.
func longestPrefix[T any](table map[string]T, model string) (T, bool) {
	var best string
	var found bool
	for key := range table {
		if len(model) >= len(key) && model[:len(key)] == key && (!found || len(key) > len(best)) {
			best, found = key, true
		}
	}
	if !found {
		var zero T
		return zero, false
	}
	return table[best], true
}

// clone copies the flags, so callers cannot change the catalogue's shared values through the returned pointers.
func (c Capabilities) clone() Capabilities {
	copyFlag := func(v *bool) *bool {
		if v == nil {
			return nil
		}
		return boolPtr(*v)
	}
	copyModalities := func(m Modalities) Modalities {
		return Modalities{
			Text:  copyFlag(m.Text),
			Image: copyFlag(m.Image),
			Video: copyFlag(m.Video),
			Audio: copyFlag(m.Audio),
			Embed: copyFlag(m.Embed),
		}
	}
	return Capabilities{
		Input:                copyModalities(c.Input),
		Output:               copyModalities(c.Output),
		ToolSupport:          copyFlag(c.ToolSupport),
		ToolSupportStreaming: copyFlag(c.ToolSupportStreaming),
	}
}
