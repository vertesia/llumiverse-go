package bedrock

import (
	"regexp"
	"strconv"
	"strings"

	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// bedrockFamilyFields returns the Converse additional model request fields for a
// non-Claude model, routed by model family as in the TS client. Each family
// receives only the knobs its Converse adapter reads, under the name it reads
// them by, and only when the caller set them: Bedrock rejects unknown fields.
func bedrockFamilyFields(options ExecutionOptions) map[string]any {
	model := strings.ToLower(options.Model)
	opts := options.ModelOptions
	out := map[string]any{}
	switch {
	case strings.Contains(model, "amazon"):
		// Titan accepts no additional fields.
		if !strings.Contains(model, "nova") {
			break
		}
		if v := optionInt(opts, "top_k"); v != nil {
			out["inferenceConfig"] = map[string]any{"topK": *v}
		}
		// Explicit effort passes through on every Nova version, including ones
		// without known reasoning support; omission keeps the provider default.
		if effort := optionString(opts, "effort"); effort == "none" {
			out["reasoningConfig"] = map[string]any{"type": "disabled"}
		} else if effort != "" {
			out["reasoningConfig"] = map[string]any{"type": "enabled", "maxReasoningEffort": effort}
		}
	case strings.Contains(model, "meta"):
		// Llama accepts no additional fields.
	case strings.Contains(model, "mistral"):
		// Only Mistral 7B and Mixtral 8x7B read top_k; newer generations take none.
		if strings.Contains(model, "7b") {
			setBedrockField(out, "top_k", optionIntValue(opts, "top_k"))
		}
	case strings.Contains(model, "ai21"):
		// Jurassic uses nested penalty scales; Jamba accepts the numeric fields directly.
		if strings.Contains(model, "j2") {
			if v := optionFloatValue(opts, "presence_penalty"); v != nil {
				out["presencePenalty"] = map[string]any{"scale": v}
			}
			if v := optionFloatValue(opts, "frequency_penalty"); v != nil {
				out["frequencyPenalty"] = map[string]any{"scale": v}
			}
		} else if strings.Contains(model, "jamba") {
			setBedrockField(out, "presence_penalty", optionFloatValue(opts, "presence_penalty"))
			setBedrockField(out, "frequency_penalty", optionFloatValue(opts, "frequency_penalty"))
		}
	case strings.Contains(model, "cohere.command"):
		setBedrockField(out, "k", optionIntValue(opts, "top_k"))
		if strings.Contains(model, "cohere.command-r") {
			setBedrockField(out, "frequency_penalty", optionFloatValue(opts, "frequency_penalty"))
			setBedrockField(out, "presence_penalty", optionFloatValue(opts, "presence_penalty"))
		}
	case strings.Contains(model, "palmyra"):
		setBedrockField(out, "seed", optionIntValue(opts, "seed"))
		setBedrockField(out, "presence_penalty", optionFloatValue(opts, "presence_penalty"))
		setBedrockField(out, "frequency_penalty", optionFloatValue(opts, "frequency_penalty"))
		setBedrockField(out, "min_tokens", optionIntValue(opts, "min_tokens"))
	case strings.Contains(model, "deepseek"):
		// DeepSeek accepts no additional fields; see bedrockInferenceConfig for
		// the stop and top_p omissions.
	case bedrockIsGPT5(model):
		if effort := bedrockOpenAIEffort(opts); effort != "" {
			out["reasoning"] = map[string]any{"effort": effort}
		}
		text := map[string]any{}
		if verbosity := optionString(opts, "verbosity"); verbosity != "" {
			text["verbosity"] = verbosity
		}
		if options.ResultSchema != nil {
			// Bedrock's OpenAI Converse adapter requires strict in addition to outputConfig.
			text["format"] = map[string]any{"strict": true}
		}
		if len(text) > 0 {
			out["text"] = text
		}
	case strings.Contains(model, "xai.grok-"):
		// Grok reads effort from the same reasoning field as GPT; omission keeps
		// the provider default.
		if effort := bedrockOpenAIEffort(opts); effort != "" {
			out["reasoning"] = map[string]any{"effort": effort}
		}
	case strings.Contains(model, "gpt-oss"):
		if v := optionString(opts, "reasoning_effort"); v != "" {
			out["reasoning_effort"] = v
		}
	}
	return out
}

// setBedrockField sets key when the option value is present.
func setBedrockField(out map[string]any, key string, value any) {
	if value != nil {
		out[key] = value
	}
}

// bedrockFoldsSystemPrompt reports whether a model family rejects Converse
// system blocks, so the system prompt travels as a user message instead:
// Mistral 7B and Mixtral 8x7B, Jurassic 2, and Cohere Command (non-R).
func bedrockFoldsSystemPrompt(model string) bool {
	model = strings.ToLower(model)
	switch {
	case strings.Contains(model, "amazon"), strings.Contains(model, "claude"), strings.Contains(model, "meta"):
		return false
	case strings.Contains(model, "mistral"):
		return strings.Contains(model, "7b")
	case strings.Contains(model, "ai21"):
		return strings.Contains(model, "j2")
	case strings.Contains(model, "cohere.command"):
		return !strings.Contains(model, "cohere.command-r")
	}
	return false
}

// foldBedrockSystemPrompt appends the system text as a final user message, merged
// into a trailing user turn, as the TS client's converseSystemToMessages does.
// The messages passed in are not modified.
func foldBedrockSystemPrompt(messages []brtypes.Message, system []brtypes.SystemContentBlock) []brtypes.Message {
	var parts []string
	for _, block := range system {
		if text, ok := block.(*brtypes.SystemContentBlockMemberText); ok {
			parts = append(parts, text.Value)
		}
	}
	text := strings.TrimSpace(strings.Join(parts, "\n"))
	if text == "" {
		return messages
	}
	out := append([]brtypes.Message(nil), messages...)
	block := &brtypes.ContentBlockMemberText{Value: text}
	if n := len(out); n > 0 && out[n-1].Role == brtypes.ConversationRoleUser {
		last := out[n-1]
		last.Content = append(append([]brtypes.ContentBlock(nil), last.Content...), block)
		out[n-1] = last
		return out
	}
	return append(out, brtypes.Message{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{block}})
}

// withoutBedrockOptions returns a copy of options without the named keys.
func withoutBedrockOptions(options map[string]any, keys ...string) map[string]any {
	out := make(map[string]any, len(options))
	for key, value := range options {
		out[key] = value
	}
	for _, key := range keys {
		delete(out, key)
	}
	return out
}

// bedrockGPTVersionPattern matches closed-weight GPT generations in bare IDs,
// regional or global profiles and ARNs. Restricted-access Daybreak models put a
// "daybreak-<color>-" segment before the version they belong to, as in
// openai.gpt-daybreak-blue-5.6-sol. gpt-oss has no version and does not match.
var bedrockGPTVersionPattern = regexp.MustCompile(`(?:^|[./:])(?:openai\.)?gpt-(?:daybreak-[a-z]+-)?(\d+)(?:\.(\d+))?(?:-|[.:@]|$)`)

// bedrockGPTVersion parses the GPT generation of a model ID.
func bedrockGPTVersion(model string) (major, minor int, ok bool) {
	match := bedrockGPTVersionPattern.FindStringSubmatch(strings.ToLower(model))
	if match == nil {
		return 0, 0, false
	}
	major, _ = strconv.Atoi(match[1])
	if match[2] != "" {
		minor, _ = strconv.Atoi(match[2])
	}
	return major, minor, true
}

// bedrockIsGPT5 reports whether a model is a closed-weight GPT of generation 5
// or later. Those read reasoning and text controls as Converse additional
// fields and accept temperature together with top_p.
func bedrockIsGPT5(model string) bool {
	major, _, ok := bedrockGPTVersion(model)
	return ok && major >= 5
}

// bedrockOpenAIEffort returns the reasoning effort for OpenAI-style models: the
// canonical effort option wins over its legacy reasoning_effort alias.
func bedrockOpenAIEffort(options map[string]any) string {
	if effort := optionString(options, "effort"); effort != "" {
		return effort
	}
	return optionString(options, "reasoning_effort")
}
