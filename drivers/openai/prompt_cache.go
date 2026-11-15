package openai

import (
	"regexp"
	"slices"
	"strings"
)

// openAIPromptCacheKey returns the prompt-cache routing key: the model option,
// or ExecutionOptions.PromptCacheKey.
func openAIPromptCacheKey(options ExecutionOptions) string {
	if key := optionString(options.ModelOptions, "prompt_cache_key"); key != "" {
		return key
	}
	return options.PromptCacheKey
}

// hasExplicitPromptCacheBreakpoint reports whether an item's content already
// carries an explicit prompt-cache breakpoint.
func hasExplicitPromptCacheBreakpoint(item responseItem) bool {
	parts, _ := item["content"].([]any)
	for _, raw := range parts {
		part, _ := raw.(map[string]any)
		breakpoint, _ := part["prompt_cache_breakpoint"].(map[string]any)
		if breakpoint["mode"] == "explicit" {
			return true
		}
	}
	return false
}

// configureOpenAIPromptCaching sets explicit cache boundaries for GPT-5.6 and
// later when the caller supplies a cache key. A caller's own breakpoint is kept;
// otherwise, with at least two user messages, the last content part of the
// latest user message carrying an image or file (else the second-to-last user
// message) becomes the breakpoint, so the stable prefix before the final turn
// is cached. It returns the input to send and the prompt_cache_options, if any.
func configureOpenAIPromptCaching(input []responseItem, model, key string) ([]responseItem, map[string]any) {
	if key == "" || !isOpenAIGptVersionGTE(model, 5, 6) {
		return input, nil
	}
	for _, item := range input {
		if hasExplicitPromptCacheBreakpoint(item) {
			return input, map[string]any{"mode": "explicit"}
		}
	}
	var users []int
	for i, item := range input {
		if item["role"] == "user" {
			users = append(users, i)
		}
	}
	if len(users) < 2 {
		return input, nil
	}
	source := users[len(users)-2]
	for i := len(users) - 1; i >= 0; i-- {
		parts, _ := input[users[i]]["content"].([]any)
		if partsHaveType(parts, "input_image", "input_file") {
			source = users[i]
			break
		}
	}
	item := cloneJSONValue(input[source]).(map[string]any)
	var content []any
	switch value := item["content"].(type) {
	case string:
		content = []any{map[string]any{"type": "input_text", "text": value}}
	case []any:
		content = value
	}
	if len(content) == 0 {
		return input, nil
	}
	last, ok := content[len(content)-1].(map[string]any)
	if !ok {
		return input, nil
	}
	last["prompt_cache_breakpoint"] = map[string]any{"mode": "explicit"}
	item["content"] = content
	marked := make([]responseItem, len(input))
	copy(marked, input)
	marked[source] = item
	return marked, map[string]any{"mode": "explicit"}
}

func partsHaveType(parts []any, types ...string) bool {
	for _, raw := range parts {
		part, _ := raw.(map[string]any)
		for _, t := range types {
			if part["type"] == t {
				return true
			}
		}
	}
	return false
}

// openAIExtendedCacheRetentionModels are the earlier models OpenAI documents
// with extended (24h) prompt-cache retention. GPT-5.6 and later take a
// lifetime through prompt_cache_options instead.
var openAIExtendedCacheRetentionModels = map[string]bool{
	"gpt-5.4": true, "gpt-5.2": true, "gpt-5.1-codex-max": true, "gpt-5.1": true, "gpt-5.1-codex": true,
	"gpt-5.1-codex-mini": true, "gpt-5.1-chat-latest": true, "gpt-5": true, "gpt-5-codex": true, "gpt-4.1": true,
}

var openAIModelDateSuffix = regexp.MustCompile(`-\d{4}-\d{2}-\d{2}$`)

// openAIPromptCacheTTLs lists the prompt-cache lifetimes a model accepts on
// the OpenAI provider: 30m and 12h on GPT-5.6 and later, in_memory and 24h on
// the earlier extended-retention models, none elsewhere.
func openAIPromptCacheTTLs(model string, provider Provider) []string {
	if provider != ProviderOpenAI {
		return nil
	}
	if i := strings.LastIndex(model, "::"); i >= 0 {
		model = model[i+2:]
	}
	if isOpenAIGptVersionGTE(model, 5, 6) {
		return []string{"30m", "12h"}
	}
	name := strings.TrimPrefix(strings.ToLower(model[strings.LastIndex(model, "/")+1:]), "openai.")
	if openAIExtendedCacheRetentionModels[openAIModelDateSuffix.ReplaceAllString(name, "")] {
		return []string{"in_memory", "24h"}
	}
	return nil
}

// applyOpenAIPromptCacheTTL sets the cache_ttl option in the parameter the
// model takes: prompt_cache_options.ttl on GPT-5.6 and later, the legacy
// prompt_cache_retention on earlier models. A lifetime the model does not
// accept is dropped rather than failing the request with a 400.
func applyOpenAIPromptCacheTTL(payload map[string]any, model string, provider Provider, ttl string) {
	if ttl == "" || !slices.Contains(openAIPromptCacheTTLs(model, provider), ttl) {
		return
	}
	if ttl == "30m" || ttl == "12h" {
		options, _ := payload["prompt_cache_options"].(map[string]any)
		merged := make(map[string]any, len(options)+1)
		for key, value := range options {
			merged[key] = value
		}
		merged["ttl"] = ttl
		payload["prompt_cache_options"] = merged
		return
	}
	payload["prompt_cache_retention"] = ttl
}
