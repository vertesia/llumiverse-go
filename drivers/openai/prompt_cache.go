package openai

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
