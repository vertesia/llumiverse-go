package claude

import (
	"errors"
	"fmt"
	"strings"
)

const (
	// claudeOutput128KBeta lets Claude 3.7 Sonnet produce more than 64K output tokens.
	claudeOutput128KBeta = "output-128k-2025-02-19"
	// claudeFastModeBeta enables speed "fast" on models whose fast mode is not generally available.
	claudeFastModeBeta = "fast-mode-2026-02-01"
)

// claudePayloadSettings are the per-provider settings of a request.
type claudePayloadSettings struct {
	// Provider selects provider-specific behaviour: tool changes and native result schemas.
	Provider string
	// Operation names the operation in errors (execute or stream).
	Operation string
	// AllowNativeResultSchema false sends the schema as prompt instructions even when the request qualifies for
	// output_config.format, for instance after the provider refused structured outputs for the model.
	AllowNativeResultSchema bool
}

// isClaudePromptCacheEnabled reports whether prompt caching is on: a prompt cache key, or cache_enabled.
func isClaudePromptCacheEnabled(options ExecutionOptions) bool {
	return options.PromptCacheKey != "" || optionBool(options.ModelOptions, "cache_enabled")
}

// claudeToolChoiceError is the 400 a forced tool choice that cannot be honoured fails with.
func claudeToolChoiceError(message string, options ExecutionOptions, settings claudePayloadSettings) error {
	return newLlumiverseError(message, boolPtr(false), LlumiverseErrorContext{
		Provider:  Provider(settings.Provider),
		Model:     options.Model,
		Operation: settings.Operation,
	}, errors.New(message), 400, "ToolChoiceConfigurationError")
}

// claudeMaxTokens is the request's max_tokens: the caller's value, else the model's output limit (64K for Claude
// 3.7 Sonnet unless its thinking budget needs the 128K beta).
func claudeMaxTokens(options ExecutionOptions) int {
	if v := optionInt(options.ModelOptions, "max_tokens"); v != nil {
		return *v
	}
	limit := claudeMaxTokensLimit(options.Model)
	if strings.Contains(options.Model, "claude-3-7-sonnet") {
		budget := optionInt(options.ModelOptions, "thinking_budget_tokens")
		if budget == nil || *budget < 48000 {
			limit = 64000
		}
	}
	return limit
}

// buildClaudePayload builds the Messages API body (without model, stream or provider version fields) and the beta
// values of a request on a prepared conversation, as the TS client's getClaudePayload does. The conversation is
// never modified: cache breakpoints and repairs are written into copies.
func buildClaudePayload(options ExecutionOptions, conversation claudePrompt, settings claudePayloadSettings) (map[string]any, []string, error) {
	model := options.Model
	modelOptions := options.ModelOptions
	var betas []string
	if strings.Contains(model, "claude-3-7-sonnet") {
		maxTokens := optionInt(modelOptions, "max_tokens")
		budget := optionInt(modelOptions, "thinking_budget_tokens")
		if (maxTokens != nil && *maxTokens > 64000) || (budget != nil && *budget > 64000) {
			betas = append(betas, claudeOutput128KBeta)
		}
	}
	fastMode := optionString(modelOptions, "speed") == "fast"
	// Generally available fast mode takes speed alone; the research preview, and any model not known to have left
	// it, needs the beta.
	if fastMode && claudeFastModeStatus(model) != "available" {
		betas = append(betas, claudeFastModeBeta)
	}

	var toolState *claudeToolState
	if canUseClaudeToolChanges(model, settings.Provider) {
		toolState = conversation.toolState()
	}
	messages := cloneMessages(conversation.Messages)
	if toolState == nil {
		messages = stripClaudeToolChanges(messages)
	}
	// Merge first so parallel tool results split across user turns are recombined, then repair both orphan
	// directions. Tool-change messages are placed first, so user turns they separated are merged before repair.
	messages = mergeConsecutiveClaudeUserMessages(placeClaudeSystemMessagesAfterUserTurns(messages))
	messages = sanitizeClaudeMessages(fixClaudeOrphanedToolResults(fixClaudeOrphanedToolUse(messages)))

	for _, tool := range options.Tools {
		if tool.InputSchema != nil && tool.InputSchema["type"] != "object" {
			return nil, nil, fmt.Errorf(`tool %q has invalid input_schema.type: expected "object", got "%v"`, tool.Name, tool.InputSchema["type"])
		}
	}
	hasRequestedTools := len(options.Tools) > 0
	// A conversation with declared tools keeps sending them on a request without tools (a final answer, a summary),
	// so earlier thinking and the prompt cache stay valid; tool_choice none keeps the model from calling them.
	hasTools := hasRequestedTools || toolState != nil
	if toolState != nil {
		deferred := false
		for _, tool := range toolState.Declared {
			deferred = deferred || tool.Deferred
		}
		if deferred || containsClaudeToolChanges(messages) {
			betas = append(betas, claudeToolChangesBeta)
		}
	}
	if !hasTools && claudeMessagesContainToolBlocks(messages) {
		messages = convertClaudeToolBlocksToText(messages)
	}
	// Claude 4.6+ rejects a conversation that ends with an assistant turn (no prefill). A minimal user turn turns
	// that 400 into a continuation; the repair runs again so a trailing tool_use gets its interrupted result.
	if isClaudeVersionGTE(model, 4, 6) && len(messages) > 0 && messages[len(messages)-1].Role == "assistant" {
		messages = fixClaudeOrphanedToolUse(append(messages, claudeContinueTurn()))
	}
	messages = mergeConsecutiveClaudeUserMessages(placeClaudeSystemMessagesAfterUserTurns(messages))
	messages = cloneMessages(messages)
	system := cloneBlocks(conversation.System)

	// Bedrock and Vertex AI enforce the result schema for Claude 4.5+. A request that ends with a prefill, or whose
	// provider refused structured outputs, keeps the prompt path, with the instructions added to the system prompt
	// when formatClaudePrompt left them out.
	qualifiesForNativeFormat := claudeNativeResultSchemaProviders[settings.Provider] && UsesNativeResultSchema(options)
	nativeFormat := qualifiesForNativeFormat && settings.AllowNativeResultSchema && !endsWithClaudePrefill(messages)
	if qualifiesForNativeFormat && !nativeFormat && !hasClaudeSchemaInstructions(options, system, messages) {
		system = append(system, claudeBlock{Type: "text", Text: claudeSchemaInstructions(options)})
	}
	var tools []map[string]any
	switch {
	case toolState != nil:
		tools = renderClaudeTools(toolState)
	case hasTools:
		for _, tool := range options.Tools {
			tools = append(tools, claudeToolDefinition(tool))
		}
	}

	if isClaudePromptCacheEnabled(options) {
		cacheControl := map[string]any{"type": "ephemeral"}
		if ttl := optionString(modelOptions, "cache_ttl"); ttl != "" {
			cacheControl["ttl"] = ttl
		}
		addClaudeCacheBreakpoints(options, system, tools, messages, cacheControl)
	}

	thinking := resolveClaudeThinking(model, modelOptions)
	betas = append(betas, thinking.Betas...)

	var forced map[string]any
	if name := optionString(modelOptions, "required_tool_name"); name != "" {
		forced = map[string]any{"type": "tool", "name": name}
	} else if choice := optionString(modelOptions, "tool_choice"); choice == "required" || choice == "any" {
		forced = map[string]any{"type": "any"}
	}
	if forced != nil && !hasRequestedTools {
		return nil, nil, claudeToolChoiceError("A forced Claude tool turn requires at least one tool definition.", options, settings)
	}
	if forced != nil && strings.Contains(strings.ToLower(model), "claude-mythos-preview") {
		return nil, nil, claudeToolChoiceError(fmt.Sprintf("Claude preview model %s does not support forced tool choice.", model), options, settings)
	}
	// Models that reject a forced tool choice get automatic tool choice; the caller's required-tool check stays
	// authoritative.
	supportedForced := forced != nil && claudeSupportsForcedToolChoice(model)
	disableManualThinking := supportedForced && thinking.Thinking != nil && thinking.Thinking["type"] == "enabled"
	var toolChoice map[string]any
	switch choice := optionString(modelOptions, "tool_choice"); {
	case !hasTools:
	case forced != nil:
		toolChoice = map[string]any{"type": "auto"}
		if supportedForced {
			toolChoice = forced
		}
		parallel, ok := modelOptions["parallel_tool_calls"].(bool)
		toolChoice["disable_parallel_tool_use"] = ok && !parallel
	case choice == "none" || !hasRequestedTools:
		toolChoice = map[string]any{"type": "none"}
	case choice == "auto":
		toolChoice = map[string]any{"type": "auto"}
	}

	payload := map[string]any{
		"messages":   messages,
		"max_tokens": claudeMaxTokens(options),
	}
	if len(system) > 0 {
		payload["system"] = system
	}
	if tools != nil {
		payload["tools"] = tools
	}
	if toolChoice != nil {
		payload["tool_choice"] = toolChoice
	}
	// Sampling: models from 4.7 reject temperature, top_p and top_k; otherwise temperature wins over top_p.
	if !thinking.HasSamplingRestriction {
		temperature := optionFloat(modelOptions, "temperature")
		if temperature != nil {
			payload["temperature"] = *temperature
		} else if topP := optionFloat(modelOptions, "top_p"); topP != nil {
			payload["top_p"] = *topP
		}
		if topK := optionInt(modelOptions, "top_k"); topK != nil {
			payload["top_k"] = *topK
		}
	}
	if stops := optionStringSlice(modelOptions, "stop_sequence"); len(stops) > 0 {
		payload["stop_sequences"] = stops
	}
	switch {
	case disableManualThinking:
		payload["thinking"] = map[string]any{"type": "disabled"}
	case thinking.Thinking != nil:
		payload["thinking"] = thinking.Thinking
	}
	if !disableManualThinking && (thinking.OutputConfig != nil || nativeFormat) {
		outputConfig := map[string]any{}
		for key, value := range thinking.OutputConfig {
			outputConfig[key] = value
		}
		if nativeFormat {
			outputConfig["format"] = map[string]any{"type": "json_schema", "schema": options.ResultSchema}
		}
		payload["output_config"] = outputConfig
	}
	// standard is the provider default and is left out: the research-preview models reject any speed without
	// the fast mode beta.
	if fastMode {
		payload["speed"] = "fast"
	}
	if options.CompactConversation != nil {
		// The API rejects context_management alongside an on-demand compaction: the explicit request wins, and
		// the threshold settings apply again on the next request.
		payload["compaction"] = claudeOnDemandCompaction(options.CompactConversation)
		betas = append(betas, claudeOnDemandCompactionBeta)
	} else if contextManagement := claudeCompactionContextManagement(options.Compaction); contextManagement != nil {
		payload["context_management"] = contextManagement
		betas = append(betas, claudeCompactionBeta)
	}
	return payload, betas, nil
}

// addClaudeCacheBreakpoints marks the prompt cache breakpoints, on the request copies.
//
// The agent layout (an agent cache key) marks the static prefix (the last system block, or the last tool loaded up
// front), the tail (the last cacheable block of the conversation, which moves every request; the provider's
// lookback finds the previous entry), and, for models whose earlier-turn thinking is pruned, the two latest plain
// user turns before the tail, so the history up to them stays readable when pruning rewrites a turn.
//
// The other layout marks the last system block and the last loaded tool, then with a cache key the stable block
// before the final dynamic one, or without one the end of the turn before the last once there are four turns.
func addClaudeCacheBreakpoints(options ExecutionOptions, system []claudeBlock, tools []map[string]any, messages []claudeMessage, cacheControl map[string]any) {
	lastLoadedTool := -1
	for i := len(tools) - 1; i >= 0; i-- {
		// Deferred tools cannot carry cache_control.
		if deferred, _ := tools[i]["defer_loading"].(bool); !deferred {
			lastLoadedTool = i
			break
		}
	}
	if isClaudeAgentPromptCacheKey(options.PromptCacheKey) {
		if len(system) > 0 {
			system[len(system)-1].CacheControl = cacheControl
		} else if lastLoadedTool >= 0 {
			tools[lastLoadedTool]["cache_control"] = cacheControl
		}
		tail := -1
		for i := len(messages) - 1; i >= 0 && tail < 0; i-- {
			if block := lastCacheableClaudeBlock(messages[i]); block != nil {
				block.CacheControl = cacheControl
				tail = i
			}
		}
		if keepsClaudePriorTurnThinking(options.Model) {
			return
		}
		markers := 0
		for i := tail - 1; i >= 0 && markers < 2; i-- {
			if messages[i].Role != "user" || claudeMessageHasBlock(messages[i], "tool_result") {
				continue
			}
			if block := lastCacheableClaudeBlock(messages[i]); block != nil {
				block.CacheControl = cacheControl
				markers++
			}
		}
		return
	}

	if len(system) > 0 {
		system[len(system)-1].CacheControl = cacheControl
	}
	if lastLoadedTool >= 0 {
		tools[lastLoadedTool]["cache_control"] = cacheControl
	}
	// Breakpoints go on conversation turns, not on a trailing tool-change system message.
	var turns []int
	for i, message := range messages {
		if message.Role != "system" {
			turns = append(turns, i)
		}
	}
	isThinking := func(block claudeBlock) bool {
		return block.Type == "thinking" || block.Type == "redacted_thinking"
	}
	if options.PromptCacheKey != "" {
		if len(turns) == 0 {
			return
		}
		content := messages[turns[len(turns)-1]].Content
		if len(content) >= 2 && !isThinking(content[len(content)-2]) {
			content[len(content)-2].CacheControl = cacheControl
		}
	} else if len(turns) >= 4 {
		content := messages[turns[len(turns)-2]].Content
		if len(content) > 0 && !isThinking(content[len(content)-1]) {
			content[len(content)-1].CacheControl = cacheControl
		}
	}
}

// lastCacheableClaudeBlock is the last block of a turn that can carry cache_control: thinking and empty text blocks
// cannot, and tool-change system messages are not marked.
func lastCacheableClaudeBlock(message claudeMessage) *claudeBlock {
	if message.Role == "system" {
		return nil
	}
	for i := len(message.Content) - 1; i >= 0; i-- {
		block := &message.Content[i]
		if block.Type == "thinking" || block.Type == "redacted_thinking" || (block.Type == "text" && block.Text == "") {
			continue
		}
		return block
	}
	return nil
}
