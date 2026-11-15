package claude

import (
	"encoding/json"
	"reflect"
)

// Append-only tool changes for Claude conversations, as in the TS client's claude-tool-changes.ts.
//
// Claude binds each thinking block, and the prompt cache, to the conversation before it, including the tools array.
// Editing tools between turns therefore invalidates earlier thinking and restarts the cache. Instead the first
// request's tools stay in tools as sent; tools that appear later are appended with defer_loading: true, and every
// change is announced by a tool_addition / tool_removal block in an appended system message (beta
// mid-conversation-tool-changes-2026-07-01). The declared tools are stored in _llumiverse_meta.claude_tools.

// claudeToolChangesBeta enables tool-change system messages and deferred tools.
const claudeToolChangesBeta = "mid-conversation-tool-changes-2026-07-01"

// Provider names that select provider-specific Claude behaviour.
const (
	ProviderNameAnthropic = "anthropic"
	ProviderNameVertexAI  = "vertexai"
	ProviderNameBedrock   = "bedrock"
	// ProviderNameBedrockMantle is Claude on the Bedrock Mantle Messages API.
	ProviderNameBedrockMantle = "bedrock_mantle"
)

// claudeToolChangeProviders are the platforms whose Claude endpoint accepts mid-conversation tool changes.
var claudeToolChangeProviders = map[string]bool{
	ProviderNameAnthropic:     true,
	ProviderNameVertexAI:      true,
	ProviderNameBedrock:       true,
	ProviderNameBedrockMantle: true,
}

// claudeToolState is stored with the conversation so later requests replay tools unchanged.
type claudeToolState struct {
	// Declared is every tool declared in this conversation, in first-declared order.
	Declared []claudeDeclaredTool `json:"declared"`
	// Visible names the tools the model can currently call.
	Visible []string `json:"visible"`
}

// claudeDeclaredTool is one declared tool.
type claudeDeclaredTool struct {
	// Definition is the tool as sent in tools, without request-only fields (cache_control, defer_loading).
	Definition map[string]any `json:"definition"`
	// Deferred marks tools first offered after the first request; they are sent with defer_loading: true.
	Deferred bool `json:"deferred"`
}

// canUseClaudeToolChanges reports whether a request to this model on this provider may use tool changes.
func canUseClaudeToolChanges(model, provider string) bool {
	return claudeToolChangeProviders[provider] && claudeSupportsMidConversationToolChanges(model)
}

// claudeContinueTurn is the user turn added after a trailing assistant turn, which Claude 4.6+ would otherwise
// treat as a prefill.
func claudeContinueTurn() claudeMessage {
	return claudeMessage{Role: "user", Content: []claudeBlock{{Type: "text", Text: "Continue."}}}
}

// claudeToolDefinition is a tool as the Messages API takes it. An omitted description stays omitted.
func claudeToolDefinition(tool ToolDefinition) map[string]any {
	definition := map[string]any{"name": tool.Name, "input_schema": tool.InputSchema}
	if tool.Description != nil {
		definition["description"] = *tool.Description
	}
	return definition
}

// sameClaudeToolDefinition compares definitions by their JSON values, ignoring key order.
func sameClaudeToolDefinition(a, b map[string]any) bool {
	normalize := func(value map[string]any) any {
		data, err := json.Marshal(value)
		if err != nil {
			return nil
		}
		var decoded any
		_ = json.Unmarshal(data, &decoded)
		return decoded
	}
	return reflect.DeepEqual(normalize(a), normalize(b))
}

// applyClaudeToolChanges records this request's tools on the conversation. The first request declares the tools
// as sent. Later requests append new tools as deferred, announce additions and removals in one system message at
// the end of the conversation, and keep every declared definition in place. A request without tools leaves the
// conversation unchanged: the declared tools are then sent with tool_choice none.
func applyClaudeToolChanges(conversation claudePrompt, tools []ToolDefinition) claudePrompt {
	if len(tools) == 0 {
		return conversation
	}
	// One definition per name; a later duplicate wins, as it does when replacing a declared tool.
	var requested []map[string]any
	index := map[string]int{}
	for _, tool := range tools {
		definition := claudeToolDefinition(tool)
		if i, ok := index[tool.Name]; ok {
			requested[i] = definition
			continue
		}
		index[tool.Name] = len(requested)
		requested = append(requested, definition)
	}
	name := func(definition map[string]any) string {
		s, _ := definition["name"].(string)
		return s
	}

	state := conversation.toolState()
	if state == nil {
		// The first declaration loads its tools up front: the API rejects a tools array whose tools are all
		// deferred. Tool-change blocks without stored state (a rebuilt conversation that lost its metadata) may
		// name tools this declaration does not include, so they are dropped.
		next := &claudeToolState{}
		for _, definition := range requested {
			next.Declared = append(next.Declared, claudeDeclaredTool{Definition: definition})
			next.Visible = append(next.Visible, name(definition))
		}
		conversation.Messages = stripClaudeToolChanges(conversation.Messages)
		return conversation.withMeta(conversation.turn(), next)
	}

	declared := append([]claudeDeclaredTool(nil), state.Declared...)
	declaredIndex := map[string]int{}
	for i, tool := range declared {
		declaredIndex[name(tool.Definition)] = i
	}
	visible := map[string]bool{}
	for _, toolName := range state.Visible {
		visible[toolName] = true
	}
	var additions []string
	requestedNames := map[string]bool{}
	nextVisible := make([]string, 0, len(requested))
	for _, definition := range requested {
		toolName := name(definition)
		requestedNames[toolName] = true
		nextVisible = append(nextVisible, toolName)
		if i, ok := declaredIndex[toolName]; !ok {
			declaredIndex[toolName] = len(declared)
			declared = append(declared, claudeDeclaredTool{Definition: definition, Deferred: true})
		} else if !sameClaudeToolDefinition(declared[i].Definition, definition) {
			// Same name, new definition (e.g. a refreshed MCP schema). There is no append-only form for it, so it
			// is replaced in place: this invalidates earlier thinking and the prompt cache once.
			declared[i] = claudeDeclaredTool{Definition: definition, Deferred: declared[i].Deferred}
		}
		if !visible[toolName] {
			additions = append(additions, toolName)
		}
	}
	var removals []string
	for _, toolName := range state.Visible {
		if !requestedNames[toolName] {
			removals = append(removals, toolName)
		}
	}
	next := &claudeToolState{Declared: declared, Visible: nextVisible}
	if len(additions) == 0 && len(removals) == 0 {
		return conversation.withMeta(conversation.turn(), next)
	}
	changes := make([]claudeBlock, 0, len(additions)+len(removals))
	for _, toolName := range additions {
		changes = append(changes, claudeBlock{Type: "tool_addition", Tool: &claudeToolReference{Type: "tool_reference", Name: toolName}})
	}
	for _, toolName := range removals {
		changes = append(changes, claudeBlock{Type: "tool_removal", Tool: &claudeToolReference{Type: "tool_reference", Name: toolName}})
	}
	// A system message must follow a user turn: prepareClaudeConversation stores the continuation turn after a
	// trailing assistant turn, and the payload builder places any change that still follows one.
	messages := append(append([]claudeMessage(nil), conversation.Messages...), claudeMessage{Role: "system", Content: changes})
	conversation.Messages = messages
	return conversation.withMeta(conversation.turn(), next)
}

// renderClaudeTools is the tools array for a conversation: every declared tool, later ones deferred. The
// definitions are copied, so request-only fields never reach the stored state.
func renderClaudeTools(state *claudeToolState) []map[string]any {
	out := make([]map[string]any, 0, len(state.Declared))
	for _, tool := range state.Declared {
		definition := make(map[string]any, len(tool.Definition)+1)
		for key, value := range tool.Definition {
			definition[key] = value
		}
		if tool.Deferred {
			definition["defer_loading"] = true
		}
		out = append(out, definition)
	}
	return out
}

// isClaudeToolChangeBlock reports whether a block announces a tool change.
func isClaudeToolChangeBlock(block claudeBlock) bool {
	return block.Type == "tool_addition" || block.Type == "tool_removal"
}

// stripClaudeToolChanges removes tool-change blocks for a request that sends tools the old way (an unsupported
// model or provider). System messages left empty are dropped.
func stripClaudeToolChanges(messages []claudeMessage) []claudeMessage {
	if !containsClaudeToolChanges(messages) {
		return messages
	}
	out := make([]claudeMessage, 0, len(messages))
	for _, message := range messages {
		content := make([]claudeBlock, 0, len(message.Content))
		for _, block := range message.Content {
			if !isClaudeToolChangeBlock(block) {
				content = append(content, block)
			}
		}
		if len(content) == len(message.Content) {
			out = append(out, message)
		} else if len(content) > 0 {
			out = append(out, claudeMessage{Role: message.Role, Content: content})
		}
	}
	return out
}

// containsClaudeToolChanges reports whether any message announces a tool change.
func containsClaudeToolChanges(messages []claudeMessage) bool {
	for _, message := range messages {
		for _, block := range message.Content {
			if isClaudeToolChangeBlock(block) {
				return true
			}
		}
	}
	return false
}

// placeClaudeSystemMessagesAfterUserTurns keeps each tool-change system message between a user turn and the next
// assistant turn, as the API requires. Orphan repair can drop the user turn a stored change followed, and an empty
// reply can leave a change before the next user turn. Well-formed histories pass through unchanged.
func placeClaudeSystemMessagesAfterUserTurns(messages []claudeMessage) []claudeMessage {
	hasSystem := false
	for _, message := range messages {
		if message.Role == "system" {
			hasSystem = true
			break
		}
	}
	if !hasSystem {
		return messages
	}
	out := make([]claudeMessage, 0, len(messages)+1)
	var pending []claudeMessage
	flush := func() {
		if len(pending) == 0 {
			return
		}
		merged := pending[0]
		if len(pending) > 1 {
			merged = mergeClaudeToolChangeMessages(pending)
		}
		pending = nil
		// Changes that cancel out leave nothing to announce.
		if len(merged.Content) == 0 {
			return
		}
		if len(out) == 0 || out[len(out)-1].Role != "user" {
			out = append(out, claudeContinueTurn())
		}
		out = append(out, merged)
	}
	for _, message := range messages {
		switch message.Role {
		case "system":
			pending = append(pending, message)
		case "user":
			out = append(out, message)
		default:
			flush()
			out = append(out, message)
		}
	}
	flush()
	return out
}

// mergeClaudeToolChangeMessages folds several pending system messages into one. Each stored change was a real
// change when it was made, so the changes to one tool alternate between addition and removal: an even number
// cancels out, an odd number leaves the last one.
func mergeClaudeToolChangeMessages(messages []claudeMessage) claudeMessage {
	var blocks []claudeBlock
	for _, message := range messages {
		blocks = append(blocks, message.Content...)
	}
	counts := map[string]int{}
	last := map[string]int{}
	for i, block := range blocks {
		if isClaudeToolChangeBlock(block) && block.Tool != nil && block.Tool.Name != "" {
			counts[block.Tool.Name]++
			last[block.Tool.Name] = i
		}
	}
	content := make([]claudeBlock, 0, len(blocks))
	for i, block := range blocks {
		if isClaudeToolChangeBlock(block) && block.Tool != nil && block.Tool.Name != "" {
			if counts[block.Tool.Name]%2 == 0 || last[block.Tool.Name] != i {
				continue
			}
		}
		content = append(content, block)
	}
	return claudeMessage{Role: "system", Content: content}
}
