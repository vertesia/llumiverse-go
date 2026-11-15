package claude

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/vertesia/llumiverse-go/core"
)

const (
	// claudeKeepRecentMessages is how many of the latest messages the streaming fallback keeps whole when text
	// truncation is requested: the agent's active working set.
	claudeKeepRecentMessages = 12
	// claudeOldMessageTextMaxTokens caps the requested text limit for older messages in the streaming fallback.
	claudeOldMessageTextMaxTokens = 2000
)

// decodeClaudeConversation reads stored conversation state: a claudePrompt (value or pointer), a message list, or
// their JSON-decoded forms. Anything else is ignored.
func decodeClaudeConversation(conversation any) claudePrompt {
	switch value := conversation.(type) {
	case nil:
		return claudePrompt{}
	case claudePrompt:
		return value
	case *claudePrompt:
		if value != nil {
			return *value
		}
		return claudePrompt{}
	case []claudeMessage:
		return claudePrompt{Messages: value}
	}
	data, err := json.Marshal(conversation)
	if err != nil {
		return claudePrompt{}
	}
	var decoded claudePrompt
	if json.Unmarshal(data, &decoded) == nil {
		return decoded
	}
	var messages []claudeMessage
	if json.Unmarshal(data, &messages) == nil {
		return claudePrompt{Messages: messages}
	}
	return claudePrompt{}
}

// updateClaudeConversation appends a prompt to stored conversation state: the system blocks are merged (keeping
// one generated schema block), empty text dropped and consecutive user turns merged. The turn number and the
// declared tools are carried forward.
func updateClaudeConversation(conversation claudePrompt, prompt claudePrompt) claudePrompt {
	messages := append(cloneMessages(conversation.Messages), cloneMessages(prompt.Messages)...)
	out := claudePrompt{
		Messages: mergeConsecutiveClaudeUserMessages(sanitizeClaudeMessages(messages)),
		System:   mergeClaudeSystemBlocks(conversation.System, prompt.System),
	}
	if len(out.System) == 0 {
		out.System = nil
	}
	return out.withMeta(conversation.turn(), conversation.toolState())
}

// prepareClaudeConversation appends this turn's prompt to the stored conversation and records its tool set. A
// resumed conversation that ends with an assistant turn gets the continuation turn the payload builder would add on
// Claude 4.6+, so the next request replays what this one sent.
func prepareClaudeConversation(options ExecutionOptions, prompt claudePrompt, provider string) claudePrompt {
	conversation := updateClaudeConversation(decodeClaudeConversation(options.Conversation), prompt)
	if isClaudeVersionGTE(options.Model, 4, 6) {
		for i := len(conversation.Messages) - 1; i >= 0; i-- {
			if conversation.Messages[i].Role == "system" {
				continue
			}
			if conversation.Messages[i].Role == "assistant" {
				conversation.Messages = append(conversation.Messages, claudeContinueTurn())
			}
			break
		}
	}
	if canUseClaudeToolChanges(options.Model, provider) {
		return applyClaudeToolChanges(conversation, options.Tools)
	}
	return conversation.withMeta(conversation.turn(), nil)
}

// sanitizeClaudeMessages drops blank text blocks and messages left empty: the API rejects both.
func sanitizeClaudeMessages(messages []claudeMessage) []claudeMessage {
	out := make([]claudeMessage, 0, len(messages))
	for _, message := range messages {
		content := make([]claudeBlock, 0, len(message.Content))
		for _, block := range message.Content {
			if block.Type == "text" && strings.TrimSpace(block.Text) == "" {
				continue
			}
			content = append(content, block)
		}
		if len(content) > 0 {
			out = append(out, claudeMessage{Role: message.Role, Content: content})
		}
	}
	return out
}

// mergeConsecutiveClaudeUserMessages merges adjacent user turns into one, with tool_result blocks first: they must
// lead a user turn.
func mergeConsecutiveClaudeUserMessages(messages []claudeMessage) []claudeMessage {
	needsMerging := false
	for i, message := range messages {
		if message.Role == "user" && ((i+1 < len(messages) && messages[i+1].Role == "user") || hasClaudeToolResultAfterOtherContent(message)) {
			needsMerging = true
			break
		}
	}
	if !needsMerging {
		return messages
	}
	out := make([]claudeMessage, 0, len(messages))
	for i := 0; i < len(messages); {
		if messages[i].Role != "user" {
			out = append(out, messages[i])
			i++
			continue
		}
		var merged []claudeBlock
		for i < len(messages) && messages[i].Role == "user" {
			merged = append(merged, messages[i].Content...)
			i++
		}
		var results, others []claudeBlock
		for _, block := range merged {
			if block.Type == "tool_result" {
				results = append(results, block)
			} else {
				others = append(others, block)
			}
		}
		out = append(out, claudeMessage{Role: "user", Content: append(results, others...)})
	}
	return out
}

// hasClaudeToolResultAfterOtherContent reports a tool_result that follows another block in the same turn.
func hasClaudeToolResultAfterOtherContent(message claudeMessage) bool {
	seenOther := false
	for _, block := range message.Content {
		if block.Type != "tool_result" {
			seenOther = true
		} else if seenOther {
			return true
		}
	}
	return false
}

// fixClaudeOrphanedToolUse answers tool_use blocks that have no result in the next user turn. An agent stopped
// after Claude asked for a tool but before it ran would otherwise send an invalid conversation. The synthetic
// results are written into a copy, never into the stored conversation.
func fixClaudeOrphanedToolUse(messages []claudeMessage) []claudeMessage {
	if len(messages) < 2 {
		return messages
	}
	out := append([]claudeMessage(nil), messages...)
	for i := 0; i < len(out)-1; i++ {
		if out[i].Role != "assistant" || out[i+1].Role != "user" {
			continue
		}
		answered := map[string]bool{}
		for _, block := range out[i+1].Content {
			if block.Type == "tool_result" {
				answered[block.ToolUseID] = true
			}
		}
		var synthetic []claudeBlock
		for _, block := range out[i].Content {
			if block.Type != "tool_use" || answered[block.ID] {
				continue
			}
			text := fmt.Sprintf(`[Tool interrupted: The user stopped the operation before "%s" could execute.]`, block.Name)
			synthetic = append(synthetic, claudeBlock{Type: "tool_result", ToolUseID: block.ID, ContentText: &text})
		}
		if len(synthetic) > 0 {
			out[i+1] = claudeMessage{Role: "user", Content: append(synthetic, out[i+1].Content...)}
		}
	}
	return out
}

// fixClaudeOrphanedToolResults drops tool_result blocks with no matching tool_use in the assistant turn just before
// them, for instance after the tool_use was trimmed away. The API rejects such a request with a non-retryable 400.
// It runs after user turns are merged, so results split across user messages are first recombined.
func fixClaudeOrphanedToolResults(messages []claudeMessage) []claudeMessage {
	out := make([]claudeMessage, 0, len(messages))
	for i, message := range messages {
		if message.Role != "user" || !claudeMessageHasBlock(message, "tool_result") {
			out = append(out, message)
			continue
		}
		allowed := map[string]bool{}
		if i > 0 && messages[i-1].Role == "assistant" {
			for _, block := range messages[i-1].Content {
				if block.Type == "tool_use" {
					allowed[block.ID] = true
				}
			}
		}
		content := make([]claudeBlock, 0, len(message.Content))
		for _, block := range message.Content {
			if block.Type != "tool_result" || allowed[block.ToolUseID] {
				content = append(content, block)
			}
		}
		// A turn made only of orphaned results is dropped rather than sent empty.
		if len(content) == 0 {
			continue
		}
		out = append(out, claudeMessage{Role: message.Role, Content: content})
	}
	return out
}

// claudeMessageHasBlock reports whether a message holds a block of the given type.
func claudeMessageHasBlock(message claudeMessage, blockType string) bool {
	for _, block := range message.Content {
		if block.Type == blockType {
			return true
		}
	}
	return false
}

// claudeMessagesContainToolBlocks reports whether any message holds a tool_use or tool_result block.
func claudeMessagesContainToolBlocks(messages []claudeMessage) bool {
	for _, message := range messages {
		if claudeMessageHasBlock(message, "tool_use") || claudeMessageHasBlock(message, "tool_result") {
			return true
		}
	}
	return false
}

// truncateClaudeToolText cuts a tool text to 500 characters for the tool-free text form.
func truncateClaudeToolText(text string) string {
	if utf8.RuneCountInString(text) <= 500 {
		return text
	}
	return string([]rune(text)[:500]) + "..."
}

// convertClaudeToolBlocksToText rewrites tool_use and tool_result blocks as text, so a conversation with tool calls
// can be sent in a request that declares no tools.
func convertClaudeToolBlocksToText(messages []claudeMessage) []claudeMessage {
	out := make([]claudeMessage, 0, len(messages))
	for _, message := range messages {
		if !claudeMessageHasBlock(message, "tool_use") && !claudeMessageHasBlock(message, "tool_result") {
			out = append(out, message)
			continue
		}
		content := make([]claudeBlock, 0, len(message.Content))
		for _, block := range message.Content {
			switch block.Type {
			case "tool_use":
				input := ""
				if block.Input != nil {
					input = claudeCompactJSON(block.Input)
				}
				content = append(content, claudeBlock{Type: "text", Text: fmt.Sprintf("[Tool call: %s(%s)]", block.Name, truncateClaudeToolText(input))})
			case "tool_result":
				result := "No content"
				switch {
				case block.ContentText != nil:
					result = truncateClaudeToolText(*block.ContentText)
				case block.Content != nil:
					var texts []string
					for _, item := range block.Content {
						if item.Type == "text" {
							texts = append(texts, truncateClaudeToolText(item.Text))
						}
					}
					result = strings.Join(texts, "\n")
					if result == "" {
						result = "No text content"
					}
				}
				content = append(content, claudeBlock{Type: "text", Text: "[Tool result: " + result + "]"})
			default:
				content = append(content, block)
			}
		}
		out = append(out, claudeMessage{Role: message.Role, Content: content})
	}
	return out
}

// findClaudeActiveTurnStart returns the index of the first message of the active tool loop: when the latest
// assistant turn calls tools, the message after the last plain user turn before it. It returns -1 otherwise.
func findClaudeActiveTurnStart(messages []claudeMessage) int {
	active := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "assistant" {
			continue
		}
		if claudeMessageHasBlock(messages[i], "tool_use") {
			active = i
		}
		break
	}
	if active < 0 {
		return -1
	}
	for i := active - 1; i >= 0; i-- {
		if messages[i].Role == "user" && !claudeMessageHasBlock(messages[i], "tool_result") {
			return i + 1
		}
	}
	return 0
}

// pruneClaudeThinking drops thinking blocks from assistant turns before the active tool loop (or before the latest
// assistant turn when no loop is active). Models that do not keep earlier turns' thinking strip it themselves on
// the next plain user turn, so pruning it here confines the cache invalidation to the latest loop.
func pruneClaudeThinking(conversation claudePrompt) claudePrompt {
	activeStart := findClaudeActiveTurnStart(conversation.Messages)
	latestAssistant := -1
	for i := len(conversation.Messages) - 1; i >= 0; i-- {
		if conversation.Messages[i].Role == "assistant" {
			latestAssistant = i
			break
		}
	}
	preserveFrom := latestAssistant
	if activeStart >= 0 {
		preserveFrom = activeStart
	}
	messages := make([]claudeMessage, len(conversation.Messages))
	for i, message := range conversation.Messages {
		if (latestAssistant >= 0 && i >= preserveFrom && i <= latestAssistant) || message.Role != "assistant" {
			messages[i] = message
			continue
		}
		content := make([]claudeBlock, 0, len(message.Content))
		for _, block := range message.Content {
			if block.Type != "thinking" && block.Type != "redacted_thinking" {
				content = append(content, block)
			}
		}
		messages[i] = claudeMessage{Role: message.Role, Content: content}
	}
	conversation.Messages = messages
	return conversation
}

// claudeRetention is the retention policy applied to a stored conversation.
type claudeRetention struct {
	stripMedia      bool
	textMaxTokens   int
	stripHeartbeats bool
}

// claudeHeartbeatKeepTurns is how many turns heartbeat messages are kept. With prompt caching they are never
// stripped: stripping rewrites an earlier user turn, so the cached prefix would stop matching there.
func claudeHeartbeatKeepTurns(options ExecutionOptions) *int {
	if isClaudePromptCacheEnabled(options) {
		keep := math.MaxInt
		return &keep
	}
	return options.StripHeartbeatsAfterTurns
}

// text applies heartbeat stripping and text truncation to one string.
func (r claudeRetention) text(value string) string {
	if r.stripHeartbeats && strings.HasPrefix(value, core.ConversationHeartbeatOpenTag) && strings.HasSuffix(value, core.ConversationHeartbeatCloseTag) {
		return core.ConversationHeartbeatPlaceholder
	}
	if strings.HasPrefix(value, "data:image/") && strings.Contains(value, ";base64,") {
		return value
	}
	return core.TruncateConversationText(value, r.textMaxTokens)
}

// blocks applies the policy to blocks. Thinking and opaque blocks (compaction, server tools) are sent back
// verbatim, so they are left untouched; base64 images and documents become placeholders once media expires.
func (r claudeRetention) blocks(blocks []claudeBlock) []claudeBlock {
	if blocks == nil {
		return nil
	}
	out := make([]claudeBlock, 0, len(blocks))
	for _, block := range blocks {
		if block.raw != nil {
			out = append(out, block)
			continue
		}
		switch block.Type {
		case "thinking", "redacted_thinking", "tool_use", "tool_addition", "tool_removal":
			out = append(out, block)
			continue
		case "image", "document":
			if block.Source != nil && block.Source.Type == "base64" {
				if r.stripMedia {
					placeholder := core.ConversationImagePlaceholder
					if block.Type == "document" {
						placeholder = core.ConversationDocumentPlaceholder
					}
					out = append(out, claudeBlock{Type: "text", Text: placeholder})
				} else {
					out = append(out, block)
				}
				continue
			}
			if block.Source != nil {
				source := *block.Source
				source.Data = r.text(source.Data)
				block.Source = &source
			}
		case "text":
			block.Text = r.text(block.Text)
		case "tool_result":
			if block.ContentText != nil {
				text := r.text(*block.ContentText)
				block.ContentText = &text
			}
			block.Content = r.blocks(block.Content)
		}
		out = append(out, block)
	}
	return out
}

// active reports whether the policy changes anything.
func (r claudeRetention) active() bool {
	return r.stripMedia || r.textMaxTokens > 0 || r.stripHeartbeats
}

// applyClaudeRetention applies a policy to the system blocks and to the messages before keepFrom; later messages
// are left whole.
func applyClaudeRetention(conversation claudePrompt, policy claudeRetention, keepFrom int) claudePrompt {
	if !policy.active() {
		return conversation
	}
	conversation.System = policy.blocks(conversation.System)
	messages := make([]claudeMessage, len(conversation.Messages))
	for i, message := range conversation.Messages {
		if i >= keepFrom {
			messages[i] = message
			continue
		}
		messages[i] = claudeMessage{Role: message.Role, Content: policy.blocks(message.Content)}
	}
	conversation.Messages = messages
	return conversation
}

// claudeRetentionPolicy is the policy for a turn: media after StripImagesAfterTurns, text truncation to maxTokens
// (none with prompt caching, which needs a byte-stable history), heartbeats after their keep window.
func claudeRetentionPolicy(options ExecutionOptions, turn int, maxTokens int) claudeRetention {
	policy := claudeRetention{
		stripMedia:      core.ShouldStripConversationMedia(turn, options.StripImagesAfterTurns),
		stripHeartbeats: core.ShouldStripConversationHeartbeats(turn, claudeHeartbeatKeepTurns(options)),
	}
	if maxTokens > 0 && !isClaudePromptCacheEnabled(options) {
		policy.textMaxTokens = maxTokens
	}
	return policy
}

// finalizeClaudeConversation appends a response to the conversation and applies the retention policy: the history
// is trimmed to the latest compaction summary, the turn counter advances, earlier turns' thinking is pruned for
// models that do not keep it, and media, long text and heartbeats are cleaned outside the active tool loop.
func finalizeClaudeConversation(conversation claudePrompt, response claudeMessage, options ExecutionOptions) claudePrompt {
	if response.Role == "" {
		response.Role = "assistant"
	}
	completed := updateClaudeConversation(conversation, claudePrompt{Messages: []claudeMessage{response}})
	completed = trimClaudeConversationToCompaction(completed)
	completed = completed.withMeta(completed.turn()+1, completed.toolState())
	if !keepsClaudePriorTurnThinking(options.Model) {
		completed = pruneClaudeThinking(completed)
	}
	keepFrom := len(completed.Messages)
	if start := findClaudeActiveTurnStart(completed.Messages); start >= 0 {
		keepFrom = start
	}
	return applyClaudeRetention(completed, claudeRetentionPolicy(options, completed.turn(), options.StripTextMaxTokens), keepFrom)
}

// buildClaudeStreamingConversation rebuilds the conversation of a stream that ended without a complete message,
// from its accumulated text and tool calls. When text truncation is requested, the latest messages stay whole and
// older text is cut to at most 2000 tokens, so long agent conversations do not balloon.
func buildClaudeStreamingConversation(conversation claudePrompt, completion *Completion, options ExecutionOptions) claudePrompt {
	if text := completionTextOnly(completion.Result); text != "" {
		conversation = updateClaudeConversation(conversation, claudePrompt{Messages: []claudeMessage{{Role: "assistant", Content: []claudeBlock{{Type: "text", Text: text}}}}})
	}
	if len(completion.ToolUse) > 0 {
		blocks := make([]claudeBlock, 0, len(completion.ToolUse))
		for _, tool := range completion.ToolUse {
			input := map[string]any{}
			if tool.ToolInput != nil {
				input = map[string]any(*tool.ToolInput)
			}
			blocks = append(blocks, claudeBlock{Type: "tool_use", ID: tool.ID, Name: tool.ToolName, Input: input})
		}
		conversation = updateClaudeConversation(conversation, claudePrompt{Messages: []claudeMessage{{Role: "assistant", Content: blocks}}})
	}
	conversation = conversation.withMeta(conversation.turn()+1, conversation.toolState())
	maxTokens := 0
	keepFrom := len(conversation.Messages)
	if options.StripTextMaxTokens > 0 && !isClaudePromptCacheEnabled(options) {
		maxTokens = min(options.StripTextMaxTokens, claudeOldMessageTextMaxTokens)
		keepFrom = core.ConversationTruncationCutoff(len(conversation.Messages), claudeKeepRecentMessages)
	}
	policy := claudeRetentionPolicy(options, conversation.turn(), maxTokens)
	// Media and heartbeat cleanup apply to every message; only text truncation spares the recent ones.
	whole := policy
	whole.textMaxTokens = 0
	conversation = applyClaudeRetention(conversation, whole, len(conversation.Messages))
	if policy.textMaxTokens > 0 {
		// Turn-aware truncation applies to older messages only, not to the system prompt.
		system := conversation.System
		conversation = applyClaudeRetention(conversation, claudeRetention{textMaxTokens: policy.textMaxTokens}, keepFrom)
		conversation.System = system
	}
	return conversation
}

// completionTextOnly joins the text results, leaving thoughts out.
func completionTextOnly(results []CompletionResult) string {
	var builder strings.Builder
	for _, result := range results {
		if result.Type() == ResultTypeText {
			builder.WriteString(result.String())
		}
	}
	return builder.String()
}
