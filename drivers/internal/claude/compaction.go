package claude

import "encoding/json"

// Server-side compaction for Claude, as in the TS client's claude-compaction.ts.
//
// Threshold compaction (beta compact-2026-01-12): when the request's input reaches the trigger, the API summarizes
// the earlier turns and returns a compaction block first in the response content. On-demand compaction (beta
// compact-2026-09-04): a request with compaction {type: summarize} returns only the compaction block, with stop
// reason "compaction", and no answer. Either way the block must be sent back unchanged; everything before it is
// superseded, so the stored conversation is trimmed to start at the latest one.

const (
	claudeCompactionBeta         = "compact-2026-01-12"
	claudeOnDemandCompactionBeta = "compact-2026-09-04"
)

// claudeOnDemandCompaction is the compaction field of an on-demand compaction request. An omitted prompt keeps the
// API's default.
func claudeOnDemandCompaction(request *CompactConversationOptions) map[string]any {
	out := map[string]any{"type": "summarize"}
	if request.Instructions != "" {
		out["instructions"] = request.Instructions
	}
	return out
}

// claudeCompactionContextManagement is the context_management field for threshold compaction; omitted fields keep
// the API defaults.
func claudeCompactionContextManagement(compaction *ConversationCompactionOptions) map[string]any {
	if compaction == nil {
		return nil
	}
	edit := map[string]any{"type": "compact_20260112"}
	if compaction.TriggerTokens > 0 {
		edit["trigger"] = map[string]any{"type": "input_tokens", "value": compaction.TriggerTokens}
	}
	if compaction.Instructions != "" {
		edit["instructions"] = compaction.Instructions
	}
	return map[string]any{"edits": []any{edit}}
}

// claudeCompactionSummary returns the summary of a compaction block. A block with null content is a failed
// compaction, which the API ignores.
func claudeCompactionSummary(block claudeBlock) (string, bool) {
	if block.Type != "compaction" {
		return "", false
	}
	raw, ok := block.field("content")
	if !ok {
		return "", false
	}
	var summary string
	if json.Unmarshal(raw, &summary) != nil || summary == "" {
		return "", false
	}
	return summary, true
}

// claudeCompactionResults are the summaries of the compaction blocks of a response, reported as the result of an
// on-demand compaction.
func claudeCompactionResults(content []claudeBlock) []CompletionResult {
	var results []CompletionResult
	for _, block := range content {
		if summary, ok := claudeCompactionSummary(block); ok {
			results = append(results, newTextResult(summary))
		}
	}
	return results
}

// trimClaudeConversationToCompaction drops everything before the latest compaction summary, which replaces it. The
// API ignores that history once the block is in the conversation, but it would still be stored and sent with every
// request, and a second summary alongside the first is rejected.
func trimClaudeConversationToCompaction(conversation claudePrompt) claudePrompt {
	for messageIndex := len(conversation.Messages) - 1; messageIndex >= 0; messageIndex-- {
		message := conversation.Messages[messageIndex]
		blockIndex := -1
		for i := len(message.Content) - 1; i >= 0; i-- {
			if _, ok := claudeCompactionSummary(message.Content[i]); ok {
				blockIndex = i
				break
			}
		}
		if blockIndex < 0 {
			continue
		}
		if messageIndex == 0 && blockIndex == 0 {
			return conversation
		}
		first := claudeMessage{Role: message.Role, Content: message.Content[blockIndex:]}
		conversation.Messages = append([]claudeMessage{first}, conversation.Messages[messageIndex+1:]...)
		return conversation
	}
	return conversation
}

// claudeIterationUsage is one entry of usage.iterations.
type claudeIterationUsage struct {
	Type                     string `json:"type"`
	InputTokens              int    `json:"input_tokens"`
	OutputTokens             int    `json:"output_tokens"`
	CacheReadInputTokens     int    `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int    `json:"cache_creation_input_tokens"`
	CacheCreation            *struct {
		Ephemeral1HInputTokens int `json:"ephemeral_1h_input_tokens"`
	} `json:"cache_creation"`
}

// claudeCompactionTotals is the usage of a request's compaction iterations.
type claudeCompactionTotals struct {
	Input, Output, CacheRead, CacheWrite, CacheWrite1H, Count int
}

// claudeCompactionUsage sums the compaction iterations, which the API leaves out of the top-level usage.
func claudeCompactionUsage(iterations []claudeIterationUsage) claudeCompactionTotals {
	var total claudeCompactionTotals
	for _, iteration := range iterations {
		if iteration.Type != "compaction" {
			continue
		}
		total.Count++
		total.Input += iteration.InputTokens
		total.Output += iteration.OutputTokens
		total.CacheRead += iteration.CacheReadInputTokens
		total.CacheWrite += iteration.CacheCreationInputTokens
		if iteration.CacheCreation != nil {
			total.CacheWrite1H += iteration.CacheCreation.Ephemeral1HInputTokens
		}
	}
	return total
}
