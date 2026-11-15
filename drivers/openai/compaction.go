package openai

import (
	"context"
	"encoding/json"

	"github.com/vertesia/llumiverse-go/common"
)

// Server-side compaction for the OpenAI Responses API. Once the rendered input
// crosses compact_threshold, the response carries an opaque compaction output
// item that replaces the context before it. The item is sent back with the
// rest of the output, and the items before the latest one may be dropped, so
// the stored conversation is trimmed to start there.

// openAIResponsesContextManagement is the context_management field for the
// caller's compaction options; an omitted trigger keeps OpenAI's default. The
// Responses API has no summarization prompt, so Instructions is not sent.
func openAIResponsesContextManagement(compaction *common.ConversationCompactionOptions) []map[string]any {
	if compaction == nil {
		return nil
	}
	entry := map[string]any{"type": "compaction"}
	if compaction.TriggerTokens > 0 {
		entry["compact_threshold"] = compaction.TriggerTokens
	}
	return []map[string]any{entry}
}

// trimOpenAIConversationToCompaction drops the items before the latest
// compaction item, which supersedes them. Conversations without one are
// unchanged.
func trimOpenAIConversationToCompaction(items []responseItem) []responseItem {
	for i := len(items) - 1; i > 0; i-- {
		if itemType(items[i]) == "compaction" {
			return items[i:]
		}
	}
	return items
}

// compactOpenAIConversation compacts the conversation on demand (POST
// responses/compact). The compacted output (the user messages, then one
// opaque compaction item) replaces the conversation as is, and there is no
// answer: the summary is encrypted. The compact endpoint has no summarization
// prompt and no threshold, so Instructions and Compaction are not sent.
func (d *OpenAICompatibleDriver) compactOpenAIConversation(ctx context.Context, input []responseItem, options ExecutionOptions, operation string) (*Completion, error) {
	payload := map[string]any{"model": options.Model, "input": input}
	if key := openAIPromptCacheKey(options); key != "" {
		payload["prompt_cache_key"] = key
	}
	var raw json.RawMessage
	if err := d.sdk.Post(ctx, "responses/compact", payload, &raw, d.requestOptions(options.HTTPTimeout)...); err != nil {
		return nil, d.formatError(err, options.Model, operation)
	}
	var response struct {
		Output []responsesOutputItem `json:"output"`
		Usage  *responsesUsage       `json:"usage"`
	}
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, err
	}
	_, turn, _ := openAIResponsesConversation(options.Conversation)
	completion := &Completion{
		Result:       []CompletionResult{},
		TokenUsage:   openAIResponseUsage(response.Usage, nil),
		FinishReason: "compaction",
		Conversation: finalizeOpenAIResponsesItems(rawOutputItems(response.Output), turn+1, options),
	}
	if options.IncludeOriginalResponse {
		var original any
		_ = json.Unmarshal(raw, &original)
		completion.OriginalResponse = original
	}
	return completion, nil
}

// assertOpenAIConversationShape refuses a conversation that is neither absent,
// an item array, nor the driver's stored state, as a non-retryable 400.
func (d *OpenAICompatibleDriver) assertOpenAIConversationShape(options ExecutionOptions, operation string) error {
	if _, _, ok := openAIResponsesConversation(options.Conversation); ok {
		return nil
	}
	message := "[" + string(d.Provider()) + "] The conversation must be an array of messages"
	return newLlumiverseError(message, boolPtr(false), LlumiverseErrorContext{Provider: d.Provider(), Model: options.Model, Operation: operation}, &invalidConversationError{message}, 400, "InvalidConversationError")
}

type invalidConversationError struct{ message string }

func (e *invalidConversationError) Error() string { return e.message }
