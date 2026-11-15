package openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vertesia/llumiverse-go/core"
)

const (
	conversationImagePlaceholder     = core.ConversationImagePlaceholder
	conversationTextTruncatedMarker  = core.ConversationTextTruncatedMarker
	conversationHeartbeatPlaceholder = core.ConversationHeartbeatPlaceholder
)

// openAIResponsesConversationState is the driver's retained conversation: the
// accumulated Responses input items plus the turn counter used by retention rules.
type openAIResponsesConversationState struct {
	Items []responseItem
	Turn  int
}

// openAIResponsesConversation reads a retained conversation in any of the forms a
// caller may pass back: the state this driver returns (by value, by pointer, or
// decoded from JSON), or a bare array of input items. ok is false for a value of
// any other shape.
func openAIResponsesConversation(conversation any) (items []responseItem, turn int, ok bool) {
	switch value := conversation.(type) {
	case nil:
		return nil, 0, true
	case openAIResponsesConversationState:
		return cloneItems(value.Items), value.Turn, true
	case *openAIResponsesConversationState:
		if value == nil {
			return nil, 0, true
		}
		return cloneItems(value.Items), value.Turn, true
	case []responseItem:
		return cloneItems(value), 0, true
	}
	// Anything else (JSON-decoded state, SDK param slices, []any) is normalized
	// through its JSON encoding, which is the shape the API accepts.
	data, err := json.Marshal(conversation)
	if err != nil {
		return nil, 0, false
	}
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, 0, false
	}
	switch value := raw.(type) {
	case nil:
		return nil, 0, true
	case []any:
		items, ok := itemsFromJSON(value)
		return items, 0, ok
	case map[string]any:
		rawItems, found := lookup(value, "Items", "items")
		if !found {
			return nil, 0, false
		}
		if rawItems == nil {
			return nil, intValue(lookupValue(value, "Turn", "turn")), true
		}
		list, isList := rawItems.([]any)
		if !isList {
			return nil, 0, false
		}
		items, ok := itemsFromJSON(list)
		return items, intValue(lookupValue(value, "Turn", "turn")), ok
	}
	return nil, 0, false
}

// itemsFromJSON converts decoded JSON array entries into input items.
func itemsFromJSON(list []any) ([]responseItem, bool) {
	items := make([]responseItem, 0, len(list))
	for _, entry := range list {
		item, ok := entry.(map[string]any)
		if !ok {
			return nil, false
		}
		items = append(items, item)
	}
	return items, true
}

// lookup returns the first value present under one of keys.
func lookup(value map[string]any, keys ...string) (any, bool) {
	for _, key := range keys {
		if raw, ok := value[key]; ok {
			return raw, true
		}
	}
	return nil, false
}

// lookupValue is lookup without the presence flag.
func lookupValue(value map[string]any, keys ...string) any {
	raw, _ := lookup(value, keys...)
	return raw
}

// intValue converts a decoded JSON number to an int, returning 0 for other values.
func intValue(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	}
	return 0
}

// cloneItems deep-copies items so request and retention processing never
// modify a conversation value the caller still holds.
func cloneItems(items []responseItem) []responseItem {
	if items == nil {
		return nil
	}
	out := make([]responseItem, len(items))
	for i, item := range items {
		out[i] = cloneJSONValue(item).(map[string]any)
	}
	return out
}

// cloneJSONValue deep-copies a JSON-shaped value (maps, slices, scalars).
func cloneJSONValue(v any) any {
	switch value := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(value))
		for key, inner := range value {
			out[key] = cloneJSONValue(inner)
		}
		return out
	case []any:
		out := make([]any, len(value))
		for i, inner := range value {
			out[i] = cloneJSONValue(inner)
		}
		return out
	case []map[string]any:
		out := make([]any, len(value))
		for i, inner := range value {
			out[i] = cloneJSONValue(inner)
		}
		return out
	default:
		return value
	}
}

// openAIResponsesInput combines retained conversation items with the new prompt
// items, repairs orphaned tool calls, and converts tool items to text when the
// current request has no tools.
func openAIResponsesInput(conversation any, prompt []responseItem, hasTools bool) []responseItem {
	stored, _, _ := openAIResponsesConversation(conversation)
	out := make([]responseItem, 0, len(stored)+len(prompt))
	out = append(out, stored...)
	out = append(out, cloneItems(prompt)...)
	out = fixOpenAIOrphanedToolResults(fixOpenAIOrphanedToolUse(out))
	if !hasTools && openAIResponsesInputHasToolItems(out) {
		// Stored conversations may include tool call/output items from checkpoint
		// summaries. If this request has no tools, convert them to text to avoid
		// Responses API validation errors.
		out = convertOpenAIResponsesToolItemsToText(out)
	}
	return out
}

// buildOpenAIResponsesConversation appends the response's output items to the
// request input, so reasoning (with its encrypted content), image generation
// calls and compaction items are replayed as the API returned them. Without
// output items it falls back to the completion's text and tool calls. It then
// increments the turn and applies the retention rules.
func buildOpenAIResponsesConversation(input []responseItem, output []responseItem, completion *Completion, options ExecutionOptions) openAIResponsesConversationState {
	out := make([]responseItem, 0, len(input)+len(output)+1)
	out = append(out, input...)
	if len(output) > 0 {
		out = append(out, output...)
	} else {
		if text := completionResultsToText(completion.Result); text != "" {
			out = append(out, responseItem{"type": "message", "role": "assistant", "content": text})
		}
		for _, tool := range completion.ToolUse {
			data, _ := json.Marshal(tool.ToolInput)
			out = append(out, responseItem{"type": "function_call", "call_id": tool.ID, "name": tool.ToolName, "arguments": string(data)})
		}
	}
	_, turn, _ := openAIResponsesConversation(options.Conversation)
	return finalizeOpenAIResponsesItems(out, turn+1, options)
}

// finalizeOpenAIResponsesItems applies the retention rules for the given turn.
// Expired generated images are replaced by a reference to the stored item.
func finalizeOpenAIResponsesItems(items []responseItem, turn int, options ExecutionOptions) openAIResponsesConversationState {
	stripMedia := shouldStripConversationMedia(turn, options.StripImagesAfterTurns)
	stripHeartbeats := shouldStripConversationHeartbeats(turn, options.StripHeartbeatsAfterTurns)
	if stripMedia {
		for i, item := range items {
			if id := stringValue(item["id"]); itemType(item) == "image_generation_call" && id != "" {
				items[i] = responseItem{"type": "item_reference", "id": id}
			}
		}
	}
	return openAIResponsesConversationState{
		Items: processOpenAIResponsesItemsForConversation(items, options, stripMedia, stripHeartbeats),
		Turn:  turn,
	}
}

// processOpenAIResponsesItemsForConversation applies the retention rules to
// retained items: expired inline images become a text placeholder, and text is
// truncated and expired heartbeats replaced.
func processOpenAIResponsesItemsForConversation(items []responseItem, options ExecutionOptions, stripMedia bool, stripHeartbeats bool) []responseItem {
	out := make([]responseItem, 0, len(items))
	for _, item := range items {
		next := cloneJSONValue(item).(map[string]any)
		switch itemType(next) {
		case "message":
			switch content := next["content"].(type) {
			case string:
				next["content"] = core.ProcessConversationText(content, options, stripHeartbeats)
			case []any:
				next["content"] = processContentParts(content, options, stripMedia, stripHeartbeats)
			}
		case "function_call":
			if args, ok := next["arguments"].(string); ok {
				next["arguments"] = core.ProcessConversationText(args, options, stripHeartbeats)
			}
		case "function_call_output":
			switch output := next["output"].(type) {
			case string:
				next["output"] = core.ProcessConversationText(output, options, stripHeartbeats)
			case []any:
				next["output"] = processContentParts(output, options, stripMedia, stripHeartbeats)
			}
		}
		out = append(out, next)
	}
	return out
}

// processContentParts applies the retention rules to a content-part list.
func processContentParts(parts []any, options ExecutionOptions, stripMedia bool, stripHeartbeats bool) []any {
	out := make([]any, 0, len(parts))
	for _, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok {
			out = append(out, raw)
			continue
		}
		switch part["type"] {
		case "input_image":
			// Only strip inline base64 images. Remote URLs are kept because the
			// conversation does not retain large local payloads for them.
			url := stringValue(part["image_url"])
			if stripMedia && strings.HasPrefix(url, "data:image/") && strings.Contains(url, ";base64,") {
				out = append(out, map[string]any{"type": "input_text", "text": conversationImagePlaceholder})
				continue
			}
		case "input_text", "output_text":
			if text, ok := part["text"].(string); ok {
				part["text"] = core.ProcessConversationText(text, options, stripHeartbeats)
			}
		}
		out = append(out, part)
	}
	return out
}

// itemType returns the type of an input item: its "type" field, or "message"
// for a role-bearing item without one.
func itemType(item responseItem) string {
	if t, ok := item["type"].(string); ok && t != "" {
		return t
	}
	if _, ok := item["role"]; ok {
		return "message"
	}
	return ""
}

// fixOpenAIOrphanedToolUse inserts synthetic function_call_output items for any
// function_call lacking a matching output, keeping the conversation valid.
func fixOpenAIOrphanedToolUse(items []responseItem) []responseItem {
	// If an agent was stopped after a function_call but before its output was
	// recorded, the next Responses request would contain an orphaned tool call.
	// Insert a synthetic interruption output so the conversation remains valid.
	if len(items) < 2 {
		return items
	}
	outputs := map[string]bool{}
	for _, item := range items {
		if itemType(item) == "function_call_output" {
			outputs[stringValue(item["call_id"])] = true
		}
	}
	out := make([]responseItem, 0, len(items))
	var pendingOrder []string
	pending := map[string]string{}
	flushPending := func() {
		for _, id := range pendingOrder {
			out = append(out, responseItem{
				"type":    "function_call_output",
				"call_id": id,
				"output":  fmt.Sprintf(`[Tool interrupted: The user stopped the operation before "%s" could execute.]`, pending[id]),
			})
		}
		pendingOrder = nil
		pending = map[string]string{}
	}
	for _, item := range items {
		switch itemType(item) {
		case "function_call":
			callID := stringValue(item["call_id"])
			if callID != "" && !outputs[callID] {
				name := stringValue(item["name"])
				if name == "" {
					name = "unknown"
				}
				if _, seen := pending[callID]; !seen {
					pendingOrder = append(pendingOrder, callID)
				}
				pending[callID] = name
			}
			out = append(out, item)
		case "function_call_output":
			out = append(out, item)
		default:
			flushPending()
			out = append(out, item)
		}
	}
	flushPending()
	return out
}

// fixOpenAIOrphanedToolResults drops function_call_output items whose call_id
// has no matching function_call anywhere in the input, as the TS client does.
// fixOpenAIOrphanedToolUse covers calls left unanswered (a cancelled run); this
// covers outputs left dangling after their call was dropped (for example by
// conversation trimming). The Responses API rejects such outputs.
func fixOpenAIOrphanedToolResults(items []responseItem) []responseItem {
	calls := map[string]bool{}
	for _, item := range items {
		if itemType(item) == "function_call" {
			calls[stringValue(item["call_id"])] = true
		}
	}
	out := make([]responseItem, 0, len(items))
	for _, item := range items {
		if itemType(item) == "function_call_output" {
			if id := stringValue(item["call_id"]); id == "" || !calls[id] {
				continue
			}
		}
		out = append(out, item)
	}
	return out
}

// openAIResponsesInputHasToolItems reports whether any input item is a tool call
// or tool output.
func openAIResponsesInputHasToolItems(items []responseItem) bool {
	for _, item := range items {
		if t := itemType(item); t == "function_call" || t == "function_call_output" {
			return true
		}
	}
	return false
}

// convertOpenAIResponsesToolItemsToText rewrites tool call/output items as
// bracketed transcript messages (truncated) for requests made without tools.
func convertOpenAIResponsesToolItemsToText(items []responseItem) []responseItem {
	// When tools are unavailable, preserve prior tool context as readable
	// transcript text instead of dropping it from conversation history.
	out := make([]responseItem, 0, len(items))
	for _, item := range items {
		switch itemType(item) {
		case "function_call":
			name := stringValue(item["name"])
			if name == "" {
				name = "unknown"
			}
			out = append(out, responseItem{
				"role":    "assistant",
				"content": fmt.Sprintf("[Tool call: %s(%s)]", name, truncateForConversation(stringValue(item["arguments"]), 500)),
			})
		case "function_call_output":
			output := functionCallOutputText(item["output"])
			if output == "" {
				output = "No output"
			}
			out = append(out, responseItem{
				"role":    "user",
				"content": "[Tool result: " + truncateForConversation(output, 500) + "]",
			})
		default:
			out = append(out, item)
		}
	}
	return out
}

// functionCallOutputText returns the text of a function_call_output: the string
// itself, or the joined input_text parts of a content list.
func functionCallOutputText(output any) string {
	switch value := output.(type) {
	case string:
		return value
	case []any:
		var texts []string
		for _, raw := range value {
			if part, ok := raw.(map[string]any); ok && part["type"] == "input_text" {
				texts = append(texts, stringValue(part["text"]))
			}
		}
		return strings.Join(texts, "\n")
	}
	return ""
}
