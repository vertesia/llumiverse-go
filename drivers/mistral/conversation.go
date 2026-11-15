package mistral

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
)

// Conversation is the conversation state the driver returns: the Mistral messages so far, in wire shape, and
// the turn counter the retention options count against.
type Conversation struct {
	Messages []map[string]any `json:"messages"`
	Turn     int              `json:"turn,omitempty"`
}

// storedConversation reads the conversation shapes the driver accepts: its own, the TS client's (whose messages
// carry the SDK's camelCase field names), and the Chat Completions conversation earlier clients stored for
// Mistral, marked _is_openai_chat_completions.
type storedConversation struct {
	Messages                []map[string]any `json:"messages"`
	Turn                    int              `json:"turn"`
	IsOpenAIChatCompletions bool             `json:"_is_openai_chat_completions"`
}

// requestConversation prepends the stored conversation to the prompt's messages.
func requestConversation(stored any, prompt any) (Conversation, error) {
	conversation, err := readConversation(stored)
	if err != nil {
		return Conversation{}, err
	}
	var messages []map[string]any
	switch value := prompt.(type) {
	case Prompt:
		messages = value.Messages
	case *Prompt:
		if value != nil {
			messages = value.Messages
		}
	default:
		return Conversation{}, fmt.Errorf("mistral requires a Mistral prompt, got %T", prompt)
	}
	for _, message := range messages {
		conversation.Messages = append(conversation.Messages, cloneValue(message).(map[string]any))
	}
	return conversation, nil
}

func readConversation(stored any) (Conversation, error) {
	switch value := stored.(type) {
	case nil:
		return Conversation{}, nil
	case Conversation:
		return Conversation{Messages: cloneMessages(value.Messages), Turn: value.Turn}, nil
	case *Conversation:
		if value == nil {
			return Conversation{}, nil
		}
		return Conversation{Messages: cloneMessages(value.Messages), Turn: value.Turn}, nil
	}
	data, err := json.Marshal(stored)
	if err != nil {
		return Conversation{}, err
	}
	var state storedConversation
	if json.Unmarshal(data, &state) != nil {
		// Not an object with messages: start a new conversation, as the TS client does.
		return Conversation{}, nil
	}
	conversation := Conversation{Turn: state.Turn, Messages: make([]map[string]any, 0, len(state.Messages))}
	for _, message := range state.Messages {
		converted := normalizeStoredMessage(message)
		if state.IsOpenAIChatCompletions {
			if converted, err = fromChatCompletionsMessage(message); err != nil {
				return Conversation{}, err
			}
		}
		conversation.Messages = append(conversation.Messages, converted)
	}
	return conversation, nil
}

// storedFieldNames maps the TS SDK's camelCase message fields to the wire names.
var storedFieldNames = map[string]string{
	"toolCalls":   "tool_calls",
	"toolCallId":  "tool_call_id",
	"imageUrl":    "image_url",
	"inputAudio":  "input_audio",
	"documentUrl": "document_url",
	"fileId":      "file_id",
}

// normalizeStoredMessage renames camelCase fields, as the TS client stores them, to their wire names.
func normalizeStoredMessage(message map[string]any) map[string]any {
	out := make(map[string]any, len(message))
	for key, value := range message {
		if name, ok := storedFieldNames[key]; ok {
			key = name
		}
		out[key] = value
	}
	if parts, ok := out["content"].([]any); ok {
		normalized := make([]any, len(parts))
		for i, raw := range parts {
			if part, ok := raw.(map[string]any); ok {
				normalized[i] = normalizeStoredMessage(part)
			} else {
				normalized[i] = raw
			}
		}
		out["content"] = normalized
	}
	return out
}

// fromChatCompletionsMessage converts a stored Chat Completions message to a Mistral message.
func fromChatCompletionsMessage(message map[string]any) (map[string]any, error) {
	content := message["content"]
	var parts []any
	if raw, ok := content.([]any); ok {
		parts = make([]any, 0, len(raw))
		for _, item := range raw {
			part, _ := item.(map[string]any)
			switch part["type"] {
			case "input_audio":
				return nil, errors.New("this inference endpoint does not support audio input")
			case "text":
				parts = append(parts, map[string]any{"type": "text", "text": part["text"]})
			case "image_url":
				image, _ := part["image_url"].(map[string]any)
				parts = append(parts, map[string]any{"type": "image_url", "image_url": image["url"]})
			}
		}
	}
	text, _ := content.(string)
	switch message["role"] {
	case "system", "developer":
		return map[string]any{"role": "system", "content": textOf(content)}, nil
	case "assistant":
		out := map[string]any{"role": "assistant", "content": content}
		if parts != nil {
			out["content"] = parts
		}
		if calls, ok := message["tool_calls"].([]any); ok && len(calls) > 0 {
			converted := make([]any, len(calls))
			for i, raw := range calls {
				call, _ := raw.(map[string]any)
				converted[i] = map[string]any{"id": call["id"], "index": i, "type": "function", "function": call["function"]}
			}
			out["tool_calls"] = converted
		}
		return out, nil
	case "tool":
		out := map[string]any{"role": "tool", "content": text, "tool_call_id": message["tool_call_id"]}
		if parts != nil {
			out["content"] = parts
		}
		return out, nil
	default:
		out := map[string]any{"role": "user", "content": text}
		if parts != nil {
			out["content"] = parts
		}
		return out, nil
	}
}

func textOf(content any) string {
	switch value := content.(type) {
	case string:
		return value
	case []any:
		var texts []string
		for _, raw := range value {
			if part, ok := raw.(map[string]any); ok && part["type"] == "text" {
				if text, ok := part["text"].(string); ok {
					texts = append(texts, text)
				}
			}
		}
		return strings.Join(texts, "\n")
	}
	return ""
}

// finalizeConversation counts the turn and applies the retention options: expired inline images and all inline
// audio become placeholders, text is truncated and expired heartbeats removed. Messages carrying signed
// thinking are kept verbatim, since Mistral checks the signature against them.
func finalizeConversation(conversation Conversation, options common.ExecutionOptions) Conversation {
	conversation.Turn++
	stripMedia := core.ShouldStripConversationMedia(conversation.Turn, options.StripImagesAfterTurns)
	stripHeartbeats := core.ShouldStripConversationHeartbeats(conversation.Turn, options.StripHeartbeatsAfterTurns)
	for i, message := range conversation.Messages {
		if hasSignedThinking(message) {
			continue
		}
		switch content := message["content"].(type) {
		case string:
			message["content"] = core.ProcessConversationText(content, options, stripHeartbeats)
		case []any:
			message["content"] = retainParts(content, options, stripMedia, stripHeartbeats)
		}
		conversation.Messages[i] = message
	}
	return conversation
}

func hasSignedThinking(message map[string]any) bool {
	parts, _ := message["content"].([]any)
	for _, raw := range parts {
		if part, ok := raw.(map[string]any); ok && part["type"] == "thinking" {
			if signature, _ := part["signature"].(string); signature != "" {
				return true
			}
		}
	}
	return false
}

func retainParts(parts []any, options common.ExecutionOptions, stripMedia, stripHeartbeats bool) []any {
	out := make([]any, 0, len(parts))
	for _, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok {
			out = append(out, raw)
			continue
		}
		switch part["type"] {
		case "image_url":
			if stripMedia && isInlineImage(part["image_url"]) {
				out = append(out, map[string]any{"type": "text", "text": core.ConversationImagePlaceholder})
				continue
			}
		case "input_audio":
			// Inline audio is not kept in history on any turn.
			out = append(out, map[string]any{"type": "text", "text": core.ConversationAudioPlaceholder})
			continue
		case "text":
			if text, ok := part["text"].(string); ok {
				part["text"] = core.ProcessConversationText(text, options, stripHeartbeats)
			}
		}
		out = append(out, part)
	}
	return out
}

// isInlineImage reports an image_url value (a URL string, or an object with url) holding base64 data.
func isInlineImage(value any) bool {
	url, ok := value.(string)
	if !ok {
		image, _ := value.(map[string]any)
		url, _ = image["url"].(string)
	}
	return strings.HasPrefix(url, "data:image/") && strings.Contains(url, ";base64,")
}

func cloneMessages(messages []map[string]any) []map[string]any {
	out := make([]map[string]any, len(messages))
	for i, message := range messages {
		out[i] = cloneValue(message).(map[string]any)
	}
	return out
}

// cloneValue deep-copies JSON-shaped values, so retention never rewrites the caller's conversation.
func cloneValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, item := range typed {
			out[key] = cloneValue(item)
		}
		return out
	case []any:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = cloneValue(item)
		}
		return out
	}
	return value
}
