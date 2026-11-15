package claude

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// claudePrompt is the stored Claude conversation and the Messages API prompt: an ordered list of user, assistant
// and (for tool changes) system messages, the system blocks, and llumiverse metadata. The metadata is stored with
// the conversation as _llumiverse_meta, like the TS client's, and is never sent to the API.
type claudePrompt struct {
	Messages []claudeMessage         `json:"messages"`
	System   []claudeBlock           `json:"system,omitempty"`
	Meta     *claudeConversationMeta `json:"_llumiverse_meta,omitempty"`
}

// claudeConversationMeta is the conversation metadata: the turn counter that drives retention, and the tools
// declared for append-only tool changes.
type claudeConversationMeta struct {
	TurnNumber  int              `json:"turnNumber"`
	ClaudeTools *claudeToolState `json:"claude_tools,omitempty"`
}

// turn returns the stored turn number.
func (p claudePrompt) turn() int {
	if p.Meta == nil {
		return 0
	}
	return p.Meta.TurnNumber
}

// toolState returns the declared tools, or nil when the conversation tracks none.
func (p claudePrompt) toolState() *claudeToolState {
	if p.Meta == nil {
		return nil
	}
	return p.Meta.ClaudeTools
}

// withMeta returns a copy of the prompt with new metadata, dropping metadata that carries nothing.
func (p claudePrompt) withMeta(turn int, tools *claudeToolState) claudePrompt {
	if turn == 0 && tools == nil {
		p.Meta = nil
		return p
	}
	p.Meta = &claudeConversationMeta{TurnNumber: turn, ClaudeTools: tools}
	return p
}

// claudeMessage is one turn: "user", "assistant", or "system" (a tool-change announcement).
type claudeMessage struct {
	Role    string        `json:"role"`
	Content []claudeBlock `json:"content"`
}

// UnmarshalJSON accepts string content, which the TS client stores for some turns, as one text block.
func (m *claudeMessage) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	m.Role = raw.Role
	m.Content = nil
	content := bytes.TrimSpace(raw.Content)
	if len(content) == 0 || bytes.Equal(content, []byte("null")) {
		return nil
	}
	if content[0] == '"' {
		var text string
		if err := json.Unmarshal(content, &text); err != nil {
			return err
		}
		m.Content = []claudeBlock{{Type: "text", Text: text}}
		return nil
	}
	return json.Unmarshal(content, &m.Content)
}

// claudeBlock is one Messages API content block. Type selects the fields that apply:
//   - text: Text
//   - image, document: Source (and Title for documents)
//   - tool_use: ID, Name, Input
//   - tool_result: ToolUseID, Content or ContentText, IsError
//   - thinking: Thinking, Signature; redacted_thinking: Data
//   - tool_addition, tool_removal: Tool
//
// Any other type (compaction, server tool blocks, ...) is kept verbatim in raw and sent back unchanged.
type claudeBlock struct {
	Type         string
	Text         string
	Title        string
	Source       *claudeSource
	ToolUseID    string
	Content      []claudeBlock
	ContentText  *string
	IsError      *bool
	ID           string
	Name         string
	Input        any
	Thinking     string
	Signature    string
	Data         string
	Tool         *claudeToolReference
	CacheControl any

	raw json.RawMessage
}

// claudeSource carries inline base64 (or text) data for image and document blocks.
type claudeSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// claudeToolReference names a tool in a tool_addition or tool_removal block.
type claudeToolReference struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

// opaqueClaudeBlock returns a block that is sent back exactly as raw.
func opaqueClaudeBlock(raw json.RawMessage) claudeBlock {
	var head struct {
		Type string `json:"type"`
	}
	_ = json.Unmarshal(raw, &head)
	return claudeBlock{Type: head.Type, raw: append(json.RawMessage(nil), raw...)}
}

// MarshalJSON writes the fields the block's type requires, so a required field such as an empty thinking text is
// never dropped. Maps marshal with sorted keys, so a block serializes to the same bytes on every request.
func (b claudeBlock) MarshalJSON() ([]byte, error) {
	if b.raw != nil {
		if b.CacheControl == nil {
			return b.raw, nil
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(b.raw, &fields); err != nil {
			return nil, err
		}
		cacheControl, err := json.Marshal(b.CacheControl)
		if err != nil {
			return nil, err
		}
		fields["cache_control"] = cacheControl
		return json.Marshal(fields)
	}
	out := map[string]any{"type": b.Type}
	switch b.Type {
	case "text":
		out["text"] = b.Text
	case "image":
		out["source"] = b.Source
	case "document":
		out["source"] = b.Source
		if b.Title != "" {
			out["title"] = b.Title
		}
	case "tool_use":
		input := b.Input
		if input == nil {
			input = map[string]any{}
		}
		out["id"] = b.ID
		out["name"] = b.Name
		out["input"] = input
	case "tool_result":
		out["tool_use_id"] = b.ToolUseID
		switch {
		case b.ContentText != nil:
			out["content"] = *b.ContentText
		case b.Content != nil:
			out["content"] = b.Content
		}
		if b.IsError != nil {
			out["is_error"] = *b.IsError
		}
	case "thinking":
		out["thinking"] = b.Thinking
		out["signature"] = b.Signature
	case "redacted_thinking":
		out["data"] = b.Data
	case "tool_addition", "tool_removal":
		out["tool"] = b.Tool
	default:
		return nil, fmt.Errorf("claude block of type %q has no stored content", b.Type)
	}
	if b.CacheControl != nil {
		out["cache_control"] = b.CacheControl
	}
	return json.Marshal(out)
}

// UnmarshalJSON reads the fields of the known block types and keeps any other block verbatim.
func (b *claudeBlock) UnmarshalJSON(data []byte) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	var typ string
	_ = json.Unmarshal(fields["type"], &typ)
	*b = claudeBlock{Type: typ}
	str := func(key string) string {
		var s string
		_ = json.Unmarshal(fields[key], &s)
		return s
	}
	switch typ {
	case "text":
		b.Text = str("text")
	case "image", "document":
		if raw, ok := fields["source"]; ok {
			b.Source = &claudeSource{}
			if err := json.Unmarshal(raw, b.Source); err != nil {
				return err
			}
		}
		b.Title = str("title")
	case "tool_use":
		b.ID = str("id")
		b.Name = str("name")
		if raw, ok := fields["input"]; ok && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			// Keep the provider's bytes so a replayed tool call is byte-identical.
			b.Input = append(json.RawMessage(nil), raw...)
		}
	case "tool_result":
		b.ToolUseID = str("tool_use_id")
		if raw, ok := fields["content"]; ok {
			trimmed := bytes.TrimSpace(raw)
			switch {
			case len(trimmed) > 0 && trimmed[0] == '"':
				text := str("content")
				b.ContentText = &text
			case len(trimmed) > 0 && trimmed[0] == '[':
				b.Content = []claudeBlock{}
				if err := json.Unmarshal(trimmed, &b.Content); err != nil {
					return err
				}
			}
		}
		if raw, ok := fields["is_error"]; ok {
			var isError bool
			if json.Unmarshal(raw, &isError) == nil {
				b.IsError = &isError
			}
		}
	case "thinking":
		b.Thinking = str("thinking")
		b.Signature = str("signature")
	case "redacted_thinking":
		b.Data = str("data")
	case "tool_addition", "tool_removal":
		if raw, ok := fields["tool"]; ok {
			b.Tool = &claudeToolReference{}
			if err := json.Unmarshal(raw, b.Tool); err != nil {
				return err
			}
		}
	default:
		delete(fields, "cache_control")
		raw, err := json.Marshal(fields)
		if err != nil {
			return err
		}
		*b = opaqueClaudeBlock(raw)
		return nil
	}
	// cache_control is request-only: it is re-applied on every request, never stored.
	return nil
}

// field reads one top-level field of an opaque block.
func (b claudeBlock) field(key string) (json.RawMessage, bool) {
	if b.raw == nil {
		return nil, false
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(b.raw, &fields) != nil {
		return nil, false
	}
	value, ok := fields[key]
	return value, ok
}

// cloneBlocks deep-copies blocks so request-only edits (cache_control) never reach the caller's conversation.
func cloneBlocks(blocks []claudeBlock) []claudeBlock {
	if blocks == nil {
		return nil
	}
	out := make([]claudeBlock, len(blocks))
	for i, block := range blocks {
		if block.Content != nil {
			block.Content = cloneBlocks(block.Content)
		}
		out[i] = block
	}
	return out
}

// cloneMessages deep-copies messages and their blocks.
func cloneMessages(messages []claudeMessage) []claudeMessage {
	if messages == nil {
		return nil
	}
	out := make([]claudeMessage, len(messages))
	for i, message := range messages {
		out[i] = claudeMessage{Role: message.Role, Content: cloneBlocks(message.Content)}
	}
	return out
}
