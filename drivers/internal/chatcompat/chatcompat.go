// Package chatcompat holds the request and listing helpers shared by the drivers built on the OpenAI Chat
// Completions protocol (Groq, Together AI, OpenRouter): walking the request messages, refusing input their
// endpoints do not take, and reading model versions and families from model IDs.
package chatcompat

import (
	"errors"
	"regexp"
	"strconv"
	"strings"
)

// ErrAudioInput is returned for a request with audio parts to an endpoint that takes no audio input.
var ErrAudioInput = errors.New("this inference endpoint does not support audio input")

// Messages returns the request messages of a Chat Completions payload.
func Messages(payload map[string]any) []map[string]any {
	raw, _ := payload["messages"].([]any)
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		if message, ok := item.(map[string]any); ok {
			out = append(out, message)
		}
	}
	return out
}

// Parts returns the content parts of a message, or nil when its content is a string or empty.
func Parts(message map[string]any) []map[string]any {
	raw, _ := message["content"].([]any)
	if raw == nil {
		return nil
	}
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		if part, ok := item.(map[string]any); ok {
			out = append(out, part)
		}
	}
	return out
}

// HasAudio reports whether any message of the payload carries an input_audio part.
func HasAudio(payload map[string]any) bool {
	for _, message := range Messages(payload) {
		for _, part := range Parts(message) {
			if part["type"] == "input_audio" {
				return true
			}
		}
	}
	return false
}

// TextContent returns the text of a message's content: the string itself, or its text parts joined by newlines.
func TextContent(content any) string {
	switch value := content.(type) {
	case string:
		return value
	case []any:
		var texts []string
		for _, item := range value {
			if part, ok := item.(map[string]any); ok && part["type"] == "text" {
				if text, ok := part["text"].(string); ok {
					texts = append(texts, text)
				}
			}
		}
		return strings.Join(texts, "\n")
	}
	return ""
}

// HasTools reports whether the payload declares tools.
func HasTools(payload map[string]any) bool {
	tools, _ := payload["tools"].([]any)
	return len(tools) > 0
}

// BaseModelID returns the last path segment of a model ID, lower-cased: "qwen/qwen3.8-flash" gives
// "qwen3.8-flash".
func BaseModelID(model string) string {
	model = strings.ToLower(model)
	if index := strings.LastIndex(model, "/"); index >= 0 {
		return model[index+1:]
	}
	return model
}

var qwenVersionPattern = regexp.MustCompile(`qwen(\d+)(?:\.(\d+))?`)

// IsQwenVersionGTE reports whether a Qwen model ID is at least major.minor. Qwen writes minor versions after a
// dot (qwen3.5); a dash starts the size (qwen3-235b), not a minor version.
func IsQwenVersionGTE(model string, major, minor int) bool {
	match := qwenVersionPattern.FindStringSubmatch(strings.ToLower(model))
	if match == nil {
		return false
	}
	gotMajor, _ := strconv.Atoi(match[1])
	gotMinor := 0
	if match[2] != "" {
		gotMinor, _ = strconv.Atoi(match[2])
	}
	return gotMajor > major || (gotMajor == major && gotMinor >= minor)
}
