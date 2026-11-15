package core

import (
	"strings"
	"unicode/utf8"
)

const (
	ConversationImagePlaceholder     = "[Image removed from conversation history]"
	ConversationDocumentPlaceholder  = "[Document removed from conversation history]"
	ConversationVideoPlaceholder     = "[Video removed from conversation history]"
	ConversationTextTruncatedMarker  = "\n\n[Content truncated - exceeded token limit]"
	ConversationHeartbeatOpenTag     = "<heartbeat>"
	ConversationHeartbeatCloseTag    = "</heartbeat>"
	ConversationHeartbeatPlaceholder = "[Heartbeat removed from conversation history]"
	ConversationCharsPerToken        = 4
)

// ShouldStripConversationMedia implements the TS client's media retention rule.
func ShouldStripConversationMedia(turn int, keepForTurns *int) bool {
	if keepForTurns == nil {
		return false
	}
	if *keepForTurns > 0 && turn < *keepForTurns {
		return false
	}
	return true
}

// ShouldStripConversationHeartbeats implements the TS client's heartbeat retention rule.
func ShouldStripConversationHeartbeats(turn int, keepForTurns *int) bool {
	keep := 1
	if keepForTurns != nil {
		keep = *keepForTurns
	}
	if keep > 0 && turn < keep {
		return false
	}
	return true
}

// ProcessConversationText removes expired heartbeat blocks and applies coarse text truncation.
func ProcessConversationText(text string, options ExecutionOptions, stripHeartbeats bool) string {
	if stripHeartbeats && strings.HasPrefix(text, ConversationHeartbeatOpenTag) && strings.HasSuffix(text, ConversationHeartbeatCloseTag) {
		return ConversationHeartbeatPlaceholder
	}
	return TruncateConversationText(text, options.StripTextMaxTokens)
}

// TruncateConversationText cuts text to about maxTokens tokens (ConversationCharsPerToken
// characters per token) and appends ConversationTextTruncatedMarker. A non-positive maxTokens
// leaves text unchanged. The cut never splits a UTF-8 sequence, so the result stays valid UTF-8.
func TruncateConversationText(text string, maxTokens int) string {
	if maxTokens <= 0 {
		return text
	}
	maxChars := maxTokens * ConversationCharsPerToken
	if len(text) <= maxChars {
		return text
	}
	return text[:runeBoundary(text, maxChars)] + ConversationTextTruncatedMarker
}

// runeBoundary returns the largest index <= n that starts a rune in text, so text[:index] never
// ends in a partial UTF-8 sequence. A split sequence encodes as U+FFFD, or is rejected outright by
// strict JSON parsers (the TS client hit this with a split surrogate pair). n must be < len(text).
func runeBoundary(text string, n int) int {
	for n > 0 && !utf8.RuneStart(text[n]) {
		n--
	}
	return n
}

// ConversationTruncationCutoff returns the index of the first message that keeps its full text
// when a driver truncates large text with turn awareness (the TS client's keepRecentMessages):
// messages before the cutoff are truncated, and the last keepRecent messages (the active working
// set: the file the agent just read, the latest diagnostics) stay intact. A non-positive
// keepRecent returns messageCount, so every message is truncated; a conversation no longer than
// keepRecent returns 0, so nothing is.
func ConversationTruncationCutoff(messageCount int, keepRecent int) int {
	if keepRecent <= 0 {
		return messageCount
	}
	if messageCount <= keepRecent {
		return 0
	}
	return messageCount - keepRecent
}
