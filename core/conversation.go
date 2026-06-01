package core

import "strings"

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
	if options.StripTextMaxTokens > 0 {
		maxChars := options.StripTextMaxTokens * ConversationCharsPerToken
		if len(text) > maxChars {
			return text[:maxChars] + ConversationTextTruncatedMarker
		}
	}
	return text
}
