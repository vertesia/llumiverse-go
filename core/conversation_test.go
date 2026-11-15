package core

import (
	"strings"
	"testing"
)

// conversationTestMessage stands in for a provider message: drivers apply the cutoff to their own
// typed message slices.
type conversationTestMessage struct {
	text string
}

func makeConversationTestMessages(n int) []conversationTestMessage {
	messages := make([]conversationTestMessage, n)
	for i := range messages {
		messages[i].text = strings.Repeat(string(rune('a'+i%26)), 50000)
	}
	return messages
}

func truncateOlderConversationTestMessages(messages []conversationTestMessage, maxTokens int, keepRecent int) {
	cutoff := ConversationTruncationCutoff(len(messages), keepRecent)
	for i := range messages[:cutoff] {
		messages[i].text = TruncateConversationText(messages[i].text, maxTokens)
	}
}

// Mirrors the TS client's "keepRecentMessages (turn-aware truncation)" tests.
func TestConversationTruncationCutoffKeepsRecentMessages(t *testing.T) {
	t.Parallel()

	t.Run("keeps the last N messages full and truncates older ones", func(t *testing.T) {
		messages := makeConversationTestMessages(10)
		truncateOlderConversationTestMessages(messages, 1000, 3)
		for i, message := range messages {
			truncated := strings.Contains(message.text, ConversationTextTruncatedMarker)
			if i < 7 && (!truncated || len(message.text) >= 50000) {
				t.Fatalf("message %d was not truncated (len %d)", i, len(message.text))
			}
			if i >= 7 && (truncated || len(message.text) != 50000) {
				t.Fatalf("recent message %d was truncated (len %d)", i, len(message.text))
			}
		}
	})

	t.Run("does not truncate a conversation shorter than keepRecent", func(t *testing.T) {
		messages := makeConversationTestMessages(3)
		truncateOlderConversationTestMessages(messages, 1000, 5)
		for i, message := range messages {
			if len(message.text) != 50000 {
				t.Fatalf("message %d truncated (len %d)", i, len(message.text))
			}
		}
	})

	t.Run("truncates every message when keepRecent is not set", func(t *testing.T) {
		messages := makeConversationTestMessages(6)
		truncateOlderConversationTestMessages(messages, 1000, 0)
		for i, message := range messages {
			if !strings.Contains(message.text, ConversationTextTruncatedMarker) {
				t.Fatalf("message %d was not truncated", i)
			}
		}
	})

	if got := ConversationTruncationCutoff(4, 4); got != 0 {
		t.Fatalf("cutoff at keepRecent = %d, want 0", got)
	}
	if got := TruncateConversationText("abc", 0); got != "abc" {
		t.Fatalf("zero cap = %q", got)
	}
}
