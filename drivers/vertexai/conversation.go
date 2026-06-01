package vertexai

import "github.com/vertesia/llumiverse-go/core"

const (
	conversationImagePlaceholder     = core.ConversationImagePlaceholder
	conversationTextTruncatedMarker  = core.ConversationTextTruncatedMarker
	conversationHeartbeatPlaceholder = core.ConversationHeartbeatPlaceholder
)

func finalizeGeminiConversation(conversation geminiPrompt, options ExecutionOptions) geminiPrompt {
	// Increment first so "strip after N turns" evaluates against the stored
	// conversation after this model response has been added.
	conversation.Turn++
	stripMedia := core.ShouldStripConversationMedia(conversation.Turn, options.StripImagesAfterTurns)
	stripHeartbeats := core.ShouldStripConversationHeartbeats(conversation.Turn, options.StripHeartbeatsAfterTurns)
	if conversation.System != nil {
		conversation.System.Parts = processGeminiPartsForConversation(conversation.System.Parts, options, stripMedia, stripHeartbeats)
	}
	for i := range conversation.Contents {
		conversation.Contents[i].Parts = processGeminiPartsForConversation(conversation.Contents[i].Parts, options, stripMedia, stripHeartbeats)
	}
	return conversation
}

func processGeminiPartsForConversation(parts []geminiPart, options ExecutionOptions, stripMedia bool, stripHeartbeats bool) []geminiPart {
	out := make([]geminiPart, 0, len(parts))
	for _, part := range parts {
		if part.InlineData != nil && stripMedia {
			// Gemini inlineData blocks can contain large base64 payloads; replace
			// the entire part so a later request still has a valid text part.
			out = append(out, geminiPart{Text: conversationImagePlaceholder})
			continue
		}
		if part.Text != "" {
			part.Text = core.ProcessConversationText(part.Text, options, stripHeartbeats)
		}
		out = append(out, part)
	}
	return out
}
