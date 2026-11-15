package vertexai

import "github.com/vertesia/llumiverse-go/core"

const (
	conversationImagePlaceholder     = core.ConversationImagePlaceholder
	conversationTextTruncatedMarker  = core.ConversationTextTruncatedMarker
	conversationHeartbeatPlaceholder = core.ConversationHeartbeatPlaceholder
	conversationAudioPlaceholder     = core.ConversationAudioPlaceholder
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
		if part.ThoughtSignature != "" {
			// A signed part must reach Gemini byte for byte, or the model rejects
			// the reasoning chain it belongs to.
			out = append(out, part)
			continue
		}
		if part.InlineData != nil && core.IsAudioMIMEType(part.InlineData.MIMEType) {
			// Inline audio never stays in history, whatever the retention options;
			// Cloud Storage references (fileData) are kept.
			out = append(out, geminiPart{Text: conversationAudioPlaceholder})
			continue
		}
		if part.InlineData != nil && stripMedia {
			// Gemini inlineData blocks can contain large base64 payloads; replace
			// the entire part so a later request still has a valid text part.
			out = append(out, geminiPart{Text: conversationImagePlaceholder})
			continue
		}
		if part.FunctionResponse != nil && len(part.FunctionResponse.Parts) > 0 && (stripMedia || geminiMediaHasInlineAudio(part.FunctionResponse.Parts)) {
			// A function response part cannot hold text, so expired inline
			// attachments, and inline audio always, are dropped; Cloud Storage
			// references are kept.
			response := *part.FunctionResponse
			response.Parts = nil
			for _, media := range part.FunctionResponse.Parts {
				if media.InlineData == nil || (!stripMedia && !core.IsAudioMIMEType(media.InlineData.MIMEType)) {
					response.Parts = append(response.Parts, media)
				}
			}
			part.FunctionResponse = &response
		}
		if part.Text != "" {
			part.Text = core.ProcessConversationText(part.Text, options, stripHeartbeats)
		}
		out = append(out, part)
	}
	return out
}

// geminiMediaHasInlineAudio reports a function response carrying inline audio.
func geminiMediaHasInlineAudio(parts []geminiFunctionMediaPart) bool {
	for _, media := range parts {
		if media.InlineData != nil && core.IsAudioMIMEType(media.InlineData.MIMEType) {
			return true
		}
	}
	return false
}
