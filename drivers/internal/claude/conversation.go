package claude

import "github.com/vertesia/llumiverse-go/core"

func finalizeClaudeConversation(conversation claudePrompt, options ExecutionOptions) claudePrompt {
	// Increment first so "strip after N turns" evaluates against the stored
	// conversation after this model response has been added.
	conversation.Turn++
	stripMedia := core.ShouldStripConversationMedia(conversation.Turn, options.StripImagesAfterTurns)
	stripHeartbeats := core.ShouldStripConversationHeartbeats(conversation.Turn, options.StripHeartbeatsAfterTurns)
	conversation.System = processClaudeBlocksForConversation(conversation.System, options, stripMedia, stripHeartbeats)
	for i := range conversation.Messages {
		conversation.Messages[i].Content = processClaudeBlocksForConversation(conversation.Messages[i].Content, options, stripMedia, stripHeartbeats)
	}
	return conversation
}

func processClaudeBlocksForConversation(blocks []claudeBlock, options ExecutionOptions, stripMedia bool, stripHeartbeats bool) []claudeBlock {
	out := make([]claudeBlock, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case "image":
			if stripMedia {
				// Replace the entire media block; keeping an empty image/document
				// wrapper would make the next Claude API request invalid.
				out = append(out, claudeBlock{Type: "text", Text: core.ConversationImagePlaceholder})
				continue
			}
		case "document":
			if stripMedia {
				// Document source payloads can be large text or base64 blobs, so
				// retention removes the whole block rather than only its data field.
				out = append(out, claudeBlock{Type: "text", Text: core.ConversationDocumentPlaceholder})
				continue
			}
		}
		if block.Text != "" {
			block.Text = core.ProcessConversationText(block.Text, options, stripHeartbeats)
		}
		if len(block.Content) > 0 {
			block.Content = processClaudeBlocksForConversation(block.Content, options, stripMedia, stripHeartbeats)
		}
		out = append(out, block)
	}
	return out
}
