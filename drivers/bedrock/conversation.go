package bedrock

import (
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/vertesia/llumiverse-go/core"
)

const (
	conversationImagePlaceholder     = core.ConversationImagePlaceholder
	conversationDocumentPlaceholder  = core.ConversationDocumentPlaceholder
	conversationVideoPlaceholder     = core.ConversationVideoPlaceholder
	conversationTextTruncatedMarker  = core.ConversationTextTruncatedMarker
	conversationHeartbeatPlaceholder = core.ConversationHeartbeatPlaceholder
)

func finalizeBedrockConversation(conversation bedrockPrompt, options ExecutionOptions) bedrockPrompt {
	// Increment first so "strip after N turns" evaluates against the stored
	// conversation after this model response has been added.
	conversation.Turn++
	stripMedia := core.ShouldStripConversationMedia(conversation.Turn, options.StripImagesAfterTurns)
	stripHeartbeats := core.ShouldStripConversationHeartbeats(conversation.Turn, options.StripHeartbeatsAfterTurns)
	conversation.System = processBedrockSystemForConversation(conversation.System, options, stripHeartbeats)
	for i := range conversation.Messages {
		conversation.Messages[i].Content = processBedrockContentForConversation(conversation.Messages[i].Content, options, stripMedia, stripHeartbeats)
	}
	return conversation
}

func processBedrockSystemForConversation(blocks []brtypes.SystemContentBlock, options ExecutionOptions, stripHeartbeats bool) []brtypes.SystemContentBlock {
	out := make([]brtypes.SystemContentBlock, 0, len(blocks))
	for _, block := range blocks {
		if text, ok := block.(*brtypes.SystemContentBlockMemberText); ok {
			next := *text
			next.Value = core.ProcessConversationText(next.Value, options, stripHeartbeats)
			out = append(out, &next)
			continue
		}
		out = append(out, block)
	}
	return out
}

func processBedrockContentForConversation(blocks []brtypes.ContentBlock, options ExecutionOptions, stripMedia bool, stripHeartbeats bool) []brtypes.ContentBlock {
	out := make([]brtypes.ContentBlock, 0, len(blocks))
	for _, block := range blocks {
		switch value := block.(type) {
		case *brtypes.ContentBlockMemberImage:
			if stripMedia {
				// Replace the whole media block, not just bytes, so Bedrock sees a
				// valid text content block on the next Converse request.
				out = append(out, &brtypes.ContentBlockMemberText{Value: conversationImagePlaceholder})
				continue
			}
		case *brtypes.ContentBlockMemberDocument:
			if stripMedia {
				// Bedrock document blocks require a source. After retention expires,
				// a text placeholder is safer than an empty document source.
				out = append(out, &brtypes.ContentBlockMemberText{Value: conversationDocumentPlaceholder})
				continue
			}
		case *brtypes.ContentBlockMemberVideo:
			if stripMedia {
				// Video may be S3-backed or inline bytes; both are replaced with a
				// simple text block once the retention window closes.
				out = append(out, &brtypes.ContentBlockMemberText{Value: conversationVideoPlaceholder})
				continue
			}
		case *brtypes.ContentBlockMemberText:
			next := *value
			next.Value = core.ProcessConversationText(next.Value, options, stripHeartbeats)
			out = append(out, &next)
			continue
		case *brtypes.ContentBlockMemberToolResult:
			next := *value
			next.Value.Content = processBedrockToolResultForConversation(next.Value.Content, options, stripMedia, stripHeartbeats)
			out = append(out, &next)
			continue
		}
		out = append(out, block)
	}
	return out
}

func processBedrockToolResultForConversation(blocks []brtypes.ToolResultContentBlock, options ExecutionOptions, stripMedia bool, stripHeartbeats bool) []brtypes.ToolResultContentBlock {
	out := make([]brtypes.ToolResultContentBlock, 0, len(blocks))
	for _, block := range blocks {
		switch value := block.(type) {
		case *brtypes.ToolResultContentBlockMemberImage:
			if stripMedia {
				// Tool-result media follows the same whole-block replacement rule as
				// normal message media so replayed conversations stay valid.
				out = append(out, &brtypes.ToolResultContentBlockMemberText{Value: conversationImagePlaceholder})
				continue
			}
		case *brtypes.ToolResultContentBlockMemberDocument:
			if stripMedia {
				out = append(out, &brtypes.ToolResultContentBlockMemberText{Value: conversationDocumentPlaceholder})
				continue
			}
		case *brtypes.ToolResultContentBlockMemberVideo:
			if stripMedia {
				out = append(out, &brtypes.ToolResultContentBlockMemberText{Value: conversationVideoPlaceholder})
				continue
			}
		case *brtypes.ToolResultContentBlockMemberText:
			next := *value
			next.Value = core.ProcessConversationText(next.Value, options, stripHeartbeats)
			out = append(out, &next)
			continue
		}
		out = append(out, block)
	}
	return out
}
