package openai

import (
	"strings"

	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/vertesia/llumiverse-go/core"
)

const (
	conversationImagePlaceholder     = core.ConversationImagePlaceholder
	conversationTextTruncatedMarker  = core.ConversationTextTruncatedMarker
	conversationHeartbeatPlaceholder = core.ConversationHeartbeatPlaceholder
)

func processOpenAIResponsesItemsForConversation(items responses.ResponseInputParam, options ExecutionOptions, stripMedia bool, stripHeartbeats bool) responses.ResponseInputParam {
	out := make(responses.ResponseInputParam, 0, len(items))
	for _, item := range items {
		switch {
		case item.OfMessage != nil:
			next := *item.OfMessage
			next.Content = processOpenAIEasyMessageContentForConversation(next.Content, options, stripMedia, stripHeartbeats)
			out = append(out, responses.ResponseInputItemUnionParam{OfMessage: &next})
		case item.OfInputMessage != nil:
			next := *item.OfInputMessage
			next.Content = processOpenAIContentPartsForConversation(next.Content, options, stripMedia, stripHeartbeats)
			out = append(out, responses.ResponseInputItemUnionParam{OfInputMessage: &next})
		case item.OfFunctionCall != nil:
			next := *item.OfFunctionCall
			next.Arguments = core.ProcessConversationText(next.Arguments, options, stripHeartbeats)
			out = append(out, responses.ResponseInputItemUnionParam{OfFunctionCall: &next})
		case item.OfFunctionCallOutput != nil:
			next := *item.OfFunctionCallOutput
			next.Output = core.ProcessConversationText(next.Output, options, stripHeartbeats)
			out = append(out, responses.ResponseInputItemUnionParam{OfFunctionCallOutput: &next})
		default:
			out = append(out, item)
		}
	}
	return out
}

func processOpenAIEasyMessageContentForConversation(content responses.EasyInputMessageContentUnionParam, options ExecutionOptions, stripMedia bool, stripHeartbeats bool) responses.EasyInputMessageContentUnionParam {
	if content.OfString.Valid() {
		content.OfString = param.NewOpt(core.ProcessConversationText(content.OfString.Value, options, stripHeartbeats))
	}
	if len(content.OfInputItemContentList) > 0 {
		content.OfInputItemContentList = processOpenAIContentPartsForConversation(content.OfInputItemContentList, options, stripMedia, stripHeartbeats)
	}
	return content
}

func processOpenAIContentPartsForConversation(parts responses.ResponseInputMessageContentListParam, options ExecutionOptions, stripMedia bool, stripHeartbeats bool) responses.ResponseInputMessageContentListParam {
	out := make(responses.ResponseInputMessageContentListParam, 0, len(parts))
	for _, part := range parts {
		// Only strip inline base64 images. Remote URLs are intentionally kept
		// because the conversation does not retain large local payloads.
		if part.OfInputImage != nil && stripMedia && part.OfInputImage.ImageURL.Valid() && strings.HasPrefix(part.OfInputImage.ImageURL.Value, "data:image/") && strings.Contains(part.OfInputImage.ImageURL.Value, ";base64,") {
			// Replace the content part with text; preserving a data URL with empty
			// bytes would still be provider-shaped but not useful history.
			out = append(out, responses.ResponseInputContentParamOfInputText(conversationImagePlaceholder))
			continue
		}
		if part.OfInputText != nil {
			next := *part.OfInputText
			next.Text = core.ProcessConversationText(next.Text, options, stripHeartbeats)
			out = append(out, responses.ResponseInputContentUnionParam{OfInputText: &next})
			continue
		}
		out = append(out, part)
	}
	return out
}
