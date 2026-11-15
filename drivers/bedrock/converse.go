package bedrock

import (
	"github.com/aws/aws-sdk-go-v2/aws"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// fixBedrockOrphanedToolResults drops toolResult blocks whose toolUseId has no
// matching toolUse in the immediately preceding assistant message, and drops a
// user message left with no content. It mirrors fixBedrockOrphanedToolUse: that
// one answers an unanswered toolUse (an interrupted run), this one removes a
// result whose toolUse is gone (for example after the caller trimmed history).
// Converse rejects either orphan, as the TS client's fixOrphanedToolResults notes.
func fixBedrockOrphanedToolResults(messages []brtypes.Message) []brtypes.Message {
	if len(messages) == 0 {
		return messages
	}
	out := make([]brtypes.Message, 0, len(messages))
	for i, message := range messages {
		if message.Role != brtypes.ConversationRoleUser || !bedrockHasToolResult(message.Content) {
			out = append(out, message)
			continue
		}
		allowed := map[string]bool{}
		if i > 0 && messages[i-1].Role == brtypes.ConversationRoleAssistant {
			for _, block := range messages[i-1].Content {
				if tool, ok := block.(*brtypes.ContentBlockMemberToolUse); ok && aws.ToString(tool.Value.ToolUseId) != "" {
					allowed[aws.ToString(tool.Value.ToolUseId)] = true
				}
			}
		}
		content := make([]brtypes.ContentBlock, 0, len(message.Content))
		for _, block := range message.Content {
			if result, ok := block.(*brtypes.ContentBlockMemberToolResult); ok && !allowed[aws.ToString(result.Value.ToolUseId)] {
				continue
			}
			content = append(content, block)
		}
		if len(content) == 0 {
			continue
		}
		message.Content = content
		out = append(out, message)
	}
	return out
}

// bedrockHasToolResult reports whether content carries a toolResult block.
func bedrockHasToolResult(content []brtypes.ContentBlock) bool {
	for _, block := range content {
		if _, ok := block.(*brtypes.ContentBlockMemberToolResult); ok {
			return true
		}
	}
	return false
}
