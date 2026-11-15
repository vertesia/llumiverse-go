package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

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

// sanitizeBedrockMessages removes what Converse rejects outright: blank text
// blocks and messages left with no content. A response the model ends without
// output (a content_filtered stop, say) would otherwise be stored as an empty
// assistant turn and fail every later request. Dropping a message can leave two
// same-role messages adjacent, so they are merged afterwards.
func sanitizeBedrockMessages(messages []brtypes.Message) []brtypes.Message {
	kept := make([]brtypes.Message, 0, len(messages))
	for _, message := range messages {
		content := make([]brtypes.ContentBlock, 0, len(message.Content))
		for _, block := range message.Content {
			if text, ok := block.(*brtypes.ContentBlockMemberText); ok && strings.TrimSpace(text.Value) == "" {
				continue
			}
			content = append(content, block)
		}
		if len(content) == 0 {
			continue
		}
		if len(content) != len(message.Content) {
			message.Content = content
		}
		kept = append(kept, message)
	}
	return mergeBedrockMessages(kept)
}

// bedrockSupportsConverseOutputConfig reports whether a Converse model family
// accepts native structured output (outputConfig.textFormat json_schema). The
// families listed reject it and keep the schema in the prompt; a new family
// inherits native structured output, as in the TS client.
func bedrockSupportsConverseOutputConfig(model string) bool {
	normalized := strings.ToLower(model)
	for _, family := range []string{
		"claude", "ai21.", "amazon.nova", "cohere.", "meta.llama", "deepseek.r1", "writer.palmyra-x",
		// Older Mistral generations; newer ones support outputConfig.
		"mistral.mistral-7b-", "mistral.mixtral-", "mistral.mistral-large-2402", "mistral.mistral-small-2402", "mistral.pixtral-large-2502",
		"google.gemma-3-4b",
	} {
		if strings.Contains(normalized, family) {
			return false
		}
	}
	return true
}

// bedrockIncludeSchemaInPrompt reports whether the result schema is written into
// the Converse prompt: for families without native structured output, and for
// MiniMax M2.5, which also needs the prompt instruction to answer parseable JSON.
func bedrockIncludeSchemaInPrompt(model string) bool {
	return !bedrockSupportsConverseOutputConfig(model) || strings.Contains(strings.ToLower(model), "minimax.minimax-m2.5")
}

// bedrockOutputConfig returns the native structured-output config for a result
// schema, or nil when the model family does not accept one.
func bedrockOutputConfig(options ExecutionOptions) *brtypes.OutputConfig {
	if options.ResultSchema == nil || !bedrockSupportsConverseOutputConfig(options.Model) {
		return nil
	}
	schema, err := json.Marshal(options.ResultSchema)
	if err != nil {
		return nil
	}
	return &brtypes.OutputConfig{TextFormat: &brtypes.OutputFormat{
		Type: brtypes.OutputFormatTypeJsonSchema,
		Structure: &brtypes.OutputFormatStructureMemberJsonSchema{Value: brtypes.JsonSchemaDefinition{
			Name:   aws.String("output"),
			Schema: aws.String(string(schema)),
		}},
	}}
}

// relocateBedrockToolImages moves images out of tool results into ordinary
// user content for models that only accept images there. AWS documents nested
// tool-result images only for Claude and Nova; other vision models (GPT, Pixtral,
// ...) would not see them. Each image follows a "Image from tool result <id>:"
// label after all the tool results, so parallel tool calls stay together, and an
// image-only result keeps a text block so its tool pairing stays valid. The
// messages passed in are not modified: the stored conversation keeps its shape.
func relocateBedrockToolImages(messages []brtypes.Message, model string) []brtypes.Message {
	if strings.Contains(model, "anthropic.claude") || strings.Contains(model, "amazon.nova") {
		return messages
	}
	out := make([]brtypes.Message, 0, len(messages))
	for _, message := range messages {
		if message.Role != brtypes.ConversationRoleUser {
			out = append(out, message)
			continue
		}
		var attachments []brtypes.ContentBlock
		content := make([]brtypes.ContentBlock, 0, len(message.Content))
		for _, block := range message.Content {
			result, ok := block.(*brtypes.ContentBlockMemberToolResult)
			if !ok || !bedrockToolResultHasImage(result.Value.Content) {
				content = append(content, block)
				continue
			}
			remaining := make([]brtypes.ToolResultContentBlock, 0, len(result.Value.Content))
			for _, part := range result.Value.Content {
				if image, ok := part.(*brtypes.ToolResultContentBlockMemberImage); ok {
					attachments = append(attachments,
						&brtypes.ContentBlockMemberText{Value: "Image from tool result " + aws.ToString(result.Value.ToolUseId) + ":"},
						&brtypes.ContentBlockMemberImage{Value: image.Value},
					)
					continue
				}
				remaining = append(remaining, part)
			}
			if len(remaining) == 0 {
				remaining = append(remaining, &brtypes.ToolResultContentBlockMemberText{Value: "See the attached tool-result image(s)."})
			}
			relocated := result.Value
			relocated.Content = remaining
			content = append(content, &brtypes.ContentBlockMemberToolResult{Value: relocated})
		}
		if len(attachments) > 0 {
			message.Content = append(content, attachments...)
		}
		out = append(out, message)
	}
	return out
}

// bedrockToolResultHasImage reports whether tool-result content carries an image.
func bedrockToolResultHasImage(content []brtypes.ToolResultContentBlock) bool {
	for _, part := range content {
		if _, ok := part.(*brtypes.ToolResultContentBlockMemberImage); ok {
			return true
		}
	}
	return false
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

// bedrockMaxInlineAudioBytes bounds audio sent inline as Converse bytes.
const bedrockMaxInlineAudioBytes = 25_000_000

// bedrockAudioFormats maps audio MIME types to Converse audio formats; unknown
// types fall back to mp3, as in the TS client.
var bedrockAudioFormats = map[string]brtypes.AudioFormat{
	"audio/aac":        brtypes.AudioFormatAac,
	"audio/flac":       brtypes.AudioFormatFlac,
	"audio/mp4":        brtypes.AudioFormatMp4,
	"audio/mpeg":       brtypes.AudioFormatMp3,
	"audio/ogg":        brtypes.AudioFormatOgg,
	"audio/opus":       brtypes.AudioFormatOpus,
	"audio/wav":        brtypes.AudioFormatWav,
	"audio/webm":       brtypes.AudioFormatWebm,
	"audio/x-aac":      brtypes.AudioFormatXAac,
	"audio/x-m4a":      brtypes.AudioFormatM4a,
	"audio/x-matroska": brtypes.AudioFormatMka,
}

// bedrockAudioFormat returns the Converse audio format for a MIME type.
func bedrockAudioFormat(mimeType string) brtypes.AudioFormat {
	if format, ok := bedrockAudioFormats[strings.ToLower(mimeType)]; ok {
		return format
	}
	return brtypes.AudioFormatMp3
}

// bedrockAudioBlock maps an audio DataSource to a Converse audio block. An S3
// location is passed by reference; anything else is read inline, bounded to
// bedrockMaxInlineAudioBytes, and an empty file is rejected. Audio is binary,
// so it is never read as text.
func bedrockAudioBlock(ctx context.Context, file DataSource) (brtypes.AudioBlock, error) {
	format := bedrockAudioFormat(file.MIMEType())
	if uri, err := file.URL(ctx); err == nil {
		if s3URI := bedrockS3URI(uri); s3URI != "" {
			return brtypes.AudioBlock{Format: format, Source: &brtypes.AudioSourceMemberS3Location{Value: brtypes.S3Location{Uri: aws.String(s3URI)}}}, nil
		}
	}
	rc, err := file.Open(ctx)
	if err != nil {
		return brtypes.AudioBlock{}, err
	}
	defer func() { _ = rc.Close() }()
	data, err := io.ReadAll(io.LimitReader(rc, bedrockMaxInlineAudioBytes+1))
	if err != nil {
		return brtypes.AudioBlock{}, err
	}
	if len(data) > bedrockMaxInlineAudioBytes {
		return brtypes.AudioBlock{}, fmt.Errorf("audio exceeds the %d byte limit", bedrockMaxInlineAudioBytes)
	}
	if len(data) == 0 {
		return brtypes.AudioBlock{}, errors.New("audio file is empty")
	}
	return brtypes.AudioBlock{Format: format, Source: &brtypes.AudioSourceMemberBytes{Value: data}}, nil
}
