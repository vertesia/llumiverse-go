package bedrock

import (
	"encoding/base64"
	"encoding/json"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/aws/aws-sdk-go-v2/aws"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// bedrockDeepSeekR1V1 matches DeepSeek-R1 v1, which returns reasoning blocks but
// rejects them when they are replayed.
var bedrockDeepSeekR1V1 = regexp.MustCompile(`^(?:(?:us|eu|apac)\.)?deepseek\.r1-v1(?::\d+)?$`)

// bedrockExcludesReasoningReplay reports whether a model rejects its own
// reasoning blocks in later turns. Every other Converse model gets its native
// reasoning replayed, as in the TS client.
func bedrockExcludesReasoningReplay(model string) bool {
	id := strings.ToLower(model)
	if index := strings.LastIndex(id, "/"); index >= 0 {
		id = id[index+1:]
	}
	return bedrockDeepSeekR1V1.MatchString(id)
}

// bedrockHasReasoningSignature reports whether a conversation carries a signed
// reasoning block. Bedrock signatures hash every prior message, so a signed
// chain must be replayed byte for byte.
func bedrockHasReasoningSignature(messages []brtypes.Message) bool {
	for _, message := range messages {
		for _, block := range message.Content {
			reasoning, ok := block.(*brtypes.ContentBlockMemberReasoningContent)
			if !ok {
				continue
			}
			if text, ok := reasoning.Value.(*brtypes.ReasoningContentBlockMemberReasoningText); ok && aws.ToString(text.Value.Signature) != "" {
				return true
			}
		}
	}
	return false
}

// bedrockReplayContent returns the response content to store for replay: all of
// it, reasoning included, except for models that reject replayed reasoning.
func bedrockReplayContent(content []brtypes.ContentBlock, model string) []brtypes.ContentBlock {
	if !bedrockExcludesReasoningReplay(model) {
		return content
	}
	return filterBedrockReasoningBlocks(content)
}

// bedrockNativeBlock accumulates one streamed content block by its native index.
type bedrockNativeBlock struct {
	kind      string // "text", "toolUse", "reasoning" or "redacted"
	text      string
	signature string
	redacted  []byte
	toolID    string
	toolName  string
	toolInput string
}

// bedrockNativeStreamBlocks rebuilds the assistant message of a ConverseStream
// response from its events, so the stored conversation replays the provider's
// own blocks (reasoning, signatures, redacted reasoning, text and tool calls)
// rather than the normalized output. It also keeps the service tier the stream
// metadata reports. It is shared between the stream goroutine and the stream
// finalizer, which can run early on Close.
type bedrockNativeStreamBlocks struct {
	mu          sync.Mutex
	blocks      map[int32]*bedrockNativeBlock
	serviceTier string
}

// tier returns the service tier reported by the stream metadata, if any.
func (b *bedrockNativeStreamBlocks) tier() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.serviceTier
}

// collect folds one stream event into the native blocks.
func (b *bedrockNativeStreamBlocks) collect(event brtypes.ConverseStreamOutput) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.blocks == nil {
		b.blocks = map[int32]*bedrockNativeBlock{}
	}
	switch value := event.(type) {
	case *brtypes.ConverseStreamOutputMemberMetadata:
		if value.Value.ServiceTier != nil {
			b.serviceTier = string(value.Value.ServiceTier.Type)
		}
	case *brtypes.ConverseStreamOutputMemberContentBlockStart:
		if tool, ok := value.Value.Start.(*brtypes.ContentBlockStartMemberToolUse); ok {
			b.blocks[bedrockContentBlockIndex(value.Value.ContentBlockIndex)] = &bedrockNativeBlock{
				kind:     "toolUse",
				toolID:   aws.ToString(tool.Value.ToolUseId),
				toolName: aws.ToString(tool.Value.Name),
			}
		}
	case *brtypes.ConverseStreamOutputMemberContentBlockDelta:
		index := bedrockContentBlockIndex(value.Value.ContentBlockIndex)
		current := b.blocks[index]
		switch delta := value.Value.Delta.(type) {
		case *brtypes.ContentBlockDeltaMemberText:
			if current == nil || current.kind != "text" {
				current = &bedrockNativeBlock{kind: "text"}
				b.blocks[index] = current
			}
			current.text += delta.Value
		case *brtypes.ContentBlockDeltaMemberToolUse:
			if current != nil && current.kind == "toolUse" {
				current.toolInput += aws.ToString(delta.Value.Input)
			}
		case *brtypes.ContentBlockDeltaMemberReasoningContent:
			switch reasoning := delta.Value.(type) {
			case *brtypes.ReasoningContentBlockDeltaMemberRedactedContent:
				if current == nil || current.kind != "redacted" {
					current = &bedrockNativeBlock{kind: "redacted"}
					b.blocks[index] = current
				}
				current.redacted = append(current.redacted, reasoning.Value...)
			case *brtypes.ReasoningContentBlockDeltaMemberText:
				if current == nil || current.kind != "reasoning" {
					current = &bedrockNativeBlock{kind: "reasoning"}
					b.blocks[index] = current
				}
				current.text += reasoning.Value
			case *brtypes.ReasoningContentBlockDeltaMemberSignature:
				if current == nil || current.kind != "reasoning" {
					current = &bedrockNativeBlock{kind: "reasoning"}
					b.blocks[index] = current
				}
				current.signature += reasoning.Value
			}
		}
	}
}

// content returns the collected blocks in native index order. A tool call whose
// streamed input is not valid JSON was cut off: it is left out, because replaying
// it would require a tool result the caller cannot produce.
func (b *bedrockNativeStreamBlocks) content() []brtypes.ContentBlock {
	b.mu.Lock()
	defer b.mu.Unlock()
	indexes := make([]int32, 0, len(b.blocks))
	for index := range b.blocks {
		indexes = append(indexes, index)
	}
	sort.Slice(indexes, func(i, j int) bool { return indexes[i] < indexes[j] })
	out := make([]brtypes.ContentBlock, 0, len(indexes))
	for _, index := range indexes {
		block := b.blocks[index]
		switch block.kind {
		case "text":
			out = append(out, &brtypes.ContentBlockMemberText{Value: block.text})
		case "reasoning":
			text := brtypes.ReasoningTextBlock{Text: aws.String(block.text)}
			if block.signature != "" {
				text.Signature = aws.String(block.signature)
			}
			out = append(out, &brtypes.ContentBlockMemberReasoningContent{Value: &brtypes.ReasoningContentBlockMemberReasoningText{Value: text}})
		case "redacted":
			out = append(out, &brtypes.ContentBlockMemberReasoningContent{Value: &brtypes.ReasoningContentBlockMemberRedactedContent{Value: block.redacted}})
		case "toolUse":
			input := strings.TrimSpace(block.toolInput)
			if input == "" {
				input = "{}"
			}
			var parsed any
			if err := json.Unmarshal([]byte(input), &parsed); err != nil {
				continue
			}
			out = append(out, &brtypes.ContentBlockMemberToolUse{Value: brtypes.ToolUseBlock{
				ToolUseId: aws.String(block.toolID),
				Name:      aws.String(block.toolName),
				Input:     brdoc.NewLazyDocument(parsed),
			}})
		}
	}
	return out
}

// bedrockStoredReasoningBlock rebuilds a reasoning block from JSON-restored
// conversation state: the provider shape ({reasoningText: {text, signature}} or
// {redactedContent}) or encoding/json's shape of the SDK union ({Value: {Text,
// Signature}} or {Value: "<base64>"}).
func bedrockStoredReasoningBlock(raw any) (brtypes.ContentBlock, bool) {
	value, ok := raw.(map[string]any)
	if !ok {
		return nil, false
	}
	if rawText, ok := bedrockMapLookup(value, "reasoningText", "ReasoningText"); ok {
		if text, ok := rawText.(map[string]any); ok {
			return bedrockReasoningTextBlock(text), true
		}
	}
	if rawRedacted, ok := bedrockMapLookup(value, "redactedContent", "RedactedContent"); ok {
		if data, ok := bedrockStoredBytes(rawRedacted); ok {
			return &brtypes.ContentBlockMemberReasoningContent{Value: &brtypes.ReasoningContentBlockMemberRedactedContent{Value: data}}, true
		}
	}
	if len(value) != 1 {
		return nil, false
	}
	switch inner := value["Value"].(type) {
	case map[string]any:
		for key := range inner {
			if key != "Text" && key != "Signature" {
				return nil, false
			}
		}
		if _, ok := inner["Text"]; !ok {
			return nil, false
		}
		return bedrockReasoningTextBlock(inner), true
	case string:
		if data, ok := bedrockStoredBytes(inner); ok {
			return &brtypes.ContentBlockMemberReasoningContent{Value: &brtypes.ReasoningContentBlockMemberRedactedContent{Value: data}}, true
		}
	}
	return nil, false
}

// bedrockReasoningTextBlock builds a reasoning text block from a decoded
// {text, signature} map in either key case.
func bedrockReasoningTextBlock(value map[string]any) brtypes.ContentBlock {
	text := brtypes.ReasoningTextBlock{Text: aws.String(bedrockStringFromMap(value, "text", "Text"))}
	if signature := bedrockStringFromMap(value, "signature", "Signature"); signature != "" {
		text.Signature = aws.String(signature)
	}
	return &brtypes.ContentBlockMemberReasoningContent{Value: &brtypes.ReasoningContentBlockMemberReasoningText{Value: text}}
}

// bedrockStoredBytes decodes stored binary data: a base64 string (encoding/json's
// []byte encoding) or the TS client's {_base64: "..."} storage shape.
func bedrockStoredBytes(raw any) ([]byte, bool) {
	if wrapped, ok := raw.(map[string]any); ok {
		raw = wrapped["_base64"]
	}
	text, ok := raw.(string)
	if !ok {
		return nil, false
	}
	data, err := base64.StdEncoding.DecodeString(text)
	if err != nil {
		return nil, false
	}
	return data, true
}
