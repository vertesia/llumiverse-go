package claude

import (
	"encoding/json"
	"fmt"
	"sort"
)

// claudeStreamBlock is a content block being streamed.
type claudeStreamBlock struct {
	block     claudeBlock
	fields    map[string]json.RawMessage // opaque blocks: their fields, patched by compaction deltas
	inputJSON string                     // tool_use: the streamed input
	toolInput bool                       // tool_use: input deltas arrived
}

// claudeStreamAccumulator turns Messages API stream events into completion chunks and rebuilds the final message,
// as the SDK's message accumulator does, so a streamed response is stored in the conversation exactly as an
// executed one: thinking with its signature, tool calls with their input, and compaction blocks with the summary
// their compaction_delta carries.
type claudeStreamAccumulator struct {
	options    ExecutionOptions
	message    claudeResponse
	started    bool
	stopped    bool
	startUsage *claudeUsage
	blocks     map[int]*claudeStreamBlock
}

// newClaudeStreamAccumulator returns an accumulator for a request.
func newClaudeStreamAccumulator(options ExecutionOptions) *claudeStreamAccumulator {
	return &claudeStreamAccumulator{options: options, blocks: map[int]*claudeStreamBlock{}}
}

// claudeStreamEvent is the union of the stream event fields read here.
type claudeStreamEvent struct {
	Type         string          `json:"type"`
	Index        int             `json:"index"`
	Message      *claudeResponse `json:"message"`
	ContentBlock json.RawMessage `json:"content_block"`
	Delta        struct {
		Type             string          `json:"type"`
		Text             string          `json:"text"`
		PartialJSON      string          `json:"partial_json"`
		Thinking         string          `json:"thinking"`
		Signature        string          `json:"signature"`
		Content          json.RawMessage `json:"content"`
		EncryptedContent json.RawMessage `json:"encrypted_content"`
		StopReason       string          `json:"stop_reason"`
		StopDetails      json.RawMessage `json:"stop_details"`
	} `json:"delta"`
	Usage *claudeStreamDeltaUsage `json:"usage"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// claudeStreamDeltaUsage is the usage of a message_delta event: cumulative counts, any of which may be absent.
type claudeStreamDeltaUsage struct {
	InputTokens              *int                   `json:"input_tokens"`
	OutputTokens             int                    `json:"output_tokens"`
	CacheReadInputTokens     *int                   `json:"cache_read_input_tokens"`
	CacheCreationInputTokens *int                   `json:"cache_creation_input_tokens"`
	Iterations               []claudeIterationUsage `json:"iterations"`
}

// claudeStreamError is an error event received in the stream.
type claudeStreamError struct {
	errorType string
	message   string
}

// Error returns the provider's message with its type.
func (e *claudeStreamError) Error() string {
	return fmt.Sprintf("%s: %s", e.errorType, e.message)
}

// retryable reports whether the stream error is worth retrying: overloaded and API errors are, invalid requests
// are not, anything else is unknown.
func (e *claudeStreamError) retryable() *bool {
	switch e.errorType {
	case "overloaded_error", "api_error":
		return boolPtr(true)
	case "invalid_request_error", "authentication_error", "permission_error", "not_found_error":
		return boolPtr(false)
	}
	return nil
}

// event processes one event. It returns the chunk to emit (nil for none) or the stream's error event.
func (a *claudeStreamAccumulator) event(data []byte) (*CompletionChunk, error) {
	var event claudeStreamEvent
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, err
	}
	includeThoughts := optionBool(a.options.ModelOptions, "include_thoughts")
	switch event.Type {
	case "message_start":
		if event.Message == nil {
			return nil, nil
		}
		a.started = true
		a.message = *event.Message
		a.message.Content = nil
		usage := event.Message.Usage
		a.startUsage = &usage
		return &CompletionChunk{TokenUsage: claudeTokenUsage(usage), ServiceTier: claudeServiceTier(usage)}, nil
	case "content_block_start":
		var block claudeBlock
		if err := json.Unmarshal(event.ContentBlock, &block); err != nil {
			return nil, err
		}
		streamed := &claudeStreamBlock{block: block}
		if block.raw != nil {
			_ = json.Unmarshal(block.raw, &streamed.fields)
		}
		a.blocks[event.Index] = streamed
		// An on-demand compaction streams its block whole, here, and answers with its summary alone.
		if a.options.CompactConversation != nil && block.Type == "compaction" {
			if results := claudeCompactionResults([]claudeBlock{block}); len(results) > 0 {
				return &CompletionChunk{Result: results}, nil
			}
			return nil, nil
		}
		if block.Type == "tool_use" {
			return &CompletionChunk{ToolUse: []ToolUseChunk{{ID: block.ID, ToolName: block.Name, ToolInput: ""}}}, nil
		}
	case "content_block_delta":
		streamed := a.blocks[event.Index]
		if streamed == nil {
			return nil, nil
		}
		switch event.Delta.Type {
		case "text_delta":
			streamed.block.Text += event.Delta.Text
			if event.Delta.Text != "" {
				return &CompletionChunk{Result: []CompletionResult{newTextResult(event.Delta.Text)}}, nil
			}
		case "input_json_delta":
			streamed.inputJSON += event.Delta.PartialJSON
			streamed.toolInput = true
			if event.Delta.PartialJSON != "" {
				return &CompletionChunk{ToolUse: []ToolUseChunk{{ID: streamed.block.ID, ToolInput: event.Delta.PartialJSON}}}, nil
			}
		case "thinking_delta":
			streamed.block.Thinking += event.Delta.Thinking
			if includeThoughts && event.Delta.Thinking != "" {
				return &CompletionChunk{Result: []CompletionResult{newThoughtsResult(event.Delta.Thinking)}}, nil
			}
		case "signature_delta":
			streamed.block.Signature += event.Delta.Signature
		case "compaction_delta":
			if streamed.fields != nil {
				if event.Delta.Content != nil {
					streamed.fields["content"] = event.Delta.Content
				}
				if event.Delta.EncryptedContent != nil {
					streamed.fields["encrypted_content"] = event.Delta.EncryptedContent
				}
			}
		}
	case "message_delta":
		a.message.StopReason = event.Delta.StopReason
		if event.Delta.StopDetails != nil {
			a.message.StopDetails = event.Delta.StopDetails
		}
		chunk := &CompletionChunk{FinishReason: claudeFinishReason(event.Delta.StopReason)}
		if event.Usage != nil {
			a.applyDeltaUsage(*event.Usage)
			chunk.TokenUsage = a.streamedUsage(*event.Usage)
		}
		return chunk, nil
	case "message_stop":
		a.stopped = true
	case "error":
		streamError := &claudeStreamError{errorType: "error"}
		if event.Error != nil {
			streamError.errorType = event.Error.Type
			streamError.message = event.Error.Message
		}
		return nil, streamError
	}
	return nil, nil
}

// applyDeltaUsage folds a message_delta's cumulative usage into the final message.
func (a *claudeStreamAccumulator) applyDeltaUsage(usage claudeStreamDeltaUsage) {
	a.message.Usage.OutputTokens = usage.OutputTokens
	if usage.InputTokens != nil {
		a.message.Usage.InputTokens = *usage.InputTokens
	}
	if usage.CacheReadInputTokens != nil {
		a.message.Usage.CacheReadInputTokens = usage.CacheReadInputTokens
	}
	if usage.CacheCreationInputTokens != nil {
		a.message.Usage.CacheCreationInputTokens = usage.CacheCreationInputTokens
	}
	if usage.Iterations != nil {
		a.message.Usage.Iterations = usage.Iterations
	}
}

// streamedUsage is the usage a message_delta reports: its output tokens, plus the request's full usage when it ran
// a compaction, whose tokens message_start could not include. Stream usage merges by taking the largest counts, so
// the full figures replace the earlier ones.
func (a *claudeStreamAccumulator) streamedUsage(delta claudeStreamDeltaUsage) *ExecutionTokenUsage {
	iterations := delta.Iterations
	if iterations == nil && a.startUsage != nil {
		iterations = a.startUsage.Iterations
	}
	if a.startUsage == nil || claudeCompactionUsage(iterations).Count == 0 {
		return &ExecutionTokenUsage{Result: tokens(delta.OutputTokens)}
	}
	usage := *a.startUsage
	usage.OutputTokens = delta.OutputTokens
	usage.Iterations = iterations
	return claudeTokenUsage(usage)
}

// complete reports whether the stream delivered a whole message.
func (a *claudeStreamAccumulator) complete() bool {
	return a.started && a.stopped
}

// final returns the rebuilt message.
func (a *claudeStreamAccumulator) final() claudeResponse {
	indexes := make([]int, 0, len(a.blocks))
	for index := range a.blocks {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	message := a.message
	message.Content = make([]claudeBlock, 0, len(indexes))
	for _, index := range indexes {
		streamed := a.blocks[index]
		block := streamed.block
		switch {
		case block.Type == "tool_use":
			if json.Valid([]byte(streamed.inputJSON)) {
				block.Input = json.RawMessage(streamed.inputJSON)
			} else if block.Input == nil {
				block.Input = map[string]any{}
			}
		case streamed.fields != nil:
			if raw, err := json.Marshal(streamed.fields); err == nil {
				block = opaqueClaudeBlock(raw)
			}
		}
		message.Content = append(message.Content, block)
	}
	return message
}
