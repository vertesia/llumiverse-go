package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"
	"unicode/utf8"
)

// MalformedStreamingToolArgumentsErrorName is the LlumiverseError.Name of the error a stream
// returns when a streamed tool call has malformed JSON arguments and the response was not cut
// off: the model's output must be regenerated, so it is not retryable as is. A caller can feed it
// back to the model for one corrective turn; IsMalformedStreamingToolArguments detects it.
const MalformedStreamingToolArgumentsErrorName = "MalformedStreamingToolArgumentsError"

// IsMalformedStreamingToolArguments reports whether err is, or wraps, the error a stream returns
// for a streamed tool call with malformed JSON arguments.
func IsMalformedStreamingToolArguments(err error) bool {
	var llumiErr *LlumiverseError
	return errors.As(err, &llumiErr) && llumiErr.Name == MalformedStreamingToolArgumentsErrorName
}

// newMalformedStreamingToolArgumentsError names the tool and the argument size, never the
// arguments themselves.
func newMalformedStreamingToolArgumentsError(tool ToolUseChunk, finishReason string, errContext LlumiverseErrorContext, original error) *LlumiverseError {
	name := tool.ToolName
	if name == "" {
		name = "unknown"
	}
	reason := finishReason
	if reason == "" {
		reason = "unknown"
	}
	text, _ := tool.ToolInput.(string)
	message := fmt.Sprintf("Received malformed JSON arguments for streamed tool %q (finish_reason=%s, argument_chars=%d)", name, reason, utf8.RuneCountInString(text))
	errContext.Operation = "stream"
	return NewLlumiverseError(message, BoolPtr(false), errContext, original, 0, MalformedStreamingToolArgumentsErrorName)
}

// accumulateToolUseChunk merges one streamed fragment into the tool call accumulated so far, the
// TS client's accumulateToolUseChunk. Argument strings are concatenated and argument objects
// merged; a later fragment may carry the tool name; thought-signature fragments are concatenated.
func accumulateToolUseChunk(existing *ToolUseChunk, tool ToolUseChunk) {
	if tool.ToolInput != nil {
		existingString, existingIsString := existing.ToolInput.(string)
		incomingString, incomingIsString := tool.ToolInput.(string)
		existingObject, existingIsObject := existing.ToolInput.(map[string]any)
		incomingObject, incomingIsObject := tool.ToolInput.(map[string]any)
		switch {
		case existing.ToolInput == nil:
			existing.ToolInput = tool.ToolInput
		case existingIsString && incomingIsString:
			existing.ToolInput = existingString + incomingString
		case existingIsObject && incomingIsObject:
			merged := make(map[string]any, len(existingObject)+len(incomingObject))
			maps.Copy(merged, existingObject)
			maps.Copy(merged, incomingObject)
			existing.ToolInput = merged
		case existingIsString && incomingIsObject && len(incomingObject) == 0, incomingIsString && incomingString == "":
			// Some OpenAI-compatible providers emit an empty placeholder between
			// argument fragments. It carries nothing and must not erase prior bytes.
		default:
			existing.ToolInput = tool.ToolInput
		}
	}
	if tool.ToolName != "" {
		existing.ToolName = tool.ToolName
	}
	if tool.ThoughtSignature != "" {
		// Gemini's thought signature is opaque base64 that must be sent back
		// verbatim. A large one can arrive in pieces, or only on a later
		// fragment, so the fragments are concatenated in arrival order like
		// argument strings; a signature sent whole is a single fragment.
		existing.ThoughtSignature += tool.ThoughtSignature
	}
}

// FinalizeStreamingToolUse turns the accumulated tool calls of a finished stream into the
// completion's tool calls, the TS client's finalizeStreamingToolUse: string arguments are parsed
// as JSON, and empty arguments (a call to a tool without parameters) become an empty object. A
// call whose arguments do not parse was cut off when the response stopped on a length limit, so
// it is left out and the other calls are kept. Without such a stop, malformed arguments are model
// output to regenerate: they are never coerced to an empty object, which could run a tool with
// invented arguments, and the complete calls are returned with a non-retryable LlumiverseError
// named MalformedStreamingToolArgumentsErrorName. errContext names the provider and model in
// that error. The shared channel stream calls it; a driver that
// accumulates its own stream can call it too.
func FinalizeStreamingToolUse(tools []ToolUseChunk, finishReason string, errContext LlumiverseErrorContext) ([]ToolUse, error) {
	return finalizeStreamingToolUse(tools, finishReason, errContext)
}

func finalizeStreamingToolUse(tools []ToolUseChunk, finishReason string, errContext LlumiverseErrorContext) ([]ToolUse, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	finalized := make([]ToolUse, 0, len(tools))
	var malformed *ToolUseChunk
	var parseErr error
	for i, tool := range tools {
		input := tool.ToolInput
		if text, ok := input.(string); ok {
			var parsed any
			if strings.TrimSpace(text) == "" {
				// Claude streams a call without arguments as one empty partial_json:
				// a complete call, unless a length stop cut it off before its first
				// argument fragment arrived.
				if finishReason == "length" {
					if malformed == nil {
						malformed = &tools[i]
					}
					continue
				}
				parsed = map[string]any{}
			} else if err := json.Unmarshal([]byte(text), &parsed); err != nil {
				if malformed == nil {
					malformed, parseErr = &tools[i], err
				}
				continue
			}
			input = parsed
		}
		final := ToolUse{ID: tool.ID, ToolName: tool.ToolName, ToolInput: ToolInputObject(input)}
		if tool.ThoughtSignature != "" {
			signature := tool.ThoughtSignature
			final.ThoughtSignature = &signature
		}
		finalized = append(finalized, final)
	}
	if len(finalized) == 0 {
		finalized = nil
	}
	if malformed == nil || finishReason == "length" {
		return finalized, nil
	}
	return finalized, newMalformedStreamingToolArgumentsError(*malformed, finishReason, errContext, parseErr)
}
