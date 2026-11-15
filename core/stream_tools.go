package core

import (
	"encoding/json"
	"maps"
)

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
		case existingIsString && incomingIsString:
			existing.ToolInput = existingString + incomingString
		case existingIsObject && incomingIsObject:
			merged := make(map[string]any, len(existingObject)+len(incomingObject))
			maps.Copy(merged, existingObject)
			maps.Copy(merged, incomingObject)
			existing.ToolInput = merged
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
// as JSON. A call whose arguments do not parse was cut off when the response stopped on a length
// limit, so it is left out and the other calls are kept. Without such a stop, malformed arguments
// mean a corrupt stream: they are never coerced to an empty object, which could run a tool with
// invented arguments, and a LlumiverseError is returned with the complete calls. errContext names
// the provider and model in that error. The shared channel stream calls it; a driver that
// accumulates its own stream can call it too.
func FinalizeStreamingToolUse(tools []ToolUseChunk, finishReason string, errContext LlumiverseErrorContext) ([]ToolUse, error) {
	return finalizeStreamingToolUse(tools, finishReason, errContext)
}

func finalizeStreamingToolUse(tools []ToolUseChunk, finishReason string, errContext LlumiverseErrorContext) ([]ToolUse, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	finalized := make([]ToolUse, 0, len(tools))
	var parseErr error
	for _, tool := range tools {
		input := tool.ToolInput
		if text, ok := input.(string); ok {
			var parsed any
			if err := json.Unmarshal([]byte(text), &parsed); err != nil {
				if parseErr == nil {
					parseErr = err
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
	if parseErr == nil || finishReason == "length" {
		return finalized, nil
	}
	errContext.Operation = "stream"
	return finalized, NewLlumiverseError("Received malformed JSON arguments for a streamed tool call", BoolPtr(true), errContext, parseErr, 0, "")
}
