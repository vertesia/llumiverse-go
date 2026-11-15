package vertexai

import (
	"crypto/rand"
	"encoding/hex"
	"strings"
)

// geminiGeneratedCallIDPrefix prefixes the function-call ids llumiverse generates
// when Vertex returns a call without one (or with one already used). Vertex never
// sees them: stripGeneratedGeminiCallIDs removes them from every request.
const geminiGeneratedCallIDPrefix = "llumiverse_call_"

// newGeminiGeneratedCallID returns a fresh generated function-call id.
func newGeminiGeneratedCallID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	h := hex.EncodeToString(b[:])
	return geminiGeneratedCallIDPrefix + h[0:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:]
}

func isGeminiGeneratedCallID(id string) bool {
	return strings.HasPrefix(id, geminiGeneratedCallIDPrefix)
}

// collectGeminiFunctionCallIDs returns the function-call ids already present in a conversation.
func collectGeminiFunctionCallIDs(contents []geminiContent) map[string]bool {
	ids := map[string]bool{}
	for _, content := range contents {
		for _, part := range content.Parts {
			if part.FunctionCall != nil && part.FunctionCall.ID != "" {
				ids[part.FunctionCall.ID] = true
			}
		}
	}
	return ids
}

// assignGeminiFunctionCallIDs gives every functionCall part an id unique within the
// conversation, in place. Vertex's own id is kept when it has not been used before;
// a call without one, or repeating an earlier call's id, gets a generated id. The id
// is what the driver returns as ToolUse.ID and it stays on the stored call, so a
// tool result can be paired with its call later. Name-based ids would collide when
// the model calls one tool more than once in a turn.
func assignGeminiFunctionCallIDs(parts []geminiPart, used map[string]bool) {
	for _, part := range parts {
		call := part.FunctionCall
		if call == nil {
			continue
		}
		if call.ID == "" || used[call.ID] {
			call.ID = newGeminiGeneratedCallID()
		}
		used[call.ID] = true
	}
}

// geminiToolUseID is the ToolUse id of a function call: its unique call id, or
// its name for a call that has none.
func geminiToolUseID(call *geminiFunctionCall) string {
	if call.ID != "" {
		return call.ID
	}
	return call.Name
}

// geminiFunctionCallNamesByID maps every functionCall id in a conversation to its function name.
func geminiFunctionCallNamesByID(contents []geminiContent) map[string]string {
	names := map[string]string{}
	for _, content := range contents {
		for _, part := range content.Parts {
			if call := part.FunctionCall; call != nil && call.ID != "" && call.Name != "" {
				names[call.ID] = call.Name
			}
		}
	}
	return names
}

// resolveGeminiFunctionResponseTarget finds the call a tool result answers: the
// function name it must be sent under, and the call id to echo when the call has
// one. Conversations stored before tool-use ids were unique used the function name
// as the id and carry no matching call id, so an unmatched id is sent as the name.
func resolveGeminiFunctionResponseTarget(toolUseID string, namesByID map[string]string) (id string, name string) {
	if name, ok := namesByID[toolUseID]; ok {
		return toolUseID, name
	}
	return "", toolUseID
}

// stripGeneratedGeminiCallIDs removes generated call ids from the contents sent to
// Vertex without touching the stored conversation. Vertex only receives call ids it
// issued; parts that had none pair by name and position as they always have.
func stripGeneratedGeminiCallIDs(contents []geminiContent) []geminiContent {
	out := make([]geminiContent, len(contents))
	for i, content := range contents {
		out[i] = content
		hasGenerated := false
		for _, part := range content.Parts {
			if (part.FunctionCall != nil && isGeminiGeneratedCallID(part.FunctionCall.ID)) ||
				(part.FunctionResponse != nil && isGeminiGeneratedCallID(part.FunctionResponse.ID)) {
				hasGenerated = true
				break
			}
		}
		if !hasGenerated {
			continue
		}
		parts := make([]geminiPart, len(content.Parts))
		for j, part := range content.Parts {
			if part.FunctionCall != nil && isGeminiGeneratedCallID(part.FunctionCall.ID) {
				call := *part.FunctionCall
				call.ID = ""
				part.FunctionCall = &call
			}
			if part.FunctionResponse != nil && isGeminiGeneratedCallID(part.FunctionResponse.ID) {
				response := *part.FunctionResponse
				response.ID = ""
				part.FunctionResponse = &response
			}
			parts[j] = part
		}
		out[i].Parts = parts
	}
	return out
}

// isGeminiFunctionResponseOnlyContent reports whether content is a user turn
// holding nothing but functionResponse parts.
func isGeminiFunctionResponseOnlyContent(content geminiContent) bool {
	if content.Role != "user" || len(content.Parts) == 0 {
		return false
	}
	for _, part := range content.Parts {
		if part.FunctionResponse == nil {
			return false
		}
	}
	return true
}

// mergeGeminiFunctionResponseContents recombines runs of consecutive user contents
// that hold nothing but functionResponse parts into one user turn. The prompt
// builder emits one content per tool-result segment, but Gemini requires every
// response to a model function-call turn in ONE user turn whose functionResponse
// count equals the call count; split parallel results are rejected with 400
// INVALID_ARGUMENT. Text contents keep their boundaries, which are explicit-cache
// breakpoints.
func mergeGeminiFunctionResponseContents(contents []geminiContent) []geminiContent {
	out := make([]geminiContent, 0, len(contents))
	for _, content := range contents {
		if n := len(out); n > 0 && isGeminiFunctionResponseOnlyContent(out[n-1]) && isGeminiFunctionResponseOnlyContent(content) {
			merged := out[n-1]
			merged.Parts = append(append([]geminiPart{}, merged.Parts...), content.Parts...)
			out[n-1] = merged
			continue
		}
		out = append(out, content)
	}
	return out
}

// fixOrphanedGeminiToolResults drops functionResponse parts with no matching
// functionCall in the immediately preceding model content. A response carrying a
// call id must match a call with that id; one without an id (conversations stored
// before tool-use ids were unique) matches by function name. A response left
// dangling after its call was dropped (conversation compaction or trimming) makes
// the API reject the request. The matching call set stays active across a run of
// user function-response contents, so split parallel results are not mistaken for
// orphans.
func fixOrphanedGeminiToolResults(contents []geminiContent) []geminiContent {
	if len(contents) == 0 {
		return contents
	}
	out := make([]geminiContent, 0, len(contents))
	allowedNames := map[string]bool{}
	allowedIDs := map[string]bool{}
	reset := func() {
		allowedNames = map[string]bool{}
		allowedIDs = map[string]bool{}
	}
	for _, content := range contents {
		if content.Role == "model" {
			reset()
			for _, part := range content.Parts {
				if call := part.FunctionCall; call != nil {
					if call.Name != "" {
						allowedNames[call.Name] = true
					}
					if call.ID != "" {
						allowedIDs[call.ID] = true
					}
				}
			}
			out = append(out, content)
			continue
		}
		hasResponse := false
		for _, part := range content.Parts {
			if part.FunctionResponse != nil {
				hasResponse = true
				break
			}
		}
		if content.Role != "user" || !hasResponse {
			reset()
			out = append(out, content)
			continue
		}
		filtered := make([]geminiPart, 0, len(content.Parts))
		for _, part := range content.Parts {
			response := part.FunctionResponse
			if response == nil ||
				(response.ID != "" && allowedIDs[response.ID]) ||
				(response.ID == "" && allowedNames[response.Name]) {
				filtered = append(filtered, part)
			}
		}
		if len(filtered) == 0 {
			continue
		}
		if len(filtered) != len(content.Parts) {
			content.Parts = filtered
		}
		out = append(out, content)
	}
	return out
}

// geminiRequestContents prepares stored contents for a generateContent request:
// split function responses are recombined, function parts become text when no
// tools are declared (checkpoint summaries may replay tool history with tools
// disabled), orphaned responses are dropped and generated call ids are removed.
// The input is never modified.
func geminiRequestContents(contents []geminiContent, hasTools bool) []geminiContent {
	out := mergeGeminiFunctionResponseContents(contents)
	if !hasTools && geminiContentsContainToolParts(out) {
		out = convertGeminiFunctionPartsToText(out)
	}
	return stripGeneratedGeminiCallIDs(fixOrphanedGeminiToolResults(out))
}
