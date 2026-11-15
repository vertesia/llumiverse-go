package core

import "strings"

// BinaryTruncatedMarker replaces the middle of a binary payload in a debug prompt.
const BinaryTruncatedMarker = "...[truncated binary]..."

// binaryDebugEdgeChars is how many characters of a binary payload a debug prompt keeps at each end.
const binaryDebugEdgeChars = 10

// DebugPromptFormatter is implemented by drivers whose provider prompt can carry
// inline binary data (base64 images, documents, audio). The shared lifecycle
// returns FormatDebugPrompt(prompt) as ExecutionResponse.Prompt, so a response
// never carries megabytes of base64, as the TS client's formatDebugPrompt does.
// Implementations must not modify prompt: it is still the request being sent.
type DebugPromptFormatter interface {
	FormatDebugPrompt(prompt any) any
}

// debugPrompt applies the driver's DebugPromptFormatter when it has one.
func debugPrompt(driver Driver, prompt any) any {
	if formatter, ok := driver.(DebugPromptFormatter); ok {
		return formatter.FormatDebugPrompt(prompt)
	}
	return prompt
}

// TruncateBinaryForDebug keeps the first and last few characters of a binary
// (usually base64) payload and replaces the rest with BinaryTruncatedMarker.
func TruncateBinaryForDebug(value string) string {
	if len(value) <= binaryDebugEdgeChars*2+len(BinaryTruncatedMarker) {
		return value
	}
	return value[:binaryDebugEdgeChars] + BinaryTruncatedMarker + value[len(value)-binaryDebugEdgeChars:]
}

// TruncateDataURLForDebug truncates the base64 payload of a data: URL and leaves
// any other string unchanged.
func TruncateDataURLForDebug(value string) string {
	const base64Marker = ";base64,"
	if !strings.HasPrefix(value, "data:") {
		return value
	}
	index := strings.Index(value, base64Marker)
	if index == -1 {
		return value
	}
	start := index + len(base64Marker)
	return value[:start] + TruncateBinaryForDebug(value[start:])
}
