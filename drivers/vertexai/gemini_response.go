package vertexai

import (
	"encoding/json"
	"strings"
)

// geminiCandidate is one candidate turn of a generateContent response.
type geminiCandidate struct {
	FinishReason  string          `json:"finishReason"`
	FinishMessage string          `json:"finishMessage"`
	SafetyRatings json.RawMessage `json:"safetyRatings,omitempty"`
	Content       geminiContent   `json:"content"`
}

// geminiUsageMetadata is the token accounting of a generateContent response.
type geminiUsageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	ToolUsePromptTokenCount int `json:"toolUsePromptTokenCount"`
	CandidatesTokensDetails []struct {
		Modality   string `json:"modality"`
		TokenCount int    `json:"tokenCount"`
	} `json:"candidatesTokensDetails,omitempty"`
	TrafficType string `json:"trafficType,omitempty"`
}

// geminiResponse is the generateContent / streamGenerateContent response: candidate
// turns, prompt-level safety feedback, and token usage metadata.
type geminiResponse struct {
	Candidates     []geminiCandidate `json:"candidates"`
	PromptFeedback struct {
		BlockReason        string `json:"blockReason"`
		BlockReasonMessage string `json:"blockReasonMessage"`
	} `json:"promptFeedback"`
	UsageMetadata geminiUsageMetadata `json:"usageMetadata"`
	ResponseID    string              `json:"responseId,omitempty"`
}

// geminiFinishReasonHandling describes a terminal finish reason: why Gemini
// stopped, and whether another attempt may succeed.
type geminiFinishReasonHandling struct {
	message   string
	retryable bool
}

// geminiFinishReasonHandlings classifies the finish reasons that end a response
// without a usable answer, as the TS client does.
var geminiFinishReasonHandlings = map[string]geminiFinishReasonHandling{
	"SAFETY":                   {"Gemini blocked the response because it may violate safety policies.", false},
	"RECITATION":               {"Gemini blocked the response because it may reproduce protected source material.", false},
	"LANGUAGE":                 {"Gemini stopped because the response used an unsupported language.", false},
	"BLOCKLIST":                {"Gemini blocked the response because it contains a forbidden term.", false},
	"PROHIBITED_CONTENT":       {"Gemini blocked the response because it may contain prohibited content.", false},
	"SPII":                     {"Gemini blocked the response because it may contain sensitive personal information.", false},
	"IMAGE_SAFETY":             {"Gemini blocked the generated image because it may violate safety policies.", false},
	"IMAGE_PROHIBITED_CONTENT": {"Gemini blocked the generated image because it may contain prohibited content.", false},
	"IMAGE_RECITATION":         {"Gemini blocked the generated image because it may reproduce protected source material.", false},
	// Model Armor, escalation rules and Prohibited Use Policy limits are policy decisions; retrying does not help.
	"MODEL_ARMOR":          {"Gemini blocked the response through a Model Armor policy.", false},
	"ESCALATION":           {"Gemini filtered the request through an escalation rule.", false},
	"PUP_LIMITED_DISABLED": {"Gemini stopped because the account is limited or disabled for Prohibited Use Policy violations.", false},
	// The request itself lacks a thought signature, so resending it unchanged fails the same way.
	"MISSING_THOUGHT_SIGNATURE": {"Gemini rejected the request because a thought signature is missing from the conversation.", false},
	"MALFORMED_FUNCTION_CALL":   {"Gemini generated an invalid function call.", true},
	// A malformed generation, like a malformed function call, may succeed on another attempt.
	"MALFORMED_RESPONSE": {"Gemini generated a malformed response.", true},
	"NO_IMAGE":           {"Gemini did not generate the requested image.", true},
	"IMAGE_OTHER":        {"Gemini stopped image generation for an unspecified reason.", true},
	"OTHER":              {"Gemini stopped generation for an unspecified reason.", true},
}

// geminiFinishReasonError is a terminal finish reason: the response carries no
// usable answer. Its retryability is unknown (nil) for a reason the driver does
// not know.
type geminiFinishReasonError struct {
	finishReason string
	message      string
	retryable    *bool
}

func (e *geminiFinishReasonError) Error() string { return e.message }

func newGeminiFinishReasonError(candidate geminiCandidate) *geminiFinishReasonError {
	handling, known := geminiFinishReasonHandlings[candidate.FinishReason]
	summary := "Gemini stopped generation with an unsupported finish reason."
	var retryable *bool
	if known {
		summary = handling.message
		retryable = ptr(handling.retryable)
	}
	details := []string{summary + " Finish reason: " + candidate.FinishReason + "."}
	if candidate.FinishMessage != "" {
		details = append(details, "Finish message: "+candidate.FinishMessage+".")
	}
	if len(candidate.SafetyRatings) > 0 && string(candidate.SafetyRatings) != "[]" && string(candidate.SafetyRatings) != "null" {
		details = append(details, "Safety ratings: "+string(candidate.SafetyRatings)+".")
	}
	return &geminiFinishReasonError{finishReason: candidate.FinishReason, message: strings.Join(details, " "), retryable: retryable}
}

// checkGeminiFinishReason returns an error for a finish reason that ends the
// response without a usable answer. STOP, MAX_TOKENS, CONTINUATION (a long decode
// that could not resume) and an unspecified reason are normal; UNEXPECTED_TOOL_CALL
// is recoverable, so its tool use is returned for the caller to answer with an
// error.
func checkGeminiFinishReason(candidate geminiCandidate) error {
	switch candidate.FinishReason {
	case "", "STOP", "MAX_TOKENS", "CONTINUATION", "FINISH_REASON_UNSPECIFIED", "UNEXPECTED_TOOL_CALL":
		return nil
	}
	return newGeminiFinishReasonError(candidate)
}

// checkGeminiResponseFinishReasons checks every candidate of a response.
func checkGeminiResponseFinishReasons(response geminiResponse) error {
	for _, candidate := range response.Candidates {
		if err := checkGeminiFinishReason(candidate); err != nil {
			return err
		}
	}
	return nil
}

// geminiResponseToCompletion converts a full response into a Completion, ensuring
// at least one (empty) text result and optionally attaching the raw response.
func geminiResponseToCompletion(response geminiResponse, options ExecutionOptions) *Completion {
	chunk := geminiResponseToChunk(response, geminiIncludeThoughts(options.ModelOptions))
	completion := &Completion{
		Result:       chunk.Result,
		TokenUsage:   chunk.TokenUsage,
		ToolUse:      geminiToolUses(chunk.ToolUse),
		FinishReason: chunk.FinishReason,
		ServiceTier:  chunk.ServiceTier,
	}
	if len(completion.Result) == 0 {
		completion.Result = []CompletionResult{newTextResult("")}
	}
	if options.IncludeOriginalResponse {
		completion.OriginalResponse = response
	}
	return completion
}

// geminiToolUses converts the function-call chunks of a full response into the
// final ToolUse entries, carrying the thought signature only when Gemini sent one.
func geminiToolUses(chunks []ToolUseChunk) []ToolUse {
	if chunks == nil {
		return nil
	}
	tools := make([]ToolUse, 0, len(chunks))
	for _, chunk := range chunks {
		tool := ToolUse{ID: chunk.ID, ToolName: chunk.ToolName, ToolInput: toolInputObject(chunk.ToolInput)}
		if chunk.ThoughtSignature != "" {
			tool.ThoughtSignature = ptr(chunk.ThoughtSignature)
		}
		tools = append(tools, tool)
	}
	return tools
}

// geminiResponseToChunk converts a response (full or streamed) into a
// CompletionChunk: thought parts become thoughts results (when includeThoughts),
// text, image and functionCall parts map in order, blocked-prompt feedback is
// surfaced as text, and a function call marks the chunk tool_use.
func geminiResponseToChunk(response geminiResponse, includeThoughts bool) CompletionChunk {
	chunk := CompletionChunk{
		TokenUsage:  geminiUsage(response.UsageMetadata),
		ServiceTier: normalizeVertexServiceTier(response.UsageMetadata.TrafficType),
	}
	if len(response.Candidates) == 0 {
		if response.PromptFeedback.BlockReasonMessage != "" {
			chunk.Result = []CompletionResult{newTextResult(response.PromptFeedback.BlockReasonMessage)}
		}
		chunk.FinishReason = response.PromptFeedback.BlockReason
		return chunk
	}
	candidate := response.Candidates[0]
	chunk.FinishReason = geminiFinishReason(candidate.FinishReason)
	for _, part := range candidate.Content.Parts {
		switch {
		case part.Text != "" && part.Thought:
			if includeThoughts {
				chunk.Result = append(chunk.Result, newThoughtsResult(part.Text))
			}
		case part.Text != "":
			chunk.Result = append(chunk.Result, newTextResult(part.Text))
		case part.InlineData != nil:
			chunk.Result = append(chunk.Result, newImageResult("data:"+part.InlineData.MIMEType+";base64,"+part.InlineData.Data))
		}
		if part.FunctionCall != nil {
			// Tool-call finish reasons are recoverable: return tool_use so the
			// caller can execute the tool or report a toolError, rather than
			// treating the model response as a provider failure.
			chunk.ToolUse = append(chunk.ToolUse, ToolUseChunk{
				ID:               geminiToolUseID(part.FunctionCall),
				ToolName:         part.FunctionCall.Name,
				ToolInput:        part.FunctionCall.Args,
				ThoughtSignature: part.ThoughtSignature,
			})
		}
	}
	if len(chunk.ToolUse) > 0 {
		chunk.FinishReason = "tool_use"
	}
	return chunk
}

// geminiUsage builds token usage as the TS client does: none without a total,
// prompt_new is prompt minus cached tokens, the result counts candidate, thinking
// and tool-use prompt tokens, and generated-image tokens are reported apart.
func geminiUsage(usage geminiUsageMetadata) *ExecutionTokenUsage {
	if usage.TotalTokenCount == 0 {
		return nil
	}
	out := &ExecutionTokenUsage{
		Total:     tokens(usage.TotalTokenCount),
		Prompt:    tokens(usage.PromptTokenCount),
		PromptNew: tokens(usage.PromptTokenCount - usage.CachedContentTokenCount),
	}
	if usage.CachedContentTokenCount > 0 {
		out.PromptCached = tokens(usage.CachedContentTokenCount)
	}
	if result := usage.CandidatesTokenCount + usage.ThoughtsTokenCount + usage.ToolUsePromptTokenCount; result > 0 {
		out.Result = tokens(result)
	}
	for _, detail := range usage.CandidatesTokensDetails {
		if detail.Modality == "IMAGE" && detail.TokenCount > 0 {
			out.ResultImage = tokens(detail.TokenCount)
			break
		}
	}
	return out
}

// geminiFinishReason maps Gemini finish reasons to the neutral vocabulary, passing
// unknown reasons through unchanged. A CONTINUATION left on the final response
// means a long decode could not resume, so it reads as length.
func geminiFinishReason(reason string) string {
	switch reason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS", "CONTINUATION":
		return "length"
	default:
		return reason
	}
}

// normalizeVertexServiceTier maps the trafficType Vertex reports to the neutral
// service tier, or "" for an unknown or absent traffic type.
func normalizeVertexServiceTier(trafficType string) string {
	switch trafficType {
	case "ON_DEMAND":
		return "default"
	case "ON_DEMAND_PRIORITY":
		return "priority"
	case "ON_DEMAND_FLEX":
		return "flex"
	case "PROVISIONED_THROUGHPUT":
		return "provisioned"
	default:
		return ""
	}
}

// geminiServiceTier resolves the requested service tier: service_tier, else the
// deprecated flex flag.
func geminiServiceTier(modelOptions map[string]any) string {
	if tier := optionString(modelOptions, "service_tier"); tier != "" {
		return tier
	}
	if flex, _ := modelOptions["flex"].(bool); flex {
		return "flex"
	}
	return ""
}

// addGeminiServiceTierHeaders requests a shared service tier (flex, priority)
// through the Vertex request-type headers. The default tier sends none.
func addGeminiServiceTierHeaders(headers map[string]string, modelOptions map[string]any) {
	if tier := geminiServiceTier(modelOptions); tier != "" && tier != "default" {
		headers["X-Vertex-AI-LLM-Request-Type"] = "shared"
		headers["X-Vertex-AI-LLM-Shared-Request-Type"] = tier
	}
}

// appendGeminiStreamParts accumulates streamed parts into the model turn kept in
// the conversation: adjacent text deltas of the same kind merge, while signed
// parts (thought signatures) stay separate so each signature keeps its own part.
func appendGeminiStreamParts(target []geminiPart, incoming []geminiPart) []geminiPart {
	for _, part := range incoming {
		if n := len(target); n > 0 && part.Text != "" {
			previous := &target[n-1]
			if previous.isTextOnly() && part.isTextOnly() && previous.ThoughtSignature == "" && part.ThoughtSignature == "" && previous.Thought == part.Thought {
				previous.Text += part.Text
				continue
			}
		}
		target = append(target, part)
	}
	return target
}

// isTextOnly reports whether a part carries text and no other data.
func (p geminiPart) isTextOnly() bool {
	return p.InlineData == nil && p.FileData == nil && p.FunctionCall == nil && p.FunctionResponse == nil
}
