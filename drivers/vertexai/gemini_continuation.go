package vertexai

// Long decodes (Gemini 4 Argon with a maxOutputTokens above what one request
// returns) end a request with a continuationToken and finish reason CONTINUATION
// or none; the next request repeats the payload with that token and continues the
// same answer. The Google Gen AI SDK the TS client uses resumes automatically, so
// the driver does the same over REST.

// isGeminiResumableFinishReason reports a finish reason a continuation token can
// resume: none (an intermediate checkpoint) or CONTINUATION.
func isGeminiResumableFinishReason(reason string) bool {
	return reason == "" || reason == "CONTINUATION"
}

// geminiContinuationToken returns the token to resume a unary response from, or
// "" when generation is complete.
func geminiContinuationToken(response geminiResponse) string {
	if len(response.Candidates) == 0 {
		return ""
	}
	candidate := response.Candidates[0]
	if candidate.ContinuationToken != "" && isGeminiResumableFinishReason(candidate.FinishReason) {
		return candidate.ContinuationToken
	}
	return ""
}

// mergeGeminiContinuation appends a continuation response to the answer so far:
// candidate parts concatenate, the latest finish state wins, and the usage of
// each request is summed, since every request is billed.
func mergeGeminiContinuation(previous, next geminiResponse) geminiResponse {
	merged := next
	merged.UsageMetadata = addGeminiUsage(previous.UsageMetadata, next.UsageMetadata)
	if merged.PromptFeedback.BlockReason == "" && merged.PromptFeedback.BlockReasonMessage == "" {
		merged.PromptFeedback = previous.PromptFeedback
	}
	merged.Candidates = append([]geminiCandidate(nil), previous.Candidates...)
	for i, candidate := range next.Candidates {
		if i >= len(merged.Candidates) {
			merged.Candidates = append(merged.Candidates, candidate)
			continue
		}
		target := &merged.Candidates[i]
		target.Content.Parts = append(append([]geminiPart(nil), target.Content.Parts...), candidate.Content.Parts...)
		if candidate.Content.Role != "" {
			target.Content.Role = candidate.Content.Role
		}
		target.FinishReason = candidate.FinishReason
		target.ContinuationToken = candidate.ContinuationToken
		if candidate.FinishMessage != "" {
			target.FinishMessage = candidate.FinishMessage
		}
		if len(candidate.SafetyRatings) > 0 {
			target.SafetyRatings = candidate.SafetyRatings
		}
	}
	return merged
}

// addGeminiUsage sums the usage of two requests of one long decode.
func addGeminiUsage(a, b geminiUsageMetadata) geminiUsageMetadata {
	out := geminiUsageMetadata{
		PromptTokenCount:        a.PromptTokenCount + b.PromptTokenCount,
		CandidatesTokenCount:    a.CandidatesTokenCount + b.CandidatesTokenCount,
		TotalTokenCount:         a.TotalTokenCount + b.TotalTokenCount,
		CachedContentTokenCount: a.CachedContentTokenCount + b.CachedContentTokenCount,
		ThoughtsTokenCount:      a.ThoughtsTokenCount + b.ThoughtsTokenCount,
		ToolUsePromptTokenCount: a.ToolUsePromptTokenCount + b.ToolUsePromptTokenCount,
		TrafficType:             b.TrafficType,
	}
	if out.TrafficType == "" {
		out.TrafficType = a.TrafficType
	}
	out.CandidatesTokensDetails = append(out.CandidatesTokensDetails, a.CandidatesTokensDetails...)
	for _, detail := range b.CandidatesTokensDetails {
		found := false
		for i := range out.CandidatesTokensDetails {
			if out.CandidatesTokensDetails[i].Modality == detail.Modality {
				out.CandidatesTokensDetails[i].TokenCount += detail.TokenCount
				found = true
				break
			}
		}
		if !found {
			out.CandidatesTokensDetails = append(out.CandidatesTokensDetails, detail)
		}
	}
	return out
}

// geminiStreamHop tracks one streamed request of a long decode.
type geminiStreamHop struct {
	startToken        string
	token             string
	finishReason      string
	hasPromptFeedback bool
}

// observe records a streamed chunk and clears an intermediate finish reason the
// stream will resume from.
func (h *geminiStreamHop) observe(response *geminiResponse) {
	if response.PromptFeedback.BlockReason != "" || response.PromptFeedback.BlockReasonMessage != "" {
		h.hasPromptFeedback = true
	}
	if len(response.Candidates) == 0 {
		return
	}
	candidate := &response.Candidates[0]
	if candidate.FinishReason != "" {
		h.finishReason = candidate.FinishReason
	}
	if candidate.ContinuationToken != "" {
		h.token = candidate.ContinuationToken
	} else if !isGeminiResumableFinishReason(candidate.FinishReason) {
		h.token = ""
	}
	if h.token != "" && candidate.FinishReason != "" && isGeminiResumableFinishReason(candidate.FinishReason) {
		candidate.FinishReason = ""
	}
}

// resumeToken returns the token the next request resumes from, or "" to stop.
// A hop must bring a new token, so a stream that makes no progress ends.
func (h *geminiStreamHop) resumeToken() string {
	if h.hasPromptFeedback || h.token == "" || h.token == h.startToken || !isGeminiResumableFinishReason(h.finishReason) {
		return ""
	}
	return h.token
}
