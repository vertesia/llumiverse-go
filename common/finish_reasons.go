package common

// cutOffFinishReasons are the provider finish reasons that can end a response in the middle of a
// tool call: a token limit, a refusal (Claude refusal), or a content filter (OpenAI
// content_filter, Bedrock content_filtered and guardrail_intervened). Drivers pass these through
// unnormalized apart from length.
var cutOffFinishReasons = map[string]bool{
	"length":               true,
	"refusal":              true,
	"content_filter":       true,
	"content_filtered":     true,
	"guardrail_intervened": true,
}

// IsCutOffFinishReason reports whether finishReason means the response may have stopped in the
// middle of its last tool call.
func IsCutOffFinishReason(finishReason string) bool {
	return cutOffFinishReasons[finishReason]
}
