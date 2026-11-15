package bedrock

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// Mirrors the TS prompt-caching.test.ts "reports the one-hour share of cache
// writes", for Converse responses and streamed metadata alike.
func TestBedrockUsageReportsOneHourCacheWrites(t *testing.T) {
	usage := &brtypes.TokenUsage{
		InputTokens:           aws.Int32(25),
		OutputTokens:          aws.Int32(10),
		TotalTokens:           aws.Int32(85),
		CacheWriteInputTokens: aws.Int32(50),
		CacheDetails: []brtypes.CacheDetail{
			{Ttl: brtypes.CacheTTLFiveMinutes, InputTokens: aws.Int32(20)},
			{Ttl: brtypes.CacheTTLOneHour, InputTokens: aws.Int32(30)},
		},
	}
	got := bedrockUsage(usage)
	if count(got.PromptCacheWrite) != 50 || count(got.PromptCacheWrite1H) != 30 || count(got.Prompt) != 75 {
		t.Fatalf("usage = %#v", got)
	}
	state := bedrockStreamState{toolBlocks: map[int32]bedrockStreamingToolBlock{}}
	chunk := state.chunk(&brtypes.ConverseStreamOutputMemberMetadata{Value: brtypes.ConverseStreamMetadataEvent{Usage: usage}}, ExecutionOptions{})
	if chunk.TokenUsage == nil || count(chunk.TokenUsage.PromptCacheWrite1H) != 30 {
		t.Fatalf("stream usage = %#v", chunk.TokenUsage)
	}
	usage.CacheDetails = []brtypes.CacheDetail{{Ttl: brtypes.CacheTTLFiveMinutes, InputTokens: aws.Int32(50)}}
	if got := bedrockUsage(usage); got.PromptCacheWrite1H != nil {
		t.Fatalf("five-minute writes reported as one hour: %#v", got)
	}
}
