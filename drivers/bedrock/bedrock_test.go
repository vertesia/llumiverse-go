package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrock "github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
	bedrockruntime "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/vertesia/llumiverse-go/common"
)

type fakeBedrockRuntime struct {
	got          *bedrockruntime.ConverseInput
	gotStream    *bedrockruntime.ConverseStreamInput
	gotInvoke    *bedrockruntime.InvokeModelInput
	gotInvokes   []*bedrockruntime.InvokeModelInput
	streamEvents []brtypes.ConverseStreamOutput
	invokeBody   []byte
}

type fakeBedrockModels struct {
	foundation *bedrock.ListFoundationModelsOutput
	custom     *bedrock.ListCustomModelsOutput
	profiles   *bedrock.ListInferenceProfilesOutput
}

func (f *fakeBedrockRuntime) Converse(_ context.Context, input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	f.got = input
	return &bedrockruntime.ConverseOutput{
		Output: &brtypes.ConverseOutputMemberMessage{Value: brtypes.Message{
			Role: brtypes.ConversationRoleAssistant,
			Content: []brtypes.ContentBlock{
				&brtypes.ContentBlockMemberText{Value: "bedrock ok"},
				&brtypes.ContentBlockMemberToolUse{Value: brtypes.ToolUseBlock{
					ToolUseId: aws.String("tool_1"),
					Name:      aws.String("lookup"),
					Input:     brdoc.NewLazyDocument(map[string]any{"q": "osaka"}),
				}},
			},
		}},
		StopReason: brtypes.StopReasonToolUse,
		Usage: &brtypes.TokenUsage{
			InputTokens:  aws.Int32(2),
			OutputTokens: aws.Int32(3),
			TotalTokens:  aws.Int32(5),
		},
	}, nil
}

func (f *fakeBedrockRuntime) ConverseStream(_ context.Context, input *bedrockruntime.ConverseStreamInput, _ ...func(*bedrockruntime.Options)) (bedrockConverseStream, error) {
	f.gotStream = input
	return newFakeBedrockConverseStream(f.streamEvents), nil
}

func (f *fakeBedrockRuntime) InvokeModel(_ context.Context, input *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	f.gotInvoke = input
	f.gotInvokes = append(f.gotInvokes, input)
	body := f.invokeBody
	if body == nil {
		body = []byte(`{}`)
	}
	return &bedrockruntime.InvokeModelOutput{Body: body}, nil
}

func (f *fakeBedrockModels) ListFoundationModels(context.Context, *bedrock.ListFoundationModelsInput, ...func(*bedrock.Options)) (*bedrock.ListFoundationModelsOutput, error) {
	if f.foundation == nil {
		return &bedrock.ListFoundationModelsOutput{}, nil
	}
	return f.foundation, nil
}

func (f *fakeBedrockModels) ListCustomModels(context.Context, *bedrock.ListCustomModelsInput, ...func(*bedrock.Options)) (*bedrock.ListCustomModelsOutput, error) {
	if f.custom == nil {
		return &bedrock.ListCustomModelsOutput{}, nil
	}
	return f.custom, nil
}

func (f *fakeBedrockModels) ListInferenceProfiles(context.Context, *bedrock.ListInferenceProfilesInput, ...func(*bedrock.Options)) (*bedrock.ListInferenceProfilesOutput, error) {
	if f.profiles == nil {
		return &bedrock.ListInferenceProfilesOutput{}, nil
	}
	return f.profiles, nil
}

type fakeBedrockConverseStream struct {
	events chan brtypes.ConverseStreamOutput
	once   sync.Once
}

func newFakeBedrockConverseStream(events []brtypes.ConverseStreamOutput) *fakeBedrockConverseStream {
	ch := make(chan brtypes.ConverseStreamOutput, len(events))
	for _, event := range events {
		ch <- event
	}
	close(ch)
	return &fakeBedrockConverseStream{events: ch}
}

func (s *fakeBedrockConverseStream) Events() <-chan brtypes.ConverseStreamOutput {
	return s.events
}

func (s *fakeBedrockConverseStream) Close() error {
	s.once.Do(func() {})
	return nil
}

func (s *fakeBedrockConverseStream) Err() error {
	return nil
}

func TestBedrockExecuteUsesConverse(t *testing.T) {
	t.Parallel()

	runtime := &fakeBedrockRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	resp, err := driver.Execute(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "System"},
		{Role: PromptRoleUser, Content: "Hi"},
	}, ExecutionOptions{
		Model:        "anthropic.claude-3-5-sonnet-20241022-v2:0",
		ModelOptions: map[string]any{"max_tokens": 128, "temperature": 0.3},
		Tools: []ToolDefinition{{
			Name:        "lookup",
			Description: "Lookup",
			InputSchema: map[string]any{"type": "object"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if runtime.got == nil {
		t.Fatal("Converse was not called")
	}
	if runtime.got.ModelId == nil || *runtime.got.ModelId != "anthropic.claude-3-5-sonnet-20241022-v2:0" {
		t.Fatalf("model id = %#v", runtime.got.ModelId)
	}
	if runtime.got.InferenceConfig == nil || runtime.got.InferenceConfig.MaxTokens == nil || *runtime.got.InferenceConfig.MaxTokens != 128 {
		t.Fatalf("inference config = %#v", runtime.got.InferenceConfig)
	}
	if len(runtime.got.ToolConfig.Tools) != 1 {
		t.Fatalf("tools = %#v", runtime.got.ToolConfig)
	}
	if len(resp.Result) != 1 || resp.Result[0].Value != "bedrock ok" {
		t.Fatalf("result = %#v", resp.Result)
	}
	if len(resp.ToolUse) != 1 || resp.ToolUse[0].ToolName != "lookup" {
		t.Fatalf("tool use = %#v", resp.ToolUse)
	}
}

func TestNewBedrockDriverWithSuppliedConfig(t *testing.T) {
	t.Parallel()

	driver, err := NewBedrockDriver(context.Background(), BedrockOptions{
		DriverOptions: DriverOptions{HTTPTimeout: HTTPTimeoutOptions{BodyTimeout: time.Second}},
		Region:        "us-east-1",
		Config:        aws.Config{Region: "us-east-1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if driver.runtime == nil || driver.models == nil {
		t.Fatalf("clients were not initialized: %#v", driver)
	}
}

func TestBedrockClaudeDocumentsAndCachePoints(t *testing.T) {
	t.Parallel()

	runtime := &fakeBedrockRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	_, err := driver.Execute(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "System"},
		{
			Role:    PromptRoleUser,
			Content: "Use the docs",
			Files: []DataSource{
				BytesDataSource{FileName: "brief.pdf", MIME: "application/pdf", Data: []byte("%PDF")},
				BytesDataSource{FileName: "notes.txt", MIME: "text/plain", Data: []byte("notes")},
			},
		},
		{Role: PromptRoleAssistant, Content: "Earlier 1"},
		{Role: PromptRoleUser, Content: "Earlier 2"},
		{Role: PromptRoleAssistant, Content: "Earlier 3"},
		{Role: PromptRoleUser, Content: "Earlier 4"},
	}, ExecutionOptions{
		Model: "anthropic.claude-3-7-sonnet-20250219-v1:0",
		ModelOptions: map[string]any{
			"cache_enabled":          true,
			"cache_ttl":              "1h",
			"thinking_budget_tokens": 1024,
			"max_tokens":             2048,
		},
		Tools: []ToolDefinition{{
			Name:        "lookup",
			Description: "Lookup",
			InputSchema: map[string]any{"type": "object"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if runtime.got == nil {
		t.Fatal("Converse was not called")
	}
	if len(runtime.got.System) < 2 {
		t.Fatalf("system = %#v", runtime.got.System)
	}
	if cache, ok := runtime.got.System[len(runtime.got.System)-1].(*brtypes.SystemContentBlockMemberCachePoint); !ok || cache.Value.Ttl != brtypes.CacheTTLOneHour {
		t.Fatalf("system cache point = %#v", runtime.got.System[len(runtime.got.System)-1])
	}
	if runtime.got.ToolConfig == nil || len(runtime.got.ToolConfig.Tools) < 2 {
		t.Fatalf("tool config = %#v", runtime.got.ToolConfig)
	}
	if _, ok := runtime.got.ToolConfig.Tools[len(runtime.got.ToolConfig.Tools)-1].(*brtypes.ToolMemberCachePoint); !ok {
		t.Fatalf("tool cache point missing: %#v", runtime.got.ToolConfig.Tools)
	}
	var sawPDF, sawTextDoc bool
	for _, msg := range runtime.got.Messages {
		for _, block := range msg.Content {
			doc, ok := block.(*brtypes.ContentBlockMemberDocument)
			if !ok {
				continue
			}
			if doc.Value.Format == brtypes.DocumentFormatPdf {
				sawPDF = true
			}
			if doc.Value.Format == brtypes.DocumentFormatTxt {
				sawTextDoc = true
			}
		}
	}
	if !sawPDF || !sawTextDoc {
		t.Fatalf("document blocks missing: %#v", runtime.got.Messages)
	}
	pivot := runtime.got.Messages[len(runtime.got.Messages)-2]
	if _, ok := pivot.Content[len(pivot.Content)-1].(*brtypes.ContentBlockMemberCachePoint); !ok {
		t.Fatalf("pivot cache point missing: %#v", pivot.Content)
	}
	if runtime.got.AdditionalModelRequestFields == nil {
		t.Fatal("missing reasoning config")
	}
}

func TestBedrockPromptFileAndConversationBranches(t *testing.T) {
	t.Parallel()

	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, &fakeBedrockRuntime{}, nil)
	promptAny, err := driver.CreatePrompt(context.Background(), []PromptSegment{{Role: PromptRoleSystem, Content: "System only"}}, ExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	prompt := promptAny.(bedrockPrompt)
	if len(prompt.System) != 0 || len(prompt.Messages) != 1 || prompt.Messages[0].Role != brtypes.ConversationRoleUser {
		t.Fatalf("system-only prompt = %#v", prompt)
	}
	if _, err := driver.CreatePrompt(context.Background(), []PromptSegment{{Role: PromptRoleTool, Content: "missing id"}}, ExecutionOptions{}); err == nil {
		t.Fatal("expected tool segment without ToolUseID to fail")
	}

	promptAny, err = driver.CreatePrompt(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "System"},
		{Role: PromptRoleAssistant, Content: "Previous answer"},
		{Role: PromptRoleUser, Content: "Question"},
		{Role: PromptRoleSafety, Content: "Safety"},
		{Role: PromptRoleNegative, Content: "ignored"},
		{Role: PromptRoleMask, Content: "ignored"},
	}, ExecutionOptions{
		ResultSchema: map[string]any{"type": "object"},
		Tools:        []ToolDefinition{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	prompt = promptAny.(bedrockPrompt)
	if len(prompt.System) != 2 {
		t.Fatalf("system blocks = %#v", prompt.System)
	}
	notice := prompt.System[1].(*brtypes.SystemContentBlockMemberText).Value
	if !strings.Contains(notice, "When not calling tools") {
		t.Fatalf("schema notice = %q", notice)
	}
	if len(prompt.Messages) != 2 || prompt.Messages[0].Role != brtypes.ConversationRoleAssistant || prompt.Messages[1].Role != brtypes.ConversationRoleUser {
		t.Fatalf("messages = %#v", prompt.Messages)
	}

	jsonBlock, err := bedrockFileBlock(context.Background(), BytesDataSource{FileName: "data.json", MIME: "application/json", Data: []byte(`{"ok":true}`)})
	if err != nil {
		t.Fatal(err)
	}
	if text, ok := jsonBlock.(*brtypes.ContentBlockMemberText); !ok || text.Value != `{"ok":true}` {
		t.Fatalf("json file block = %#v", jsonBlock)
	}
	textDocBlock, err := bedrockFileBlock(context.Background(), BytesDataSource{FileName: "notes.txt", MIME: "text/plain", Data: []byte("notes")})
	if err != nil {
		t.Fatal(err)
	}
	if doc, ok := textDocBlock.(*brtypes.ContentBlockMemberDocument); !ok {
		t.Fatalf("text document block = %#v", textDocBlock)
	} else if _, ok := doc.Value.Source.(*brtypes.DocumentSourceMemberText); !ok {
		t.Fatalf("text document source = %#v", doc.Value.Source)
	}
	videoBlock, err := bedrockVideoBlock(context.Background(), BytesDataSource{FileName: "clip.mp4", MIME: "video/mp4", Data: []byte("mp4"), URI: "https://example.com/clip.mp4"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := videoBlock.Source.(*brtypes.VideoSourceMemberBytes); !ok {
		t.Fatalf("video source = %#v", videoBlock.Source)
	}
	fallbackBlock, err := bedrockFileBlock(context.Background(), BytesDataSource{FileName: "raw.bin", MIME: "application/octet-stream", Data: []byte("raw")})
	if err != nil {
		t.Fatal(err)
	}
	if text, ok := fallbackBlock.(*brtypes.ContentBlockMemberText); !ok || text.Value != "raw" {
		t.Fatalf("fallback block = %#v", fallbackBlock)
	}
}

func TestBedrockListModelsIncludesCustomAndInferenceProfiles(t *testing.T) {
	t.Parallel()

	modelsClient := &fakeBedrockModels{
		foundation: &bedrock.ListFoundationModelsOutput{ModelSummaries: []bedrocktypes.FoundationModelSummary{
			{
				ModelId:                    aws.String("anthropic.claude-3-5-sonnet"),
				ModelArn:                   aws.String("arn:foundation:claude"),
				ProviderName:               aws.String("Anthropic"),
				ModelName:                  aws.String("Claude 3.5 Sonnet"),
				ResponseStreamingSupported: aws.Bool(true),
				InputModalities:            []bedrocktypes.ModelModality{bedrocktypes.ModelModalityText},
				OutputModalities:           []bedrocktypes.ModelModality{bedrocktypes.ModelModalityText},
			},
		}},
		custom: &bedrock.ListCustomModelsOutput{ModelSummaries: []bedrocktypes.CustomModelSummary{
			{
				ModelArn:      aws.String("arn:custom:model"),
				ModelName:     aws.String("custom-claude"),
				BaseModelName: aws.String("Claude 3.5 Sonnet"),
			},
		}},
		profiles: &bedrock.ListInferenceProfilesOutput{InferenceProfileSummaries: []bedrocktypes.InferenceProfileSummary{
			{
				InferenceProfileArn:  aws.String("arn:profile:anthropic"),
				InferenceProfileId:   aws.String("anthropic.profile"),
				InferenceProfileName: aws.String("Anthropic profile"),
			},
		}},
	}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, &fakeBedrockRuntime{}, modelsClient)
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 3 {
		t.Fatalf("models = %#v", models)
	}
	byID := map[string]AIModel{}
	for _, model := range models {
		byID[model.ID] = model
	}
	if !byID["arn:custom:model"].IsCustom || byID["arn:custom:model"].Owner != "custom" {
		t.Fatalf("custom model = %#v", byID["arn:custom:model"])
	}
	if byID["arn:profile:anthropic"].Owner != "anthropic" {
		t.Fatalf("profile model = %#v", byID["arn:profile:anthropic"])
	}
}

func TestBedrockCompletionStreamingAndConversationHelperBranches(t *testing.T) {
	t.Parallel()

	output := &bedrockruntime.ConverseOutput{
		Output: &brtypes.ConverseOutputMemberMessage{Value: brtypes.Message{Content: []brtypes.ContentBlock{
			&brtypes.ContentBlockMemberReasoningContent{Value: &brtypes.ReasoningContentBlockMemberReasoningText{Value: brtypes.ReasoningTextBlock{Text: aws.String("think"), Signature: aws.String("sig")}}},
			&brtypes.ContentBlockMemberToolUse{Value: brtypes.ToolUseBlock{ToolUseId: aws.String("tool_1"), Name: aws.String("lookup")}},
		}}},
		Usage: &brtypes.TokenUsage{CacheReadInputTokens: aws.Int32(1), CacheWriteInputTokens: aws.Int32(2)},
	}
	completion := extractBedrockCompletion(output, ExecutionOptions{IncludeOriginalResponse: true, ModelOptions: map[string]any{"include_thoughts": true}})
	if completion.OriginalResponse == nil || len(completion.Result) != 1 || completion.Result[0].Value != "think\n\n" {
		t.Fatalf("completion = %#v", completion)
	}
	if len(completion.ToolUse) != 1 || completion.ToolUse[0].ToolInput != nil || completion.FinishReason != "tool_use" {
		t.Fatalf("tool completion = %#v", completion)
	}
	if completion.TokenUsage == nil || completion.TokenUsage.Prompt != 3 || completion.TokenUsage.PromptCached != 1 || completion.TokenUsage.PromptCacheWrite != 2 {
		t.Fatalf("usage = %#v", completion.TokenUsage)
	}
	empty := extractBedrockCompletion(&bedrockruntime.ConverseOutput{}, ExecutionOptions{})
	if len(empty.Result) != 1 || empty.Result[0].Value != "" {
		t.Fatalf("empty completion = %#v", empty)
	}
	if documentToAny(nil) != nil || bedrockUsage(nil) != nil {
		t.Fatal("nil document or usage should stay nil")
	}

	state := bedrockStreamState{toolBlocks: map[int32]bedrockStreamingToolBlock{}}
	if chunk := state.contentBlockStart(brtypes.ContentBlockStartEvent{}); len(chunk.ToolUse) != 0 || len(chunk.Result) != 0 {
		t.Fatalf("non-tool start = %#v", chunk)
	}
	reasoning := state.contentBlockDelta(brtypes.ContentBlockDeltaEvent{
		Delta: &brtypes.ContentBlockDeltaMemberReasoningContent{Value: &brtypes.ReasoningContentBlockDeltaMemberText{Value: "stream think"}},
	}, ExecutionOptions{ModelOptions: map[string]any{"include_thoughts": true}})
	if len(reasoning.Result) != 1 || reasoning.Result[0].Value != "stream think" {
		t.Fatalf("reasoning delta = %#v", reasoning)
	}
	hidden := state.contentBlockDelta(brtypes.ContentBlockDeltaEvent{
		Delta: &brtypes.ContentBlockDeltaMemberReasoningContent{Value: &brtypes.ReasoningContentBlockDeltaMemberText{Value: "hidden"}},
	}, ExecutionOptions{})
	if len(hidden.Result) != 0 {
		t.Fatalf("hidden reasoning = %#v", hidden)
	}
	if chunk := state.chunk(nil, ExecutionOptions{}); len(chunk.Result) != 0 || chunk.FinishReason != "" {
		t.Fatalf("unknown stream event = %#v", chunk)
	}
	if got := bedrockContentBlockIndex(nil); got != -1 {
		t.Fatalf("nil content block index = %d", got)
	}

	processed := processBedrockToolResultForConversation([]brtypes.ToolResultContentBlock{
		&brtypes.ToolResultContentBlockMemberImage{},
		&brtypes.ToolResultContentBlockMemberDocument{},
		&brtypes.ToolResultContentBlockMemberVideo{},
		&brtypes.ToolResultContentBlockMemberText{Value: "<heartbeat>working</heartbeat>"},
		&brtypes.ToolResultContentBlockMemberJson{Value: brdoc.NewLazyDocument(map[string]any{"ok": true})},
	}, ExecutionOptions{}, true, true)
	texts := make([]string, 0, len(processed))
	for _, block := range processed {
		if text, ok := block.(*brtypes.ToolResultContentBlockMemberText); ok {
			texts = append(texts, text.Value)
		}
	}
	for _, want := range []string{conversationImagePlaceholder, conversationDocumentPlaceholder, conversationVideoPlaceholder, conversationHeartbeatPlaceholder} {
		if !containsText(texts, want) {
			t.Fatalf("missing %q in processed blocks: %#v", want, processed)
		}
	}
}

func TestBedrockToolResultFiles(t *testing.T) {
	t.Parallel()

	runtime := &fakeBedrockRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	_, err := driver.Execute(context.Background(), []PromptSegment{{
		Role:      PromptRoleTool,
		ToolUseID: "tool_1",
		Content:   "tool text",
		Files: []DataSource{
			BytesDataSource{FileName: "data.json", MIME: "application/json", Data: []byte(`{"ok":true}`)},
			BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte("png")},
			BytesDataSource{FileName: "brief.pdf", MIME: "application/pdf", Data: []byte("%PDF")},
			BytesDataSource{FileName: "clip.mp4", MIME: "video/mp4", Data: []byte("mp4"), URI: "s3://bucket/clip.mp4"},
		},
	}}, ExecutionOptions{
		Model: "anthropic.claude-3-5-sonnet-20241022-v2:0",
		Tools: []ToolDefinition{{
			Name:        "lookup",
			Description: "Lookup",
			InputSchema: map[string]any{"type": "object"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if runtime.got == nil || len(runtime.got.Messages) == 0 {
		t.Fatalf("messages = %#v", runtime.got)
	}
	result, ok := runtime.got.Messages[0].Content[0].(*brtypes.ContentBlockMemberToolResult)
	if !ok {
		t.Fatalf("content = %#v", runtime.got.Messages[0].Content)
	}
	var sawText, sawJSON, sawImage, sawDocument, sawVideo bool
	for _, block := range result.Value.Content {
		switch value := block.(type) {
		case *brtypes.ToolResultContentBlockMemberText:
			sawText = value.Value != ""
		case *brtypes.ToolResultContentBlockMemberJson:
			sawJSON = true
		case *brtypes.ToolResultContentBlockMemberImage:
			sawImage = true
		case *brtypes.ToolResultContentBlockMemberDocument:
			sawDocument = true
		case *brtypes.ToolResultContentBlockMemberVideo:
			sawVideo = true
		}
	}
	if !sawText || !sawJSON || !sawImage || !sawDocument || !sawVideo {
		t.Fatalf("tool result content = %#v", result.Value.Content)
	}
}

func TestBedrockMarengoHelperBranches(t *testing.T) {
	t.Parallel()

	textReq, err := bedrockMarengoRequest(context.Background(), EmbeddingInput{Type: embeddingInputText, Text: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if textReq["inputType"] != "text" || textReq["inputText"] != "hello" {
		t.Fatalf("text request = %#v", textReq)
	}
	imageReq, err := bedrockMarengoRequest(context.Background(), EmbeddingInput{Type: embeddingInputImage, Source: BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte("img")}})
	if err != nil {
		t.Fatal(err)
	}
	imageMedia := imageReq["mediaSource"].(map[string]any)
	if imageReq["inputType"] != "image" || imageMedia["base64String"] != "aW1n" {
		t.Fatalf("image request = %#v", imageReq)
	}
	start := 1.0
	length := 2.0
	audioReq, err := bedrockMarengoRequest(context.Background(), EmbeddingInput{
		Type:      embeddingInputAudio,
		Source:    BytesDataSource{FileName: "sound.wav", MIME: "audio/wav", Data: []byte("wav")},
		StartSec:  &start,
		LengthSec: &length,
	})
	if err != nil {
		t.Fatal(err)
	}
	if audioReq["inputType"] != embeddingInputAudio || audioReq["startSec"] != 1.0 || audioReq["lengthSec"] != 2.0 {
		t.Fatalf("audio request = %#v", audioReq)
	}
	_, err = bedrockMarengoRequest(context.Background(), EmbeddingInput{
		Type:            embeddingInputVideo,
		Source:          BytesDataSource{FileName: "clip.mp4", MIME: "video/mp4", Data: []byte("mp4")},
		EmbeddingOption: []string{"visual-text", "audio"},
	})
	if err == nil {
		t.Fatal("expected multiple embedding option error")
	}
	if _, err := bedrockMarengoRequest(context.Background(), EmbeddingInput{Type: "document"}); err == nil {
		t.Fatal("expected unsupported Marengo input error")
	}

	media, err := bedrockMediaSource(context.Background(), BytesDataSource{FileName: "image.png", MIME: "image/png", URI: "https://bucket.s3.us-east-1.amazonaws.com/path/image.png"})
	if err != nil {
		t.Fatal(err)
	}
	if media["s3Location"].(map[string]any)["uri"] != "s3://bucket/path/image.png" {
		t.Fatalf("media source = %#v", media)
	}
	segments, err := bedrockParseMarengoSegments(json.RawMessage(`{"embedding":[0.1],"startSec":0,"endSec":1}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(segments) != 1 || len(segments[0].Embedding) != 1 {
		t.Fatalf("bare segment = %#v", segments)
	}
	if _, err := bedrockParseMarengoSegments(json.RawMessage(`not json`)); err == nil {
		t.Fatal("expected invalid Marengo response error")
	}
	formats := map[string]string{
		"video/mp4":        "mp4",
		"video/quicktime":  "mov",
		"video/x-matroska": "mkv",
		"video/webm":       "webm",
		"video/x-flv":      "flv",
		"video/mpeg":       "mpeg",
		"video/mpg":        "mpg",
		"video/x-ms-wmv":   "wmv",
	}
	for mimeType, want := range formats {
		if got, err := bedrockEmbeddingVideoFormat(mimeType); err != nil || got != want {
			t.Fatalf("bedrockEmbeddingVideoFormat(%q) = %q, %v", mimeType, got, err)
		}
	}
}

func TestBedrockConversationRetentionStripsMediaTextAndHeartbeats(t *testing.T) {
	t.Parallel()

	runtime := &fakeBedrockRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	stripNow := 0
	resp, err := driver.Execute(context.Background(), []PromptSegment{
		{
			Role:    PromptRoleUser,
			Content: "abcdefghijklmnop",
			Files: []DataSource{
				BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte("png")},
				BytesDataSource{FileName: "brief.pdf", MIME: "application/pdf", Data: []byte("%PDF")},
			},
		},
		{Role: PromptRoleAssistant, Content: "<heartbeat>working</heartbeat>"},
	}, ExecutionOptions{
		Model:                     "anthropic.claude-3-5-sonnet-20241022-v2:0",
		StripImagesAfterTurns:     &stripNow,
		StripTextMaxTokens:        2,
		StripHeartbeatsAfterTurns: &stripNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	conversation, ok := resp.Conversation.(bedrockPrompt)
	if !ok {
		t.Fatalf("conversation = %#v", resp.Conversation)
	}
	texts := bedrockConversationTexts(conversation)
	if !containsText(texts, conversationImagePlaceholder) {
		t.Fatalf("image placeholder missing: %#v", texts)
	}
	if !containsText(texts, conversationDocumentPlaceholder) {
		t.Fatalf("document placeholder missing: %#v", texts)
	}
	if !containsText(texts, "abcdefgh"+conversationTextTruncatedMarker) {
		t.Fatalf("truncated text missing: %#v", texts)
	}
	if !containsText(texts, conversationHeartbeatPlaceholder) {
		t.Fatalf("heartbeat placeholder missing: %#v", texts)
	}
}

func TestBedrockS3URIConversion(t *testing.T) {
	t.Parallel()

	tests := map[string]string{
		"s3://bucket/key.mp4": "s3://bucket/key.mp4",
		"https://bucket.s3.us-east-1.amazonaws.com/path/clip.mp4":  "s3://bucket/path/clip.mp4",
		"https://s3.us-east-1.amazonaws.com/bucket/path/clip.mp4":  "s3://bucket/path/clip.mp4",
		"https://example.com/bucket/path/clip.mp4":                 "",
		"https://evil.s3.us-east-1.amazonaws.com.evil.com/key.mp4": "",
	}
	for input, want := range tests {
		if got := bedrockS3URI(input); got != want {
			t.Fatalf("bedrockS3URI(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestBedrockNovaMultimodalEmbeddings(t *testing.T) {
	t.Parallel()

	runtime := &fakeBedrockRuntime{invokeBody: []byte(`{"embeddings":[{"embeddingType":"TEXT","embedding":[0.1,0.2]}]}`)}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	result, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{
		Dimensions: 256,
		TaskType:   EmbeddingTaskQuery,
		Inputs: []EmbeddingInput{
			{Type: embeddingInputText, Text: "find documents"},
			{Type: embeddingInputImage, Source: BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte("img")}},
			{Type: embeddingInputAudio, Source: BytesDataSource{FileName: "sound.mp3", MIME: "audio/mpeg", Data: []byte("mp3"), URI: "https://bucket.s3.us-east-1.amazonaws.com/sound.mp3"}},
			{Type: embeddingInputVideo, Source: BytesDataSource{FileName: "clip.mp4", MIME: "video/mp4", Data: []byte("mp4")}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Model != defaultBedrockEmbeddingModel || len(result.Results) != 4 {
		t.Fatalf("result = %#v", result)
	}
	if len(runtime.gotInvokes) != 4 {
		t.Fatalf("invoke calls = %d", len(runtime.gotInvokes))
	}
	first := bedrockInvokeBody(t, runtime.gotInvokes[0])
	if first["taskType"] != "SINGLE_EMBEDDING" {
		t.Fatalf("taskType = %#v", first["taskType"])
	}
	firstParams := first["singleEmbeddingParams"].(map[string]any)
	if firstParams["embeddingPurpose"] != "GENERIC_RETRIEVAL" || firstParams["embeddingDimension"] != float64(256) {
		t.Fatalf("singleEmbeddingParams = %#v", firstParams)
	}
	text := firstParams["text"].(map[string]any)
	if text["truncationMode"] != "END" || text["value"] != "find documents" {
		t.Fatalf("text params = %#v", text)
	}

	imageParams := bedrockInvokeBody(t, runtime.gotInvokes[1])["singleEmbeddingParams"].(map[string]any)
	image := imageParams["image"].(map[string]any)
	if image["format"] != "png" {
		t.Fatalf("image params = %#v", image)
	}
	imageSource := image["source"].(map[string]any)
	if imageSource["bytes"] != "aW1n" {
		t.Fatalf("image source = %#v", imageSource)
	}

	audioParams := bedrockInvokeBody(t, runtime.gotInvokes[2])["singleEmbeddingParams"].(map[string]any)
	audio := audioParams["audio"].(map[string]any)
	audioSource := audio["source"].(map[string]any)
	if audio["format"] != "mp3" || audioSource["s3Location"].(map[string]any)["uri"] != "s3://bucket/sound.mp3" {
		t.Fatalf("audio params = %#v", audio)
	}

	videoParams := bedrockInvokeBody(t, runtime.gotInvokes[3])["singleEmbeddingParams"].(map[string]any)
	video := videoParams["video"].(map[string]any)
	if video["format"] != "mp4" || video["embeddingMode"] != "AUDIO_VIDEO_COMBINED" {
		t.Fatalf("video params = %#v", video)
	}
}

func TestBedrockTitanImageEmbeddings(t *testing.T) {
	t.Parallel()

	runtime := &fakeBedrockRuntime{invokeBody: []byte(`{"embedding":[0.3,0.4],"inputTextTokenCount":7}`)}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	result, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{
		Model:      "amazon.titan-embed-image-v1",
		Dimensions: 384,
		Inputs: []EmbeddingInput{{
			Type:   embeddingInputImage,
			Source: BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte("img")},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 1 || result.Results[0].Outputs[0].Modality != embeddingInputImage {
		t.Fatalf("result = %#v", result)
	}
	body := bedrockInvokeBody(t, runtime.gotInvoke)
	if body["inputImage"] != "aW1n" {
		t.Fatalf("body = %#v", body)
	}
	config := body["embeddingConfig"].(map[string]any)
	if config["outputEmbeddingLength"] != float64(384) {
		t.Fatalf("embeddingConfig = %#v", config)
	}
	if result.Usage == nil || result.Usage.InputTokens != 7 {
		t.Fatalf("usage = %#v", result.Usage)
	}
}

func TestBedrockCohereEmbeddingsTextAndImage(t *testing.T) {
	t.Parallel()

	runtime := &fakeBedrockRuntime{invokeBody: []byte(`{"embeddings":[[0.1,0.2],[0.3,0.4]]}`)}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	result, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{
		Model:    "cohere.embed-english-v3",
		TaskType: EmbeddingTaskDocument,
		Inputs: []EmbeddingInput{
			{Type: embeddingInputText, Text: "first"},
			{Type: embeddingInputText, Text: "second"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 2 || result.Results[1].Outputs[0].Values[0] != 0.3 {
		t.Fatalf("result = %#v", result)
	}
	body := bedrockInvokeBody(t, runtime.gotInvoke)
	if body["input_type"] != "search_document" {
		t.Fatalf("body = %#v", body)
	}
	texts := body["texts"].([]any)
	if len(texts) != 2 || texts[0] != "first" || texts[1] != "second" {
		t.Fatalf("texts = %#v", texts)
	}

	runtime = &fakeBedrockRuntime{invokeBody: []byte(`{"embeddings":[[0.5,0.6]]}`)}
	driver = NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	result, err = driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{
		Model: "cohere.embed-multilingual-v3",
		Inputs: []EmbeddingInput{{
			Type:   embeddingInputImage,
			Source: BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte("img")},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Results[0].Outputs[0].Modality != embeddingInputImage {
		t.Fatalf("result = %#v", result)
	}
	body = bedrockInvokeBody(t, runtime.gotInvoke)
	if body["input_type"] != "image" {
		t.Fatalf("body = %#v", body)
	}
	images := body["images"].([]any)
	if len(images) != 1 || images[0] != "data:image/png;base64,aW1n" {
		t.Fatalf("images = %#v", images)
	}
}

func TestBedrockMarengoEmbeddings(t *testing.T) {
	t.Parallel()

	runtime := &fakeBedrockRuntime{invokeBody: []byte(`{"embeddings":[{"embedding":[0.1,0.2],"startSec":1,"endSec":2,"embeddingOption":"visual-text"},{"embedding":[0.3,0.4],"startSec":2,"endSec":3,"embeddingOption":"audio"}]}`)}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	start := 1.0
	length := 4.0
	fixed := true
	minClip := 2.0
	result, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{
		Model: "twelvelabs.marengo-embed-2-7-v1:0",
		Inputs: []EmbeddingInput{{
			Type:            embeddingInputVideo,
			Source:          BytesDataSource{FileName: "clip.mp4", MIME: "video/mp4", Data: []byte("mp4"), URI: "s3://bucket/clip.mp4"},
			StartSec:        &start,
			LengthSec:       &length,
			UseFixedLength:  &fixed,
			MinClipSec:      &minClip,
			EmbeddingOption: []string{"visual-text"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 1 || len(result.Results[0].Outputs) != 2 {
		t.Fatalf("result = %#v", result)
	}
	if result.Results[0].Outputs[0].StartSec == nil || *result.Results[0].Outputs[0].StartSec != 1 {
		t.Fatalf("outputs = %#v", result.Results[0].Outputs)
	}
	body := bedrockInvokeBody(t, runtime.gotInvoke)
	if body["inputType"] != "video" || body["startSec"] != float64(1) || body["lengthSec"] != float64(4) {
		t.Fatalf("body = %#v", body)
	}
	if body["embeddingOption"] != "visual-text" || body["useFixedLengthSec"] != true || body["minClipSec"] != float64(2) {
		t.Fatalf("body = %#v", body)
	}
	media := body["mediaSource"].(map[string]any)
	if media["s3Location"].(map[string]any)["uri"] != "s3://bucket/clip.mp4" {
		t.Fatalf("mediaSource = %#v", media)
	}
}

func TestBedrockStreamUsesConverseStream(t *testing.T) {
	t.Parallel()

	runtime := &fakeBedrockRuntime{streamEvents: []brtypes.ConverseStreamOutput{
		&brtypes.ConverseStreamOutputMemberContentBlockStart{Value: brtypes.ContentBlockStartEvent{
			ContentBlockIndex: aws.Int32(0),
			Start: &brtypes.ContentBlockStartMemberToolUse{Value: brtypes.ToolUseBlockStart{
				ToolUseId: aws.String("tool_1"),
				Name:      aws.String("lookup"),
			}},
		}},
		&brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(1),
			Delta:             &brtypes.ContentBlockDeltaMemberText{Value: "hello "},
		}},
		&brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(1),
			Delta:             &brtypes.ContentBlockDeltaMemberText{Value: "world"},
		}},
		&brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(0),
			Delta:             &brtypes.ContentBlockDeltaMemberToolUse{Value: brtypes.ToolUseBlockDelta{Input: aws.String(`{"q"`)}},
		}},
		&brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(0),
			Delta:             &brtypes.ContentBlockDeltaMemberToolUse{Value: brtypes.ToolUseBlockDelta{Input: aws.String(`:"tokyo"}`)}},
		}},
		&brtypes.ConverseStreamOutputMemberMessageStop{Value: brtypes.MessageStopEvent{StopReason: brtypes.StopReasonToolUse}},
		&brtypes.ConverseStreamOutputMemberMetadata{Value: brtypes.ConverseStreamMetadataEvent{Usage: &brtypes.TokenUsage{
			InputTokens:  aws.Int32(2),
			OutputTokens: aws.Int32(3),
			TotalTokens:  aws.Int32(5),
		}}},
	}}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	stream, err := driver.Stream(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "System"},
		{Role: PromptRoleUser, Content: "Hi"},
	}, ExecutionOptions{
		Model:        "anthropic.claude-3-5-sonnet-20241022-v2:0",
		ModelOptions: map[string]any{"max_tokens": 128},
		Tools: []ToolDefinition{{
			Name:        "lookup",
			Description: "Lookup",
			InputSchema: map[string]any{"type": "object"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}

	var chunks []CompletionChunk
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}

	if runtime.gotStream == nil {
		t.Fatal("ConverseStream was not called")
	}
	if len(chunks) != 7 {
		t.Fatalf("chunks = %#v", chunks)
	}
	completion := stream.Completion()
	if completion == nil {
		t.Fatal("completion is nil")
	}
	if len(completion.Result) != 1 || completion.Result[0].Value != "hello world" {
		t.Fatalf("completion result = %#v", completion.Result)
	}
	if len(completion.ToolUse) != 1 || completion.ToolUse[0].ToolName != "lookup" {
		t.Fatalf("completion tool use = %#v", completion.ToolUse)
	}
	input, ok := completion.ToolUse[0].ToolInput.(map[string]any)
	if !ok || input["q"] != "tokyo" {
		t.Fatalf("completion tool input = %#v", completion.ToolUse[0].ToolInput)
	}
	if completion.FinishReason != "tool_use" {
		t.Fatalf("finish reason = %q", completion.FinishReason)
	}
	if completion.TokenUsage == nil || completion.TokenUsage.Total != 5 {
		t.Fatalf("token usage = %#v", completion.TokenUsage)
	}
}

func TestBedrockConversationToolBlocksBecomeTextWithoutTools(t *testing.T) {
	t.Parallel()

	runtime := &fakeBedrockRuntime{}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	_, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "summarize"}}, ExecutionOptions{
		Model: "anthropic.claude-3-5-sonnet-20241022-v2:0",
		Conversation: bedrockPrompt{Messages: []brtypes.Message{{
			Role: brtypes.ConversationRoleAssistant,
			Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberToolUse{Value: brtypes.ToolUseBlock{
				ToolUseId: aws.String("tool_1"),
				Name:      aws.String("lookup"),
				Input:     brdoc.NewLazyDocument(map[string]any{"q": "tokyo"}),
			}}},
		}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if runtime.got == nil {
		t.Fatal("Converse was not called")
	}
	if runtime.got.ToolConfig != nil {
		t.Fatalf("tool config should not be set: %#v", runtime.got.ToolConfig)
	}
	if len(runtime.got.Messages) == 0 || len(runtime.got.Messages[0].Content) == 0 {
		t.Fatalf("messages = %#v", runtime.got.Messages)
	}
	if _, ok := runtime.got.Messages[0].Content[0].(*brtypes.ContentBlockMemberText); !ok {
		t.Fatalf("tool block was not converted: %#v", runtime.got.Messages[0].Content[0])
	}
}

func TestBedrockImageGenerationUsesInvokeModel(t *testing.T) {
	t.Parallel()

	runtime := &fakeBedrockRuntime{invokeBody: []byte(`{"images":["aW1n"]}`)}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	resp, err := driver.Execute(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "Editorial image"},
		{Role: PromptRoleUser, Content: "A cube"},
		{Role: PromptRoleNegative, Content: "blur"},
	}, ExecutionOptions{
		Model:        "amazon.nova-canvas-v1:0",
		ModelOptions: map[string]any{"numberOfImages": 1, "width": 512, "height": 512, "cfgScale": 7.5},
	})
	if err != nil {
		t.Fatal(err)
	}
	if runtime.gotInvoke == nil {
		t.Fatal("InvokeModel was not called")
	}
	if aws.ToString(runtime.gotInvoke.ModelId) != "amazon.nova-canvas-v1:0" {
		t.Fatalf("model = %s", aws.ToString(runtime.gotInvoke.ModelId))
	}
	var got map[string]any
	if err := json.Unmarshal(runtime.gotInvoke.Body, &got); err != nil {
		t.Fatal(err)
	}
	if got["taskType"] != "TEXT_IMAGE" {
		t.Fatalf("taskType = %v", got["taskType"])
	}
	params := got["textToImageParams"].(map[string]any)
	if params["text"] != "A cube\n\n\nIMPORTANT: Editorial image" || params["negativeText"] != "blur" {
		t.Fatalf("textToImageParams = %#v", params)
	}
	config := got["imageGenerationConfig"].(map[string]any)
	if config["width"] != float64(512) || config["cfgScale"] != 7.5 {
		t.Fatalf("imageGenerationConfig = %#v", config)
	}
	if len(resp.Result) != 1 || resp.Result[0].Type != ResultTypeImage || resp.Result[0].Value != "data:image/png;base64,aW1n" {
		t.Fatalf("result = %#v", resp.Result)
	}
}

func TestBedrockFormatHelpers(t *testing.T) {
	t.Parallel()

	imageFormats := map[string]brtypes.ImageFormat{
		"image/jpeg": brtypes.ImageFormatJpeg,
		"image/jpg":  brtypes.ImageFormatJpeg,
		"image/gif":  brtypes.ImageFormatGif,
		"image/webp": brtypes.ImageFormatWebp,
		"image/png":  brtypes.ImageFormatPng,
		"":           brtypes.ImageFormatPng,
	}
	for mimeType, want := range imageFormats {
		if got := bedrockImageFormat(mimeType); got != want {
			t.Fatalf("bedrockImageFormat(%q) = %s, want %s", mimeType, got, want)
		}
	}

	videoFormats := map[string]brtypes.VideoFormat{
		"video/quicktime":  brtypes.VideoFormatMov,
		"video/x-matroska": brtypes.VideoFormatMkv,
		"video/webm":       brtypes.VideoFormatWebm,
		"video/x-flv":      brtypes.VideoFormatFlv,
		"video/mpeg":       brtypes.VideoFormatMpeg,
		"video/mpg":        brtypes.VideoFormatMpg,
		"video/x-ms-wmv":   brtypes.VideoFormatWmv,
		"video/3gpp":       brtypes.VideoFormatThreeGp,
		"video/mp4":        brtypes.VideoFormatMp4,
	}
	for mimeType, want := range videoFormats {
		if got := bedrockVideoFormat(mimeType); got != want {
			t.Fatalf("bedrockVideoFormat(%q) = %s, want %s", mimeType, got, want)
		}
	}

	documentCases := []struct {
		source BytesDataSource
		want   brtypes.DocumentFormat
		ok     bool
	}{
		{BytesDataSource{FileName: "brief.pdf", MIME: "application/pdf"}, brtypes.DocumentFormatPdf, true},
		{BytesDataSource{FileName: "table.csv", MIME: "text/csv"}, brtypes.DocumentFormatCsv, true},
		{BytesDataSource{FileName: "README.md", MIME: "text/markdown"}, brtypes.DocumentFormatMd, true},
		{BytesDataSource{FileName: "notes.txt", MIME: "text/plain"}, brtypes.DocumentFormatTxt, true},
		{BytesDataSource{FileName: "legacy.doc", MIME: "application/msword"}, brtypes.DocumentFormatDoc, true},
		{BytesDataSource{FileName: "modern.docx", MIME: "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}, brtypes.DocumentFormatDocx, true},
		{BytesDataSource{FileName: "legacy.xls", MIME: "application/vnd.ms-excel"}, brtypes.DocumentFormatXls, true},
		{BytesDataSource{FileName: "modern.xlsx", MIME: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}, brtypes.DocumentFormatXlsx, true},
		{BytesDataSource{FileName: "page.htm"}, brtypes.DocumentFormatHtml, true},
		{BytesDataSource{FileName: "archive.bin", MIME: "application/octet-stream"}, "", false},
	}
	for _, tc := range documentCases {
		got, ok := bedrockDocumentFormat(tc.source)
		if ok != tc.ok || got != tc.want {
			t.Fatalf("bedrockDocumentFormat(%s) = %s, %v", tc.source.FileName, got, ok)
		}
	}
	if got := bedrockDocumentName("  % weird\tname!.pdf  "); got != "weird namepdf" {
		t.Fatalf("document name = %q", got)
	}
	if got := bedrockDocumentName("!!!"); got != "document" {
		t.Fatalf("empty document name = %q", got)
	}
}

func TestBedrockReasoningFinishAndRetryHelpers(t *testing.T) {
	t.Parallel()

	if got := bedrockFinishReason(brtypes.StopReasonEndTurn); got != "stop" {
		t.Fatalf("end turn = %q", got)
	}
	if got := bedrockFinishReason(brtypes.StopReasonMaxTokens); got != "length" {
		t.Fatalf("max tokens = %q", got)
	}
	if got := bedrockFinishReason(brtypes.StopReasonToolUse); got != "tool_use" {
		t.Fatalf("tool use = %q", got)
	}
	if got := bedrockFinishReason(brtypes.StopReason("guardrail_intervened")); got != "guardrail_intervened" {
		t.Fatalf("custom stop = %q", got)
	}
	if !bedrockIncludeThoughts(ExecutionOptions{Model: "deepseek.r1-v1"}) {
		t.Fatal("DeepSeek R1 should include thoughts")
	}
	if !bedrockIncludeThoughts(ExecutionOptions{ModelOptions: map[string]any{"include_thoughts": true}}) {
		t.Fatal("include_thoughts option should include thoughts")
	}
	if bedrockIncludeThoughts(ExecutionOptions{Model: "anthropic.claude-3-haiku"}) {
		t.Fatal("thoughts should be omitted by default")
	}
	if got := bedrockReasoningDeltaText(&brtypes.ReasoningContentBlockDeltaMemberText{Value: "think"}); got != "think" {
		t.Fatalf("reasoning delta text = %q", got)
	}
	if got := bedrockReasoningDeltaText(&brtypes.ReasoningContentBlockDeltaMemberRedactedContent{Value: []byte("secret")}); got != "[Redacted thinking: secret]" {
		t.Fatalf("reasoning delta redacted = %q", got)
	}
	if got := bedrockReasoningDeltaText(&brtypes.ReasoningContentBlockDeltaMemberSignature{Value: "sig"}); got != "\n\n" {
		t.Fatalf("reasoning delta signature = %q", got)
	}
	if got := bedrockReasoningContentText(&brtypes.ReasoningContentBlockMemberReasoningText{Value: brtypes.ReasoningTextBlock{Text: aws.String("full")}}); got != "full" {
		t.Fatalf("reasoning content = %q", got)
	}
	if got := bedrockReasoningContentText(&brtypes.ReasoningContentBlockMemberReasoningText{Value: brtypes.ReasoningTextBlock{Text: aws.String("full"), Signature: aws.String("sig")}}); got != "full\n\n" {
		t.Fatalf("reasoning signed content = %q", got)
	}
	if got := bedrockReasoningContentText(&brtypes.ReasoningContentBlockMemberRedactedContent{Value: []byte("sealed")}); got != "[Redacted thinking: sealed]" {
		t.Fatalf("reasoning redacted content = %q", got)
	}
	assertRetryable(t, bedrockRetryable("ThrottlingException", 0, ""), true)
	assertRetryable(t, bedrockRetryable("ValidationException", 0, ""), false)
	assertRetryable(t, bedrockRetryable("Other", 0, "server"), true)
	assertRetryable(t, bedrockRetryable("Other", 0, "client"), false)
	assertRetryable(t, bedrockRetryable("Other", 503, ""), true)
	if got := bedrockRetryable("Other", 0, ""); got != nil {
		t.Fatalf("unknown retryability = %#v", got)
	}
}

func TestBedrockImagePayloadHelpers(t *testing.T) {
	t.Parallel()

	prompt := bedrockImagePrompt{
		Text:     "draw",
		System:   "editorial",
		Negative: "blur",
		Images:   []string{"image-base64"},
		Masks:    []string{"mask-base64"},
	}
	if got := bedrockImageText(prompt); got != "draw\n\n\nIMPORTANT: editorial" {
		t.Fatalf("image text = %q", got)
	}
	if got := optionStringDefault(map[string]any{"mode": "CUSTOM"}, "mode", "DEFAULT"); got != "CUSTOM" {
		t.Fatalf("optionStringDefault set = %q", got)
	}
	if got := optionStringDefault(nil, "mode", "DEFAULT"); got != "DEFAULT" {
		t.Fatalf("optionStringDefault fallback = %q", got)
	}
	stable := bedrockStableDiffusionPayload(prompt, ExecutionOptions{ModelOptions: map[string]any{"cfgScale": 7.5, "seed": 1, "steps": 30, "width": 512, "height": 512}})
	prompts := stable["text_prompts"].([]map[string]any)
	if len(prompts) != 2 || prompts[1]["weight"] != -1 || stable["cfg_scale"] != 7.5 {
		t.Fatalf("stable payload = %#v", stable)
	}

	tasks := map[string]string{
		"IMAGE_VARIATION":         "imageVariationParams",
		"COLOR_GUIDED_GENERATION": "colorGuidedGenerationParams",
		"BACKGROUND_REMOVAL":      "backgroundRemovalParams",
		"INPAINTING":              "inPaintingParams",
		"OUTPAINTING":             "outPaintingParams",
		"TEXT_IMAGE":              "textToImageParams",
	}
	for taskType, key := range tasks {
		payload := bedrockNovaTitanImagePayload(prompt, ExecutionOptions{ModelOptions: map[string]any{
			"taskType":        taskType,
			"colors":          []any{"red", "blue"},
			"outPaintingMode": "PRECISE",
		}})
		if payload["taskType"] != taskType || payload[key] == nil {
			t.Fatalf("%s payload = %#v", taskType, payload)
		}
	}
	if got := firstString(nil); got != nil {
		t.Fatalf("firstString nil = %#v", got)
	}
	if got := firstString([]string{"x"}); got != "x" {
		t.Fatalf("firstString = %#v", got)
	}
}

func TestBedrockModelAndProviderHelpers(t *testing.T) {
	t.Parallel()

	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, &fakeBedrockRuntime{}, &fakeBedrockModels{})
	if driver.Provider() != ProviderBedrock {
		t.Fatalf("provider = %s", driver.Provider())
	}
	if err := driver.ValidateConnection(context.Background()); err != nil {
		t.Fatal(err)
	}
	if bedrockModelSupported(bedrocktypes.FoundationModelSummary{ProviderName: aws.String("Unsupported")}) {
		t.Fatal("unsupported publisher should be filtered")
	}
	if bedrockModelSupported(bedrocktypes.FoundationModelSummary{
		ProviderName:            aws.String("Anthropic"),
		InferenceTypesSupported: []bedrocktypes.InferenceType{bedrocktypes.InferenceTypeProvisioned},
	}) {
		t.Fatal("model without on-demand inference should be filtered")
	}
	if bedrockModelSupported(bedrocktypes.FoundationModelSummary{
		ProviderName:     aws.String("Amazon"),
		OutputModalities: []bedrocktypes.ModelModality{bedrocktypes.ModelModalityEmbedding},
	}) {
		t.Fatal("embedding foundation model should be filtered")
	}
	if bedrockModelSupported(bedrocktypes.FoundationModelSummary{
		ModelId:      aws.String("cohere.rerank-v3"),
		ProviderName: aws.String("Cohere"),
	}) {
		t.Fatal("unsupported model family should be filtered")
	}
	if !bedrockModelSupported(bedrocktypes.FoundationModelSummary{
		ModelId:      aws.String("anthropic.claude-3-5-sonnet"),
		ProviderName: aws.String("Anthropic"),
	}) {
		t.Fatal("supported Claude model was filtered")
	}
	modalities := bedrockModalities([]bedrocktypes.ModelModality{bedrocktypes.ModelModalityText, bedrocktypes.ModelModalityImage, bedrocktypes.ModelModalityEmbedding, bedrocktypes.ModelModality("AUDIO")})
	if len(modalities) != 4 || modalities[3] != "audio" {
		t.Fatalf("modalities = %#v", modalities)
	}
	if got := bedrockCapabilitiesInput("amazon.nova-canvas-v1:0"); len(got) != 2 || got[1] != "image" {
		t.Fatalf("image input capabilities = %#v", got)
	}
	if got := bedrockCapabilitiesInput("amazon.nova-embed-v1"); len(got) != 4 {
		t.Fatalf("nova embed capabilities = %#v", got)
	}
	if got := bedrockCapabilitiesInput("amazon.titan-embed-image-v1"); len(got) != 2 || got[1] != "image" {
		t.Fatalf("titan image capabilities = %#v", got)
	}
	if got := bedrockCapabilitiesOutput("amazon.nova-canvas-v1:0"); len(got) != 1 || got[0] != "image" {
		t.Fatalf("image output capabilities = %#v", got)
	}
	if got := bedrockCapabilitiesOutput("amazon.embed-v1"); len(got) != 1 || got[0] != "embedding" {
		t.Fatalf("embedding output capabilities = %#v", got)
	}
	if got := bedrockUnsupportedModels("Amazon"); len(got) == 0 || got[0] != "titan-image-generator" {
		t.Fatalf("amazon unsupported = %#v", got)
	}
	if got := bedrockUnsupportedModels("Unknown"); got != nil {
		t.Fatalf("unknown unsupported = %#v", got)
	}
	if got := bedrockCohereInputType(EmbeddingTaskQuery); got != "search_query" {
		t.Fatalf("cohere query = %q", got)
	}
	if got := bedrockCohereInputType(""); got != "" {
		t.Fatalf("cohere default = %q", got)
	}
}

func TestBedrockHTTPTimeoutRuntimeOptions(t *testing.T) {
	t.Parallel()

	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, nil, nil)
	if got := driver.runtimeOptions(HTTPTimeoutOptions{}); got != nil {
		t.Fatalf("empty runtime options = %#v", got)
	}
	opts := driver.runtimeOptions(HTTPTimeoutOptions{BodyTimeout: 3 * time.Second})
	if len(opts) != 1 {
		t.Fatalf("runtime options = %#v", opts)
	}
	var runtimeOptions bedrockruntime.Options
	opts[0](&runtimeOptions)
	client, ok := runtimeOptions.HTTPClient.(*http.Client)
	if !ok || client.Timeout != 3*time.Second {
		t.Fatalf("runtime HTTP client = %#v", runtimeOptions.HTTPClient)
	}
}

func TestBedrockAdditionalFieldsAndErrorFormatting(t *testing.T) {
	t.Parallel()

	additional := bedrockAdditionalFields(ExecutionOptions{
		Model: "anthropic.claude-sonnet-4-6-20260601",
		ModelOptions: map[string]any{
			"effort":           "high",
			"include_thoughts": true,
			"top_k":            10,
		},
	})
	if additional["top_k"] != nil || additional["output_config"].(map[string]any)["effort"] != "high" {
		t.Fatalf("adaptive fields = %#v", additional)
	}
	reasoning := additional["reasoning_config"].(map[string]any)
	if reasoning["type"] != "adaptive" || reasoning["display"] != "summarized" {
		t.Fatalf("reasoning = %#v", reasoning)
	}
	additional = bedrockAdditionalFields(ExecutionOptions{
		Model:        "mistral.large",
		ModelOptions: map[string]any{"top_k": 40, "reasoning_effort": "medium"},
	})
	if additional["top_k"] != 40 || additional["reasoning_effort"] != "medium" {
		t.Fatalf("non-Claude fields = %#v", additional)
	}

	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, nil, nil)
	wrapped := driver.formatError(fakeBedrockAPIError{code: "ThrottlingException", message: "slow", fault: "server"}, "model", "execute")
	var llumiErr *common.LlumiverseError
	if !errors.As(wrapped, &llumiErr) || llumiErr.Name != "ThrottlingException" || llumiErr.Retryable == nil || !*llumiErr.Retryable {
		t.Fatalf("wrapped error = %#v", wrapped)
	}
	if err := driver.formatError(errors.New("plain"), "model", "execute"); err.Error() != "plain" {
		t.Fatalf("plain error = %v", err)
	}
}

func TestBedrockEmbeddingFormatErrors(t *testing.T) {
	t.Parallel()

	if got, err := bedrockEmbeddingImageFormat("image/jpeg"); err != nil || got != "jpeg" {
		t.Fatalf("image format = %q, %v", got, err)
	}
	if _, err := bedrockEmbeddingImageFormat("image/bmp"); err == nil {
		t.Fatal("expected unsupported image format error")
	}
	if got, err := bedrockEmbeddingAudioFormat("audio/x-wav"); err != nil || got != "wav" {
		t.Fatalf("audio format = %q, %v", got, err)
	}
	if _, err := bedrockEmbeddingAudioFormat("audio/flac"); err == nil {
		t.Fatal("expected unsupported audio format error")
	}
	if got, err := bedrockEmbeddingVideoFormat("video/3gpp"); err != nil || got != "3gp" {
		t.Fatalf("video format = %q, %v", got, err)
	}
	if _, err := bedrockEmbeddingVideoFormat("video/unknown"); err == nil {
		t.Fatal("expected unsupported video format error")
	}
}

func bedrockConversationTexts(conversation bedrockPrompt) []string {
	var out []string
	for _, msg := range conversation.Messages {
		for _, block := range msg.Content {
			switch value := block.(type) {
			case *brtypes.ContentBlockMemberText:
				out = append(out, value.Value)
			case *brtypes.ContentBlockMemberToolResult:
				for _, result := range value.Value.Content {
					if text, ok := result.(*brtypes.ToolResultContentBlockMemberText); ok {
						out = append(out, text.Value)
					}
				}
			}
		}
	}
	return out
}

func bedrockInvokeBody(t *testing.T, input *bedrockruntime.InvokeModelInput) map[string]any {
	t.Helper()
	if input == nil {
		t.Fatal("InvokeModel was not called")
	}
	var got map[string]any
	if err := json.Unmarshal(input.Body, &got); err != nil {
		t.Fatal(err)
	}
	return got
}

func containsText(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func assertRetryable(t *testing.T, got *bool, want bool) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("retryable = %#v, want %v", got, want)
	}
}

type fakeBedrockAPIError struct {
	code    string
	message string
	fault   string
}

func (e fakeBedrockAPIError) Error() string {
	return e.message
}

func (e fakeBedrockAPIError) ErrorCode() string {
	return e.code
}

func (e fakeBedrockAPIError) ErrorMessage() string {
	return e.message
}

func (e fakeBedrockAPIError) ErrorFault() string {
	return e.fault
}
