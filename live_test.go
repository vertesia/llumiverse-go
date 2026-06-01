package llumiverse

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

const (
	liveShortTimeout = 90 * time.Second
	liveLongTimeout  = 5 * time.Minute
	tinyPNGB64       = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADklEQVQI12P4z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)

type liveDriverCase struct {
	name        string
	driver      Driver
	textModel   string
	visionModel string
	toolModel   string
	claude      bool
}

func TestLiveListModelsAndBasicCompletion(t *testing.T) {
	skipUnlessLiveTestsRequested(t)
	cases := liveDriverCases(t)
	if len(cases) == 0 {
		t.Skip("set OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_PROJECT_ID/GOOGLE_REGION, or BEDROCK_REGION to run live provider tests")
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := liveContext(t, liveShortTimeout)
			if err := tc.driver.ValidateConnection(ctx); err != nil {
				t.Fatalf("validate connection: %v", err)
			}
			models, err := tc.driver.ListModels(ctx, nil)
			if err != nil {
				t.Fatalf("list models: %v", err)
			}
			if len(models) == 0 {
				if tc.driver.Provider() == ProviderVertexAI {
					t.Log("Gemini Enterprise Agent Platform returned no listable publisher models for this project/region; continuing with direct model calls")
				} else {
					t.Fatal("expected at least one listed model")
				}
			}
			prompt := []PromptSegment{{Role: PromptRoleUser, Content: "What color is the sky? Reply with one short sentence."}}
			if prepared, err := tc.driver.CreatePrompt(ctx, prompt, ExecutionOptions{Model: tc.textModel}); err != nil || prepared == nil {
				t.Fatalf("create prompt = %#v, %v", prepared, err)
			}

			resp, err := tc.driver.Execute(ctx, prompt, liveTextOptions(tc.textModel))
			if err != nil {
				t.Fatalf("execute: %v", err)
			}
			assertLiveCompletion(t, resp)

			stream, err := tc.driver.Stream(ctx, prompt, liveTextOptions(tc.textModel))
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			text := consumeLiveStream(t, stream)
			if len(strings.TrimSpace(text)) <= 2 {
				t.Fatalf("stream text too short: %q", text)
			}
			assertLiveCompletion(t, stream.Completion())

			schemaResp, err := tc.driver.Execute(ctx, []PromptSegment{{
				Role:    PromptRoleUser,
				Content: `Return JSON for the sky color, for example {"color":"blue"}.`,
			}}, liveSchemaOptions(tc.textModel))
			if err != nil {
				t.Fatalf("execute with schema: %v", err)
			}
			assertLiveCompletion(t, schemaResp)
			if schemaResp.Result[0].Type != ResultTypeJSON {
				t.Fatalf("expected normalized JSON result, got %#v", schemaResp.Result)
			}
		})
	}
}

func TestLiveMultiTurnConversations(t *testing.T) {
	skipUnlessLiveTestsRequested(t)
	cases := liveDriverCases(t)
	if len(cases) == 0 {
		t.Skip("set live provider credentials to run multi-turn live tests")
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := liveContext(t, liveShortTimeout)
			options := liveTextOptions(tc.textModel)
			turn1, err := tc.driver.Execute(ctx, []PromptSegment{{
				Role:    PromptRoleUser,
				Content: "I am thinking of a number between 1 and 10. The number is 7. Remember this number.",
			}}, options)
			if err != nil {
				t.Fatalf("turn 1: %v", err)
			}
			verifyLiveConversationSerializable(t, turn1.Conversation)

			turn2, err := tc.driver.Execute(ctx, []PromptSegment{{
				Role:    PromptRoleUser,
				Content: "What number did I tell you I was thinking of?",
			}}, withConversation(options, turn1.Conversation))
			if err != nil {
				t.Fatalf("turn 2: %v", err)
			}
			if !strings.Contains(resultText(turn2.Result), "7") {
				t.Fatalf("turn 2 did not preserve context: %#v", turn2.Result)
			}
			verifyLiveConversationSerializable(t, turn2.Conversation)

			turn3, err := tc.driver.Execute(ctx, []PromptSegment{{
				Role:    PromptRoleUser,
				Content: "Add 3 to that number. What is the result?",
			}}, withConversation(options, turn2.Conversation))
			if err != nil {
				t.Fatalf("turn 3: %v", err)
			}
			if !strings.Contains(strings.ToLower(resultText(turn3.Result)), "10") && !strings.Contains(strings.ToLower(resultText(turn3.Result)), "ten") {
				t.Fatalf("turn 3 did not preserve context: %#v", turn3.Result)
			}
			verifyLiveConversationSerializable(t, turn3.Conversation)
		})
	}
}

func TestLiveStreamingConversationContext(t *testing.T) {
	skipUnlessLiveTestsRequested(t)
	cases := liveDriverCases(t)
	if len(cases) == 0 {
		t.Skip("set live provider credentials to run streaming conversation live tests")
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := liveContext(t, liveShortTimeout)
			options := liveTextOptions(tc.textModel)
			stream1, err := tc.driver.Stream(ctx, []PromptSegment{{
				Role:    PromptRoleUser,
				Content: "Remember this code: ALPHA-7749.",
			}}, options)
			if err != nil {
				t.Fatalf("stream turn 1: %v", err)
			}
			_ = consumeLiveStream(t, stream1)
			verifyLiveConversationSerializable(t, stream1.Completion().Conversation)

			stream2, err := tc.driver.Stream(ctx, []PromptSegment{{
				Role:    PromptRoleUser,
				Content: "What code did I tell you to remember?",
			}}, withConversation(options, stream1.Completion().Conversation))
			if err != nil {
				t.Fatalf("stream turn 2: %v", err)
			}
			_ = consumeLiveStream(t, stream2)
			text2 := resultText(stream2.Completion().Result)
			if !strings.Contains(text2, "ALPHA-7749") && !strings.Contains(text2, "7749") {
				t.Fatalf("stream turn 2 did not preserve context: %#v", stream2.Completion().Result)
			}

			turn3, err := tc.driver.Execute(ctx, []PromptSegment{{
				Role:    PromptRoleUser,
				Content: "What were the numbers in that code?",
			}}, withConversation(options, stream2.Completion().Conversation))
			if err != nil {
				t.Fatalf("execute after stream: %v", err)
			}
			if !strings.Contains(resultText(turn3.Result), "7749") {
				t.Fatalf("execute after stream did not preserve context: %#v", turn3.Result)
			}
		})
	}
}

func TestLiveToolCallingAndCheckpointConversion(t *testing.T) {
	skipUnlessLiveTestsRequested(t)
	cases := liveDriverCases(t)
	if len(cases) == 0 {
		t.Skip("set live provider credentials to run live tool tests")
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := liveContext(t, liveShortTimeout)
			options := liveToolOptions(tc.toolModel)
			toolTurn, err := tc.driver.Execute(ctx, []PromptSegment{{
				Role:    PromptRoleUser,
				Content: "What is the weather in Paris? Use the get_weather tool.",
			}}, options)
			if err != nil {
				t.Fatalf("tool turn: %v", err)
			}
			if len(toolTurn.ToolUse) == 0 {
				t.Fatalf("expected tool use, got %#v", toolTurn)
			}
			if toolTurn.ToolUse[0].ID == "" || toolTurn.ToolUse[0].ToolName != "get_weather" || toolTurn.ToolUse[0].ToolInput == nil {
				t.Fatalf("tool use = %#v", toolTurn.ToolUse)
			}

			continued, err := tc.driver.Execute(ctx, []PromptSegment{{
				Role:      PromptRoleTool,
				ToolUseID: toolTurn.ToolUse[0].ID,
				Content:   "15 degrees celsius, sunny",
			}}, withConversation(options, toolTurn.Conversation))
			if err != nil {
				t.Fatalf("tool result turn: %v", err)
			}
			if !strings.Contains(resultText(continued.Result), "15") {
				t.Fatalf("tool result was not used: %#v", continued.Result)
			}

			checkpoint, err := tc.driver.Execute(ctx, []PromptSegment{{
				Role:    PromptRoleUser,
				Content: "Summarize what happened in this conversation. Do not call tools; output text only.",
			}}, ExecutionOptions{
				Model:        tc.toolModel,
				ModelOptions: liveNonReasoningModelOptions(256),
				Tools:        []ToolDefinition{},
				Conversation: continued.Conversation,
			})
			if err != nil {
				t.Fatalf("checkpoint turn: %v", err)
			}
			if checkpoint.FinishReason == "tool_use" {
				t.Fatalf("checkpoint unexpectedly requested a tool: %#v", checkpoint)
			}
			assertLiveCompletion(t, checkpoint)
		})
	}
}

func TestLiveEmbeddings(t *testing.T) {
	skipUnlessLiveTestsRequested(t)
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		t.Run("openai", func(t *testing.T) {
			endpoint := getenvDefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
			driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: apiKey, Endpoint: endpoint})
			if err != nil {
				t.Fatal(err)
			}
			ctx := liveContext(t, liveShortTimeout)
			result, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{Inputs: []EmbeddingInput{{Text: "Hello"}}})
			if err != nil {
				t.Fatalf("text embedding: %v", err)
			}
			assertEmbeddingResult(t, result, 1)
			if result.Model != "text-embedding-3-small" {
				t.Fatalf("model = %q", result.Model)
			}
			batch, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{Inputs: []EmbeddingInput{{Text: "alpha"}, {Text: "beta"}}})
			if err != nil {
				t.Fatalf("batch embedding: %v", err)
			}
			assertEmbeddingResult(t, batch, 2)
		})
	}

	if project, region, ok := googleLiveProjectRegion(); ok {
		t.Run("vertexai", func(t *testing.T) {
			driver, err := NewVertexAIDriver(VertexAIOptions{Project: project, Region: region})
			if err != nil {
				t.Fatal(err)
			}
			ctx := liveContext(t, liveShortTimeout)
			textModel := getenvDefault("VERTEX_LIVE_TEXT_EMBEDDING_MODEL", "text-embedding-005")
			result, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{
				Model:    textModel,
				TaskType: EmbeddingTaskDocument,
				Inputs:   []EmbeddingInput{{Text: "The quick brown fox jumps over the lazy dog"}},
			})
			if err != nil {
				t.Fatalf("%s document: %v", textModel, err)
			}
			assertEmbeddingResult(t, result, 1)

			query, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{
				Model:    textModel,
				TaskType: EmbeddingTaskQuery,
				Inputs:   []EmbeddingInput{{Text: "what is the capital of France?"}},
			})
			if err != nil {
				t.Fatalf("%s query: %v", textModel, err)
			}
			assertEmbeddingResult(t, query, 1)

			dim, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{
				Model:      textModel,
				Dimensions: 128,
				Inputs:     []EmbeddingInput{{Text: "dimension test"}},
			})
			if err != nil {
				t.Fatalf("%s dimensions: %v", textModel, err)
			}
			assertEmbeddingResult(t, dim, 1)
			if got := len(dim.Results[0].Outputs[0].Values); got != 128 {
				t.Fatalf("dimension length = %d", got)
			}

			batch, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{
				Model:    textModel,
				TaskType: EmbeddingTaskDocument,
				Inputs:   []EmbeddingInput{{Text: "first document"}, {Text: "second document"}},
			})
			if err != nil {
				t.Fatalf("%s batch: %v", textModel, err)
			}
			assertEmbeddingResult(t, batch, 2)

			gemini, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{
				Model:    getenvDefault("VERTEX_LIVE_GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
				TaskType: EmbeddingTaskQuery,
				Inputs:   []EmbeddingInput{{Text: "find something relevant"}},
			})
			if err != nil {
				t.Fatalf("gemini-embedding-2: %v", err)
			}
			assertEmbeddingResult(t, gemini, 1)

			legacyModel := getenvDefault("VERTEX_LIVE_MULTIMODAL_EMBEDDING_MODEL", "multimodalembedding@001")
			legacyText, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{
				Model:  legacyModel,
				Inputs: []EmbeddingInput{{Text: "The quick brown fox"}},
			})
			if err != nil {
				t.Fatalf("%s text: %v", legacyModel, err)
			}
			assertEmbeddingResult(t, legacyText, 1)

			imageSource := fetchLiveImageOrSkip(t, "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png", "google-logo.png", "image/png")
			image, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{
				Model:  legacyModel,
				Inputs: []EmbeddingInput{{Type: EmbeddingInputImage, Source: imageSource}},
			})
			if err != nil {
				t.Fatalf("%s image: %v", legacyModel, err)
			}
			assertEmbeddingResult(t, image, 1)
		})
	}

	if region := bedrockLiveRegion(); region != "" {
		t.Run("bedrock", func(t *testing.T) {
			driver, err := NewBedrockDriver(context.Background(), BedrockOptions{Region: region})
			if err != nil {
				t.Fatal(err)
			}
			ctx := liveContext(t, liveShortTimeout)
			textModel := getenvDefault("BEDROCK_LIVE_TEXT_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
			imageModel := getenvDefault("BEDROCK_LIVE_IMAGE_EMBEDDING_MODEL", "amazon.titan-embed-image-v1")
			text, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{
				Model:  textModel,
				Inputs: []EmbeddingInput{{Text: "The quick brown fox jumps over the lazy dog"}},
			})
			if err != nil {
				t.Fatalf("%s text: %v", textModel, err)
			}
			assertEmbeddingResult(t, text, 1)

			query, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{
				Model:    textModel,
				TaskType: EmbeddingTaskQuery,
				Inputs:   []EmbeddingInput{{Text: "what is the capital of France?"}},
			})
			if err != nil {
				t.Fatalf("%s query: %v", textModel, err)
			}
			assertEmbeddingResult(t, query, 1)

			image, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{
				Model:  imageModel,
				Inputs: []EmbeddingInput{{Type: EmbeddingInputImage, Source: tinyPNGSource()}},
			})
			if err != nil {
				t.Fatalf("%s image: %v", imageModel, err)
			}
			assertEmbeddingResult(t, image, 1)

			dim, err := driver.GenerateEmbeddings(ctx, EmbeddingsOptions{
				Model:      textModel,
				Dimensions: 256,
				Inputs:     []EmbeddingInput{{Text: "dimension test"}},
			})
			if err != nil {
				t.Fatalf("%s dimensions: %v", textModel, err)
			}
			assertEmbeddingResult(t, dim, 1)
			if got := len(dim.Results[0].Outputs[0].Values); got != 256 {
				t.Fatalf("dimension length = %d", got)
			}
		})
	}
}

func TestLiveVisionConversations(t *testing.T) {
	skipUnlessLiveTestsRequested(t)
	cases := liveDriverCases(t)
	var vision []liveDriverCase
	for _, tc := range cases {
		if tc.visionModel != "" {
			vision = append(vision, tc)
		}
	}
	if len(vision) == 0 {
		t.Skip("set live provider credentials with vision-capable models to run vision live tests")
	}

	for _, tc := range vision {
		t.Run(tc.name, func(t *testing.T) {
			ctx := liveContext(t, liveLongTimeout)
			image := fetchLiveImageOrSkip(t, "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png", "image.png", "image/png")
			options := liveTextOptions(tc.visionModel)
			turn1, err := tc.driver.Execute(ctx, []PromptSegment{{
				Role:    PromptRoleUser,
				Content: "What company logo is shown in this image? Describe the colors briefly.",
				Files:   []DataSource{image},
			}}, options)
			if err != nil {
				t.Fatalf("vision turn 1: %v", err)
			}
			if !strings.Contains(strings.ToLower(resultText(turn1.Result)), "google") {
				t.Fatalf("expected Google logo recognition, got %#v", turn1.Result)
			}
			verifyLiveConversationSerializable(t, turn1.Conversation)

			turn2, err := tc.driver.Execute(ctx, []PromptSegment{{
				Role:    PromptRoleUser,
				Content: "How many colors are in the logo you just described?",
			}}, withConversation(options, turn1.Conversation))
			if err != nil {
				t.Fatalf("vision turn 2: %v", err)
			}
			text2 := strings.ToLower(resultText(turn2.Result))
			if !strings.Contains(text2, "4") && !strings.Contains(text2, "four") {
				t.Fatalf("expected four colors, got %#v", turn2.Result)
			}
			verifyLiveConversationSerializable(t, turn2.Conversation)
		})
	}
}

func TestLiveBedrockImageGeneration(t *testing.T) {
	skipUnlessLiveTestsRequested(t)
	region := bedrockLiveRegion()
	if region == "" {
		t.Skip("set BEDROCK_REGION or AWS_REGION to run Bedrock image live tests")
	}
	driver, err := NewBedrockDriver(context.Background(), BedrockOptions{Region: region})
	if err != nil {
		t.Fatal(err)
	}
	ctx := liveContext(t, liveLongTimeout)
	imageModel := getenvDefault("BEDROCK_LIVE_IMAGE_MODEL", "amazon.titan-image-generator-v2:0")
	resp, err := driver.Execute(ctx, []PromptSegment{{
		Role:    PromptRoleUser,
		Content: "A blue sky with a purple unicorn flying",
	}}, ExecutionOptions{
		Model: imageModel,
		ModelOptions: map[string]any{
			"taskType":       "TEXT_IMAGE",
			"numberOfImages": 1,
		},
	})
	if err != nil {
		if os.Getenv("BEDROCK_LIVE_IMAGE_MODEL") == "" && liveProviderAccessOrAvailabilityError(err) {
			t.Skipf("skipping Bedrock image generation; default model %s is not enabled in this AWS account/region: %v", imageModel, err)
		}
		t.Fatalf("bedrock image generation: %v", err)
	}
	if len(resp.Result) != 1 || resp.Result[0].Type != ResultTypeImage || !strings.HasPrefix(fmt.Sprint(resp.Result[0].Value), "data:image/") {
		t.Fatalf("image result = %#v", resp.Result)
	}
}

func liveDriverCases(t *testing.T) []liveDriverCase {
	t.Helper()

	var cases []liveDriverCase
	if project, region, ok := googleLiveProjectRegion(); ok {
		gemini, err := NewVertexAIDriver(VertexAIOptions{Project: project, Region: region})
		if err != nil {
			t.Fatalf("vertex gemini driver: %v", err)
		}
		cases = append(cases, liveDriverCase{
			name:        "vertexai-gemini",
			driver:      gemini,
			textModel:   getenvDefault("VERTEX_LIVE_GEMINI_MODEL", "publishers/google/models/gemini-2.5-flash"),
			visionModel: getenvDefault("VERTEX_LIVE_GEMINI_VISION_MODEL", "publishers/google/models/gemini-2.5-flash"),
			toolModel:   getenvDefault("VERTEX_LIVE_GEMINI_TOOL_MODEL", "publishers/google/models/gemini-2.5-flash"),
		})

		claudeVertex, err := NewVertexAIDriver(VertexAIOptions{Project: project, Region: region})
		if err != nil {
			t.Fatalf("vertex claude driver: %v", err)
		}
		vertexClaudeModel := getenvDefault("VERTEX_LIVE_CLAUDE_MODEL", "locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5")
		cases = append(cases, liveDriverCase{
			name:        "vertexai-claude",
			driver:      claudeVertex,
			textModel:   vertexClaudeModel,
			visionModel: getenvDefault("VERTEX_LIVE_CLAUDE_VISION_MODEL", vertexClaudeModel),
			toolModel:   getenvDefault("VERTEX_LIVE_CLAUDE_TOOL_MODEL", vertexClaudeModel),
			claude:      true,
		})
	}

	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		endpoint := getenvDefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
		driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: apiKey, Endpoint: endpoint})
		if err != nil {
			t.Fatalf("openai driver: %v", err)
		}
		textModel := getenvDefault("OPENAI_LIVE_TEXT_MODEL", "gpt-5")
		cases = append(cases, liveDriverCase{
			name:        "openai",
			driver:      driver,
			textModel:   textModel,
			visionModel: getenvDefault("OPENAI_LIVE_VISION_MODEL", textModel),
			toolModel:   getenvDefault("OPENAI_LIVE_TOOL_MODEL", "gpt-4o-mini"),
		})
	}

	if region := bedrockLiveRegion(); region != "" {
		driver, err := NewBedrockDriver(context.Background(), BedrockOptions{Region: region})
		if err != nil {
			t.Fatalf("bedrock driver: %v", err)
		}
		textModel := getenvDefault("BEDROCK_LIVE_TEXT_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
		cases = append(cases, liveDriverCase{
			name:        "bedrock-claude",
			driver:      driver,
			textModel:   textModel,
			visionModel: getenvDefault("BEDROCK_LIVE_VISION_MODEL", textModel),
			toolModel:   getenvDefault("BEDROCK_LIVE_TOOL_MODEL", textModel),
			claude:      true,
		})
	}

	if apiKey := os.Getenv("ANTHROPIC_API_KEY"); apiKey != "" {
		driver, err := NewAnthropicDriver(AnthropicOptions{APIKey: apiKey})
		if err != nil {
			t.Fatalf("anthropic driver: %v", err)
		}
		cases = append(cases, liveDriverCase{
			name:        "anthropic-claude",
			driver:      driver,
			textModel:   getenvDefault("ANTHROPIC_LIVE_TEXT_MODEL", "claude-3-5-haiku-20241022"),
			visionModel: getenvDefault("ANTHROPIC_LIVE_VISION_MODEL", "claude-3-5-sonnet-20241022"),
			toolModel:   getenvDefault("ANTHROPIC_LIVE_TOOL_MODEL", "claude-3-5-haiku-20241022"),
			claude:      true,
		})
	}
	return cases
}

func liveContext(t *testing.T, timeout time.Duration) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	t.Cleanup(cancel)
	return ctx
}

func liveTextOptions(model string) ExecutionOptions {
	return ExecutionOptions{Model: model, ModelOptions: liveModelOptions(model, 256)}
}

func liveSchemaOptions(model string) ExecutionOptions {
	return ExecutionOptions{
		Model:        model,
		ModelOptions: liveModelOptions(model, 512),
		ResultSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"color": map[string]any{"type": "string"}},
			"required":   []string{"color"},
		},
	}
}

func liveToolOptions(model string) ExecutionOptions {
	return ExecutionOptions{
		Model:        model,
		ModelOptions: liveModelOptions(model, 512),
		Tools: []ToolDefinition{{
			Name:        "get_weather",
			Description: "Get the current weather in a given location",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{"type": "string", "description": "The city to get weather for"},
				},
				"required": []string{"location"},
			},
		}},
	}
}

func liveModelOptions(model string, maxTokens int) map[string]any {
	if isLiveReasoningModel(model) {
		return map[string]any{"max_tokens": 3000, "effort": "low"}
	}
	return liveNonReasoningModelOptions(maxTokens)
}

func liveNonReasoningModelOptions(maxTokens int) map[string]any {
	return map[string]any{"max_tokens": maxTokens, "temperature": 0.3}
}

func withConversation(options ExecutionOptions, conversation any) ExecutionOptions {
	options.Conversation = conversation
	return options
}

func skipUnlessLiveTestsRequested(t *testing.T) {
	t.Helper()
	if !liveTestsExplicitlyRequested() {
		t.Skip("set LLUMIVERSE_LIVE_TESTS=1, USE_REAL_API=1, or run with -run Live to run live provider tests")
	}
}

func isLiveReasoningModel(model string) bool {
	model = strings.ToLower(model)
	return strings.Contains(model, "gpt-5") || strings.Contains(model, "o1") || strings.Contains(model, "o3")
}

func consumeLiveStream(t *testing.T, stream CompletionStream) string {
	t.Helper()
	defer func() { _ = stream.Close() }()
	var chunks []string
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("stream recv: %v", err)
		}
		chunks = append(chunks, resultText(chunk.Result))
	}
	return strings.Join(chunks, "")
}

func assertLiveCompletion(t *testing.T, resp *ExecutionResponse) {
	t.Helper()
	if resp == nil {
		t.Fatal("response is nil")
	}
	if len(resp.Result) == 0 {
		t.Fatalf("empty result: %#v", resp)
	}
	if strings.TrimSpace(resultText(resp.Result)) == "" {
		t.Fatalf("empty result text: %#v", resp.Result)
	}
	if resp.FinishReason == "" {
		t.Fatalf("missing finish reason: %#v", resp)
	}
}

func resultText(results []CompletionResult) string {
	var b strings.Builder
	for _, result := range results {
		switch result.Type {
		case ResultTypeText, ResultTypeImage:
			_, _ = fmt.Fprint(&b, result.Value)
		case ResultTypeJSON:
			data, _ := json.Marshal(result.Value)
			b.Write(data)
		}
	}
	return b.String()
}

func verifyLiveConversationSerializable(t *testing.T, conversation any) {
	t.Helper()
	if conversation == nil {
		t.Fatal("conversation is nil")
	}
	data, err := json.Marshal(conversation)
	if err != nil {
		t.Fatalf("conversation JSON marshal: %v", err)
	}
	var decoded any
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("conversation JSON unmarshal: %v", err)
	}
	if hasNumericByteObject(decoded) {
		t.Fatalf("conversation contains a JSON object that looks like corrupted binary bytes: %s", data)
	}
}

func hasNumericByteObject(value any) bool {
	switch v := value.(type) {
	case map[string]any:
		if len(v) > 10 {
			allNumeric := true
			for key := range v {
				for _, r := range key {
					if r < '0' || r > '9' {
						allNumeric = false
						break
					}
				}
				if !allNumeric {
					break
				}
			}
			if allNumeric {
				return true
			}
		}
		for _, child := range v {
			if hasNumericByteObject(child) {
				return true
			}
		}
	case []any:
		for _, child := range v {
			if hasNumericByteObject(child) {
				return true
			}
		}
	}
	return false
}

func assertEmbeddingResult(t *testing.T, result *EmbeddingsResult, wantItems int) {
	t.Helper()
	if result == nil {
		t.Fatal("embedding result is nil")
	}
	if len(result.Results) != wantItems {
		t.Fatalf("embedding result count = %d, want %d: %#v", len(result.Results), wantItems, result)
	}
	for i, item := range result.Results {
		if len(item.Outputs) == 0 {
			t.Fatalf("embedding item %d has no outputs: %#v", i, item)
		}
		if len(item.Outputs[0].Values) == 0 {
			t.Fatalf("embedding item %d has empty vector: %#v", i, item.Outputs[0])
		}
		for _, value := range item.Outputs[0].Values {
			if value != value {
				t.Fatalf("embedding item %d contains NaN", i)
			}
		}
	}
}

func tinyPNGSource() BytesDataSource {
	data, _ := base64.StdEncoding.DecodeString(tinyPNGB64)
	return BytesDataSource{FileName: "pixel.png", MIME: "image/png", Data: data}
}

func fetchLiveImageOrSkip(t *testing.T, url string, name string, mimeType string) BytesDataSource {
	t.Helper()
	client := &http.Client{Timeout: 15 * time.Second}
	res, err := client.Get(url)
	if err != nil {
		t.Skipf("skipping vision test; could not fetch image %s: %v", url, err)
	}
	defer func() { _ = res.Body.Close() }()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		t.Skipf("skipping vision test; fetching %s returned HTTP %d", url, res.StatusCode)
	}
	data, err := io.ReadAll(res.Body)
	if err != nil {
		t.Skipf("skipping vision test; could not read image %s: %v", url, err)
	}
	return BytesDataSource{FileName: name, MIME: mimeType, Data: data}
}

func googleLiveProjectRegion() (string, string, bool) {
	project := firstEnv("GOOGLE_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
	region := firstEnv("GOOGLE_REGION", "GOOGLE_CLOUD_REGION")
	if liveTestsExplicitlyRequested() {
		// Vertex auth uses Application Default Credentials. Env vars only select
		// project/region; when they are absent, use the active gcloud config so
		// CLI-authenticated developers can run the live suite.
		if project == "" {
			project = commandValue("gcloud", "config", "get-value", "project")
		}
		if region == "" {
			region = firstCommandValue(
				[]string{"gcloud", "config", "get-value", "ai/region"},
				[]string{"gcloud", "config", "get-value", "compute/region"},
			)
		}
	}
	if region == "" {
		region = "us-central1"
	}
	return project, region, project != ""
}

func bedrockLiveRegion() string {
	region := firstEnv("BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")
	if region == "" && liveTestsExplicitlyRequested() {
		// Bedrock auth comes from the AWS SDK credential chain. The CLI config is
		// only used to discover a region when no env var provides one.
		region = commandValue("aws", "configure", "get", "region")
	}
	return region
}

func firstEnv(keys ...string) string {
	for _, key := range keys {
		if value := strings.TrimSpace(os.Getenv(key)); value != "" {
			return value
		}
	}
	return ""
}

func getenvDefault(key string, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func liveTestsExplicitlyRequested() bool {
	if truthyEnv("LLUMIVERSE_LIVE_TESTS") || truthyEnv("USE_REAL_API") {
		return true
	}
	if runFlag := flag.Lookup("test.run"); runFlag != nil {
		return strings.Contains(strings.ToLower(runFlag.Value.String()), "live")
	}
	return false
}

func truthyEnv(key string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "1", "true", "yes", "y", "on":
		return true
	default:
		return false
	}
}

func liveProviderAccessOrAvailabilityError(err error) bool {
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "access denied") ||
		strings.Contains(msg, "not found") ||
		strings.Contains(msg, "resourcenotfound") ||
		strings.Contains(msg, "not enabled") ||
		strings.Contains(msg, "isn't supported") ||
		strings.Contains(msg, "isn’t supported")
}

func firstCommandValue(commands ...[]string) string {
	for _, command := range commands {
		if len(command) == 0 {
			continue
		}
		if value := commandValue(command[0], command[1:]...); value != "" {
			return value
		}
	}
	return ""
}

func commandValue(name string, args ...string) string {
	if _, err := exec.LookPath(name); err != nil {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		return ""
	}
	value := strings.TrimSpace(string(out))
	if value == "" || strings.HasPrefix(value, "(") {
		return ""
	}
	return value
}
