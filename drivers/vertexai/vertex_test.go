package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vertesia/llumiverse-go/common"
	"golang.org/x/oauth2"
)

func TestVertexGeminiExecuteUsesGenerateContentEndpoint(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/publishers/google/models/gemini-test:generateContent") {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer vertex-token" {
			t.Fatalf("missing auth header: %q", r.Header.Get("Authorization"))
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{
			"candidates":[{
				"finishReason":"STOP",
				"content":{"role":"model","parts":[{"text":"gemini ok"}]}
			}],
			"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}
		}`))
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "System"},
		{Role: PromptRoleUser, Content: "Hello"},
	}, ExecutionOptions{Model: "gemini-test"})
	if err != nil {
		t.Fatal(err)
	}

	if got["systemInstruction"] == nil {
		t.Fatalf("missing system instruction: %#v", got)
	}
	if len(resp.Result) != 1 || resp.Result[0].Value != "gemini ok" {
		t.Fatalf("result = %#v", resp.Result)
	}
}

func TestVertexClaudeExecuteUsesRawPredictEndpoint(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/publishers/anthropic/models/claude-test:rawPredict") {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{
			"content":[{"type":"text","text":"claude vertex ok"}],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":2}
		}`))
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-east5",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{Model: "claude-test"})
	if err != nil {
		t.Fatal(err)
	}
	if got["anthropic_version"] != "vertex-2023-10-16" {
		t.Fatalf("anthropic_version = %v", got["anthropic_version"])
	}
	if _, ok := got["model"]; ok {
		t.Fatalf("rawPredict payload should not include model: %#v", got)
	}
	if resp.FinishReason != "stop" {
		t.Fatalf("finish reason = %q", resp.FinishReason)
	}
}

func TestVertexClaudeForwardsClaudeHeadersAndDocuments(t *testing.T) {
	t.Parallel()

	var got map[string]any
	var betaHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/publishers/anthropic/models/claude-3-7-sonnet-20250219:rawPredict") {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		betaHeader = r.Header.Get("anthropic-beta")
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{
			"content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":2}
		}`))
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-east5",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = driver.Execute(context.Background(), []PromptSegment{{
		Role:    PromptRoleUser,
		Content: "Read this",
		Files: []DataSource{BytesDataSource{
			FileName: "notes.txt",
			MIME:     "text/plain",
			Data:     []byte("notes"),
		}},
	}}, ExecutionOptions{
		Model: "claude-3-7-sonnet-20250219",
		ModelOptions: map[string]any{
			"max_tokens":             65000,
			"thinking_budget_tokens": 1024,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if betaHeader != "output-128k-2025-02-19" {
		t.Fatalf("anthropic-beta = %q", betaHeader)
	}
	messages := got["messages"].([]any)
	content := messages[0].(map[string]any)["content"].([]any)
	var sawDocument bool
	for _, raw := range content {
		block := raw.(map[string]any)
		if block["type"] == "document" {
			sawDocument = true
		}
	}
	if !sawDocument {
		t.Fatalf("document block missing: %#v", got)
	}
}

func TestVertexGeminiConvertsToolConversationAndThinkingConfig(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/publishers/google/models/gemini-2.5-pro:generateContent") {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{
			"candidates":[{
				"finishReason":"STOP",
				"content":{"role":"model","parts":[{"text":"ok"}]}
			}],
			"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":5,"totalTokenCount":9}
		}`))
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "summarize"}}, ExecutionOptions{
		Model: "gemini-2.5-pro",
		Conversation: geminiPrompt{Contents: []geminiContent{{
			Role:  "model",
			Parts: []geminiPart{{FunctionCall: &geminiFunctionCall{Name: "lookup", Args: map[string]any{"q": "tokyo"}}}},
		}}},
		ModelOptions: map[string]any{"effort": "low", "include_thoughts": true, "presence_penalty": 0.1, "seed": 7},
		Labels:       map[string]string{"job": "test"},
	})
	if err != nil {
		t.Fatal(err)
	}

	config := got["generationConfig"].(map[string]any)
	thinking := config["thinkingConfig"].(map[string]any)
	if thinking["thinkingBudget"] != float64(128) || thinking["includeThoughts"] != true {
		t.Fatalf("thinkingConfig = %#v", thinking)
	}
	if config["presencePenalty"] != 0.1 || config["seed"] != float64(7) {
		t.Fatalf("generationConfig = %#v", config)
	}
	contents := got["contents"].([]any)
	firstPart := contents[0].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if firstPart["functionCall"] != nil || !strings.Contains(firstPart["text"].(string), "[Tool call: lookup") {
		t.Fatalf("tool part was not converted: %#v", got["contents"])
	}
	if resp.Conversation == nil {
		t.Fatal("conversation was not returned")
	}
}

func TestVertexGeminiPromptFormattingAndHelperBranches(t *testing.T) {
	t.Parallel()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := driver.formatGeminiPrompt(context.Background(), []PromptSegment{{
		Role:    PromptRoleSystem,
		Content: "System",
		Files:   []DataSource{BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte("img")}},
	}}, ExecutionOptions{}); err == nil {
		t.Fatal("expected system file error")
	}
	if _, err := driver.formatGeminiPrompt(context.Background(), []PromptSegment{{Role: PromptRoleTool, Content: "missing id"}}, ExecutionOptions{}); err == nil {
		t.Fatal("expected missing ToolUseID error")
	}

	prompt, err := driver.formatGeminiPrompt(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "System"},
		{Role: PromptRoleAssistant, Content: "Previous answer"},
		{Role: PromptRoleUser, Content: "Question"},
		{Role: PromptRoleTool, ToolUseID: "lookup", ThoughtSignature: "sig", Content: `{"ok":true}`},
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
	if prompt.System == nil || len(prompt.System.Parts) != 2 || !strings.Contains(prompt.System.Parts[1].Text, "When not calling tools") {
		t.Fatalf("system = %#v", prompt.System)
	}
	if len(prompt.Contents) != 2 || prompt.Contents[0].Role != "model" || prompt.Contents[1].Role != "user" {
		t.Fatalf("contents = %#v", prompt.Contents)
	}
	toolPart := prompt.Contents[1].Parts[1]
	if toolPart.FunctionResponse == nil || toolPart.FunctionResponse.Name != "lookup" || toolPart.ThoughtSignature != "sig" {
		t.Fatalf("tool part = %#v", toolPart)
	}
	if prompt.Contents[1].Parts[2].Text != "Safety" {
		t.Fatalf("safety part = %#v", prompt.Contents[1].Parts)
	}
	sparse, err := driver.formatGeminiPrompt(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Question"}}, ExecutionOptions{
		ResultSchema: map[string]any{"type": "object"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if sparse.System == nil || sparse.System.Parts[0].Text != "Fill all appropriate fields in the JSON output." {
		t.Fatalf("sparse schema prompt = %#v", sparse.System)
	}

	processed := processGeminiPartsForConversation([]geminiPart{
		{InlineData: &geminiInlineData{MIMEType: "image/png", Data: "aW1n"}},
		{Text: "abcdefghijklmnop"},
		{Text: "<heartbeat>working</heartbeat>"},
	}, ExecutionOptions{StripTextMaxTokens: 2}, true, true)
	if processed[0].Text != conversationImagePlaceholder || processed[1].Text != "abcdefgh"+conversationTextTruncatedMarker || processed[2].Text != conversationHeartbeatPlaceholder {
		t.Fatalf("processed parts = %#v", processed)
	}
}

func TestVertexGeminiDefaultThinkingAndImageConfig(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/publishers/google/models/gemini-3-pro-image-preview:generateContent") {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{
			"candidates":[{
				"finishReason":"STOP",
				"content":{"role":"model","parts":[{"text":"ok"}]}
			}]
		}`))
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "draw"}}, ExecutionOptions{
		Model: "gemini-3-pro-image-preview",
		ModelOptions: map[string]any{
			"include_thoughts":           true,
			"image_aspect_ratio":         "16:9",
			"image_size":                 "2K",
			"person_generation":          "ALLOW_ALL",
			"prominent_people":           "BLOCK_PROMINENT_PEOPLE",
			"output_mime_type":           "image/jpeg",
			"output_compression_quality": 90,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	config := got["generationConfig"].(map[string]any)
	thinking := config["thinkingConfig"].(map[string]any)
	if thinking["thinkingLevel"] != "HIGH" || thinking["includeThoughts"] != true {
		t.Fatalf("thinkingConfig = %#v", thinking)
	}
	modalities := config["responseModalities"].([]any)
	if len(modalities) != 2 || modalities[0] != "TEXT" || modalities[1] != "IMAGE" {
		t.Fatalf("responseModalities = %#v", modalities)
	}
	imageConfig := config["imageConfig"].(map[string]any)
	if imageConfig["aspectRatio"] != "16:9" || imageConfig["imageSize"] != "2K" || imageConfig["outputCompressionQuality"] != float64(90) {
		t.Fatalf("imageConfig = %#v", imageConfig)
	}
}

func TestVertexThinkingAndEmbeddingHelperBranches(t *testing.T) {
	t.Parallel()

	thinking := geminiThinkingConfig(ExecutionOptions{
		Model:        "gemini-3-pro",
		ModelOptions: map[string]any{"thinking_level": "HIGH", "include_thoughts": true},
	})
	if thinking["thinkingLevel"] != "HIGH" || thinking["includeThoughts"] != true {
		t.Fatalf("explicit thinking = %#v", thinking)
	}
	thinking = geminiThinkingConfig(ExecutionOptions{
		Model:        "gemini-2.5-flash",
		ModelOptions: map[string]any{"include_thoughts": true},
	})
	if thinking["thinkingBudget"] != 0 || thinking["includeThoughts"] != true {
		t.Fatalf("default flash thinking = %#v", thinking)
	}

	if got := vertexEmbeddingPrefixText(EmbeddingInput{TaskType: EmbeddingTaskQuery, Text: "find"}); got != "task: search result | query: find" {
		t.Fatalf("query prefix = %q", got)
	}
	if got := vertexEmbeddingPrefixText(EmbeddingInput{TaskType: EmbeddingTaskDocument, Text: "body"}); got != "title: none | text: body" {
		t.Fatalf("document prefix = %q", got)
	}
	if got := vertexEmbeddingPrefixText(EmbeddingInput{Text: "plain"}); got != "plain" {
		t.Fatalf("plain prefix = %q", got)
	}
	if got := vertexEmbeddingTaskType(EmbeddingTaskDocument); got != "RETRIEVAL_DOCUMENT" {
		t.Fatalf("document task type = %q", got)
	}
	if got := vertexEmbeddingTaskType(""); got != "" {
		t.Fatalf("empty task type = %q", got)
	}
	dataPart, err := vertexEmbeddingDataPart(context.Background(), BytesDataSource{FileName: "image.png", MIME: "image/png", URI: "gs://bucket/image.png"})
	if err != nil {
		t.Fatal(err)
	}
	if dataPart["fileData"].(map[string]any)["fileUri"] != "gs://bucket/image.png" {
		t.Fatalf("data part = %#v", dataPart)
	}

	audioInstance, modality, err := vertexLegacyMultimodalInstance(context.Background(), EmbeddingInput{
		Type:   embeddingInputAudio,
		Source: BytesDataSource{FileName: "sound.mp3", MIME: "audio/mpeg", Data: []byte("mp3")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if modality != embeddingInputAudio || audioInstance["video"].(map[string]any)["bytesBase64Encoded"] != "bXAz" {
		t.Fatalf("audio instance = %#v modality=%s", audioInstance, modality)
	}
	if _, _, err := vertexLegacyMultimodalInstance(context.Background(), EmbeddingInput{Type: "document"}); err == nil {
		t.Fatal("expected unsupported legacy input error")
	}
	imagePart, err := vertexLegacyImagePart(context.Background(), BytesDataSource{FileName: "image.png", MIME: "image/png", URI: "gs://bucket/image.png"})
	if err != nil {
		t.Fatal(err)
	}
	if imagePart["gcsUri"] != "gs://bucket/image.png" {
		t.Fatalf("legacy image part = %#v", imagePart)
	}
	if _, err := vertexLegacyImagePart(context.Background(), nil); err == nil {
		t.Fatal("expected missing image source error")
	}
	if _, err := vertexLegacyVideoPart(context.Background(), EmbeddingInput{Type: embeddingInputVideo}); err == nil {
		t.Fatal("expected missing video source error")
	}

	outputs, err := vertexLegacyMultimodalOutputs(vertexLegacyMultimodalPrediction{ImageEmbedding: []float64{0.1}}, embeddingInputText)
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 1 || outputs[0].Modality != embeddingInputImage {
		t.Fatalf("fallback outputs = %#v", outputs)
	}
	start := 1.0
	end := 2.0
	outputs, err = vertexLegacyMultimodalOutputs(vertexLegacyMultimodalPrediction{VideoEmbeddings: []vertexLegacyVideoEmbedding{{Embedding: []float64{0.2}, StartOffsetSec: &start, EndOffsetSec: &end}}}, embeddingInputImage)
	if err != nil {
		t.Fatal(err)
	}
	if outputs[0].Modality != embeddingInputVideo || outputs[0].StartSec == nil || *outputs[0].StartSec != 1 {
		t.Fatalf("video fallback outputs = %#v", outputs)
	}
	if _, err := vertexLegacyMultimodalOutputs(vertexLegacyMultimodalPrediction{}, embeddingInputText); err == nil {
		t.Fatal("expected empty legacy prediction error")
	}
}

func TestVertexEmbedContentDefaultModelUsesGlobalAndMultimodalContent(t *testing.T) {
	t.Parallel()

	var calls []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/locations/global/publishers/google/models/gemini-embedding-2:embedContent") {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		var got map[string]any
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		calls = append(calls, got)
		if len(calls) == 1 {
			_, _ = w.Write([]byte(`{"embedding":{"values":[0.1,0.2]},"usageMetadata":{"promptTokenCount":3,"promptTokensDetails":[{"modality":"TEXT","tokenCount":3}]}}`))
			return
		}
		_, _ = w.Write([]byte(`{"embedding":{"values":[0.3,0.4]},"usageMetadata":{"promptTokenCount":5,"promptTokensDetails":[{"modality":"IMAGE","tokenCount":5}]}}`))
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{
		Inputs: []EmbeddingInput{
			{Type: embeddingInputText, Text: "policy text", TaskType: EmbeddingTaskDocument, Title: "Policy"},
			{Type: embeddingInputImage, Source: BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte("img")}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Model != defaultVertexEmbeddingModel || len(result.Results) != 2 {
		t.Fatalf("result = %#v", result)
	}
	if result.Usage == nil || result.Usage.InputTokens != 8 || result.Usage.InputTextTokens != 3 || result.Usage.InputImageTokens != 5 {
		t.Fatalf("usage = %#v", result.Usage)
	}
	if len(calls) != 2 {
		t.Fatalf("calls = %#v", calls)
	}
	firstContent := calls[0]["content"].(map[string]any)
	firstPart := firstContent["parts"].([]any)[0].(map[string]any)
	if firstPart["text"] != "title: Policy | text: policy text" {
		t.Fatalf("first call = %#v", calls[0])
	}
	if _, ok := calls[0]["taskType"]; ok {
		t.Fatalf("prefix model should not send taskType: %#v", calls[0])
	}
	if calls[0]["title"] != "Policy" {
		t.Fatalf("first call = %#v", calls[0])
	}
	secondContent := calls[1]["content"].(map[string]any)
	secondPart := secondContent["parts"].([]any)[0].(map[string]any)
	inlineData := secondPart["inlineData"].(map[string]any)
	if inlineData["mimeType"] != "image/png" || inlineData["data"] != "aW1n" {
		t.Fatalf("second call = %#v", calls[1])
	}
}

func TestVertexTextEmbeddingPredictTaskTypeAndDimensions(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/publishers/google/models/text-embedding-005:predict") {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{"predictions":[{"embeddings":{"values":[0.1,0.2],"statistics":{"token_count":4}}}]}`))
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{
		Model:      "text-embedding-005",
		Dimensions: 128,
		Inputs: []EmbeddingInput{{
			Type:     embeddingInputText,
			Text:     "find this",
			TaskType: EmbeddingTaskQuery,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 1 || result.Results[0].InputTokens != 4 {
		t.Fatalf("result = %#v", result)
	}
	parameters := got["parameters"].(map[string]any)
	if parameters["outputDimensionality"] != float64(128) {
		t.Fatalf("got = %#v", got)
	}
	instances := got["instances"].([]any)
	instance := instances[0].(map[string]any)
	if instance["content"] != "find this" || instance["task_type"] != "RETRIEVAL_QUERY" {
		t.Fatalf("instances = %#v", instances)
	}
}

func TestVertexTextEmbeddingPredictBatchesAndPreservesOrder(t *testing.T) {
	t.Parallel()

	var calls []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got map[string]any
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		calls = append(calls, got)
		_, _ = w.Write([]byte(`{"predictions":[{"embeddings":{"values":[1,2]}},{"embeddings":{"values":[3,4]}}]}`))
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{
		Model: "text-embedding-005",
		Inputs: []EmbeddingInput{
			{Type: embeddingInputText, Text: "first"},
			{Type: embeddingInputText, Text: "second", TaskType: EmbeddingTaskDocument, Title: "Second"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("calls = %#v", calls)
	}
	if len(result.Results) != 2 || result.Results[0].Outputs[0].Values[0] != 1 || result.Results[1].Outputs[0].Values[0] != 3 {
		t.Fatalf("result = %#v", result)
	}
	instances := calls[0]["instances"].([]any)
	second := instances[1].(map[string]any)
	if second["task_type"] != "RETRIEVAL_DOCUMENT" || second["title"] != "Second" {
		t.Fatalf("second instance = %#v", second)
	}
}

func TestVertexTextEmbeddingPredictRejectsMedia(t *testing.T) {
	t.Parallel()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{
		Model: "text-embedding-005",
		Inputs: []EmbeddingInput{{
			Type:   embeddingInputImage,
			Source: BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte("img")},
		}},
	})
	if err == nil || !strings.Contains(err.Error(), "only supports text input") {
		t.Fatalf("err = %v", err)
	}
}

func TestVertexLegacyMultimodalEmbeddingsPredict(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/publishers/google/models/multimodalembedding@001:predict") {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{
			"predictions":[
				{"textEmbedding":[0.1,0.2]},
				{"imageEmbedding":[0.3,0.4]},
				{"videoEmbeddings":[{"embedding":[0.5,0.6],"startOffsetSec":1,"endOffsetSec":3}]}
			]
		}`))
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	start := 1.0
	length := 2.0
	interval := 0.5
	result, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{
		Model:      "multimodalembedding@001",
		Dimensions: 1408,
		Inputs: []EmbeddingInput{
			{Type: embeddingInputText, Text: "hello"},
			{Type: embeddingInputImage, Source: BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte("img")}},
			{Type: embeddingInputVideo, Source: BytesDataSource{FileName: "clip.mp4", MIME: "video/mp4", Data: []byte("mp4"), URI: "gs://bucket/clip.mp4"}, StartSec: &start, LengthSec: &length, IntervalSec: &interval},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 3 || result.Results[2].Outputs[0].StartSec == nil || *result.Results[2].Outputs[0].StartSec != 1 {
		t.Fatalf("result = %#v", result)
	}
	instances := got["instances"].([]any)
	if len(instances) != 3 {
		t.Fatalf("instances = %#v", instances)
	}
	if instances[0].(map[string]any)["text"] != "hello" {
		t.Fatalf("instances = %#v", instances)
	}
	image := instances[1].(map[string]any)["image"].(map[string]any)
	if image["bytesBase64Encoded"] != "aW1n" || image["mimeType"] != "image/png" {
		t.Fatalf("image instance = %#v", image)
	}
	video := instances[2].(map[string]any)["video"].(map[string]any)
	if video["gcsUri"] != "gs://bucket/clip.mp4" {
		t.Fatalf("video instance = %#v", video)
	}
	segmentConfig := video["videoSegmentConfig"].(map[string]any)
	if segmentConfig["startOffsetSec"] != float64(1) || segmentConfig["endOffsetSec"] != float64(3) || segmentConfig["intervalSec"] != 0.5 {
		t.Fatalf("videoSegmentConfig = %#v", segmentConfig)
	}
	parameters := got["parameters"].(map[string]any)
	if parameters["dimension"] != float64(1408) {
		t.Fatalf("parameters = %#v", parameters)
	}
}

func TestVertexImagenUsesPredictEndpoint(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/publishers/google/models/imagen-test:predict") {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{"predictions":[{"bytesBase64Encoded":"aW1n"}]}`))
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := driver.Execute(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "Editorial image"},
		{Role: PromptRoleUser, Content: "A cube"},
		{Role: PromptRoleNegative, Content: "blur"},
	}, ExecutionOptions{
		Model:        "imagen-test",
		ModelOptions: map[string]any{"number_of_images": 2, "aspect_ratio": "1:1", "add_watermark": false},
	})
	if err != nil {
		t.Fatal(err)
	}
	instances := got["instances"].([]any)
	instance := instances[0].(map[string]any)
	if !strings.Contains(instance["prompt"].(string), "Editorial image") || !strings.Contains(instance["prompt"].(string), "A cube") {
		t.Fatalf("instance = %#v", instance)
	}
	params := got["parameters"].(map[string]any)
	if params["sampleCount"] != float64(2) || params["aspectRatio"] != "1:1" || params["negativePrompt"] != "blur" {
		t.Fatalf("parameters = %#v", params)
	}
	if len(resp.Result) != 1 || resp.Result[0].Value != "data:image/png;base64,aW1n" {
		t.Fatalf("result = %#v", resp.Result)
	}
}

func TestVertexHelperBranches(t *testing.T) {
	t.Parallel()

	if _, err := NewVertexAIDriver(VertexAIOptions{Region: "us-central1"}); err == nil {
		t.Fatal("expected missing project error")
	}
	if _, err := NewVertexAIDriver(VertexAIOptions{Project: "project-1"}); err == nil {
		t.Fatal("expected missing region error")
	}
	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     "https://example.test/v1",
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if driver.Provider() != ProviderVertexAI {
		t.Fatalf("provider = %s", driver.Provider())
	}
	if err := driver.ValidateConnection(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := vertexDefaultBaseURL("global"); got != "https://aiplatform.googleapis.com/v1" {
		t.Fatalf("global base = %q", got)
	}
	if got := vertexDefaultBaseURL("us-central1"); got != "https://us-central1-aiplatform.googleapis.com/v1" {
		t.Fatalf("regional base = %q", got)
	}
	base, path := driver.vertexModelPathForRegion("locations/europe-west4/publishers/google/models/gemini-test", "")
	if !strings.Contains(base, "/locations/europe-west4") || path != "publishers/google/models/gemini-test" {
		t.Fatalf("region override = %q %q", base, path)
	}
	_, claudePath := driver.vertexModelPath("claude-3-5-sonnet")
	if claudePath != "publishers/anthropic/models/claude-3-5-sonnet" {
		t.Fatalf("claude path = %q", claudePath)
	}
	if !isClaudeModel("publishers/anthropic/models/claude") || !isVertexImagenModel("imagen-4") || !isGeminiVersionGTE("gemini-3-pro", "2.5") || isGeminiVersionGTE("not-gemini", "2.5") {
		t.Fatal("model helper returned unexpected result")
	}
	if got := withShortModel(ExecutionOptions{Model: "publishers/google/models/gemini"}).Model; got != "gemini" {
		t.Fatalf("short model = %q", got)
	}
	if got := optionFloatValue(map[string]any{"x": 1.25}, "x"); got != 1.25 {
		t.Fatalf("optionFloatValue = %#v", got)
	}
	if got := optionFloatValue(nil, "x"); got != nil {
		t.Fatalf("optionFloatValue nil = %#v", got)
	}
	if got := formatGeminiFunctionResponse(`{"ok":true}`); got["ok"] != true {
		t.Fatalf("json function response = %#v", got)
	}
	if got := formatGeminiFunctionResponse("plain"); got["output"] != "plain" {
		t.Fatalf("plain function response = %#v", got)
	}
	tools := geminiTools([]ToolDefinition{{Name: "lookup", Description: "Lookup", InputSchema: map[string]any{"type": "object"}}})
	if len(tools) != 1 || tools[0]["name"] != "lookup" || tools[0]["parametersJsonSchema"] == nil {
		t.Fatalf("tools = %#v", tools)
	}
	streamed := buildGeminiStreamingConversation(geminiPrompt{}, &Completion{
		Result:  []CompletionResult{{Type: ResultTypeText, Value: "hello"}},
		ToolUse: []ToolUse{{ID: "lookup", ToolName: "lookup", ToolInput: map[string]any{"q": "tokyo"}, ThoughtSignature: "sig"}},
	})
	if len(streamed.Contents) != 1 || len(streamed.Contents[0].Parts) != 2 || streamed.Contents[0].Parts[1].ThoughtSignature != "sig" {
		t.Fatalf("streamed conversation = %#v", streamed)
	}
	if got := geminiThinkingLevelForEffort("gemini-3-pro-image-preview", "low"); got != "HIGH" {
		t.Fatalf("image level = %q", got)
	}
	if got := geminiThinkingLevelForEffort("gemini-3.1-flash-image", "low"); got != "MINIMAL" {
		t.Fatalf("flash image low = %q", got)
	}
	if got := geminiThinkingLevelForEffort("gemini-3.1-flash-image", "medium"); got != "HIGH" {
		t.Fatalf("flash image medium = %q", got)
	}
	if got := geminiThinkingLevelForEffort("gemini-3-pro", "medium"); got != "MEDIUM" {
		t.Fatalf("medium level = %q", got)
	}
	if got := geminiThinkingLevelForEffort("gemini-3-pro", "unknown"); got != "" {
		t.Fatalf("unknown level = %q", got)
	}
	if got := geminiBudgetForEffort("gemini-2.5-pro", "high"); got != 32768 {
		t.Fatalf("pro high budget = %d", got)
	}
	if got := geminiBudgetForEffort("gemini-2.5-flash-lite", "low"); got != 512 {
		t.Fatalf("flash-lite low budget = %d", got)
	}
	if got := geminiBudgetForEffort("gemini-2.5-flash", "low"); got != 1 {
		t.Fatalf("flash low budget = %d", got)
	}
	if got := geminiBudgetForEffort("gemini-2.5-flash", "high"); got != 24576 {
		t.Fatalf("flash high budget = %d", got)
	}
	if got := geminiBudgetForEffort("gemini-2.5", "unknown"); got != 0 {
		t.Fatalf("unknown budget = %d", got)
	}
	if got := geminiFinishReason("MAX_TOKENS"); got != "length" {
		t.Fatalf("finish = %q", got)
	}
	if geminiUsage(0, 0, 0, 0) != nil {
		t.Fatal("zero Gemini usage should be nil")
	}
}

func TestVertexHTTPTimeoutConfiguration(t *testing.T) {
	t.Parallel()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
		DriverOptions: DriverOptions{HTTPTimeout: HTTPTimeoutOptions{
			BodyTimeout:    3 * time.Second,
			ConnectTimeout: 2 * time.Second,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if driver.client.Timeout != 3*time.Second {
		t.Fatalf("constructor timeout = %s", driver.client.Timeout)
	}
	override := driver.httpClient(HTTPTimeoutOptions{BodyTimeout: time.Second})
	if override == driver.client || override.Timeout != time.Second {
		t.Fatalf("override client = %#v", override)
	}
	if got := driver.httpClient(HTTPTimeoutOptions{}); got != driver.client {
		t.Fatal("empty per-request timeout should reuse the driver client")
	}
}

func TestVertexGeminiPartsAndResponseHelpers(t *testing.T) {
	t.Parallel()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	filePart, err := driver.geminiFilePart(context.Background(), BytesDataSource{FileName: "image.png", MIME: "image/png", URI: "gs://bucket/image.png"})
	if err != nil {
		t.Fatal(err)
	}
	if filePart.FileData == nil || filePart.FileData.FileURI != "gs://bucket/image.png" {
		t.Fatalf("file data part = %#v", filePart)
	}
	inlinePart, err := driver.geminiFilePart(context.Background(), BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte("img")})
	if err != nil {
		t.Fatal(err)
	}
	if inlinePart.InlineData == nil || inlinePart.InlineData.Data != "aW1n" {
		t.Fatalf("inline part = %#v", inlinePart)
	}
	if !isVertexGCSURL("https://storage.googleapis.com/bucket/image.png") || !isVertexGCSURL("https://storage.cloud.google.com/bucket/image.png") || isVertexGCSURL("https://example.com/image.png") {
		t.Fatal("GCS URL helper returned unexpected result")
	}

	blocked := geminiResponseToChunk(geminiResponse{})
	if blocked.TokenUsage != nil || blocked.FinishReason != "" || len(blocked.Result) != 0 {
		t.Fatalf("empty chunk = %#v", blocked)
	}
	blocked = geminiResponseToChunk(geminiResponse{PromptFeedback: struct {
		BlockReason        string `json:"blockReason"`
		BlockReasonMessage string `json:"blockReasonMessage"`
	}{BlockReason: "SAFETY", BlockReasonMessage: "blocked"}})
	if blocked.FinishReason != "SAFETY" || blocked.Result[0].Value != "blocked" {
		t.Fatalf("blocked chunk = %#v", blocked)
	}
	response := geminiResponse{}
	if err := json.Unmarshal([]byte(`{
		"candidates":[{
			"finishReason":"STOP",
			"content":{"role":"model","parts":[
				{"text":"hello"},
				{"inlineData":{"mimeType":"image/png","data":"aW1n"}},
				{"functionCall":{"name":"lookup","args":{"q":"tokyo"}},"thoughtSignature":"sig"}
			]}
		}],
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":6,"totalTokenCount":11,"cachedContentTokenCount":2,"thoughtsTokenCount":1,"toolUsePromptTokenCount":1}
	}`), &response); err != nil {
		t.Fatal(err)
	}
	completion := geminiResponseToCompletion(response, true)
	if completion.OriginalResponse == nil || completion.FinishReason != "tool_use" || len(completion.Result) != 2 || len(completion.ToolUse) != 1 {
		t.Fatalf("completion = %#v", completion)
	}
	if completion.TokenUsage == nil || completion.TokenUsage.Result != 8 || completion.TokenUsage.PromptNew != 3 {
		t.Fatalf("usage = %#v", completion.TokenUsage)
	}
	if empty := geminiResponseToCompletion(geminiResponse{}, false); len(empty.Result) != 1 || empty.Result[0].Value != "" {
		t.Fatalf("empty completion = %#v", empty)
	}
}

func TestVertexImagenPromptAndParametersHelpers(t *testing.T) {
	t.Parallel()

	prompt, err := formatVertexImagenPrompt(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "system"},
		{Role: PromptRoleUser, Content: "draw", Files: []DataSource{BytesDataSource{FileName: "raw.png", MIME: "image/png", Data: []byte("raw")}}},
		{Role: PromptRoleSafety, Content: "safe"},
		{Role: PromptRoleNegative, Content: "blur"},
		{Role: PromptRoleMask, Files: []DataSource{BytesDataSource{FileName: "mask.png", MIME: "image/png", Data: []byte("mask")}}},
	}, ExecutionOptions{ModelOptions: map[string]any{"mask_mode": "MASK_MODE_USER_PROVIDED", "mask_dilation": 0.2}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt.Prompt, "system") || !strings.Contains(prompt.Prompt, "draw") || !strings.Contains(prompt.Prompt, "safe") || prompt.NegativePrompt != "blur" {
		t.Fatalf("prompt = %#v", prompt)
	}
	if len(prompt.ReferenceImages) != 2 || prompt.ReferenceImages[1]["referenceType"] != "REFERENCE_TYPE_MASK" {
		t.Fatalf("reference images = %#v", prompt.ReferenceImages)
	}
	params := vertexImagenParameters(prompt, ExecutionOptions{ModelOptions: map[string]any{
		"edit_mode":    "EDIT_MODE_INPAINT_INSERTION",
		"edit_steps":   12,
		"aspect_ratio": "ignored-for-edit",
	}})
	if params["editMode"] != "EDIT_MODE_INPAINT_INSERTION" || params["editConfig"].(map[string]any)["baseSteps"] != 12 {
		t.Fatalf("edit params = %#v", params)
	}
	if got := optionBoolPtrValue(map[string]any{"x": false}, "x"); got != false {
		t.Fatalf("bool ptr value = %#v", got)
	}
	if got := optionBoolPtrValue(nil, "x"); got != nil {
		t.Fatalf("bool ptr nil = %#v", got)
	}
}

func TestVertexListModels(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/publishers/google/models"):
			_, _ = w.Write([]byte(`{"publisherModels":[{"name":"publishers/google/models/gemini-b","displayName":"Gemini B"},{"name":"publishers/google/models/other"}]}`))
		case strings.Contains(r.URL.Path, "/publishers/anthropic/models"):
			_, _ = w.Write([]byte(`{"models":[{"name":"publishers/anthropic/models/claude-a","displayName":"Claude A"}]}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	models, err := driver.ListModels(context.Background(), &ModelSearchPayload{Text: "gemini"})
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].ID != "publishers/google/models/gemini-b" || models[0].Owner != "google" {
		t.Fatalf("filtered models = %#v", models)
	}
	models, err = driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 3 || models[0].ID != "publishers/anthropic/models/claude-a" {
		t.Fatalf("models = %#v", models)
	}
}

func TestVertexGeminiStreamParsesSSE(t *testing.T) {
	t.Parallel()

	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/publishers/google/models/gemini-test:streamGenerateContent") || r.URL.Query().Get("alt") != "sse" {
			t.Fatalf("unexpected URL: %s", r.URL.String())
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		writeVertexSSE(t, w, `{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"text":"hello "}]}}],"usageMetadata":{"promptTokenCount":2}}`)
		writeVertexSSE(t, w, `{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"text":"world"}]}}],"usageMetadata":{"candidatesTokenCount":3,"totalTokenCount":5}}`)
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{Model: "gemini-test"})
	if err != nil {
		t.Fatal(err)
	}
	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if got["contents"] == nil {
		t.Fatalf("payload = %#v", got)
	}
	completion := stream.Completion()
	if len(completion.Result) != 1 || completion.Result[0].Value != "hello world" {
		t.Fatalf("result = %#v", completion.Result)
	}
	if completion.TokenUsage == nil || completion.TokenUsage.Total != 5 {
		t.Fatalf("usage = %#v", completion.TokenUsage)
	}
	if conversation, ok := completion.Conversation.(geminiPrompt); !ok || len(conversation.Contents) != 2 {
		t.Fatalf("conversation = %#v", completion.Conversation)
	}
}

func TestVertexErrorFormattingFromHTTPStatus(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "overloaded", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	driver, err := NewVertexAIDriver(VertexAIOptions{
		Project:     "project-1",
		Region:      "us-central1",
		BaseURL:     server.URL,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "vertex-token"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Hi"}}, ExecutionOptions{Model: "gemini-test"})
	var llumiErr *common.LlumiverseError
	if !errors.As(err, &llumiErr) || llumiErr.Name != "GeminiEnterpriseAgentPlatformError" || llumiErr.Code != http.StatusServiceUnavailable || llumiErr.Context.Provider != ProviderVertexAI {
		t.Fatalf("error = %#v", err)
	}
	if err := driver.formatError(errors.New("plain"), "gemini-test", "execute"); err.Error() != "plain" {
		t.Fatalf("plain error = %v", err)
	}
}

func writeVertexSSE(t *testing.T, w http.ResponseWriter, data string) {
	t.Helper()
	_, err := w.Write([]byte("data: " + data + "\n\n"))
	if err != nil {
		t.Fatal(err)
	}
}
