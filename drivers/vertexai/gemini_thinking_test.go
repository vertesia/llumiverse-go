package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// TestGeminiThinkingConfig mirrors gemini-thinking-options.test.ts and the thinking
// cases of gemini-conversation-mutation.test.ts.
func TestGeminiThinkingConfig(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		model        string
		modelOptions map[string]any
		want         map[string]any
	}{
		{"no control on Gemini 3", "gemini-3.5-flash", nil, nil},
		{"no control on Gemini 2.5", "gemini-2.5-pro", nil, nil},
		{"Gemini 3 minimal effort", "gemini-3.5-flash", map[string]any{"effort": "minimal"}, map[string]any{"includeThoughts": true, "thinkingLevel": "MINIMAL"}},
		{"Gemini 3 medium effort", "gemini-3.1-pro", map[string]any{"effort": "medium"}, map[string]any{"includeThoughts": true, "thinkingLevel": "MEDIUM"}},
		{"xhigh effort", "gemini-4-hydrogen", map[string]any{"effort": "xhigh"}, map[string]any{"includeThoughts": true, "thinkingLevel": "XHIGH"}},
		{"effort passes through on image models", "gemini-3.1-flash-image", map[string]any{"effort": "low"}, map[string]any{"includeThoughts": true, "thinkingLevel": "LOW"}},
		{"minimal effort on an image model", "gemini-3-pro-image", map[string]any{"effort": "minimal"}, map[string]any{"includeThoughts": true, "thinkingLevel": "MINIMAL"}},
		{"Nano Banana effort", "gemini-nano-banana-2.1", map[string]any{"effort": "minimal"}, map[string]any{"includeThoughts": true, "thinkingLevel": "MINIMAL"}},
		{"Gemini 2.5 effort", "gemini-2.5-flash", map[string]any{"effort": "minimal"}, map[string]any{"includeThoughts": true, "thinkingBudget": 1}},
		{"include_thoughts alone", "gemini-3.5-flash", map[string]any{"include_thoughts": true}, map[string]any{"includeThoughts": true}},
		{"native level wins over effort", "gemini-3.5-flash", map[string]any{"effort": "high", "thinking_level": "low"}, map[string]any{"includeThoughts": true, "thinkingLevel": "low"}},
		{"zero budget disables summaries", "gemini-2.5-flash", map[string]any{"thinking_budget_tokens": 0}, map[string]any{"includeThoughts": false, "thinkingBudget": 0}},
		{"positive budget includes summaries", "gemini-2.5-flash", map[string]any{"thinking_budget_tokens": 128}, map[string]any{"includeThoughts": true, "thinkingBudget": 128}},
		{"explicit inclusion with a zero budget", "gemini-2.5-flash", map[string]any{"thinking_budget_tokens": 0, "include_thoughts": true}, map[string]any{"includeThoughts": true, "thinkingBudget": 0}},
		{"exclusion with a level", "gemini-3.8-flash", map[string]any{"thinking_level": "LOW", "include_thoughts": false}, map[string]any{"includeThoughts": false, "thinkingLevel": "LOW"}},
	}
	for _, budget := range []int{0, -1, 8192} {
		cases = append(cases, struct {
			name         string
			model        string
			modelOptions map[string]any
			want         map[string]any
		}{"budget with exclusion", "gemini-2.5-flash", map[string]any{"thinking_budget_tokens": budget, "include_thoughts": false, "effort": "high"}, map[string]any{"includeThoughts": false, "thinkingBudget": budget}})
	}
	for _, tc := range cases {
		got := geminiThinkingConfig(ExecutionOptions{Model: tc.model, ModelOptions: tc.modelOptions})
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: thinkingConfig = %#v, want %#v", tc.name, got, tc.want)
		}
	}

	// No thinking control leaves thinkingConfig out of the request.
	payload := (&VertexAIDriver{}).geminiPayload(geminiPrompt{}, ExecutionOptions{Model: "gemini-2.5-flash"})
	if _, ok := payload["generationConfig"].(map[string]any)["thinkingConfig"]; ok {
		t.Fatalf("generationConfig = %#v", payload["generationConfig"])
	}
}

// TestGeminiSamplingRestriction mirrors the Flash generation-parameter cases of
// gemini-thinking-options.test.ts.
func TestGeminiSamplingRestriction(t *testing.T) {
	t.Parallel()

	modelOptions := map[string]any{"temperature": 0.7, "top_p": 0.9, "top_k": 10, "presence_penalty": 0.5, "frequency_penalty": 0.5, "max_tokens": 512, "effort": "low"}
	for _, model := range []string{"gemini-3.7-flash", "publishers/google/models/gemini-3.8-flash-cyber", "gemini-4.0-flash", "locations/global/publishers/google/models/gemini-4-argon"} {
		config := (&VertexAIDriver{}).geminiPayload(geminiPrompt{}, ExecutionOptions{Model: model, ModelOptions: modelOptions})["generationConfig"].(map[string]any)
		if _, ok := config["candidateCount"]; ok {
			t.Errorf("%s: candidateCount sent: %#v", model, config)
		}
		for key, want := range map[string]any{"temperature": 0.7, "topP": 0.9, "topK": 10, "presencePenalty": 0.5, "frequencyPenalty": 0.5, "maxOutputTokens": 512} {
			if config[key] != want {
				t.Errorf("%s: %s = %#v, want %#v", model, key, config[key], want)
			}
		}
		if config["thinkingConfig"].(map[string]any)["thinkingLevel"] != "LOW" {
			t.Errorf("%s: thinkingConfig = %#v", model, config["thinkingConfig"])
		}
	}
	config := (&VertexAIDriver{}).geminiPayload(geminiPrompt{}, ExecutionOptions{Model: "gemini-3.6-flash", ModelOptions: map[string]any{"temperature": 0.7, "top_p": 0.9}})["generationConfig"].(map[string]any)
	if config["candidateCount"] != 1 || config["temperature"] != 0.7 || config["topP"] != 0.9 {
		t.Fatalf("earlier Flash config = %#v", config)
	}
}

// TestGeminiNanoBananaImageRequest mirrors the Nano Banana payload cases of
// gemini-conversation-mutation.test.ts.
func TestGeminiNanoBananaImageRequest(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gemini-nano-banana-2.1", "gemini-nano-banana-3.0"} {
		config := (&VertexAIDriver{}).geminiPayload(geminiPrompt{}, ExecutionOptions{
			Model:        model,
			ResultSchema: map[string]any{"type": "object"},
			ModelOptions: map[string]any{"temperature": 1.0, "top_p": 0.9, "top_k": 40, "seed": 42, "image_size": "2K", "image_aspect_ratio": "16:9", "effort": "minimal"},
		})["generationConfig"].(map[string]any)
		if !reflect.DeepEqual(config["responseModalities"], []string{"TEXT", "IMAGE"}) {
			t.Errorf("%s: responseModalities = %#v", model, config["responseModalities"])
		}
		if image := config["imageConfig"].(map[string]any); image["imageSize"] != "2K" || image["aspectRatio"] != "16:9" {
			t.Errorf("%s: imageConfig = %#v", model, image)
		}
		if config["thinkingConfig"].(map[string]any)["thinkingLevel"] != "MINIMAL" {
			t.Errorf("%s: thinkingConfig = %#v", model, config["thinkingConfig"])
		}
		for _, key := range []string{"temperature", "topP", "topK", "seed", "responseJsonSchema"} {
			if _, ok := config[key]; ok {
				t.Errorf("%s: %s sent: %#v", model, key, config)
			}
		}
	}
}

func finishReasonResponse(reason string, safetyRatings string) string {
	body := `{"candidates":[{"finishReason":"` + reason + `","content":{"role":"model"}`
	if safetyRatings != "" {
		body += `,"safetyRatings":` + safetyRatings
	}
	return body + `}]}`
}

// TestGeminiFinishReasonErrors mirrors the finish-reason cases of
// gemini-error-handling.test.ts.
func TestGeminiFinishReasonErrors(t *testing.T) {
	t.Parallel()

	run := func(t *testing.T, body string, stream bool) error {
		t.Helper()
		server := &recordingGeminiServer{bodies: []string{body}}
		httpServer := httptest.NewServer(server.handler(t))
		defer httpServer.Close()
		driver := newTestVertexDriver(t, httpServer.URL)
		segments := []PromptSegment{{Role: PromptRoleUser, Content: "Describe this image"}}
		options := ExecutionOptions{Model: "publishers/google/models/gemini-2.0-flash"}
		if !stream {
			_, err := driver.Execute(context.Background(), segments, options)
			return err
		}
		completionStream, err := driver.Stream(context.Background(), segments, options)
		if err != nil {
			return err
		}
		defer func() { _ = completionStream.Close() }()
		for {
			if _, err := completionStream.Recv(); err != nil {
				if errors.Is(err, io.EOF) {
					return nil
				}
				return err
			}
		}
	}

	safety := asLlumiverseError(t, run(t, finishReasonResponse("SAFETY", `[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","probability":"HIGH","blocked":true}]`), false))
	if retryableString(safety.Retryable) != "false" || safety.Name != "SAFETY" ||
		!strings.Contains(safety.Message, "Gemini blocked the response because it may violate safety policies") ||
		!strings.Contains(safety.Message, "Finish reason: SAFETY") ||
		!strings.Contains(safety.Message, "HARM_CATEGORY_DANGEROUS_CONTENT") ||
		strings.Contains(safety.Message, "content:") {
		t.Fatalf("SAFETY error = %#v", safety)
	}
	for _, reason := range []string{"SAFETY", "RECITATION", "LANGUAGE", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION", "MODEL_ARMOR", "ESCALATION", "PUP_LIMITED_DISABLED"} {
		got := asLlumiverseError(t, run(t, finishReasonResponse(reason, ""), false))
		if retryableString(got.Retryable) != "false" || got.Name != reason {
			t.Errorf("%s: error = %#v", reason, got)
		}
	}
	missing := asLlumiverseError(t, run(t, finishReasonResponse("MISSING_THOUGHT_SIGNATURE", ""), false))
	if retryableString(missing.Retryable) != "false" || !strings.Contains(missing.Message, "thought signature is missing") {
		t.Fatalf("MISSING_THOUGHT_SIGNATURE error = %#v", missing)
	}
	for _, reason := range []string{"MALFORMED_FUNCTION_CALL", "MALFORMED_RESPONSE", "NO_IMAGE", "IMAGE_OTHER", "OTHER"} {
		for _, stream := range []bool{false, true} {
			got := asLlumiverseError(t, run(t, finishReasonResponse(reason, ""), stream))
			if retryableString(got.Retryable) != "true" || got.Name != reason {
				t.Errorf("%s (stream=%v): error = %#v", reason, stream, got)
			}
		}
	}
	unknown := asLlumiverseError(t, run(t, finishReasonResponse("FUTURE_FINISH_REASON", ""), false))
	if unknown.Retryable != nil || !strings.Contains(unknown.Message, "unsupported finish reason") {
		t.Fatalf("unknown finish reason error = %#v", unknown)
	}
	for _, reason := range []string{"STOP", "MAX_TOKENS", "CONTINUATION", "FINISH_REASON_UNSPECIFIED", "UNEXPECTED_TOOL_CALL"} {
		if err := run(t, finishReasonResponse(reason, ""), false); err != nil {
			t.Errorf("%s: %v", reason, err)
		}
	}
	if got := geminiFinishReason("CONTINUATION"); got != "length" {
		t.Fatalf("CONTINUATION finish reason = %q", got)
	}
}

func roundTripConversation(t *testing.T, conversation any) any {
	t.Helper()
	data, err := json.Marshal(conversation)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// TestGeminiPreservesSignedParts mirrors "preserves signatures on arbitrary Parts
// and empty terminal Parts across JSON roundtrip".
func TestGeminiPreservesSignedParts(t *testing.T) {
	t.Parallel()

	native := `{"role":"model","parts":[` +
		`{"text":"plan","thought":true,"thoughtSignature":"thought-signature"},` +
		`{"functionCall":{"name":"first","args":{"value":1}},"thoughtSignature":"call-signature"},` +
		`{"text":"answer","thoughtSignature":"text-signature"},` +
		`{"text":"","thoughtSignature":"terminal-signature"}]}`
	body := `{"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"candidates":[{"finishReason":"STOP","content":` + native + `}]}`
	server := &recordingGeminiServer{bodies: []string{body, body, body}}
	httpServer := httptest.NewServer(server.handler(t))
	defer httpServer.Close()
	driver := newTestVertexDriver(t, httpServer.URL)
	tools := []ToolDefinition{{Name: "first", InputSchema: map[string]any{"type": "object"}}, {Name: "second", InputSchema: map[string]any{"type": "object"}}}
	keep := 0

	first, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "question"}}, ExecutionOptions{
		Model: "publishers/google/models/gemini-3-flash", Tools: tools, StripTextMaxTokens: 1, StripImagesAfterTurns: &keep,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Result) != 2 || first.Result[0].Type() != "thoughts" || first.Result[0].String() != "plan" || first.Result[1].Type() != "text" || first.Result[1].String() != "answer" {
		t.Fatalf("result = %#v", first.Result)
	}

	callID := first.ToolUse[0].ID
	_, err = driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleTool, ToolUseID: callID, Content: `{"output":"one"}`}}, ExecutionOptions{
		Model: "publishers/google/models/gemini-3-flash", Tools: tools, Conversation: roundTripConversation(t, first.Conversation),
	})
	if err != nil {
		t.Fatal(err)
	}
	var want any
	if err := json.Unmarshal([]byte(native), &want); err != nil {
		t.Fatal(err)
	}
	if got := requestTurn(server.last(), 1); !reflect.DeepEqual(got, want) {
		t.Fatalf("echoed model turn = %#v, want %#v", got, want)
	}

	hidden, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "hidden"}}, ExecutionOptions{
		Model: "publishers/google/models/gemini-3-flash", ModelOptions: map[string]any{"include_thoughts": false},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(hidden.Result) != 1 || hidden.Result[0].String() != "answer" {
		t.Fatalf("hidden result = %#v", hidden.Result)
	}
	if data, _ := json.Marshal(hidden.Conversation); !strings.Contains(string(data), "terminal-signature") {
		t.Fatalf("conversation lost the terminal signature: %s", data)
	}
}

func streamedTurn(t *testing.T, chunks ...string) []any {
	t.Helper()
	server := &recordingGeminiServer{bodies: []string{strings.Join(chunks, "\n\ndata: ")}}
	httpServer := httptest.NewServer(server.handler(t))
	defer httpServer.Close()
	driver := newTestVertexDriver(t, httpServer.URL)
	stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "question"}}, ExecutionOptions{Model: "publishers/google/models/gemini-3-flash"})
	if err != nil {
		t.Fatal(err)
	}
	response := drainStream(t, stream)
	conversation := roundTripConversation(t, response.Conversation).(map[string]any)
	contents := conversation["contents"].([]any)
	return contents[len(contents)-1].(map[string]any)["parts"].([]any)
}

// TestGeminiStreamKeepsSignedParts mirrors "reconstructs streamed thought,
// parallel function-call, and terminal signatures in order".
func TestGeminiStreamKeepsSignedParts(t *testing.T) {
	t.Parallel()

	parts := streamedTurn(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"plan ","thought":true},{"functionCall":{"name":"first","args":{"value":1}},"thoughtSignature":"first-signature"}]}}]}`,
		`{"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3},"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[`+
			`{"text":"continued","thought":true,"thoughtSignature":"thought-signature"},`+
			`{"functionCall":{"name":"second","args":{"value":2}},"thoughtSignature":"second-signature"},`+
			`{"text":"answer","thoughtSignature":"answer-signature"},`+
			`{"text":"","thoughtSignature":"terminal-signature"}]}}]}`,
	)
	if len(parts) != 6 {
		t.Fatalf("parts = %#v", parts)
	}
	for i, call := range map[int]string{1: "first", 3: "second"} {
		functionCall := parts[i].(map[string]any)["functionCall"].(map[string]any)
		id, _ := functionCall["id"].(string)
		if functionCall["name"] != call || !strings.HasPrefix(id, geminiGeneratedCallIDPrefix) {
			t.Fatalf("part %d = %#v", i, parts[i])
		}
	}
	want := []map[string]any{
		{"text": "plan ", "thought": true},
		{"thoughtSignature": "first-signature"},
		{"text": "continued", "thought": true, "thoughtSignature": "thought-signature"},
		{"thoughtSignature": "second-signature"},
		{"text": "answer", "thoughtSignature": "answer-signature"},
		{"text": "", "thoughtSignature": "terminal-signature"},
	}
	for i, fields := range want {
		for key, value := range fields {
			if parts[i].(map[string]any)[key] != value {
				t.Fatalf("part %d = %#v, want %s=%v", i, parts[i], key, value)
			}
		}
	}
}

// TestGeminiStreamDoesNotMergeSignedPart mirrors "does not merge a signed streamed
// Part into an unsigned Part".
func TestGeminiStreamDoesNotMergeSignedPart(t *testing.T) {
	t.Parallel()

	parts := streamedTurn(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"first ","thought":true}]}}]}`,
		`{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"text":"second","thought":true,"thoughtSignature":"signed-second"}]}}]}`,
	)
	want := []any{
		map[string]any{"text": "first ", "thought": true},
		map[string]any{"text": "second", "thought": true, "thoughtSignature": "signed-second"},
	}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("parts = %#v", parts)
	}
	unsigned := streamedTurn(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"first ","thought":true}]}}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"thought","thought":true}]}}]}`,
		`{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"text":"answer"}]}}]}`,
	)
	if !reflect.DeepEqual(unsigned, []any{map[string]any{"text": "first thought", "thought": true}, map[string]any{"text": "answer"}}) {
		t.Fatalf("unsigned parts = %#v", unsigned)
	}
}

// TestGeminiPruningKeepsSignedThoughts mirrors "prunes adjacent conversation
// content while preserving signed thought Parts".
func TestGeminiPruningKeepsSignedThoughts(t *testing.T) {
	t.Parallel()

	body := `{"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"text":"signed plan","thought":true,"thoughtSignature":"signed-plan"}]}}]}`
	server := &recordingGeminiServer{bodies: []string{body}}
	httpServer := httptest.NewServer(server.handler(t))
	defer httpServer.Close()
	driver := newTestVertexDriver(t, httpServer.URL)
	keep := 0
	completion, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "old tool output that should be truncated"}}, ExecutionOptions{
		Model: "publishers/google/models/gemini-3-flash", StripTextMaxTokens: 1, StripImagesAfterTurns: &keep,
	})
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(completion.Conversation)
	for _, want := range []string{"signed plan", "signed-plan", "[Content truncated - exceeded token limit]"} {
		if !strings.Contains(string(data), want) {
			t.Fatalf("conversation %s lacks %q", data, want)
		}
	}
}

// TestGeminiResultImageTokens checks that generated-image tokens are reported
// apart, and usage fields follow the TS client.
func TestGeminiResultImageTokens(t *testing.T) {
	t.Parallel()

	var response geminiResponse
	if err := json.Unmarshal([]byte(`{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":1300,"totalTokenCount":1310,"candidatesTokensDetails":[{"modality":"TEXT","tokenCount":10},{"modality":"IMAGE","tokenCount":1290}]}}`), &response); err != nil {
		t.Fatal(err)
	}
	usage := geminiUsage(response.UsageMetadata)
	if usage == nil || usage.ResultImage == nil || *usage.ResultImage != 1290 || usage.Result == nil || *usage.Result != 1300 || usage.PromptCached != nil || *usage.PromptNew != 10 {
		t.Fatalf("usage = %#v", usage)
	}
	noResult := geminiUsage(geminiUsageMetadata{PromptTokenCount: 4, TotalTokenCount: 4})
	if noResult.Result != nil || noResult.ResultImage != nil || *noResult.Total != 4 {
		t.Fatalf("usage without output = %#v", noResult)
	}
}
