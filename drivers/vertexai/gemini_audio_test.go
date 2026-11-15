package vertexai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

var audioTestBytes = []byte{1, 2, 3, 4}

// audioServer answers every generateContent request with response and records
// the request bodies.
func audioServer(t *testing.T, response string) (*VertexAIDriver, func() []map[string]any) {
	t.Helper()
	var (
		mu     sync.Mutex
		bodies []map[string]any
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		if !strings.HasSuffix(r.URL.Path, ":generateContent") {
			t.Errorf("unexpected request path %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(response))
	}))
	t.Cleanup(server.Close)
	return newTestVertexDriver(t, server.URL), func() []map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return append([]map[string]any(nil), bodies...)
	}
}

func storeAudioTo(t *testing.T, uri string) func(context.Context, io.Reader, AudioResult) (string, error) {
	return func(_ context.Context, audio io.Reader, metadata AudioResult) (string, error) {
		data, err := io.ReadAll(audio)
		if err != nil {
			return "", err
		}
		if string(data) != string(audioTestBytes) {
			t.Errorf("stored audio = %v", data)
		}
		if metadata.SampleRate == nil || *metadata.SampleRate != 24000 || metadata.Value != "" {
			t.Errorf("store metadata = %#v", metadata)
		}
		return uri, nil
	}
}

// TestGeminiSpeechStoresPCM mirrors "stores Vertex PCM and delivers the fallback
// completion without inline audio" (c1df2ee) and the speech synthesis finish
// reason case (e81a976).
func TestGeminiSpeechStoresPCM(t *testing.T) {
	t.Parallel()

	encoded := base64.StdEncoding.EncodeToString(audioTestBytes)
	response := `{"candidates":[{"finishReason":"STOP","content":{"parts":[{"inlineData":{"mimeType":"audio/L16;codec=pcm;rate=24000","data":"` + encoded + `"}}]}}]}`
	for _, stream := range []bool{false, true} {
		driver, bodies := audioServer(t, response)
		segments := []PromptSegment{{Role: PromptRoleUser, Content: "Hello."}}
		options := ExecutionOptions{Model: "gemini-3.1-flash-tts-preview", StoreAudio: storeAudioTo(t, "gs://bucket/speech.pcm"), IncludeOriginalResponse: true}
		var completion *ExecutionResponse
		if stream {
			completionStream, err := driver.Stream(context.Background(), segments, options)
			if err != nil {
				t.Fatal(err)
			}
			completion = drainStream(t, completionStream)
		} else {
			var err error
			if completion, err = driver.Execute(context.Background(), segments, options); err != nil {
				t.Fatal(err)
			}
		}
		config := bodies()[0]["generationConfig"].(map[string]any)
		if mustJSON(t, config["responseModalities"]) != `["AUDIO"]` ||
			mustJSON(t, config["speechConfig"]) != `{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}}` {
			t.Fatalf("stream=%v: generationConfig = %v", stream, config)
		}
		if len(completion.Result) != 1 {
			t.Fatalf("stream=%v: results = %v", stream, completion.Result)
		}
		audio, err := completion.Result[0].AsAudioResult()
		if err != nil || audio.Value != "gs://bucket/speech.pcm" || *audio.SampleRate != 24000 || *audio.Channels != 1 || string(*audio.ByteOrder) != "little" {
			t.Fatalf("stream=%v: audio = %#v (%v)", stream, audio, err)
		}
		if completion.FinishReason != "stop" || completion.OriginalResponse == nil || completion.Conversation != nil {
			t.Fatalf("stream=%v: finish %q, original %v, conversation %v", stream, completion.FinishReason, completion.OriginalResponse, completion.Conversation)
		}
		if strings.Contains(mustJSON(t, completion.Completion), encoded) || strings.Contains(mustJSON(t, completion.OriginalResponse), encoded) {
			t.Fatalf("stream=%v: the completion kept inline audio", stream)
		}
	}
}

// TestGeminiTranscriptionFileData mirrors "passes Vertex GCS transcription as
// fileData and returns transcript metadata" (c1df2ee) and the transcription
// finish reason case (e81a976).
func TestGeminiTranscriptionFileData(t *testing.T) {
	t.Parallel()

	driver, bodies := audioServer(t, `{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"Hello","audioTranscription":{"text":"Hello","speakerLabel":"A","words":[{"word":"Hello","startOffset":"0s","endOffset":"1s"}]}}]}}]}`)
	completion, err := driver.Execute(context.Background(), []PromptSegment{
		{Role: PromptRoleSystem, Content: "Preserve punctuation."},
		{Role: PromptRoleSystem, Content: "Keep speaker labels."},
		{Role: PromptRoleUser, Files: []DataSource{gcsSource{t: t, uri: "gs://bucket/recording.wav", mime: "audio/wav"}}},
	}, ExecutionOptions{Model: "gemini-3.5-transcribe-preview", ModelOptions: map[string]any{"transcription_diarization": true}})
	if err != nil {
		t.Fatal(err)
	}
	body := bodies()[0]
	if got := mustJSON(t, body["contents"]); got != `[{"parts":[{"fileData":{"fileUri":"gs://bucket/recording.wav","mimeType":"audio/wav"}}],"role":"user"}]` {
		t.Fatalf("contents = %s", got)
	}
	if got := mustJSON(t, body["systemInstruction"]); got != `{"parts":[{"text":"Preserve punctuation."},{"text":"Keep speaker labels."}],"role":"user"}` {
		t.Fatalf("systemInstruction = %s", got)
	}
	if got := mustJSON(t, body["generationConfig"]); got != `{"audioTranscriptionConfig":{"diarization":true}}` {
		t.Fatalf("generationConfig = %s", got)
	}
	if len(completion.Result) != 2 || completion.Result[0].String() != "Hello" || completion.FinishReason != "stop" {
		t.Fatalf("results %v, finish %q", completion.Result, completion.FinishReason)
	}
	transcript, err := completion.Result[1].AsJSONResult()
	if err != nil || mustJSON(t, transcript.Value) != `{"language_code":null,"speaker_label":"A","text":"Hello","words":[{"end_offset":"1s","start_offset":"0s","word":"Hello"}]}` {
		t.Fatalf("transcript = %#v (%v)", transcript, err)
	}
}

// TestGeminiFileAudioRejectsInvalidInput mirrors the GeminiModelDefinition
// createPrompt checks of c1df2ee: they run before any request is made.
func TestGeminiFileAudioRejectsInvalidInput(t *testing.T) {
	t.Parallel()

	driver, bodies := audioServer(t, `{}`)
	store := storeAudioTo(t, "gs://bucket/speech.pcm")
	audio := gcsSource{t: t, uri: "gs://bucket/recording.wav", mime: "audio/wav"}
	for label, tc := range map[string]struct {
		segments []PromptSegment
		options  ExecutionOptions
		want     string
	}{
		"speech without a sink": {[]PromptSegment{{Role: PromptRoleUser, Content: "Hello."}}, ExecutionOptions{Model: "gemini-4-tts"}, "durable audio storage sink"},
		"speech with a file":    {[]PromptSegment{{Role: PromptRoleUser, Content: "Hello.", Files: []DataSource{audio}}}, ExecutionOptions{Model: "gemini-4-tts", StoreAudio: store}, "1–4096 characters and no files"},
		"empty speech":          {[]PromptSegment{{Role: PromptRoleUser, Content: "  "}}, ExecutionOptions{Model: "gemini-4-tts", StoreAudio: store}, "1–4096 characters"},
		"long speech":           {[]PromptSegment{{Role: PromptRoleUser, Content: strings.Repeat("a", 4097)}}, ExecutionOptions{Model: "gemini-4-tts", StoreAudio: store}, "1–4096 characters"},
		"no audio file":         {[]PromptSegment{{Role: PromptRoleUser, Content: "Transcribe"}}, ExecutionOptions{Model: "gemini-4-transcribe"}, "exactly one audio file"},
		"two audio files":       {[]PromptSegment{{Role: PromptRoleUser, Files: []DataSource{audio, audio}}}, ExecutionOptions{Model: "gemini-4-transcribe"}, "exactly one audio file"},
		"assistant turn":        {[]PromptSegment{{Role: PromptRoleAssistant, Content: "Hi"}, {Role: PromptRoleUser, Files: []DataSource{audio}}}, ExecutionOptions{Model: "gemini-4-transcribe"}, "only user and system input"},
		"tools":                 {[]PromptSegment{{Role: PromptRoleUser, Files: []DataSource{audio}}}, ExecutionOptions{Model: "gemini-4-transcribe", Tools: []ToolDefinition{{Name: "lookup"}}}, "do not accept conversation, tools"},
	} {
		if _, err := driver.Execute(context.Background(), tc.segments, tc.options); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error = %v, want %q", label, err, tc.want)
		}
	}
	if len(bodies()) != 0 {
		t.Fatalf("invalid input reached the provider: %d requests", len(bodies()))
	}
	if isGeminiFileAudioModel("gemini-live-2.5-flash-tts") || isGeminiFileAudioModel("gemini-2.5-flash-native-audio-transcribe") || !isGeminiFileAudioModel("gemini-2.5-flash-preview-tts") {
		t.Fatal("file audio model detection is wrong")
	}
}

// TestGeminiAudioStaysOutOfConversation covers the c1df2ee retention rule: inline
// audio is replaced by a placeholder on every turn while a Cloud Storage audio
// reference is passed by URI and kept. A chat model returning inline audio fails.
func TestGeminiAudioStaysOutOfConversation(t *testing.T) {
	t.Parallel()

	encoded := base64.StdEncoding.EncodeToString(audioTestBytes)
	conversation := finalizeGeminiConversation(geminiPrompt{Contents: []geminiContent{
		{Role: "user", Parts: []geminiPart{
			{Text: "Transcribe"},
			{InlineData: &geminiInlineData{MIMEType: "audio/mpeg", Data: encoded}},
			{FileData: &geminiFileData{MIMEType: "audio/mpeg", FileURI: "gs://bucket/recording.mp3"}},
			{InlineData: &geminiInlineData{MIMEType: "image/png", Data: "aW1n"}},
		}},
		{Role: "user", Parts: []geminiPart{{FunctionResponse: &geminiFunctionResult{Name: "fetch", Response: map[string]any{"ok": true}, Parts: []geminiFunctionMediaPart{
			{InlineData: &geminiInlineData{MIMEType: "audio/wav", Data: encoded}},
			{InlineData: &geminiInlineData{MIMEType: "image/png", Data: "aW1n"}},
		}}}}},
	}}, ExecutionOptions{})
	parts := conversation.Contents[0].Parts
	if parts[1].Text != conversationAudioPlaceholder || parts[2].FileData == nil || parts[3].InlineData == nil {
		t.Fatalf("parts = %#v", parts)
	}
	if media := conversation.Contents[1].Parts[0].FunctionResponse.Parts; len(media) != 1 || media[0].InlineData.MIMEType != "image/png" {
		t.Fatalf("function response media = %#v", media)
	}

	driver, _ := audioServer(t, `{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/wav","data":"`+encoded+`"}}]}}]}`)
	if _, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Say hi"}}, ExecutionOptions{Model: "gemini-2.5-flash"}); err == nil || !strings.Contains(err.Error(), "requires a file speech model") {
		t.Fatalf("chat audio output error = %v", err)
	}
}

// TestVertexListsFileAudioModels mirrors the c1df2ee listing change: speech and
// transcription models are listed as non-streaming audio models.
func TestVertexListsFileAudioModels(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/publishers/anthropic/models") {
			_, _ = w.Write([]byte(`{"publisherModels":[]}`))
			return
		}
		_, _ = w.Write([]byte(`{"publisherModels":[{"name":"publishers/google/models/gemini-2.0-flash-tts"},{"name":"publishers/google/models/gemini-2.0-flash"}]}`))
	}))
	defer server.Close()
	models, err := newTestVertexDriver(t, server.URL).ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	types := map[string]string{}
	for _, model := range models {
		types[model.ID] = string(*model.Type) + "/" + mustJSON(t, model.CanStream)
	}
	if types["publishers/google/models/gemini-2.0-flash-tts"] != "audio/false" || types["publishers/google/models/gemini-2.0-flash"] != "text/true" {
		t.Fatalf("listed = %v", types)
	}
}
