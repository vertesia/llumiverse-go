package openai

import (
	"context"
	"encoding/base64"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

// audioServer answers the file-audio endpoints and records each request.
type audioServer struct {
	paths  []string
	bodies []string
	types  []string
}

func newAudioServer(t *testing.T, responses map[string]func(w http.ResponseWriter)) (*audioServer, *OpenAICompatibleDriver) {
	t.Helper()
	recorder := &audioServer{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		recorder.paths = append(recorder.paths, r.URL.Path)
		recorder.bodies = append(recorder.bodies, string(body))
		recorder.types = append(recorder.types, r.Header.Get("Content-Type"))
		for suffix, respond := range responses {
			if strings.HasSuffix(r.URL.Path, suffix) {
				respond(w)
				return
			}
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(server.Close)
	driver, err := NewOpenAIDriver(OpenAICompatibleOptions{APIKey: "k", Endpoint: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	return recorder, driver
}

func jsonReply(body string) func(http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}
}

func storeTo(uri string, stored *[]byte) func(context.Context, io.Reader, common.AudioResult) (string, error) {
	return func(_ context.Context, audio io.Reader, _ common.AudioResult) (string, error) {
		data, err := io.ReadAll(audio)
		if err != nil {
			return "", err
		}
		if stored != nil {
			*stored = data
		}
		return uri, nil
	}
}

func TestOpenAIAudioTask(t *testing.T) {
	t.Parallel()

	for model, want := range map[string]string{
		"whisper-1": "transcription", "gpt-4o-transcribe": "transcription", "gpt-4o-transcribe-diarize": "transcription",
		"tts-1": "speech", "gpt-4o-mini-tts": "speech", "gpt-audio": "understanding", "gpt-4o-audio-preview": "understanding",
		"gpt-realtime": "", "gpt-4o-realtime-preview": "", "gpt-audio-realtime": "", "gpt-4.1": "", "attestation-model": "",
	} {
		if got := openAIAudioTask(model); got != want {
			t.Errorf("openAIAudioTask(%q) = %q, want %q", model, got, want)
		}
	}
}

// Mirrors the TS transcription test: one file goes up as multipart, the
// transcript comes back as text with token usage, and nothing is retained.
func TestOpenAIAudioTranscription(t *testing.T) {
	t.Parallel()

	recorder, driver := newAudioServer(t, map[string]func(http.ResponseWriter){
		"/audio/transcriptions": jsonReply(`{"text":"Hello.","usage":{"type":"tokens","input_tokens":8,"output_tokens":2,"total_tokens":10}}`),
	})
	file := &BytesDataSource{FileName: "recording.wav", MIME: "audio/wav", Data: []byte("RIFF")}
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "names: Ada", Files: []DataSource{file}}},
		ExecutionOptions{Model: "gpt-4o-transcribe", ModelOptions: map[string]any{"language": "en"}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(recorder.types[0], "multipart/form-data") {
		t.Fatalf("content type = %q", recorder.types[0])
	}
	for _, want := range []string{`filename="recording.wav"`, "gpt-4o-transcribe", "names: Ada", `name="language"`, "RIFF"} {
		if !strings.Contains(recorder.bodies[0], want) {
			t.Fatalf("multipart body missing %q:\n%s", want, recorder.bodies[0])
		}
	}
	if completionText(resp.Result) != "Hello." || count(resp.TokenUsage.Prompt) != 8 || count(resp.TokenUsage.Total) != 10 {
		t.Fatalf("response = %#v", resp)
	}
	if resp.Conversation != nil {
		t.Fatalf("conversation = %#v", resp.Conversation)
	}

	recorder, driver = newAudioServer(t, map[string]func(http.ResponseWriter){
		"/audio/transcriptions": jsonReply(`{"text":"Hi","segments":[{"id":"s1","speaker":"A","start":0,"end":1,"text":"Hi"}],"usage":{"type":"duration","seconds":1}}`),
	})
	resp, err = driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Files: []DataSource{file}}}, ExecutionOptions{Model: "gpt-4o-transcribe-diarize"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(recorder.bodies[0], "diarized_json") || !strings.Contains(recorder.bodies[0], `name="chunking_strategy"`) {
		t.Fatalf("diarize body = %s", recorder.bodies[0])
	}
	if got := marshalNoEscape(resp.Result); !strings.Contains(got, `"speaker":"A"`) || resp.TokenUsage != nil {
		t.Fatalf("diarize result %s, usage %#v", got, resp.TokenUsage)
	}
}

// Mirrors the TS speech tests: the audio is streamed to the storage sink, the
// result is its durable URI with format metadata, and streaming falls back to
// one chunk.
func TestOpenAIAudioSpeech(t *testing.T) {
	t.Parallel()

	recorder, driver := newAudioServer(t, map[string]func(http.ResponseWriter){
		"/audio/speech": func(w http.ResponseWriter) { _, _ = w.Write([]byte("mp3-bytes")) },
	})
	var stored []byte
	options := ExecutionOptions{Model: "gpt-4o-mini-tts", ModelOptions: map[string]any{"voice": "verse", "speed": 1.5, "instructions": "Calm"}, StoreAudio: storeTo("gs://bucket/speech.mp3", &stored)}
	stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Say hi"}}, options)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()
	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if string(stored) != "mp3-bytes" {
		t.Fatalf("stored = %q", stored)
	}
	if got := marshalNoEscape(recorder.bodies[0]); !strings.Contains(got, `\"voice\":\"verse\"`) || !strings.Contains(got, `\"speed\":1.5`) || !strings.Contains(got, `\"response_format\":\"mp3\"`) {
		t.Fatalf("speech body = %s", recorder.bodies[0])
	}
	if got := marshalNoEscape(stream.Completion().Result); !strings.Contains(got, `"value":"gs://bucket/speech.mp3"`) || !strings.Contains(got, `"mime_type":"audio/mpeg"`) {
		t.Fatalf("result = %s", got)
	}

	prompt := []PromptSegment{{Role: PromptRoleUser, Content: "Say hi"}}
	for name, tc := range map[string]struct {
		segments []PromptSegment
		options  ExecutionOptions
		want     string
	}{
		"no sink":      {prompt, ExecutionOptions{Model: "gpt-4o-mini-tts"}, "storage sink"},
		"too long":     {[]PromptSegment{{Role: PromptRoleUser, Content: strings.Repeat("a", 4097)}}, ExecutionOptions{Model: "gpt-4o-mini-tts", StoreAudio: storeTo("gs://b/o", nil)}, "4096"},
		"audio input":  {[]PromptSegment{{Role: PromptRoleUser, Content: "x", Files: []DataSource{&BytesDataSource{FileName: "a.mp3", MIME: "audio/mpeg", Data: []byte("a")}}}}, ExecutionOptions{Model: "gpt-4o-mini-tts", StoreAudio: storeTo("gs://b/o", nil)}, "text only"},
		"instructions": {prompt, ExecutionOptions{Model: "tts-1", ModelOptions: map[string]any{"instructions": "Whisper"}, StoreAudio: storeTo("gs://b/o", nil)}, "do not support speech instructions"},
		"tools":        {prompt, ExecutionOptions{Model: "tts-1", Tools: []ToolDefinition{{Name: "t"}}}, "do not accept"},
	} {
		recorder, driver := newAudioServer(t, nil)
		_, err := driver.Execute(context.Background(), tc.segments, tc.options)
		if err == nil || !strings.Contains(err.Error(), tc.want) || len(recorder.paths) != 0 {
			t.Errorf("%s: error %v, requests %v", name, err, recorder.paths)
		}
	}

	_, driver = newAudioServer(t, map[string]func(http.ResponseWriter){
		"/audio/speech": func(w http.ResponseWriter) { _, _ = w.Write([]byte("mp3")) },
	})
	_, err = driver.Execute(context.Background(), prompt, ExecutionOptions{Model: "tts-1", StoreAudio: storeTo("https://example.com/signed.mp3", nil)})
	if err == nil || !strings.Contains(err.Error(), "durable object URI") {
		t.Fatalf("transient URI error = %v", err)
	}
	_, err = driver.Execute(context.Background(), prompt, ExecutionOptions{Model: "tts-1", StoreAudio: func(context.Context, io.Reader, common.AudioResult) (string, error) {
		return "", errors.New("storage unavailable")
	}})
	if err == nil || !strings.Contains(err.Error(), "storage unavailable") {
		t.Fatalf("storage error = %v", err)
	}

	_, driver = newAudioServer(t, map[string]func(http.ResponseWriter){
		"/audio/speech": func(w http.ResponseWriter) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"Rate limited","type":"rate_limit_error"}}`))
		},
	})
	_, err = driver.Execute(context.Background(), prompt, ExecutionOptions{Model: "tts-1", StoreAudio: storeTo("gs://b/o", nil)})
	var llumiverseErr *LlumiverseError
	if !errors.As(err, &llumiverseErr) || llumiverseErr.Code != 429 {
		t.Fatalf("rate limit error = %#v", err)
	}
}

// Mirrors the TS gpt-audio tests: the spoken answer is stored as audio with
// PCM metadata, the transcript is the text, and no base64 is retained.
func TestOpenAIAudioUnderstanding(t *testing.T) {
	t.Parallel()

	data := base64.StdEncoding.EncodeToString([]byte("pcm-bytes"))
	recorder, driver := newAudioServer(t, map[string]func(http.ResponseWriter){
		"/chat/completions": jsonReply(`{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":null,"audio":{"id":"a","data":"` + data + `","transcript":"Hello"}}}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`),
	})
	var stored []byte
	file := &BytesDataSource{FileName: "q.wav", MIME: "audio/wav", Data: []byte("RIFF")}
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Answer", Files: []DataSource{file}}},
		ExecutionOptions{Model: "gpt-audio", ModelOptions: map[string]any{"response_format": "pcm16"}, StoreAudio: storeTo("gs://bucket/output.pcm", &stored)})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"modalities":["text","audio"]`, `"audio":{"format":"pcm16","voice":"alloy"}`, `"type":"input_audio"`} {
		if !strings.Contains(recorder.bodies[0], want) {
			t.Fatalf("body %s\nmissing %s", recorder.bodies[0], want)
		}
	}
	if string(stored) != "pcm-bytes" || completionText(resp.Result) != "Hello" {
		t.Fatalf("stored %q, result %#v", stored, resp.Result)
	}
	got := marshalNoEscape(resp.Result)
	if !strings.Contains(got, `"byte_order":"little","channels":1,"codec":"pcm","container":"raw","mime_type":"audio/pcm","sample_encoding":"int16","sample_rate":24000,"type":"audio","value":"gs://bucket/output.pcm"`) || strings.Contains(marshalNoEscape(resp), data) {
		t.Fatalf("result = %s", got)
	}
	if count(resp.TokenUsage.Prompt) != 10 || count(resp.TokenUsage.Result) != 20 {
		t.Fatalf("usage = %#v", resp.TokenUsage)
	}
}

func TestBoundedAudioReader(t *testing.T) {
	t.Parallel()

	if _, err := io.ReadAll(&boundedAudioReader{source: strings.NewReader("123456"), limit: 5}); err == nil || !strings.Contains(err.Error(), "byte limit") {
		t.Fatalf("over limit = %v", err)
	}
	if _, err := io.ReadAll(&boundedAudioReader{source: strings.NewReader(""), limit: 5}); err == nil || !strings.Contains(err.Error(), "empty") {
		t.Fatalf("empty = %v", err)
	}
}
