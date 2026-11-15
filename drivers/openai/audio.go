package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/openai/openai-go/option"
	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/wire"
)

// openAIMaxAudioBytes bounds an uploaded audio file, as the API does (25 MB).
const openAIMaxAudioBytes = 25_000_000

var (
	openAIRealtimePattern      = regexp.MustCompile(`realtime|live`)
	openAITranscriptionPattern = regexp.MustCompile(`whisper|transcribe`)
	openAISpeechPattern        = regexp.MustCompile(`(?:^|[-_.])tts(?:[-_.]|$)`)
	openAIAudioPattern         = regexp.MustCompile(`(?:^|[-_.])audio(?:[-_.]|$)`)
	openAITTS1Pattern          = regexp.MustCompile(`^tts-1(?:-|$)`)
)

// openAIAudioTask classifies the file-audio models served outside text
// generation: "transcription" (whisper, *-transcribe), "speech" (tts) and
// "understanding" (audio chat models). Realtime models are not file audio.
func openAIAudioTask(model string) string {
	name := strings.ToLower(model)
	if i := strings.LastIndex(name, "::"); i >= 0 {
		name = name[i+2:]
	}
	name = shortResourceName(name)
	switch {
	case openAIRealtimePattern.MatchString(name) || strings.Contains(name, "native-audio"):
		return ""
	case openAITranscriptionPattern.MatchString(name):
		return "transcription"
	case openAISpeechPattern.MatchString(name):
		return "speech"
	case openAIAudioPattern.MatchString(name):
		return "understanding"
	}
	return ""
}

// isOpenAIAudioModel reports whether a request goes to the file-audio
// endpoints: an audio model family on the OpenAI or OpenAI-compatible provider.
func (d *OpenAICompatibleDriver) isOpenAIAudioModel(model string) bool {
	provider := d.Provider()
	return (provider == ProviderOpenAI || provider == ProviderOpenAICompatible) && openAIAudioTask(model) != ""
}

// executeOpenAIAudio runs a file-audio request: transcription, speech
// synthesis or audio chat. Generated audio is handed to
// ExecutionOptions.StoreAudio, and the result carries its URI, never bytes.
func (d *OpenAICompatibleDriver) executeOpenAIAudio(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (*ExecutionResponse, error) {
	start := time.Now()
	completion, err := d.openAIAudioCompletion(ctx, segments, options)
	if err != nil {
		var llumiverseErr *LlumiverseError
		if errors.As(err, &llumiverseErr) {
			return nil, err
		}
		return nil, newLlumiverseError(err.Error(), boolPtr(false), LlumiverseErrorContext{Provider: d.Provider(), Model: options.Model, Operation: "execute"}, err, 400, "OpenAIAudioError")
	}
	return &ExecutionResponse{Completion: *completion, Prompt: openAIResponsesPrompt{}, ExecutionTime: time.Since(start)}, nil
}

func (d *OpenAICompatibleDriver) openAIAudioCompletion(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (*Completion, error) {
	if options.Conversation != nil || len(options.Tools) > 0 || options.ResultSchema != nil {
		return nil, errors.New("file audio operations do not accept conversation, tools, or result schemas")
	}
	var files []DataSource
	var texts []string
	for _, segment := range segments {
		if segment.Role == PromptRoleTool || segment.Role == PromptRoleAssistant {
			return nil, errors.New("file audio operations accept only user and system input")
		}
		files = append(files, segment.Files...)
		texts = append(texts, segment.Content)
	}
	text := strings.TrimSpace(strings.Join(texts, "\n"))
	switch openAIAudioTask(options.Model) {
	case "understanding":
		return d.openAIAudioChat(ctx, files, text, options)
	case "transcription":
		return d.openAITranscription(ctx, files, text, options)
	default:
		return d.openAISpeech(ctx, files, text, options)
	}
}

// openAIAudioChat asks an audio chat model, with at most one MP3 or WAV file,
// for a spoken answer.
func (d *OpenAICompatibleDriver) openAIAudioChat(ctx context.Context, files []DataSource, text string, options ExecutionOptions) (*Completion, error) {
	if len(files) > 1 {
		return nil, errors.New("audio understanding accepts at most one audio file")
	}
	if len(files) == 0 && text == "" {
		return nil, errors.New("audio understanding requires text or an audio file")
	}
	content := []any{map[string]any{"type": "text", "text": text}}
	if len(files) == 1 {
		part, err := openAIInputAudioPart(ctx, files[0])
		if err != nil {
			return nil, errors.New("OpenAI audio chat requires MP3 or WAV input")
		}
		content = append(content, part)
	}
	if options.StoreAudio == nil {
		return nil, errors.New("audio generation requires a durable audio storage sink")
	}
	format := optionString(options.ModelOptions, "response_format")
	if format == "" {
		format = "wav"
	}
	voice := optionString(options.ModelOptions, "voice")
	if voice == "" {
		voice = "alloy"
	}
	payload := map[string]any{
		"model":      options.Model,
		"messages":   []any{map[string]any{"role": "user", "content": content}},
		"modalities": []string{"text", "audio"},
		"audio":      map[string]any{"voice": voice, "format": format},
	}
	var raw json.RawMessage
	if err := d.sdk.Post(ctx, "chat/completions", payload, &raw, d.requestOptions(options.HTTPTimeout)...); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	var response struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content *string `json:"content"`
				Audio   *struct {
					Data       string `json:"data"`
					Transcript string `json:"transcript"`
				} `json:"audio"`
			} `json:"message"`
		} `json:"choices"`
		Usage *chatCompletionsUsage `json:"usage"`
	}
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, err
	}
	if len(response.Choices) == 0 || response.Choices[0].Message.Audio == nil || response.Choices[0].Message.Audio.Data == "" {
		return nil, errors.New("OpenAI audio chat returned no audio data")
	}
	choice := response.Choices[0]
	data, err := base64.StdEncoding.DecodeString(choice.Message.Audio.Data)
	if err != nil {
		return nil, err
	}
	audio, err := storeOpenAIAudio(ctx, bytes.NewReader(data), format, options)
	if err != nil {
		return nil, err
	}
	var results []CompletionResult
	answer := choice.Message.Audio.Transcript
	if choice.Message.Content != nil && *choice.Message.Content != "" {
		answer = *choice.Message.Content
	}
	if answer != "" {
		results = append(results, newTextResult(answer))
	}
	completion := &Completion{
		Result:       append(results, audio),
		FinishReason: choice.FinishReason,
		TokenUsage:   openAIChatCompletionsUsage(response.Usage),
	}
	if options.IncludeOriginalResponse {
		var original any
		_ = json.Unmarshal(raw, &original)
		completion.OriginalResponse = original
	}
	return completion, nil
}

// openAITranscription transcribes exactly one audio file; diarize models
// return speaker segments as a JSON result.
func (d *OpenAICompatibleDriver) openAITranscription(ctx context.Context, files []DataSource, text string, options ExecutionOptions) (*Completion, error) {
	if len(files) != 1 {
		return nil, errors.New("transcription requires exactly one audio file")
	}
	diarize := strings.Contains(options.Model, "diarize")
	if diarize && text != "" {
		return nil, errors.New("diarized transcription does not accept a prompt")
	}
	file := files[0]
	reader, err := file.Open(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { _ = reader.Close() }()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	fields := map[string]string{"model": options.Model, "response_format": "json"}
	if diarize {
		fields["response_format"] = "diarized_json"
		fields["chunking_strategy"] = "auto"
	} else if text != "" {
		fields["prompt"] = text
	}
	if language := optionString(options.ModelOptions, "language"); language != "" {
		fields["language"] = language
	}
	for key, value := range fields {
		if err := form.WriteField(key, value); err != nil {
			return nil, err
		}
	}
	part, err := form.CreateFormFile("file", file.Name())
	if err != nil {
		return nil, err
	}
	// The endpoint owns codec validation; the size bound is the API's.
	n, err := io.Copy(part, io.LimitReader(reader, openAIMaxAudioBytes+1))
	if err != nil {
		return nil, err
	}
	if n > openAIMaxAudioBytes {
		return nil, fmt.Errorf("audio exceeds the %d byte limit", openAIMaxAudioBytes)
	}
	if n == 0 {
		return nil, errors.New("audio file is empty")
	}
	if err := form.Close(); err != nil {
		return nil, err
	}
	var raw json.RawMessage
	requestOptions := append(d.requestOptions(options.HTTPTimeout), option.WithRequestBody(form.FormDataContentType(), body.Bytes()))
	if err := d.sdk.Post(ctx, "audio/transcriptions", nil, &raw, requestOptions...); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	var response struct {
		Text     string `json:"text"`
		Segments []struct {
			ID      any     `json:"id"`
			Speaker string  `json:"speaker"`
			Start   float64 `json:"start"`
			End     float64 `json:"end"`
			Text    string  `json:"text"`
		} `json:"segments"`
		Usage *struct {
			Type         string `json:"type"`
			InputTokens  int    `json:"input_tokens"`
			OutputTokens int    `json:"output_tokens"`
			TotalTokens  int    `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, err
	}
	results := []CompletionResult{newTextResult(response.Text)}
	if diarize {
		segments := make([]any, len(response.Segments))
		for i, segment := range response.Segments {
			segments[i] = map[string]any{"id": segment.ID, "speaker": segment.Speaker, "start": segment.Start, "end": segment.End, "text": segment.Text}
		}
		results = append(results, common.NewJSONResult(map[string]any{"segments": segments}))
	}
	completion := &Completion{Result: results, FinishReason: "stop"}
	// Duration-billed transcription has no token counts; none are invented.
	if response.Usage != nil && response.Usage.Type == "tokens" {
		completion.TokenUsage = &ExecutionTokenUsage{Prompt: tokens(response.Usage.InputTokens), Result: tokens(response.Usage.OutputTokens), Total: tokens(response.Usage.TotalTokens)}
	}
	if options.IncludeOriginalResponse {
		var original any
		_ = json.Unmarshal(raw, &original)
		completion.OriginalResponse = original
	}
	return completion, nil
}

// openAISpeech synthesizes 1 to 4096 characters of text.
func (d *OpenAICompatibleDriver) openAISpeech(ctx context.Context, files []DataSource, text string, options ExecutionOptions) (*Completion, error) {
	if len(files) > 0 {
		return nil, errors.New("speech synthesis accepts text only")
	}
	if text == "" || len([]rune(text)) > 4096 {
		return nil, errors.New("speech synthesis requires 1-4096 characters")
	}
	if options.StoreAudio == nil {
		return nil, errors.New("speech synthesis requires a durable audio storage sink")
	}
	mo := options.ModelOptions
	instructions := optionString(mo, "instructions")
	if instructions != "" && openAITTS1Pattern.MatchString(shortResourceName(options.Model)) {
		return nil, errors.New("tts-1 models do not support speech instructions")
	}
	format := optionString(mo, "response_format")
	if format == "" {
		format = "mp3"
	}
	voice := optionString(mo, "voice")
	if voice == "" {
		voice = "alloy"
	}
	payload := map[string]any{"model": options.Model, "input": text, "voice": voice, "response_format": format}
	if speed := optionNumber(mo, "speed"); speed != nil {
		payload["speed"] = speed
	}
	if instructions != "" {
		payload["instructions"] = instructions
	}
	var res *http.Response
	if err := d.sdk.Post(ctx, "audio/speech", payload, &res, d.requestOptions(options.HTTPTimeout)...); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	defer func() { _ = res.Body.Close() }()
	audio, err := storeOpenAIAudio(ctx, res.Body, format, options)
	if err != nil {
		return nil, err
	}
	return &Completion{Result: []CompletionResult{audio}, FinishReason: "stop"}, nil
}

// openAIMaxStoredAudioBytes bounds generated audio handed to the storage sink.
const openAIMaxStoredAudioBytes = 50_000_000

// durableAudioURIPattern matches the object URIs a storage sink must return:
// signed or transient URLs would expire with the stored completion.
var durableAudioURIPattern = regexp.MustCompile(`^(?:(?:gs|s3)://[^/]+/.+|artifact:.+)`)

// storeOpenAIAudio hands generated audio to the caller's sink, bounded and
// non-empty, and returns the audio result for the durable URI it gives back.
func storeOpenAIAudio(ctx context.Context, audio io.Reader, format string, options ExecutionOptions) (CompletionResult, error) {
	metadata := openAIAudioMetadata(format)
	uri, err := options.StoreAudio(ctx, &boundedAudioReader{source: audio, limit: openAIMaxStoredAudioBytes}, metadata)
	if err != nil {
		return CompletionResult{}, err
	}
	if !durableAudioURIPattern.MatchString(uri) {
		return CompletionResult{}, errors.New("audio storage must return a durable object URI")
	}
	metadata.Value = uri
	return common.NewAudioResult(metadata), nil
}

// boundedAudioReader fails once more than limit bytes are read, and at EOF
// when the source was empty, so neither reaches storage as valid audio.
type boundedAudioReader struct {
	source io.Reader
	limit  int64
	read   int64
}

func (r *boundedAudioReader) Read(p []byte) (int, error) {
	n, err := r.source.Read(p)
	r.read += int64(n)
	if r.read > r.limit {
		return 0, fmt.Errorf("audio exceeds the %d byte limit", r.limit)
	}
	if errors.Is(err, io.EOF) && r.read == 0 {
		return 0, errors.New("audio file is empty")
	}
	return n, err
}

// openAIAudioMetadata describes the audio format OpenAI returns.
func openAIAudioMetadata(format string) common.AudioResult {
	if format == "pcm16" {
		format = "pcm"
	}
	mimeTypes := map[string]string{"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac", "pcm": "audio/pcm"}
	container := format
	switch format {
	case "opus":
		container = "ogg"
	case "pcm":
		container = "raw"
	}
	metadata := common.AudioResult{MimeType: mimeTypes[format], Container: &container}
	switch format {
	case "mp3":
		metadata.Codec = ptr("mp3")
	case "pcm":
		order := wire.AudioResultByteOrderLittle
		metadata.Codec = ptr("pcm")
		metadata.SampleRate = ptr(24000)
		metadata.Channels = ptr(1)
		metadata.SampleEncoding = ptr("int16")
		metadata.ByteOrder = &order
	}
	return metadata
}

// openAIAudioStream runs a file-audio request and returns it as a one-chunk stream.
func (d *OpenAICompatibleDriver) openAIAudioStream(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (CompletionStream, error) {
	resp, err := d.executeOpenAIAudio(ctx, segments, options)
	if err != nil {
		return nil, err
	}
	ch := make(chan streamItem, 1)
	ch <- streamItem{Chunk: CompletionChunk{Result: resp.Result, TokenUsage: resp.TokenUsage, FinishReason: resp.FinishReason}}
	close(ch)
	completion := &ExecutionResponse{Prompt: resp.Prompt}
	return newChannelStreamWithFinalizer(ch, completion, func() error { return nil }, nil), nil
}
