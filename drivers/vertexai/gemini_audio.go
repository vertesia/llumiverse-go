package vertexai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/vertesia/llumiverse-go/core"
	"github.com/vertesia/llumiverse-go/wire"
)

// Gemini file audio: speech synthesis (tts) and transcription (transcribe) models
// answer one generateContent request with a whole file, so they take no
// conversation, tools or schema and are not streamed incrementally. Live and
// native-audio models use separate APIs and are not file audio models.

const (
	// geminiSpeechMaxChars bounds the text of one speech synthesis request.
	geminiSpeechMaxChars = 4096
	// geminiGeneratedAudioMaxBytes bounds a generated audio file.
	geminiGeneratedAudioMaxBytes = 50_000_000
	// geminiDefaultSpeechVoice is the prebuilt voice used when speech_voice is unset.
	geminiDefaultSpeechVoice = "Kore"
)

// durableAudioURIPattern is what StoreAudio must return: an object URI, never
// inline bytes or a signed download URL.
var durableAudioURIPattern = regexp.MustCompile(`^(?:(?:gs|s3)://[^/]+/.+|artifact:.+)`)

// isGeminiFileAudioModel reports a speech synthesis or transcription model.
func isGeminiFileAudioModel(model string) bool {
	model = strings.ToLower(model)
	return (strings.Contains(model, "tts") || strings.Contains(model, "transcribe")) &&
		!strings.Contains(model, "live") && !strings.Contains(model, "native-audio")
}

// isGeminiSpeechModel reports a speech synthesis model.
func isGeminiSpeechModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "tts")
}

// validateGeminiFileAudioPrompt rejects input a file audio model cannot take,
// before any file is read.
func validateGeminiFileAudioPrompt(segments []PromptSegment, options ExecutionOptions) error {
	if options.Conversation != nil || len(options.Tools) > 0 || options.ResultSchema != nil {
		return errors.New("file audio operations do not accept conversation, tools, result schemas, or custom formatting")
	}
	var files []DataSource
	var text []string
	for _, segment := range segments {
		if segment.Role == PromptRoleTool || segment.Role == PromptRoleAssistant {
			return errors.New("file audio operations accept only user and system input")
		}
		files = append(files, segment.Files...)
		text = append(text, segment.Content)
	}
	if isGeminiSpeechModel(options.Model) {
		joined := strings.TrimSpace(strings.Join(text, "\n"))
		if len(files) > 0 || joined == "" || utf8.RuneCountInString(joined) > geminiSpeechMaxChars {
			return fmt.Errorf("speech synthesis requires 1–%d characters and no files", geminiSpeechMaxChars)
		}
		if options.StoreAudio == nil {
			return errors.New("speech synthesis requires a durable audio storage sink")
		}
		return nil
	}
	if len(files) != 1 || !core.IsAudioMIMEType(files[0].MIMEType()) {
		return errors.New("transcription requires exactly one audio file")
	}
	return nil
}

// geminiFileAudioPayload builds the generateContent request of a file audio
// model. Speech sends all the text as one user turn; transcription keeps the
// system instruction and the audio part.
func geminiFileAudioPayload(prompt geminiPrompt, options ExecutionOptions) map[string]any {
	if isGeminiSpeechModel(options.Model) {
		var parts []geminiPart
		if prompt.System != nil {
			parts = append(parts, prompt.System.Parts...)
		}
		for _, content := range prompt.Contents {
			parts = append(parts, content.Parts...)
		}
		speech := map[string]any{
			"voiceConfig": map[string]any{"prebuiltVoiceConfig": map[string]any{"voiceName": geminiSpeechVoice(options.ModelOptions)}},
		}
		if language := optionString(options.ModelOptions, "speech_language"); language != "" {
			speech["languageCode"] = language
		}
		return map[string]any{
			"contents": []geminiContent{{Role: "user", Parts: parts}},
			"generationConfig": map[string]any{
				"responseModalities": []string{"AUDIO"},
				"speechConfig":       speech,
			},
		}
	}
	transcription := removeNilMapValues(map[string]any{
		"languageCodes":    nilIfEmptyStrings(optionStringSlice(options.ModelOptions, "transcription_language_codes")),
		"diarization":      optionBoolPtrValue(options.ModelOptions, "transcription_diarization"),
		"wordTimestamp":    optionBoolPtrValue(options.ModelOptions, "transcription_word_timestamps"),
		"customVocabulary": nilIfEmptyStrings(optionStringSlice(options.ModelOptions, "transcription_vocabulary")),
	})
	payload := map[string]any{
		"contents":         prompt.Contents,
		"generationConfig": map[string]any{"audioTranscriptionConfig": transcription},
	}
	if prompt.System != nil && len(prompt.System.Parts) > 0 {
		payload["systemInstruction"] = prompt.System
	}
	return payload
}

func geminiSpeechVoice(options map[string]any) string {
	if voice := optionString(options, "speech_voice"); voice != "" {
		return voice
	}
	return geminiDefaultSpeechVoice
}

func nilIfEmptyStrings(values []string) any {
	if len(values) == 0 {
		return nil
	}
	return values
}

// geminiFileAudioResponse is the generateContent response of a file audio model.
type geminiFileAudioResponse struct {
	Candidates []struct {
		FinishReason string `json:"finishReason"`
		Content      struct {
			Parts []struct {
				Text       string            `json:"text"`
				InlineData *geminiInlineData `json:"inlineData"`
				// AudioTranscription is the structured transcript of a part.
				AudioTranscription *struct {
					Text         string `json:"text"`
					LanguageCode string `json:"languageCode"`
					SpeakerLabel string `json:"speakerLabel"`
					Words        []struct {
						Word        string `json:"word"`
						StartOffset string `json:"startOffset"`
						EndOffset   string `json:"endOffset"`
					} `json:"words"`
				} `json:"audioTranscription"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata geminiUsageMetadata `json:"usageMetadata"`
}

// nullableString is a transcript field: its value, or JSON null when unset.
func nullableString(value string) any {
	if value == "" {
		return nil
	}
	return value
}

// requestGeminiFileAudio runs a speech synthesis or transcription request. Speech
// audio is handed to ExecutionOptions.StoreAudio and returned as a file result;
// no conversation is returned, so audio never enters history.
func (d *VertexAIDriver) requestGeminiFileAudio(ctx context.Context, prompt geminiPrompt, options ExecutionOptions) (*Completion, error) {
	endpoint, err := d.geminiEndpoint(options.Model, "generateContent")
	if err != nil {
		return nil, err
	}
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	addGeminiServiceTierHeaders(headers, options.ModelOptions)
	var body json.RawMessage
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, geminiFileAudioPayload(prompt, options), &body); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	var response geminiFileAudioResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	speech := isGeminiSpeechModel(options.Model)
	completion := &Completion{TokenUsage: geminiUsage(response.UsageMetadata)}
	if len(response.Candidates) > 0 {
		candidate := response.Candidates[0]
		completion.FinishReason = geminiFinishReason(candidate.FinishReason)
		hasText := false
		for _, part := range candidate.Content.Parts {
			hasText = hasText || part.Text != ""
		}
		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				completion.Result = append(completion.Result, newTextResult(part.Text))
			}
			if transcription := part.AudioTranscription; transcription != nil {
				if transcription.Text != "" && !hasText {
					completion.Result = append(completion.Result, newTextResult(transcription.Text))
				}
				words := make([]map[string]any, 0, len(transcription.Words))
				for _, word := range transcription.Words {
					words = append(words, map[string]any{"word": word.Word, "start_offset": nullableString(word.StartOffset), "end_offset": nullableString(word.EndOffset)})
				}
				completion.Result = append(completion.Result, newJSONResult(map[string]any{
					"text":          transcription.Text,
					"language_code": nullableString(transcription.LanguageCode),
					"speaker_label": nullableString(transcription.SpeakerLabel),
					"words":         words,
				}))
			}
			if part.InlineData != nil && core.IsAudioMIMEType(part.InlineData.MIMEType) {
				if !speech {
					return nil, d.formatError(errors.New("unexpected audio output from transcription model"), options.Model, "execute")
				}
				result, err := storeGeminiSpeech(ctx, part.InlineData, options)
				if err != nil {
					return nil, d.formatError(err, options.Model, "execute")
				}
				completion.Result = append(completion.Result, result)
			}
		}
	}
	if len(completion.Result) == 0 || (speech && !hasAudioResult(completion.Result)) {
		return nil, d.formatError(errors.New("audio model returned no usable result"), options.Model, "execute")
	}
	if options.IncludeOriginalResponse {
		// The original response is reported without its inline audio bytes.
		var raw any
		if json.Unmarshal(body, &raw) == nil {
			completion.OriginalResponse = stripInlineAudio(raw)
		}
	}
	return completion, nil
}

// storeGeminiSpeech decodes Gemini's 24 kHz mono PCM output, within the generated
// audio cap, and hands it to StoreAudio.
func storeGeminiSpeech(ctx context.Context, inline *geminiInlineData, options ExecutionOptions) (CompletionResult, error) {
	if options.StoreAudio == nil {
		return CompletionResult{}, errors.New("speech synthesis requires a durable audio storage sink")
	}
	if len(inline.Data) > (geminiGeneratedAudioMaxBytes+2)/3*4 {
		return CompletionResult{}, fmt.Errorf("audio exceeds the %d byte limit", geminiGeneratedAudioMaxBytes)
	}
	data, err := base64.StdEncoding.DecodeString(inline.Data)
	if err != nil {
		return CompletionResult{}, fmt.Errorf("decode generated audio: %w", err)
	}
	if len(data) == 0 {
		return CompletionResult{}, errors.New("audio file is empty")
	}
	metadata := AudioResult{
		MimeType:       inline.MIMEType,
		Container:      ptr("raw"),
		Codec:          ptr("pcm"),
		SampleRate:     ptr(24000),
		Channels:       ptr(1),
		SampleEncoding: ptr("int16"),
		ByteOrder:      ptr(wire.AudioResultByteOrder("little")),
	}
	value, err := options.StoreAudio(ctx, bytes.NewReader(data), metadata)
	if err != nil {
		return CompletionResult{}, err
	}
	if err := ctx.Err(); err != nil {
		return CompletionResult{}, err
	}
	if !durableAudioURIPattern.MatchString(value) {
		return CompletionResult{}, errors.New("audio storage must return a durable object URI")
	}
	metadata.Value = value
	return newAudioResult(metadata), nil
}

func hasAudioResult(results []CompletionResult) bool {
	for _, result := range results {
		if result.Type() == ResultTypeAudio {
			return true
		}
	}
	return false
}

// stripInlineAudio returns a copy of a decoded response with inline audio data
// removed.
func stripInlineAudio(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, item := range typed {
			if inline, ok := item.(map[string]any); ok && key == "inlineData" {
				if mime, _ := inline["mimeType"].(string); core.IsAudioMIMEType(mime) {
					stripped := make(map[string]any, len(inline))
					for k, v := range inline {
						if k != "data" {
							stripped[k] = v
						}
					}
					out[key] = stripped
					continue
				}
			}
			out[key] = stripInlineAudio(item)
		}
		return out
	case []any:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = stripInlineAudio(item)
		}
		return out
	default:
		return value
	}
}

// requestGeminiFileAudioStream runs a file audio request and delivers the whole
// completion as a single chunk.
func (d *VertexAIDriver) requestGeminiFileAudioStream(ctx context.Context, prompt geminiPrompt, options ExecutionOptions) (CompletionStream, error) {
	completion, err := d.requestGeminiFileAudio(ctx, prompt, options)
	if err != nil {
		return nil, err
	}
	ch := make(chan streamItem, 1)
	ch <- streamItem{Chunk: CompletionChunk{Result: completion.Result, FinishReason: completion.FinishReason, TokenUsage: completion.TokenUsage}}
	close(ch)
	return newChannelStreamWithFinalizer(ch, &ExecutionResponse{Prompt: prompt}, func() error { return nil }, func(resp *ExecutionResponse) {
		resp.Result = completion.Result
		resp.OriginalResponse = completion.OriginalResponse
		resp.Conversation = nil
	}), nil
}

// checkGeminiAudioOutput rejects inline audio from a model outside the file
// audio path, which has nowhere durable to put it.
func checkGeminiAudioOutput(response geminiResponse) error {
	for _, candidate := range response.Candidates {
		for _, part := range candidate.Content.Parts {
			if part.InlineData != nil && core.IsAudioMIMEType(part.InlineData.MIMEType) {
				return errors.New("audio output requires a file speech model")
			}
		}
	}
	return nil
}
