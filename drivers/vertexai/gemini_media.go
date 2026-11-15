package vertexai

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// geminiInlineMediaMaxBytes is the largest source read for inline media, matching
// Gemini's 100 MB inline file limit; anything bigger must be a gs:// URI. An
// uncapped source would be buffered whole and base64-encoded in memory.
const geminiInlineMediaMaxBytes = 100_000_000

// geminiInlineMediaTotalMaxBytes is the largest total of inline media for one
// request, counting the inline media still in the conversation history. Every
// inlined file is held as base64 until the request is sent, so the per-file cap
// alone does not bound memory.
const geminiInlineMediaTotalMaxBytes = 150_000_000

// geminiInlineAudioMaxBytes is the tighter per-file cap for inline audio input.
const geminiInlineAudioMaxBytes = 25_000_000

// geminiInlineMediaLimitError reports media too large to send inline. The
// caller's input is at fault, so it is a non-retryable 413 rather than an
// unclassified failure attributed to the provider.
type geminiInlineMediaLimitError struct {
	message string
}

func (e *geminiInlineMediaLimitError) Error() string { return e.message }

// StatusCode reports 413 Content Too Large.
func (e *geminiInlineMediaLimitError) StatusCode() int { return http.StatusRequestEntityTooLarge }

// geminiInlineBudget counts the inline bytes read for one request.
type geminiInlineBudget struct {
	bytes int
}

// newGeminiInlineBudget starts a request budget with the inline media an earlier
// turn left in the conversation history, which is sent with this request too.
func newGeminiInlineBudget(conversation any) *geminiInlineBudget {
	stored, _ := geminiStoredConversation(conversation)
	budget := &geminiInlineBudget{}
	add := func(inline *geminiInlineData) {
		if inline != nil {
			budget.bytes += len(inline.Data) * 3 / 4
		}
	}
	contents := stored.Contents
	if stored.System != nil {
		contents = append([]geminiContent{*stored.System}, contents...)
	}
	for _, content := range contents {
		for _, part := range content.Parts {
			add(part.InlineData)
			if part.FunctionResponse != nil {
				for _, media := range part.FunctionResponse.Parts {
					add(media.InlineData)
				}
			}
		}
	}
	return budget
}

// isGeminiFileURI reports whether a URI points at Google Cloud Storage and can be
// passed by reference rather than read and inlined.
func isGeminiFileURI(uri string) bool {
	return isVertexGCSURL(uri)
}

// geminiMediaPart maps a file to a media part: Cloud Storage objects are passed
// by URI without being read; anything else is read, within the per-file and
// per-request inline caps, and inlined as base64.
func geminiMediaPart(ctx context.Context, file DataSource, budget *geminiInlineBudget) (geminiFunctionMediaPart, error) {
	if uri, err := file.URL(ctx); err == nil && isGeminiFileURI(uri) {
		return geminiFunctionMediaPart{FileData: &geminiFileData{FileURI: uri, MIMEType: file.MIMEType()}}, nil
	}
	isAudio := strings.HasPrefix(file.MIMEType(), "audio/")
	limit := geminiInlineMediaMaxBytes
	if isAudio {
		limit = geminiInlineAudioMaxBytes
	}
	rc, err := file.Open(ctx)
	if err != nil {
		return geminiFunctionMediaPart{}, err
	}
	defer func() { _ = rc.Close() }()
	data, err := io.ReadAll(io.LimitReader(rc, int64(limit)+1))
	if err != nil {
		return geminiFunctionMediaPart{}, err
	}
	budget.bytes += len(data)
	if len(data) > limit {
		if isAudio {
			return geminiFunctionMediaPart{}, fmt.Errorf("audio exceeds the %d byte limit", limit)
		}
		return geminiFunctionMediaPart{}, &geminiInlineMediaLimitError{message: fmt.Sprintf(
			"Media file %s exceeds the %d byte limit for inline Gemini input; provide it as a gs:// URI instead", file.Name(), limit)}
	}
	if budget.bytes > geminiInlineMediaTotalMaxBytes {
		return geminiFunctionMediaPart{}, &geminiInlineMediaLimitError{message: fmt.Sprintf(
			"Inline media for this request exceeds the %d byte limit at %s; provide the larger files as gs:// URIs instead", geminiInlineMediaTotalMaxBytes, file.Name())}
	}
	if isAudio && len(data) == 0 {
		return geminiFunctionMediaPart{}, errors.New("audio file is empty")
	}
	return geminiFunctionMediaPart{InlineData: &geminiInlineData{Data: base64.StdEncoding.EncodeToString(data), MIMEType: file.MIMEType()}}, nil
}
