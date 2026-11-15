package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"
)

// Gemini Omni video models run on the Vertex Interactions API (v1beta1, global)
// and write the generated videos to a Cloud Storage prefix.
const (
	geminiOmniVideoModel   = "gemini-omni-flash-preview"
	geminiOmni11VideoModel = "gemini-omni-1.1-flash-preview"
	geminiOmni12VideoModel = "gemini-omni-1.2-flash"
)

// geminiOmniVideoModels supplements the listing, which does not return them.
// Execution routes every gemini-omni-* model, so a model missing here still runs.
var geminiOmniVideoModels = []string{geminiOmniVideoModel, geminiOmni11VideoModel, geminiOmni12VideoModel}

var (
	geminiOmniModelPattern = regexp.MustCompile(`^gemini-omni(?:-(\d+)(?:\.(\d+))?)?-[a-z][a-z0-9]*(?:-.+)?$`)
	omniGCSURIPattern      = regexp.MustCompile(`^gs://([^/]+)/(.*)$`)
)

var omniImageMIMETypes = map[string]bool{"image/png": true, "image/jpeg": true, "image/webp": true, "image/heic": true, "image/heif": true}

var omniVideoMIMETypes = map[string]bool{
	"video/3gpp": true, "video/mp4": true, "video/mpeg": true, "video/mpegs": true, "video/mpg": true,
	"video/quicktime": true, "video/webm": true, "video/wmv": true, "video/x-flv": true, "video/x-ms-wmv": true,
}

// omniUpscaleResolutions are the resolutions Omni upscales to.
var omniUpscaleResolutions = map[string]bool{"720p": true, "1080p": true, "4k": true}

// geminiOmniVersion parses the Omni version of a model ID:
// gemini-omni-flash-preview is 1.0, later releases carry a version before the
// tier (gemini-omni-1.1-flash-preview, gemini-omni-1.2-flash).
func geminiOmniVersion(model string) (major int, minor int, ok bool) {
	name := strings.ToLower(model)
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i+1:]
	}
	match := geminiOmniModelPattern.FindStringSubmatch(name)
	if match == nil {
		return 0, 0, false
	}
	major, minor = 1, 0
	if match[1] != "" {
		major, _ = strconv.Atoi(match[1])
	}
	if match[2] != "" {
		minor, _ = strconv.Atoi(match[2])
	}
	return major, minor, true
}

// isGeminiOmniVideoModel reports whether a model ID is a Gemini Omni video model.
func isGeminiOmniVideoModel(model string) bool {
	_, _, ok := geminiOmniVersion(model)
	return ok
}

// isGeminiOmniVersionGTE reports whether an Omni model is at least major.minor.
func isGeminiOmniVersionGTE(model string, major, minor int) bool {
	gotMajor, gotMinor, ok := geminiOmniVersion(model)
	return ok && (gotMajor > major || (gotMajor == major && gotMinor >= minor))
}

// omniVideoTerminalError is an Omni failure another attempt does not fix, such as
// invalid provider output.
type omniVideoTerminalError struct{ message string }

func (e *omniVideoTerminalError) Error() string { return e.message }

// omniVideoRequestError is a request this model cannot run, rejected before
// anything is sent: the caller's to fix, never retried.
type omniVideoRequestError struct{ message string }

func (e *omniVideoRequestError) Error() string { return e.message }

// omniVideoInteractionError is an interaction that did not complete; its
// retryability follows the interaction status and errors.
type omniVideoInteractionError struct {
	message   string
	retryable *bool
}

func (e *omniVideoInteractionError) Error() string { return e.message }

func newOmniRequestError(format string, args ...any) error {
	return &omniVideoRequestError{message: fmt.Sprintf(format, args...)}
}

func isOmniGCSURI(value string, requireObject bool) bool {
	match := omniGCSURIPattern.FindStringSubmatch(value)
	return match != nil && (!requireObject || match[2] != "")
}

// omniVideoMedia is one image or video input of an interaction.
type omniVideoMedia struct {
	Type     string `json:"type"`
	URI      string `json:"uri"`
	MIMEType string `json:"mime_type"`
}

// omniVideoPrompt is the text and the Cloud Storage media of an Omni request.
type omniVideoPrompt struct {
	Text  string           `json:"text"`
	Media []omniVideoMedia `json:"media"`
}

// formatOmniVideoPrompt joins the segment text and maps each file to an image or
// video input by its Cloud Storage URI (the bytes are never read). Conversation
// resume, tools and result schemas are not supported.
func formatOmniVideoPrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (omniVideoPrompt, error) {
	if options.Conversation != nil {
		return omniVideoPrompt{}, newOmniRequestError("Gemini Omni video does not support conversation resume")
	}
	if len(options.Tools) > 0 {
		return omniVideoPrompt{}, newOmniRequestError("Gemini Omni video does not support tools")
	}
	if options.ResultSchema != nil {
		return omniVideoPrompt{}, newOmniRequestError("Gemini Omni video does not support result schemas")
	}
	var lines []string
	for _, segment := range segments {
		if text := strings.TrimSpace(segment.Content); text != "" {
			lines = append(lines, text)
		}
	}
	if len(lines) == 0 {
		return omniVideoPrompt{}, newOmniRequestError("Gemini Omni video requires a non-empty text prompt")
	}
	prompt := omniVideoPrompt{Text: strings.Join(lines, "\n"), Media: []omniVideoMedia{}}
	for _, segment := range segments {
		for _, file := range segment.Files {
			mimeType := file.MIMEType()
			kind := ""
			switch {
			case omniImageMIMETypes[mimeType]:
				kind = "image"
			case omniVideoMIMETypes[mimeType]:
				kind = "video"
			default:
				return omniVideoPrompt{}, newOmniRequestError("Gemini Omni video does not support input MIME type %s", mimeType)
			}
			uri, err := file.URL(ctx)
			if err != nil {
				return omniVideoPrompt{}, err
			}
			if !isOmniGCSURI(uri, true) {
				return omniVideoPrompt{}, newOmniRequestError("Gemini Omni video media inputs must expose a GCS object URI")
			}
			prompt.Media = append(prompt.Media, omniVideoMedia{Type: kind, URI: uri, MIMEType: mimeType})
		}
	}
	if _, err := resolveOmniVideoTask(options.Model, options.ModelOptions, prompt.Media); err != nil {
		return omniVideoPrompt{}, err
	}
	return prompt, nil
}

// resolveOmniVideoTask returns the video task (option task, else text_to_video
// without media) after checking the media each task and model version accepts.
func resolveOmniVideoTask(model string, modelOptions map[string]any, media []omniVideoMedia) (string, error) {
	images, videos := 0, 0
	for _, item := range media {
		if item.Type == "image" {
			images++
		} else {
			videos++
		}
	}
	task := optionString(modelOptions, "task")
	if task == "" {
		if len(media) > 0 {
			return "", newOmniRequestError("Gemini Omni video generation with media requires an explicit task")
		}
		task = "text_to_video"
	}
	name := shortResourceName(model)
	omni11 := isGeminiOmniVersionGTE(name, 1, 1)
	// Omni 1.2 accepts up to two reference images with the video to edit or extend.
	maxReferenceImages := 0
	if isGeminiOmniVersionGTE(name, 1, 2) {
		maxReferenceImages = 2
	}
	resolution := optionString(modelOptions, "resolution")
	if task == "extend" && !omni11 {
		return "", newOmniRequestError("%s does not support extend", name)
	}
	if resolution != "" && !omni11 && resolution != "720p" {
		return "", newOmniRequestError("%s only supports 720p output", name)
	}
	switch task {
	case "text_to_video":
		if len(media) != 0 {
			return "", newOmniRequestError("text_to_video does not accept media inputs")
		}
	case "image_to_video":
		maxImages, expected := 1, "exactly one image input"
		if omni11 {
			maxImages, expected = 2, "one or two image inputs"
		}
		if videos != 0 || images < 1 || images > maxImages {
			return "", newOmniRequestError("image_to_video requires %s", expected)
		}
	case "reference_to_video":
		if len(media) == 0 || images > 10 || videos > 3 {
			return "", newOmniRequestError("reference_to_video requires media with at most ten images and three videos")
		}
	case "edit", "extend":
		if images > maxReferenceImages || videos != 1 {
			suffix := ""
			if maxReferenceImages > 0 {
				suffix = fmt.Sprintf(" and at most %d reference images", maxReferenceImages)
			}
			return "", newOmniRequestError("%s requires exactly one video input%s", task, suffix)
		}
	case "upscale":
		if !isGeminiOmniVersionGTE(name, 1, 2) {
			return "", newOmniRequestError("%s does not support upscale", name)
		}
		if images != 0 || videos != 1 {
			return "", newOmniRequestError("upscale requires exactly one video input")
		}
		// Vertex rejects an upscale without a target resolution, and the driver
		// cannot infer one from the input.
		if !omniUpscaleResolutions[resolution] {
			return "", newOmniRequestError("upscale requires a target resolution: 720p, 1080p or 4k")
		}
	}
	return task, nil
}

// omniInteraction is the Interactions API response fields the driver reads.
type omniInteraction struct {
	ID     string            `json:"id"`
	Status string            `json:"status"`
	Errors []json.RawMessage `json:"errors,omitempty"`
	Usage  *struct {
		TotalTokens       *int `json:"total_tokens"`
		TotalInputTokens  *int `json:"total_input_tokens"`
		TotalOutputTokens *int `json:"total_output_tokens"`
	} `json:"usage,omitempty"`
	Steps []struct {
		Type    string `json:"type"`
		Content []struct {
			Type     string  `json:"type"`
			Text     string  `json:"text"`
			URI      string  `json:"uri"`
			Data     *string `json:"data"`
			MIMEType string  `json:"mime_type"`
		} `json:"content"`
	} `json:"steps,omitempty"`
}

// omniInteractionFailureRetryable classifies a failed interaction by its errors.
func omniInteractionFailureRetryable(errors []json.RawMessage) *bool {
	data, _ := json.Marshal(errors)
	details := strings.ToLower(string(data))
	for _, marker := range []string{"invalid_argument", "unauthenticated", "permission_denied", "not_found", "failed_precondition", "out_of_range", "unimplemented"} {
		if strings.Contains(details, marker) {
			return ptr(false)
		}
	}
	for _, marker := range []string{"resource_exhausted", "rate_limit", "throttl", "aborted", "internal", "unavailable", "deadline_exceeded", "timeout"} {
		if strings.Contains(details, marker) {
			return ptr(true)
		}
	}
	return nil
}

// parseOmniResults returns the text and video results of a completed
// interaction, rejecting incomplete interactions and videos that are inline,
// outside the output prefix, or not videos.
func parseOmniResults(response omniInteraction, outputPrefix string) ([]CompletionResult, error) {
	if response.Status != "completed" {
		details := ""
		if len(response.Errors) > 0 {
			data, _ := json.Marshal(response.Errors)
			details = ": " + string(data)
		}
		var retryable *bool
		switch response.Status {
		case "requires_action", "cancelled", "budget_exceeded":
			retryable = ptr(false)
		case "queued", "in_progress", "incomplete":
			retryable = ptr(true)
		default:
			retryable = omniInteractionFailureRetryable(response.Errors)
		}
		return nil, &omniVideoInteractionError{message: fmt.Sprintf("Gemini Omni video interaction did not complete (status: %s)%s", response.Status, details), retryable: retryable}
	}
	results := []CompletionResult{}
	hasVideo := false
	for _, step := range response.Steps {
		if step.Type != "model_output" {
			continue
		}
		for _, part := range step.Content {
			switch part.Type {
			case "text":
				if part.Text != "" {
					results = append(results, newTextResult(part.Text))
				}
			case "video":
				hasVideo = true
				switch {
				case part.Data != nil:
					return nil, &omniVideoTerminalError{"Gemini Omni returned inline video data instead of URI delivery"}
				case part.URI == "" || !isOmniGCSURI(part.URI, true):
					return nil, &omniVideoTerminalError{"Gemini Omni returned a missing or invalid GCS video URI"}
				case !strings.HasPrefix(part.URI, outputPrefix):
					return nil, &omniVideoTerminalError{"Gemini Omni returned a video URI outside the requested output prefix"}
				case part.MIMEType != "" && !strings.HasPrefix(part.MIMEType, "video/"):
					return nil, &omniVideoTerminalError{"Gemini Omni returned an invalid video MIME type: " + part.MIMEType}
				}
				results = append(results, newVideoResult(part.URI))
			}
		}
	}
	if !hasVideo {
		return nil, &omniVideoTerminalError{"Gemini Omni completed without a video output URI"}
	}
	return results, nil
}

// omniVideoPayload builds the interactions request: the text and media input, a
// URI-delivered video response format under the output prefix, and the task.
func omniVideoPayload(model string, prompt omniVideoPrompt, task string, outputPrefix string, modelOptions map[string]any) map[string]any {
	format := map[string]any{"type": "video", "delivery": "uri", "gcs_uri": outputPrefix}
	// Generation requests send a 5-second duration when omitted, as the documented
	// request bodies always carry one; an upscale keeps the input duration.
	duration := optionInt(modelOptions, "duration_seconds")
	if task != "upscale" || duration != nil {
		seconds := 5
		if duration != nil {
			seconds = *duration
		}
		format["duration"] = fmt.Sprintf("%ds", seconds)
	}
	if aspect := optionString(modelOptions, "aspect_ratio"); aspect != "" {
		format["aspect_ratio"] = aspect
	}
	if resolution := optionString(modelOptions, "resolution"); resolution != "" {
		format["resolution"] = resolution
	}
	input := []any{map[string]any{"type": "text", "text": prompt.Text}}
	for _, media := range prompt.Media {
		input = append(input, media)
	}
	generation := map[string]any{"video_config": map[string]any{"task": task}}
	if seed := optionInt(modelOptions, "seed"); seed != nil {
		generation["seed"] = *seed
	}
	return map[string]any{
		"model":             model,
		"input":             input,
		"response_format":   []any{format},
		"generation_config": generation,
	}
}

// requestOmniVideo creates a Gemini Omni interaction and returns its video URIs.
func (d *VertexAIDriver) requestOmniVideo(ctx context.Context, prompt omniVideoPrompt, options ExecutionOptions) (*Completion, error) {
	model := shortResourceName(options.Model)
	task, err := resolveOmniVideoTask(model, options.ModelOptions, prompt.Media)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	outputPrefix := options.OutputStorageURI
	if outputPrefix == "" || !isOmniGCSURI(outputPrefix, false) {
		return nil, d.formatError(newOmniRequestError("Gemini Omni video generation requires a valid GCS output prefix"), options.Model, "execute")
	}
	if !strings.HasSuffix(outputPrefix, "/") {
		outputPrefix += "/"
	}
	base := d.options.BaseURL
	if base == "" {
		base = "https://aiplatform.googleapis.com/v1beta1"
	}
	endpoint, err := joinEndpoint(base, "projects", d.options.Project, "locations", "global", "interactions")
	if err != nil {
		return nil, err
	}
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	var response omniInteraction
	var raw json.RawMessage
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, omniVideoPayload(model, prompt, task, outputPrefix, options.ModelOptions), &raw); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	results, err := parseOmniResults(response, outputPrefix)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	completion := &Completion{Result: results, FinishReason: "stop"}
	if usage := response.Usage; usage != nil {
		completion.TokenUsage = &ExecutionTokenUsage{}
		if usage.TotalTokens != nil {
			completion.TokenUsage.Total = tokens(*usage.TotalTokens)
		}
		if usage.TotalInputTokens != nil {
			completion.TokenUsage.Prompt = tokens(*usage.TotalInputTokens)
		}
		if usage.TotalOutputTokens != nil {
			completion.TokenUsage.Result = tokens(*usage.TotalOutputTokens)
		}
	}
	if options.IncludeOriginalResponse {
		var original any
		_ = json.Unmarshal(raw, &original)
		completion.OriginalResponse = original
	}
	return completion, nil
}

// formatOmniVideoError classifies Omni errors: a rejected request is a
// non-retryable 400, invalid provider output and the caller's cancellation are
// non-retryable, an incomplete interaction follows its status, and anything else
// (transport, HTTP status) goes through the shared status classification.
func formatOmniVideoError(err error, model string, operation string) error {
	if err == nil {
		return nil
	}
	var llumiErr *LlumiverseError
	if errors.As(err, &llumiErr) {
		return err
	}
	canceled := errors.Is(err, context.Canceled)
	context := LlumiverseErrorContext{Provider: ProviderVertexAI, Model: model, Operation: operation}
	var requestErr *omniVideoRequestError
	var terminalErr *omniVideoTerminalError
	var interactionErr *omniVideoInteractionError
	switch {
	case errors.As(err, &requestErr):
		return newLlumiverseError(requestErr.Error(), ptr(false), context, err, 400, "OmniVideoRequestError")
	case errors.As(err, &terminalErr):
		return newLlumiverseError(terminalErr.Error(), ptr(false), context, err, 0, "OmniVideoTerminalError")
	case errors.As(err, &interactionErr):
		return newLlumiverseError(interactionErr.Error(), interactionErr.retryable, context, err, 0, "OmniVideoInteractionError")
	case canceled:
		return newLlumiverseError(err.Error(), ptr(false), context, err, 0, "AbortError")
	}
	code, name := errorStatusAndName(err)
	return withVertexStorageHint(newLlumiverseError(err.Error(), retryableFromStatusAndMessage(code, err.Error()), context, err, code, name))
}
