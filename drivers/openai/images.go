package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/textproto"
	"regexp"
	"strings"

	"github.com/openai/openai-go/v3/option"
	"github.com/vertesia/llumiverse-go/capability"
	"github.com/vertesia/llumiverse-go/wire"
)

// isImageModel reports image generation models (GPT Image, DALL-E and the
// other image families of the capability catalogue), which run on the Images
// API, as the TS client decides by the model's catalogue family.
func (d *OpenAICompatibleDriver) isImageModel(model string) bool {
	return capability.ResolveProfile(model, d.Provider()).Family == "image"
}

var openAIImageVersionPattern = regexp.MustCompile(`(?:^|[./:])(?:openai\.)?gpt-image-(\d+)(?:\.(\d+))?(?:-|$)`)

// isOpenAIImageVersionGTE reports whether a GPT Image model is at least the
// given version.
func isOpenAIImageVersionGTE(model string, major, minor int) bool {
	match := openAIImageVersionPattern.FindStringSubmatch(strings.ToLower(model))
	if match == nil {
		return false
	}
	return versionGTE(atoiOrZero(match[1]), atoiOrZero(match[2]), major, minor)
}

// openAIImageKeepsInputFidelity reports whether the model takes input_fidelity:
// GPT Image 2 before 2.5 does not.
func openAIImageKeepsInputFidelity(model string) bool {
	return !isOpenAIImageVersionGTE(model, 2, 0) || isOpenAIImageVersionGTE(model, 2, 5)
}

// openAIImageOutputDataURL wraps base64 image output in a data URL for the
// format the provider returned, else the one requested, else PNG.
func openAIImageOutputDataURL(data, returned, requested string) string {
	if strings.HasPrefix(data, "data:") {
		return data
	}
	format := returned
	if format == "" {
		format = requested
	}
	if format == "" {
		format = "png"
	}
	if format == "svg" {
		format = "svg+xml"
	}
	return "data:image/" + format + ";base64," + data
}

// openAIImageInputs splits a prompt's files into reference images and masks,
// in order: files of mask segments are masks, image files of the other
// segments (bar negative prompts) are references.
func openAIImageInputs(segments []PromptSegment) (images, masks []DataSource) {
	for _, segment := range segments {
		for _, file := range segment.Files {
			if mime := file.MIMEType(); mime != "" && !strings.HasPrefix(mime, "image/") {
				continue
			}
			switch segment.Role {
			case PromptRoleMask:
				masks = append(masks, file)
			case PromptRoleNegative, PromptRoleTool:
			default:
				images = append(images, file)
			}
		}
	}
	return images, masks
}

// openAIImageRequest builds the Images API fields. DALL-E keeps its fixed
// defaults (1024x1024, standard quality, base64 output); an unset GPT Image
// size is omitted so the model's automatic sizing applies, and the GPT Image
// options (including GIF frames and SVG settings) are passed through.
func openAIImageRequest(model, text string, mo map[string]any) (map[string]any, error) {
	legacy := strings.Contains(strings.ToLower(model), "dall-e")
	width, height := optionInt(mo, "width"), optionInt(mo, "height")
	// The Images API has no single-dimension size, so a lone width or height cannot be translated.
	if !legacy && (width == nil) != (height == nil) {
		return nil, errors.New("set both width and height, or neither to let the model choose the image size")
	}
	fields := map[string]any{"model": model, "prompt": text, "n": 1}
	if n := optionInt(mo, "n"); n != nil {
		fields["n"] = *n
	}
	switch {
	case !legacy && width != nil:
		fields["size"] = fmt.Sprintf("%dx%d", *width, *height)
	case optionString(mo, "size") != "":
		fields["size"] = optionString(mo, "size")
	case legacy:
		fields["size"] = "1024x1024"
	}
	quality := optionString(mo, "image_quality", "quality")
	if legacy {
		if quality == "" {
			quality = "standard"
		}
		fields["quality"] = quality
		fields["response_format"] = "b64_json"
		if format := optionString(mo, "response_format"); format != "" {
			fields["response_format"] = format
		}
		if style := optionString(mo, "style"); style != "" {
			fields["style"] = style
		}
		return fields, nil
	}
	if quality == "" {
		quality = "auto"
	}
	fields["quality"] = quality
	fields["output_format"] = "png"
	if format := optionString(mo, "output_format"); format != "" {
		fields["output_format"] = format
	}
	for _, key := range []string{"background", "moderation", "complexity", "input_fidelity"} {
		if value := optionString(mo, key); value != "" {
			fields[key] = value
		}
	}
	for _, key := range []string{"output_compression", "partial_images", "frame_count", "frame_duration_ms"} {
		if value := optionInt(mo, key); value != nil {
			fields[key] = *value
		}
	}
	if svg := optionMap(mo, "svg"); svg != nil {
		fields["svg"] = svg
	}
	return fields, nil
}

// requestOpenAIImageGeneration calls the Images API: reference images select
// an edit (with an optional mask), a text-only prompt a generation.
func (d *OpenAICompatibleDriver) requestOpenAIImageGeneration(ctx context.Context, prompt openAIResponsesPrompt, options ExecutionOptions) (*Completion, error) {
	if len(prompt.Masks) > 1 {
		return nil, errors.New("only one image mask is supported")
	}
	if len(prompt.Masks) == 1 && len(prompt.Images) == 0 {
		return nil, errors.New("an image mask requires a reference image")
	}
	fields, err := openAIImageRequest(options.Model, strings.TrimSpace(prompt.Text), options.ModelOptions)
	if err != nil {
		return nil, err
	}
	if !openAIImageKeepsInputFidelity(options.Model) || len(prompt.Images) == 0 {
		delete(fields, "input_fidelity")
	}
	var raw json.RawMessage
	if len(prompt.Images) == 0 {
		err = d.sdk.Post(ctx, "images/generations", fields, &raw, d.requestOptions(options.HTTPTimeout)...)
	} else {
		var body []byte
		var contentType string
		body, contentType, err = openAIImageEditBody(ctx, fields, prompt.Images, prompt.Masks)
		if err != nil {
			return nil, err
		}
		requestOptions := append(d.requestOptions(options.HTTPTimeout), option.WithRequestBody(contentType, body))
		err = d.sdk.Post(ctx, "images/edits", nil, &raw, requestOptions...)
	}
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	var response struct {
		Data []struct {
			B64JSON string `json:"b64_json"`
			URL     string `json:"url"`
		} `json:"data"`
		OutputFormat string           `json:"output_format"`
		Usage        *openAIImageUsed `json:"usage"`
		SVGUsage     *openAISVGUsage  `json:"svg_usage"`
	}
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, err
	}
	requested := optionString(options.ModelOptions, "output_format")
	results := []CompletionResult{}
	for _, image := range response.Data {
		if image.B64JSON != "" {
			results = append(results, newImageResult(openAIImageOutputDataURL(image.B64JSON, response.OutputFormat, requested)))
		} else if image.URL != "" {
			results = append(results, newImageResult(image.URL))
		}
	}
	if len(results) == 0 {
		return nil, errors.New("image response did not contain a completed image")
	}
	completion := &Completion{Result: results, TokenUsage: openAIImageUsage(response.Usage, response.SVGUsage)}
	if options.IncludeOriginalResponse {
		var original any
		_ = json.Unmarshal(raw, &original)
		completion.OriginalResponse = original
	}
	return completion, nil
}

// openAIImageEditBody encodes an edit request as multipart. Moderation,
// response_format and style are generation-only, and hd quality is standard
// for edits. An SVG output format is forwarded so the API reports it.
func openAIImageEditBody(ctx context.Context, fields map[string]any, images, masks []DataSource) ([]byte, string, error) {
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	for key, value := range fields {
		switch key {
		case "moderation", "response_format", "style":
			continue
		case "quality":
			if value == "hd" {
				value = "standard"
			}
		}
		text := fmt.Sprint(value)
		if nested, ok := value.(map[string]any); ok {
			data, _ := json.Marshal(nested)
			text = string(data)
		}
		if err := form.WriteField(key, text); err != nil {
			return nil, "", err
		}
	}
	write := func(field string, file DataSource) error {
		reader, err := file.Open(ctx)
		if err != nil {
			return err
		}
		defer func() { _ = reader.Close() }()
		mime := file.MIMEType()
		if mime == "" {
			mime = "image/png"
		}
		name := file.Name()
		if name == "" {
			name = "image." + strings.TrimPrefix(mime, "image/")
		}
		header := textproto.MIMEHeader{}
		header.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"; filename="%s"`, field, strings.ReplaceAll(name, `"`, "")))
		header.Set("Content-Type", mime)
		part, err := form.CreatePart(header)
		if err != nil {
			return err
		}
		_, err = io.Copy(part, reader)
		return err
	}
	for _, image := range images {
		if err := write("image[]", image); err != nil {
			return nil, "", err
		}
	}
	for _, mask := range masks {
		if err := write("mask", mask); err != nil {
			return nil, "", err
		}
	}
	if err := form.Close(); err != nil {
		return nil, "", err
	}
	return body.Bytes(), form.FormDataContentType(), nil
}

// openAIImageUsed is the token usage the Images API reports.
type openAIImageUsed struct {
	InputTokens         int64 `json:"input_tokens"`
	OutputTokens        int64 `json:"output_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	OutputTokensDetails *struct {
		ImageTokens int64 `json:"image_tokens"`
	} `json:"output_tokens_details"`
}

// openAISVGUsage is the usage of the model that authored an SVG output.
type openAISVGUsage struct {
	Model        string `json:"model"`
	InputTokens  int64  `json:"input_tokens"`
	OutputTokens int64  `json:"output_tokens"`
	TotalTokens  int64  `json:"total_tokens"`
}

// openAIImageUsage maps Images API usage. Image output tokens are reported only
// when the provider splits them out, since output tokens can include text. An
// SVG output's authoring model is billed apart, so it goes to tool_usage.svg.
func openAIImageUsage(usage *openAIImageUsed, svg *openAISVGUsage) *ExecutionTokenUsage {
	if usage == nil && svg == nil {
		return nil
	}
	out := &ExecutionTokenUsage{}
	if usage != nil {
		out.Prompt = tokens(int(usage.InputTokens))
		out.PromptNew = tokens(int(usage.InputTokens))
		out.Result = tokens(int(usage.OutputTokens))
		out.Total = tokens(int(usage.TotalTokens))
		if usage.OutputTokensDetails != nil {
			out.ResultImage = tokens(int(usage.OutputTokensDetails.ImageTokens))
		}
	}
	if svg != nil {
		model := svg.Model
		out.ToolUsage = &wire.ToolUsage{Svg: &wire.ToolTokenUsage{Model: &model, Prompt: tokens(int(svg.InputTokens)), Result: tokens(int(svg.OutputTokens)), Total: tokens(int(svg.TotalTokens))}}
	}
	return out
}

// openAIImageGenerationTool is the Responses image_generation tool for the
// image_generation model option: the settings are passed through, the image
// model is the source model of a composite id, and a mask segment becomes the
// input_image_mask. It returns nil without the option.
func openAIImageGenerationTool(ctx context.Context, mo map[string]any, masks []DataSource) (map[string]any, bool, error) {
	config := optionMap(mo, "image_generation")
	if config == nil {
		return nil, false, nil
	}
	model, _ := config["model"].(string)
	if model == "" {
		return nil, false, errors.New("image generation tool requires an image model")
	}
	if len(masks) > 1 || (len(masks) == 1 && config["input_image_mask"] != nil) {
		return nil, false, errors.New("only one image mask is supported")
	}
	tool := make(map[string]any, len(config)+1)
	for key, value := range config {
		if key != "force" {
			tool[key] = value
		}
	}
	if i := strings.LastIndex(model, "::"); i >= 0 {
		model = model[i+2:]
	}
	tool["model"] = model
	tool["type"] = "image_generation"
	if !openAIImageKeepsInputFidelity(model) {
		delete(tool, "input_fidelity")
	}
	if len(masks) == 1 {
		url, err := openAIImageDataURL(ctx, masks[0])
		if err != nil {
			return nil, false, err
		}
		tool["input_image_mask"] = map[string]any{"image_url": url}
	}
	force, _ := config["force"].(bool)
	return tool, force, nil
}

// openAIRequestedImageFormat is the output format requested for the
// Responses image_generation tool, if any.
func openAIRequestedImageFormat(mo map[string]any) string {
	config := optionMap(mo, "image_generation")
	format, _ := config["output_format"].(string)
	return format
}

// openAIImageStream runs an image request and returns it as a one-chunk
// stream: the Images API path is not streamed here.
func (d *OpenAICompatibleDriver) openAIImageStream(ctx context.Context, prompt any, options ExecutionOptions) (CompletionStream, error) {
	completion, err := d.requestOpenAIImageGeneration(ctx, prompt.(openAIResponsesPrompt), options)
	if err != nil {
		return nil, err
	}
	ch := make(chan streamItem, 1)
	ch <- streamItem{Chunk: CompletionChunk{Result: completion.Result, TokenUsage: completion.TokenUsage, FinishReason: "stop"}}
	close(ch)
	return newChannelStreamWithFinalizer(ch, &ExecutionResponse{Prompt: prompt}, func() error { return nil }, nil), nil
}
