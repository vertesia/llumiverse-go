package openai

import (
	"context"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

// isOpenAIImageModel detects image-generation models (dall-e/gpt-image/
// chatgpt-image), which route to the Images API instead of Responses.
func isOpenAIImageModel(model string) bool {
	model = strings.ToLower(model)
	return strings.Contains(model, "dall-e") || strings.Contains(model, "gpt-image") || strings.Contains(model, "chatgpt-image")
}

// requestOpenAIImageGeneration calls the Images API using the flattened prompt
// text and image ModelOptions (size, n, quality, style, background, format,
// etc.), defaulting base64 output for dall-e, and returns image results as data
// URLs or URLs.
func (d *OpenAICompatibleDriver) requestOpenAIImageGeneration(ctx context.Context, prompt openAIResponsesPrompt, options ExecutionOptions) (*Completion, error) {
	payload := openai.ImageGenerateParams{
		Model:  openai.ImageModel(options.Model),
		Prompt: strings.TrimSpace(prompt.Text),
		Size:   openai.ImageGenerateParamsSize1024x1024,
	}
	if size := optionString(options.ModelOptions, "size"); size != "" {
		payload.Size = openai.ImageGenerateParamsSize(size)
	}
	if n := optionInt(options.ModelOptions, "n"); n != nil {
		payload.N = param.NewOpt(int64(*n))
	}
	if responseFormat := optionString(options.ModelOptions, "response_format"); responseFormat != "" {
		payload.ResponseFormat = openai.ImageGenerateParamsResponseFormat(responseFormat)
	} else if strings.Contains(options.Model, "dall-e") {
		payload.ResponseFormat = openai.ImageGenerateParamsResponseFormatB64JSON
	}
	if quality := optionString(options.ModelOptions, "image_quality", "quality"); quality != "" {
		payload.Quality = openai.ImageGenerateParamsQuality(quality)
	} else if strings.Contains(options.Model, "dall-e-3") {
		payload.Quality = openai.ImageGenerateParamsQualityStandard
	}
	if style := optionString(options.ModelOptions, "style"); style != "" {
		payload.Style = openai.ImageGenerateParamsStyle(style)
	}
	if background := optionString(options.ModelOptions, "background"); background != "" {
		payload.Background = openai.ImageGenerateParamsBackground(background)
	}
	if outputFormat := optionString(options.ModelOptions, "output_format"); outputFormat != "" {
		payload.OutputFormat = openai.ImageGenerateParamsOutputFormat(outputFormat)
	}
	if outputCompression := optionInt(options.ModelOptions, "output_compression"); outputCompression != nil {
		payload.OutputCompression = param.NewOpt(int64(*outputCompression))
	}
	if moderation := optionString(options.ModelOptions, "moderation"); moderation != "" {
		payload.Moderation = openai.ImageGenerateParamsModeration(moderation)
	}
	response, err := d.sdk.Images.Generate(ctx, payload, d.requestOptions(options.HTTPTimeout)...)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	results := []CompletionResult{}
	for _, image := range response.Data {
		if image.B64JSON != "" {
			results = append(results, newImageResult("data:image/png;base64,"+image.B64JSON))
		} else if image.URL != "" {
			results = append(results, newImageResult(image.URL))
		}
	}
	return &Completion{Result: results}, nil
}
