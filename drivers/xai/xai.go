package xai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/vertesia/llumiverse-go/capability"
	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
	"github.com/vertesia/llumiverse-go/drivers/openai"
	"github.com/vertesia/llumiverse-go/wire"
)

// DefaultEndpoint is the xAI API base URL.
const DefaultEndpoint = "https://api.x.ai/v1"

// usdTicks is the number of cost ticks in one US dollar: xAI reports what a request cost in ticks of 10^-10 USD.
const usdTicks = 1e10

// XAIOptions configures an xAI driver.
type XAIOptions struct {
	common.DriverOptions
	// APIKey is the xAI API key. Required.
	APIKey string
	// Endpoint overrides the API base URL. It defaults to DefaultEndpoint.
	Endpoint string
	// HTTPClient overrides the HTTP client.
	HTTPClient *http.Client
}

// XAIDriver implements Driver for xAI: Grok language models on the Responses API, through the shared
// drivers/openai implementation, and Grok image models on the Images API.
type XAIDriver struct {
	*openai.OpenAICompatibleDriver
}

// NewXAIDriver creates an xAI driver.
func NewXAIDriver(options XAIOptions) (*XAIDriver, error) {
	if options.APIKey == "" {
		return nil, errors.New("xai api key is required")
	}
	endpoint := strings.TrimRight(options.Endpoint, "/")
	if endpoint == "" {
		endpoint = DefaultEndpoint
	}
	responses, err := openai.NewOpenAICompatibleDriver(openai.OpenAICompatibleOptions{
		DriverOptions: options.DriverOptions,
		APIKey:        options.APIKey,
		Endpoint:      endpoint,
		HTTPClient:    options.HTTPClient,
		Provider:      common.ProviderXAI,
	})
	if err != nil {
		return nil, err
	}
	return &XAIDriver{OpenAICompatibleDriver: responses}, nil
}

// Provider returns ProviderXAI.
func (d *XAIDriver) Provider() common.Provider {
	return common.ProviderXAI
}

// IsImageModel reports whether a model is a Grok image generation model (an ID naming both grok and image).
func IsImageModel(model string) bool {
	id := model
	if index := strings.LastIndex(id, "/"); index >= 0 {
		id = id[index+1:]
	}
	id = strings.ToLower(id)
	return strings.Contains(id, "grok") && strings.Contains(id, "image")
}

// ImagePrompt is the prompt of a Grok image model: the text of every segment, and the images to edit.
type ImagePrompt struct {
	Prompt string       `json:"prompt"`
	Images []ImageInput `json:"images,omitempty"`
}

// ImageInput is an image sent to the Images API, by URL (a data URL for attached files).
type ImageInput struct {
	Type string `json:"type"`
	URL  string `json:"url"`
}

// CreatePrompt builds a Responses prompt, or an ImagePrompt for image models.
func (d *XAIDriver) CreatePrompt(ctx context.Context, segments []common.PromptSegment, options common.ExecutionOptions) (any, error) {
	if !IsImageModel(options.Model) {
		return d.OpenAICompatibleDriver.CreatePrompt(ctx, segments, options)
	}
	var texts []string
	prompt := ImagePrompt{}
	for _, segment := range segments {
		if segment.Content != "" {
			texts = append(texts, segment.Content)
		}
		for _, file := range segment.Files {
			if !strings.HasPrefix(file.MIMEType(), "image/") {
				continue
			}
			url, err := core.DataSourceToDataURL(ctx, file)
			if err != nil {
				return nil, err
			}
			prompt.Images = append(prompt.Images, ImageInput{Type: "image_url", URL: url})
		}
	}
	prompt.Prompt = strings.TrimSpace(strings.Join(texts, "\n"))
	return prompt, nil
}

// FormatDebugPrompt returns a copy of the prompt with inline payloads truncated.
func (d *XAIDriver) FormatDebugPrompt(prompt any) any {
	value, ok := prompt.(ImagePrompt)
	if !ok {
		return d.OpenAICompatibleDriver.FormatDebugPrompt(prompt)
	}
	images := make([]ImageInput, len(value.Images))
	for i, image := range value.Images {
		images[i] = ImageInput{Type: image.Type, URL: core.TruncateDataURLForDebug(image.URL)}
	}
	value.Images = images
	return value
}

// Execute runs a completion: image generation for image models, a Responses request otherwise.
func (d *XAIDriver) Execute(ctx context.Context, segments []common.PromptSegment, options common.ExecutionOptions) (*common.ExecutionResponse, error) {
	if !IsImageModel(options.Model) {
		return d.OpenAICompatibleDriver.Execute(ctx, segments, options)
	}
	return core.ExecuteWithPrompt(ctx, d, segments, options, d.requestImageGeneration)
}

// Stream runs a streaming completion. Image generation is not streamed: it returns one chunk.
func (d *XAIDriver) Stream(ctx context.Context, segments []common.PromptSegment, options common.ExecutionOptions) (common.CompletionStream, error) {
	if !IsImageModel(options.Model) {
		return d.OpenAICompatibleDriver.Stream(ctx, segments, options)
	}
	return core.StreamWithPrompt(ctx, d, segments, options, func(ctx context.Context, prompt any, options common.ExecutionOptions) (common.CompletionStream, error) {
		completion, err := d.requestImageGeneration(ctx, prompt, options)
		if err != nil {
			return nil, err
		}
		ch := make(chan core.StreamItem, 1)
		ch <- core.StreamItem{Chunk: common.CompletionChunk{Result: completion.Result, TokenUsage: completion.TokenUsage, FinishReason: "stop"}}
		close(ch)
		return core.NewChannelStreamWithFinalizer(ch, &common.ExecutionResponse{Prompt: prompt}, func() error { return nil }, func(resp *common.ExecutionResponse) {
			resp.Error = completion.Error
		}), nil
	})
}

type imageResponse struct {
	Data []*struct {
		B64JSON  string `json:"b64_json"`
		MIMEType string `json:"mime_type"`
		URL      string `json:"url"`
	} `json:"data"`
	// Images are billed per image, so xAI reports the cost rather than token counts.
	Usage *struct {
		CostInUSDTicks *float64 `json:"cost_in_usd_ticks"`
	} `json:"usage"`
}

// imageOptions are the ModelOptions passed through to the Images API.
var imageOptions = []string{"aspect_ratio", "resolution", "quality", "response_format", "n"}

// requestImageGeneration generates images, or edits the prompt's images. A rejected request is returned as the
// completion's error, as the TS client does, so callers see a content policy violation as a result.
func (d *XAIDriver) requestImageGeneration(ctx context.Context, prompt any, options common.ExecutionOptions) (*common.Completion, error) {
	image, ok := prompt.(ImagePrompt)
	if !ok {
		return nil, fmt.Errorf("xai image models require an image prompt, got %T", prompt)
	}
	payload := map[string]any{"model": options.Model, "prompt": image.Prompt}
	for _, key := range imageOptions {
		if value, ok := options.ModelOptions[key]; ok && value != nil && value != "" && value != 0 {
			payload[key] = value
		}
	}
	switch len(image.Images) {
	case 0:
	case 1:
		payload["image"] = image.Images[0]
	default:
		payload["images"] = image.Images
	}
	path := "images/generations"
	if len(image.Images) > 0 {
		path = "images/edits"
	}
	var response imageResponse
	if err := d.DoAPIRequest(ctx, http.MethodPost, path, payload, &response, openai.APIRequestOptions{Model: options.Model, Operation: "execute", HTTPTimeout: options.HTTPTimeout}); err != nil {
		code := "validation_error"
		var apiErr *openaisdk.Error
		if errors.As(err, &apiErr) && apiErr.Code == "content_policy_violation" {
			code = apiErr.Code
		}
		return &common.Completion{Result: []common.CompletionResult{}, Error: &common.ResultValidationError{Code: code, Message: err.Error()}}, nil
	}
	results := []common.CompletionResult{}
	for _, item := range response.Data {
		switch {
		case item == nil:
		case strings.TrimSpace(item.B64JSON) != "":
			mime := item.MIMEType
			if mime == "" {
				mime = "image/jpeg"
			}
			results = append(results, wire.NewImageResult("data:"+mime+";base64,"+item.B64JSON))
		case strings.TrimSpace(item.URL) != "":
			results = append(results, wire.NewImageResult(item.URL))
		}
	}
	completion := &common.Completion{Result: results}
	if len(results) == 0 {
		completion.Error = &common.ResultValidationError{Code: "validation_error", Message: "image generation returned no usable images"}
	}
	if response.Usage != nil && response.Usage.CostInUSDTicks != nil {
		cost := *response.Usage.CostInUSDTicks / usdTicks
		completion.TokenUsage = &common.ExecutionTokenUsage{ProviderCostUsd: &cost}
	}
	return completion, nil
}

type xaiModel struct {
	ID               string   `json:"id"`
	OwnedBy          string   `json:"owned_by"`
	Version          string   `json:"version"`
	InputModalities  []string `json:"input_modalities"`
	OutputModalities []string `json:"output_modalities"`
}

type xaiModels struct {
	Models []xaiModel `json:"models"`
}

// ListModels lists the language and image generation models. It fails only when both listings fail.
func (d *XAIDriver) ListModels(ctx context.Context, _ *common.ModelSearchPayload) ([]common.AIModel, error) {
	var language, images xaiModels
	languageErr := d.DoAPIRequest(ctx, http.MethodGet, "language-models", nil, &language, openai.APIRequestOptions{Operation: "listModels"})
	imageErr := d.DoAPIRequest(ctx, http.MethodGet, "image-generation-models", nil, &images, openai.APIRequestOptions{Operation: "listModels"})
	if languageErr != nil && imageErr != nil {
		return nil, languageErr
	}
	models := make([]common.AIModel, 0, len(language.Models)+len(images.Models))
	for _, model := range language.Models {
		listing := capability.Listing{ID: model.ID, InputModalities: model.InputModalities, OutputModalities: model.OutputModalities}
		if capability.IsEmbeddingModel(listing, common.ProviderXAI) {
			continue
		}
		// xAI's listed modalities have been incomplete, so the catalogue describes the model.
		aiModel := common.AIModel{
			ID:          model.ID,
			Name:        model.ID,
			Description: ptr(model.ID + " by " + model.OwnedBy),
			Provider:    string(common.ProviderXAI),
			Type:        ptr(wire.ModelTypeText),
			CanStream:   ptr(true),
		}
		capability.ForModel(model.ID, common.ProviderXAI).Apply(&aiModel)
		aiModel.Tags = modalityTags(aiModel)
		models = append(models, aiModel)
	}
	for _, model := range images.Models {
		aiModel := common.AIModel{
			ID:          model.ID,
			Name:        model.ID,
			Description: ptr(model.ID + " by " + model.OwnedBy),
			Provider:    string(common.ProviderXAI),
			Owner:       ptr(model.OwnedBy),
			Type:        ptr(wire.ModelTypeImage),
			CanStream:   ptr(false),
			ToolSupport: ptr(false),
		}
		if model.Version != "" {
			aiModel.Version = ptr(model.Version)
		}
		capability.ForModel(model.ID, common.ProviderXAI).Apply(&aiModel)
		aiModel.IsMultimodal = ptr(len(*aiModel.InputModalities) > 1)
		aiModel.ToolSupport = ptr(false)
		aiModel.Tags = modalityTags(aiModel)
		models = append(models, aiModel)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	return models, nil
}

// ValidateConnection checks the API key against the language model listing.
func (d *XAIDriver) ValidateConnection(ctx context.Context) error {
	return d.DoAPIRequest(ctx, http.MethodGet, "language-models", nil, nil, openai.APIRequestOptions{Operation: "validateConnection"})
}

func ptr[T any](v T) *T { return &v }

var _ common.Driver = (*XAIDriver)(nil)

// modalityTags tags a listed model with its modalities: i:text, o:image.
func modalityTags(model common.AIModel) *[]string {
	tags := []string{}
	for _, modality := range *model.InputModalities {
		tags = append(tags, "i:"+modality)
	}
	for _, modality := range *model.OutputModalities {
		tags = append(tags, "o:"+modality)
	}
	return &tags
}
