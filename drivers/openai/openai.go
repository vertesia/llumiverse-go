package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"regexp"
	"slices"
	"sort"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/vertesia/llumiverse-go/capability"
	"github.com/vertesia/llumiverse-go/core"
)

const defaultOpenAIBaseURL = "https://api.openai.com/v1"
const defaultOpenAIEmbeddingModel = "text-embedding-3-small"

// OpenAICompatibleOptions configures OpenAI and OpenAI-compatible endpoints.
type OpenAICompatibleOptions struct {
	DriverOptions
	// APIKey is sent as the OpenAI-compatible bearer token.
	APIKey string
	// Endpoint is the API base URL. NewOpenAIDriver defaults it to the official OpenAI API.
	Endpoint string
	// DefaultHeaders are added to every SDK request.
	DefaultHeaders map[string]string
	// HTTPClient overrides the SDK HTTP client.
	HTTPClient *http.Client
	// Provider defaults to ProviderOpenAICompatible; NewOpenAIDriver sets ProviderOpenAI.
	Provider Provider
}

// OpenAICompatibleDriver implements Driver for first-party OpenAI and
// OpenAI-compatible Responses API endpoints.
type OpenAICompatibleDriver struct {
	options OpenAICompatibleOptions
	client  *http.Client
	sdk     openai.Client
}

// NewOpenAIDriver creates the first-party OpenAI driver using openai-go.
func NewOpenAIDriver(options OpenAICompatibleOptions) (*OpenAICompatibleDriver, error) {
	if options.Endpoint == "" {
		options.Endpoint = defaultOpenAIBaseURL
	}
	options.Provider = ProviderOpenAI
	return NewOpenAICompatibleDriver(options)
}

// NewOpenAICompatibleDriver creates a driver for endpoints that implement the
// OpenAI Responses-compatible API surface.
func NewOpenAICompatibleDriver(options OpenAICompatibleOptions) (*OpenAICompatibleDriver, error) {
	if options.APIKey == "" {
		return nil, errors.New("api key is required")
	}
	if options.Endpoint == "" {
		return nil, errors.New("endpoint is required")
	}
	if options.Provider == "" {
		options.Provider = ProviderOpenAICompatible
	}
	client := options.HTTPClient
	if client == nil {
		client = newHTTPClient(nil, options.HTTPTimeout)
	} else if hasHTTPTimeout(options.HTTPTimeout) {
		client = newHTTPClient(client, options.HTTPTimeout)
	}
	sdkOptions := []option.RequestOption{
		option.WithAPIKey(options.APIKey),
		option.WithBaseURL(options.Endpoint),
		option.WithHTTPClient(client),
	}
	for key, value := range options.DefaultHeaders {
		sdkOptions = append(sdkOptions, option.WithHeader(key, value))
	}
	return &OpenAICompatibleDriver{options: options, client: client, sdk: newSDKClient(sdkOptions...)}, nil
}

// newSDKClient builds the openai-go client from the given options alone. openai.NewClient would also apply
// OPENAI_* environment variables (organization, project, admin key, custom headers), which the library does not
// read: configuration comes from OpenAICompatibleOptions.
func newSDKClient(opts ...option.RequestOption) openai.Client {
	opts = append([]option.RequestOption{option.WithEnvironmentProduction()}, opts...)
	return openai.Client{
		Options:    opts,
		Models:     openai.NewModelService(opts...),
		Embeddings: openai.NewEmbeddingService(opts...),
	}
}

// Provider returns the configured provider (ProviderOpenAI for the first-party
// driver, ProviderOpenAICompatible otherwise).
func (d *OpenAICompatibleDriver) Provider() Provider {
	return d.options.Provider
}

// CreatePrompt builds the Responses API request body from prompt segments.
func (d *OpenAICompatibleDriver) CreatePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (any, error) {
	return d.createResponsesPrompt(ctx, segments, options)
}

// Execute runs a single non-streaming completion.
func (d *OpenAICompatibleDriver) Execute(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (*ExecutionResponse, error) {
	if d.isOpenAIAudioModel(options.Model) {
		return d.executeOpenAIAudio(ctx, segments, options)
	}
	return executeWithPrompt(ctx, d, segments, options, d.requestTextCompletion)
}

// Stream runs a streaming completion, emitting chunks as they arrive.
func (d *OpenAICompatibleDriver) Stream(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (CompletionStream, error) {
	if d.isOpenAIAudioModel(options.Model) {
		return d.openAIAudioStream(ctx, segments, options)
	}
	stream, err := streamWithPrompt(ctx, d, segments, options, d.requestTextCompletionStream)
	if err != nil {
		return nil, err
	}
	return core.LeaseCompletionStream(stream, d.options.StreamStartTimeout), nil
}

// requestTextCompletion issues a Responses API request.
func (d *OpenAICompatibleDriver) requestTextCompletion(ctx context.Context, prompt any, options ExecutionOptions) (*Completion, error) {
	return d.requestResponsesCompletion(ctx, prompt, options)
}

// requestTextCompletionStream opens a Responses API stream.
func (d *OpenAICompatibleDriver) requestTextCompletionStream(ctx context.Context, prompt any, options ExecutionOptions) (CompletionStream, error) {
	return d.requestResponsesCompletionStream(ctx, prompt, options)
}

func (d *OpenAICompatibleDriver) requestOptions(timeout HTTPTimeoutOptions) []option.RequestOption {
	if !hasHTTPTimeout(timeout) {
		return nil
	}
	return []option.RequestOption{option.WithHTTPClient(newHTTPClient(d.client, timeout))}
}

// openAIUsage normalizes OpenAI token counts into ExecutionTokenUsage, deriving
// PromptNew from the cached prompt tokens; it returns nil when no usage is set.
func openAIUsage(prompt, completion, total, cached int64) *ExecutionTokenUsage {
	if prompt == 0 && completion == 0 && total == 0 {
		return nil
	}
	return &ExecutionTokenUsage{
		Prompt:       tokens(int(prompt)),
		Result:       tokens(int(completion)),
		Total:        tokens(int(total)),
		PromptCached: tokens(int(cached)),
		PromptNew:    tokens(int(prompt - cached)),
	}
}

// openAIUnsupportedEndpointPattern matches the model families served by
// dedicated or legacy endpoints rather than text generation.
var openAIUnsupportedEndpointPattern = regexp.MustCompile(`(?:^|[-_.:/])(?:embed|embeddings?|whisper|transcribe|audio|moderation|tts|realtime|babbage|davinci|computer[-_.:]use|sora)(?:[-_.:/]|$)`)

// APIRequestOptions describes a request sent with DoAPIRequest.
type APIRequestOptions struct {
	// Model and Operation label errors.
	Model     string
	Operation string
	// Query is added to the request URL.
	Query map[string]string
	// HTTPTimeout applies per-request timeouts.
	HTTPTimeout HTTPTimeoutOptions
}

// DoAPIRequest sends a JSON request to a provider endpoint the shared drivers do not cover (model catalogues,
// image generation, embeddings in another shape), with the driver's credentials, headers, retries and error
// formatting. path is relative to the endpoint ("../v2/models" leaves its version). out receives the decoded
// JSON body, or the raw response when it is a **http.Response (the caller closes its body).
func (d *OpenAICompatibleDriver) DoAPIRequest(ctx context.Context, method, path string, body, out any, request APIRequestOptions) error {
	opts := d.requestOptions(request.HTTPTimeout)
	for key, value := range request.Query {
		opts = append(opts, option.WithQuery(key, value))
	}
	if err := d.sdk.Execute(ctx, method, path, body, out, opts...); err != nil {
		return d.formatError(err, request.Model, request.Operation)
	}
	return nil
}

// ListModels returns the endpoint's models sorted by ID, with the capability catalogue's modalities and tool
// support, and the "system" owner normalized to "openai" for first-party OpenAI. File audio models are listed as
// audio models, and on OpenAI image models as image models; other dedicated-endpoint models are left out.
func (d *OpenAICompatibleDriver) ListModels(ctx context.Context, _ *ModelSearchPayload) ([]AIModel, error) {
	response, err := d.sdk.Models.List(ctx)
	if err != nil {
		return nil, d.formatError(err, "", "listModels")
	}
	provider := d.Provider()
	models := make([]AIModel, 0, len(response.Data))
	for _, model := range response.Data {
		if !d.isListedModel(model.ID) {
			continue
		}
		owner := model.OwnedBy
		if owner == "system" && provider == ProviderOpenAI {
			owner = "openai"
		}
		modelType := modelTypeText
		switch {
		case d.isOpenAIAudioModel(model.ID):
			modelType = modelTypeAudio
		case d.isImageModel(model.ID):
			modelType = modelTypeImage
		}
		aiModel := AIModel{
			ID:        model.ID,
			Name:      model.ID,
			Provider:  string(provider),
			Type:      ptr(modelType),
			CanStream: ptr(true),
		}
		if owner != "" {
			aiModel.Owner = ptr(owner)
		}
		metadata := capability.ResolveListingMetadata(model.ID, provider, nil, nil)
		metadata.Apply(&aiModel)
		// As in the TS client, OpenAI-compatible listings flag image input; first-party OpenAI listings do not.
		if provider != ProviderOpenAI {
			aiModel.IsMultimodal = ptr(slices.Contains(metadata.InputModalities, "image"))
		}
		models = append(models, aiModel)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	return models, nil
}

// isListedModel reports whether ListModels returns a model: file audio models always, and otherwise models of the
// standard inference path. OpenAI listing metadata does not describe execution, so OpenAI also leaves out the
// dedicated and legacy endpoint families by name, and DALL-E, while unknown future IDs stay visible; its image models
// are listed, since the driver runs them on the Images API.
func (d *OpenAICompatibleDriver) isListedModel(id string) bool {
	provider := d.Provider()
	if d.isOpenAIAudioModel(id) {
		return true
	}
	if provider == ProviderOpenAI {
		lower := strings.ToLower(id)
		return !openAIUnsupportedEndpointPattern.MatchString(lower) && !strings.Contains(lower, "dall-e") &&
			(d.isImageModel(id) || !capability.IsDedicatedInferenceModel(id, provider))
	}
	return !capability.IsEmbeddingModel(capability.Listing{ID: id}, provider) &&
		!capability.IsDedicatedInferenceModel(id, provider)
}

// ValidateConnection verifies the endpoint and credentials by listing models.
func (d *OpenAICompatibleDriver) ValidateConnection(ctx context.Context) error {
	_, err := d.ListModels(ctx, nil)
	return err
}

// GenerateEmbeddings embeds text inputs using the OpenAI embeddings API.
func (d *OpenAICompatibleDriver) GenerateEmbeddings(ctx context.Context, options EmbeddingsOptions) (*EmbeddingsResult, error) {
	options, err := normalizeEmbeddingsOptions(options)
	if err != nil {
		return nil, err
	}
	model := options.Model
	if model == "" {
		model = defaultOpenAIEmbeddingModel
	}
	texts := make([]string, 0, len(options.Inputs))
	for _, input := range options.Inputs {
		if input.Type != embeddingInputText {
			return nil, errors.New("openai-compatible embeddings support text inputs only")
		}
		texts = append(texts, input.Text)
	}
	payload := openai.EmbeddingNewParams{
		Model:          openai.EmbeddingModel(model),
		Input:          openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: texts},
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	}
	if options.Dimensions > 0 {
		payload.Dimensions = openai.Int(int64(options.Dimensions))
	}
	response, err := d.sdk.Embeddings.New(ctx, payload, d.requestOptions(options.HTTPTimeout)...)
	if err != nil {
		return nil, d.formatError(err, model, "embeddings")
	}
	sort.Slice(response.Data, func(i, j int) bool { return response.Data[i].Index < response.Data[j].Index })
	items := make([]EmbeddingResultItem, 0, len(response.Data))
	for _, item := range response.Data {
		items = append(items, EmbeddingResultItem{Outputs: []EmbeddingOutput{{Values: item.Embedding, Modality: ptr(embeddingOutputModalityText)}}})
	}
	usage := &EmbeddingsTokenUsage{
		InputTokens:     tokens(int(response.Usage.PromptTokens)),
		InputTextTokens: tokens(int(response.Usage.PromptTokens)),
	}
	if response.Usage.PromptTokens == 0 && response.Usage.TotalTokens == 0 {
		usage = nil
	}
	return &EmbeddingsResult{Model: model, Results: items, Usage: usage}, nil
}

// isLikelyNonToolOpenAIModel heuristically flags models that do not support
// tool calls (image, embedding, moderation, audio, and video models) by name.
func isLikelyNonToolOpenAIModel(model string) bool {
	normalized := strings.ToLower(model)
	for _, pattern := range []string{"image", "embed", "moderation", "whisper", "sora", "dall-e", "tts"} {
		if strings.Contains(normalized, pattern) {
			return true
		}
	}
	return false
}

// formatError wraps an SDK *openai.Error as a LlumiverseError carrying provider,
// model, operation, status, and a computed retryable flag; other errors pass through.
func (d *OpenAICompatibleDriver) formatError(err error, model string, operation string) error {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		message := apiErr.Message
		if message == "" {
			message = apiErr.Error()
		}
		return newLlumiverseError(message, openAIRetryable(apiErr), LlumiverseErrorContext{
			Provider:  d.Provider(),
			Model:     model,
			Operation: operation,
		}, err, apiErr.StatusCode, "OpenAIError")
	}
	return err
}

// openAIRetryable classifies an API error as retryable based on its code, type,
// and HTTP status; nil means "unknown" when no signal is available.
func openAIRetryable(apiErr *openai.Error) *bool {
	if apiErr == nil {
		return nil
	}
	switch apiErr.Code {
	case "timeout", "server_error", "service_unavailable", "rate_limit_exceeded":
		return boolPtr(true)
	case "invalid_api_key", "invalid_request_error", "model_not_found", "insufficient_quota", "invalid_model":
		return boolPtr(false)
	}
	if strings.HasPrefix(apiErr.Code, "invalid_") {
		return boolPtr(false)
	}
	if apiErr.Type == "invalid_request_error" || apiErr.Type == "authentication_error" {
		return boolPtr(false)
	}
	return retryableFromStatusAndMessage(apiErr.StatusCode, apiErr.Message)
}

// safeJSONParse decodes a JSON string, returning the original string on parse
// failure and nil for empty input.
func safeJSONParse(value string) any {
	if value == "" {
		return nil
	}
	var out any
	if err := json.Unmarshal([]byte(value), &out); err != nil {
		return value
	}
	return out
}
