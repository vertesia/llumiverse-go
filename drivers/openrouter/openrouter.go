package openrouter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
	"github.com/vertesia/llumiverse-go/drivers/internal/chatcompat"
	"github.com/vertesia/llumiverse-go/drivers/openai"
	"github.com/vertesia/llumiverse-go/wire"
)

// DefaultEndpoint is the OpenRouter API base URL.
const DefaultEndpoint = "https://openrouter.ai/api/v1"

// OpenRouterOptions configures an OpenRouter driver.
type OpenRouterOptions struct {
	common.DriverOptions
	// APIKey is the OpenRouter API key. Required.
	APIKey string
	// Endpoint overrides the API base URL. It defaults to DefaultEndpoint.
	Endpoint string
	// HTTPReferer, AppTitle and AppCategories attribute requests to an app on OpenRouter (the HTTP-Referer,
	// X-OpenRouter-Title and X-OpenRouter-Categories headers).
	HTTPReferer   string
	AppTitle      string
	AppCategories string
	// DefaultMaxTokens is sent as max_tokens when the request sets none.
	DefaultMaxTokens int
	// HTTPClient overrides the HTTP client.
	HTTPClient *http.Client
}

// OpenRouterDriver implements Driver for OpenRouter on its OpenAI-compatible Chat Completions API.
type OpenRouterDriver struct {
	*openai.OpenAIChatCompletionsDriver
}

// NewOpenRouterDriver creates an OpenRouter driver.
func NewOpenRouterDriver(options OpenRouterOptions) (*OpenRouterDriver, error) {
	if options.APIKey == "" {
		return nil, errors.New("openrouter api key is required")
	}
	endpoint := strings.TrimRight(options.Endpoint, "/")
	if endpoint == "" {
		endpoint = DefaultEndpoint
	}
	headers := map[string]string{}
	for name, value := range map[string]string{
		"HTTP-Referer":            options.HTTPReferer,
		"X-OpenRouter-Title":      options.AppTitle,
		"X-OpenRouter-Categories": options.AppCategories,
	} {
		if value != "" {
			headers[name] = value
		}
	}
	chat, err := openai.NewOpenAIChatCompletionsDriver(openai.OpenAIChatCompletionsOptions{
		OpenAICompatibleOptions: openai.OpenAICompatibleOptions{
			DriverOptions:  options.DriverOptions,
			APIKey:         options.APIKey,
			Endpoint:       endpoint,
			DefaultHeaders: headers,
			HTTPClient:     options.HTTPClient,
			Provider:       common.ProviderOpenRouter,
		},
		ChatCompletionsProtocolOptions: openai.ChatCompletionsProtocolOptions{
			Provider:                            common.ProviderOpenRouter,
			DefaultMaxTokens:                    options.DefaultMaxTokens,
			IncludeResultSchemaInPromptForModel: supplementsResultSchemaInPrompt,
			CompatibleToolSchemas:               true,
			PreparePayload:                      preparePayload,
			NormalizeResponse:                   normalizeResponse,
		},
	})
	if err != nil {
		return nil, err
	}
	return &OpenRouterDriver{OpenAIChatCompletionsDriver: chat}, nil
}

// Provider returns ProviderOpenRouter.
func (d *OpenRouterDriver) Provider() common.Provider {
	return common.ProviderOpenRouter
}

var (
	glm53Pattern   = regexp.MustCompile(`(?i)(?:^|/)glm-5\.3(?:$|[-:])`)
	minimaxPattern = regexp.MustCompile(`(?i)^minimax/`)
)

// supplementsResultSchemaInPrompt reports the families whose native structured output is unreliable on
// OpenRouter, which also get the schema as a prompt instruction: GLM 5.3 answers in prose despite
// response_format, and most of the hosts OpenRouter routes MiniMax to ignore json_schema or downgrade it to JSON
// mode. Hosts that enforce the schema still receive it natively.
func supplementsResultSchemaInPrompt(model string) bool {
	return glm53Pattern.MatchString(model) || minimaxPattern.MatchString(model)
}

// providerPreferences maps the provider_* model options to OpenRouter's provider routing preferences.
var providerPreferences = map[string]string{
	"provider_sort":               "sort",
	"provider_order":              "order",
	"provider_only":               "only",
	"provider_ignore":             "ignore",
	"provider_allow_fallbacks":    "allow_fallbacks",
	"provider_require_parameters": "require_parameters",
	"provider_data_collection":    "data_collection",
	"provider_zdr":                "zdr",
	"provider_quantizations":      "quantizations",
}

// preparePayload adapts a Chat Completions request to OpenRouter: assistant reasoning under reasoning, text-only
// system messages, provider routing preferences, and no n (OpenRouter returns one choice).
func preparePayload(payload map[string]any, options common.ExecutionOptions, _ bool) error {
	if chatcompat.HasAudio(payload) {
		return errors.New("openrouter does not support audio input")
	}
	for _, message := range chatcompat.Messages(payload) {
		switch role := message["role"]; role {
		case "assistant":
			if message["reasoning"] == nil && message["reasoning_content"] != nil {
				message["reasoning"] = message["reasoning_content"]
			}
			delete(message, "reasoning_content")
		case "system", "developer":
			if parts := chatcompat.Parts(message); parts != nil {
				texts := make([]any, 0, len(parts))
				for _, part := range parts {
					if part["type"] == "text" {
						texts = append(texts, part)
					}
				}
				message["content"] = texts
			} else if _, ok := message["content"].(string); !ok {
				message["content"] = ""
			}
		case "tool":
			if id, _ := message["tool_call_id"].(string); id == "" {
				return errors.New("openrouter tool messages require tool_call_id")
			}
			if message["content"] == nil {
				message["content"] = ""
			}
		case "user":
			if message["content"] == nil {
				return errors.New("openrouter user messages require content")
			}
		default:
			return fmt.Errorf("unsupported openrouter message role: %v", role)
		}
	}
	delete(payload, "n")
	preferences := map[string]any{}
	for option, field := range providerPreferences {
		if value, ok := options.ModelOptions[option]; ok && value != nil {
			preferences[field] = value
		}
	}
	if len(preferences) > 0 {
		payload["provider"] = preferences
	}
	return nil
}

// normalizeResponse turns the error OpenRouter sends inside a response or stream chunk, after the HTTP status
// was already sent, into an error carrying its status code.
func normalizeResponse(data []byte, options common.ExecutionOptions) ([]byte, error) {
	if !strings.Contains(string(data), `"error"`) {
		return data, nil
	}
	var chunk struct {
		Error *struct {
			Message string          `json:"message"`
			Code    json.RawMessage `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &chunk); err != nil || chunk.Error == nil {
		return data, nil
	}
	code, _ := strconv.Atoi(strings.Trim(string(chunk.Error.Code), `"`))
	message := chunk.Error.Message
	if message == "" {
		message = "openrouter returned an error"
	}
	return nil, common.NewLlumiverseError(message, core.RetryableFromStatusAndMessage(code, message), common.LlumiverseErrorContext{
		Provider:  common.ProviderOpenRouter,
		Model:     options.Model,
		Operation: "stream",
	}, errors.New(message), code, "OpenRouterStreamError")
}

type openRouterModel struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	Description  string `json:"description"`
	Architecture struct {
		InputModalities  []string `json:"input_modalities"`
		OutputModalities []string `json:"output_modalities"`
	} `json:"architecture"`
	SupportedParameters []string `json:"supported_parameters"`
}

// modelsPageSize is the page size of the model listing; a full page means more may follow.
const (
	modelsPageSize = 1000
	maxModelPages  = 20
)

// dedicatedModelPattern matches the model families that need a dedicated endpoint rather than chat
// completions, which listings leave out with their output modalities.
var dedicatedModelPattern = regexp.MustCompile(`(?i)(?:^|[-_./:])(?:moderation|whisper|transcribe|tts|realtime)(?:[-_./:]|$)`)

// ListModels lists the OpenRouter models that answer in text, with the modalities and tool support the listing
// reports, leaving out embedding and dedicated-endpoint models.
func (d *OpenRouterDriver) ListModels(ctx context.Context, _ *common.ModelSearchPayload) ([]common.AIModel, error) {
	var models []common.AIModel
	for page := range maxModelPages {
		var response struct {
			Data []openRouterModel `json:"data"`
		}
		query := map[string]string{"limit": strconv.Itoa(modelsPageSize)}
		if page > 0 {
			query["offset"] = strconv.Itoa(page * modelsPageSize)
		}
		if err := d.DoAPIRequest(ctx, http.MethodGet, "models", nil, &response, openai.APIRequestOptions{Operation: "listModels", Query: query}); err != nil {
			return nil, err
		}
		for _, model := range response.Data {
			if !isChatModel(model) {
				continue
			}
			input := append([]string(nil), model.Architecture.InputModalities...)
			output := append([]string(nil), model.Architecture.OutputModalities...)
			owner, _, _ := strings.Cut(model.ID, "/")
			aiModel := common.AIModel{
				ID:               model.ID,
				Name:             model.Name,
				Provider:         string(common.ProviderOpenRouter),
				Owner:            &owner,
				Type:             ptr(wire.ModelTypeText),
				CanStream:        ptr(true),
				IsMultimodal:     ptr(len(input) > 1),
				InputModalities:  &input,
				OutputModalities: &output,
				ToolSupport:      ptr(contains(model.SupportedParameters, "tools")),
			}
			if aiModel.Name == "" {
				aiModel.Name = model.ID
			}
			if model.Description != "" {
				aiModel.Description = ptr(model.Description)
			}
			models = append(models, aiModel)
		}
		if len(response.Data) < modelsPageSize {
			break
		}
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	return models, nil
}

func isChatModel(model openRouterModel) bool {
	if !contains(model.Architecture.OutputModalities, "text") || chatcompat.IsEmbeddingModelID(model.ID) ||
		dedicatedModelPattern.MatchString(model.ID) {
		return false
	}
	for _, modality := range append(append([]string(nil), model.Architecture.InputModalities...), model.Architecture.OutputModalities...) {
		if chatcompat.IsEmbeddingModality(modality) {
			return false
		}
	}
	return true
}

// ValidateConnection checks the API key against a one-model listing.
func (d *OpenRouterDriver) ValidateConnection(ctx context.Context) error {
	return d.DoAPIRequest(ctx, http.MethodGet, "models", nil, nil, openai.APIRequestOptions{Operation: "validateConnection", Query: map[string]string{"limit": "1"}})
}

type embeddingsResponse struct {
	Model string `json:"model"`
	Data  []struct {
		Index     *int            `json:"index"`
		Embedding json.RawMessage `json:"embedding"`
	} `json:"data"`
	Usage *struct {
		PromptTokens        float64 `json:"prompt_tokens"`
		PromptTokensDetails *struct {
			TextTokens  *float64 `json:"text_tokens"`
			ImageTokens *float64 `json:"image_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

// GenerateEmbeddings embeds text, image, audio and video inputs with an explicit embedding model.
func (d *OpenRouterDriver) GenerateEmbeddings(ctx context.Context, options common.EmbeddingsOptions) (*common.EmbeddingsResult, error) {
	options, err := core.NormalizeEmbeddingsOptions(options)
	if err != nil {
		return nil, err
	}
	if options.Model == "" {
		return nil, errors.New("openrouter requires an explicit embedding model")
	}
	inputs := make([]any, 0, len(options.Inputs))
	for _, input := range options.Inputs {
		part, err := embeddingInput(ctx, input)
		if err != nil {
			return nil, err
		}
		inputs = append(inputs, map[string]any{"content": []any{part}})
	}
	body := map[string]any{"model": options.Model, "input": inputs, "encoding_format": "float"}
	if options.Dimensions > 0 {
		body["dimensions"] = options.Dimensions
	}
	if options.TaskType != "" {
		body["input_type"] = string(options.TaskType)
	}
	var response embeddingsResponse
	if err := d.DoAPIRequest(ctx, http.MethodPost, "embeddings", body, &response, openai.APIRequestOptions{Model: options.Model, Operation: "embeddings", HTTPTimeout: options.HTTPTimeout}); err != nil {
		return nil, err
	}
	sort.SliceStable(response.Data, func(i, j int) bool { return indexOf(response.Data[i].Index, i) < indexOf(response.Data[j].Index, j) })
	results := make([]common.EmbeddingResultItem, 0, len(response.Data))
	for position, entry := range response.Data {
		index := indexOf(entry.Index, position)
		var values []float64
		if json.Unmarshal(entry.Embedding, &values) != nil || len(values) == 0 {
			return nil, fmt.Errorf("openrouter embedding empty or non-float for input index %d", index)
		}
		if index < 0 || index >= len(options.Inputs) {
			return nil, fmt.Errorf("openrouter embedding references unknown input index %d", index)
		}
		modality := wire.EmbeddingOutputModality(embeddingType(options.Inputs[index]))
		results = append(results, common.EmbeddingResultItem{Outputs: []common.EmbeddingOutput{{Values: values, Modality: &modality}}})
	}
	return &common.EmbeddingsResult{Model: response.Model, Results: results, Usage: embeddingsUsage(response, options.Inputs)}, nil
}

func embeddingsUsage(response embeddingsResponse, inputs []common.EmbeddingInput) *common.EmbeddingsTokenUsage {
	if response.Usage == nil {
		return nil
	}
	usage := &common.EmbeddingsTokenUsage{InputTokens: ptr(response.Usage.PromptTokens)}
	onlyText := true
	for _, input := range inputs {
		if embeddingType(input) != "text" {
			onlyText = false
		}
	}
	details := response.Usage.PromptTokensDetails
	switch {
	case details != nil && details.TextTokens != nil:
		usage.InputTextTokens = ptr(*details.TextTokens)
	case onlyText:
		usage.InputTextTokens = ptr(response.Usage.PromptTokens)
	}
	if details != nil && details.ImageTokens != nil {
		usage.InputImageTokens = ptr(*details.ImageTokens)
	}
	return usage
}

func embeddingType(input common.EmbeddingInput) string {
	if input.Type == "" {
		return "text"
	}
	return input.Type
}

// embeddingInput converts an embedding input to an OpenRouter content part: text, an image URL, or base64 audio
// and video with their format.
func embeddingInput(ctx context.Context, input common.EmbeddingInput) (map[string]any, error) {
	kind := embeddingType(input)
	if kind == "text" {
		return map[string]any{"type": "text", "text": input.Text}, nil
	}
	if input.Source == nil {
		return nil, fmt.Errorf("openrouter %s embedding input requires a source", kind)
	}
	switch kind {
	case "image":
		url, err := core.DataSourceToDataURL(ctx, input.Source)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}, nil
	case "audio", "video":
		data, err := core.DataSourceToBase64(ctx, input.Source)
		if err != nil {
			return nil, err
		}
		mime := input.Source.MIMEType()
		format := mime
		if _, subtype, ok := strings.Cut(mime, "/"); ok {
			format = subtype
		}
		field := "input_" + kind
		return map[string]any{"type": field, field: map[string]any{"data": data, "format": format}}, nil
	}
	return nil, fmt.Errorf("openrouter does not support %q embeddings", kind)
}

func indexOf(index *int, fallback int) int {
	if index != nil {
		return *index
	}
	return fallback
}

func contains(values []string, value string) bool {
	for _, item := range values {
		if item == value {
			return true
		}
	}
	return false
}

func ptr[T any](v T) *T { return &v }

var _ common.Driver = (*OpenRouterDriver)(nil)
