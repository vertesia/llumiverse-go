package togetherai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"regexp"
	"sort"
	"strings"

	"github.com/vertesia/llumiverse-go/capability"
	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/drivers/internal/chatcompat"
	"github.com/vertesia/llumiverse-go/drivers/openai"
	"github.com/vertesia/llumiverse-go/wire"
)

// DefaultEndpoint is the Together AI API root; requests go under /v1, and the model catalogue under /v2.
const DefaultEndpoint = "https://api.together.ai"

// TogetherAIOptions configures a Together AI driver.
type TogetherAIOptions struct {
	common.DriverOptions
	// APIKey is the Together AI API key. Required.
	APIKey string
	// Endpoint overrides the API root. A trailing /v1 is removed, so https://api.together.ai/v1 also works.
	// It defaults to DefaultEndpoint.
	Endpoint string
	// DefaultMaxTokens is sent as max_tokens when the request sets none.
	DefaultMaxTokens int
	// HTTPClient overrides the HTTP client.
	HTTPClient *http.Client
}

// TogetherAIDriver implements Driver for Together AI on its Chat Completions API. Together has no usable
// Responses API for hosted open models, so every model runs on Chat Completions.
type TogetherAIDriver struct {
	*openai.OpenAIChatCompletionsDriver
}

var versionSuffix = regexp.MustCompile(`/v1/?$`)

// NewTogetherAIDriver creates a Together AI driver.
func NewTogetherAIDriver(options TogetherAIOptions) (*TogetherAIDriver, error) {
	if options.APIKey == "" {
		return nil, errors.New("together ai api key is required")
	}
	root := strings.TrimRight(versionSuffix.ReplaceAllString(options.Endpoint, ""), "/")
	if root == "" {
		root = DefaultEndpoint
	}
	chat, err := openai.NewOpenAIChatCompletionsDriver(openai.OpenAIChatCompletionsOptions{
		OpenAICompatibleOptions: openai.OpenAICompatibleOptions{
			DriverOptions: options.DriverOptions,
			APIKey:        options.APIKey,
			Endpoint:      root + "/v1",
			HTTPClient:    options.HTTPClient,
			Provider:      common.ProviderTogetherAI,
		},
		ChatCompletionsProtocolOptions: openai.ChatCompletionsProtocolOptions{
			Provider:             common.ProviderTogetherAI,
			DefaultMaxTokens:     options.DefaultMaxTokens,
			ResultSchemaInPrompt: true,
			PreparePayload:       preparePayload,
			NormalizeResponse:    normalizeResponse,
		},
	})
	if err != nil {
		return nil, err
	}
	return &TogetherAIDriver{OpenAIChatCompletionsDriver: chat}, nil
}

// Provider returns ProviderTogetherAI.
func (d *TogetherAIDriver) Provider() common.Provider {
	return common.ProviderTogetherAI
}

// preparePayload adapts a Chat Completions request to Together: assistant and tool content as text, non-strict
// tools, and reasoning switched off with reasoning.enabled on hybrid models that have no "none" effort.
func preparePayload(payload map[string]any, options common.ExecutionOptions, _ bool) error {
	if chatcompat.HasAudio(payload) {
		return chatcompat.ErrAudioInput
	}
	for _, message := range chatcompat.Messages(payload) {
		switch message["role"] {
		case "system", "developer":
			message["role"] = "system"
			message["content"] = chatcompat.TextContent(message["content"])
		case "assistant", "tool":
			if parts := chatcompat.Parts(message); parts != nil {
				message["content"] = flattenContent(parts)
			}
		}
	}
	if tools, ok := payload["tools"].([]any); ok {
		for _, raw := range tools {
			if tool, ok := raw.(map[string]any); ok {
				if function, ok := tool["function"].(map[string]any); ok {
					function["strict"] = false
				}
			}
		}
	}
	if payload["reasoning_effort"] == "none" && hasReasoningToggleOnly(options.Model) {
		delete(payload, "reasoning_effort")
		payload["reasoning"] = map[string]any{"enabled": false}
	}
	return nil
}

// flattenContent joins content parts into text, naming images by URL: Together takes assistant and tool content
// as text only.
func flattenContent(parts []map[string]any) string {
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		switch part["type"] {
		case "text":
			text, _ := part["text"].(string)
			texts = append(texts, text)
		case "image_url":
			image, _ := part["image_url"].(map[string]any)
			url, _ := image["url"].(string)
			texts = append(texts, "[Image: "+url+"]")
		}
	}
	return strings.Join(texts, "\n")
}

// hasReasoningToggleOnly reports the hybrid reasoning models whose only reasoning control the catalogue lists is
// switching it off: Together serves Qwen 3.5 to 3.7 (but not the Coder models) with thinking on by default.
func hasReasoningToggleOnly(model string) bool {
	levels := capability.ResolveProfile(model, common.ProviderTogetherAI).ReasoningEffortLevels
	return len(levels) == 1 && levels[0] == "none"
}

// normalizeResponse maps Together's eos finish reason to stop, and moves the usage some models (Kimi K3) report
// inside a stream choice to the chunk's usage.
func normalizeResponse(data []byte, _ common.ExecutionOptions) ([]byte, error) {
	if !strings.Contains(string(data), `"eos"`) && !strings.Contains(string(data), `"usage"`) {
		return data, nil
	}
	var response map[string]any
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, err
	}
	choices, _ := response["choices"].([]any)
	for _, raw := range choices {
		choice, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if choice["finish_reason"] == "eos" {
			choice["finish_reason"] = "stop"
		}
		if usage, ok := choice["usage"].(map[string]any); ok && response["usage"] == nil {
			response["usage"] = usage
		}
	}
	return json.Marshal(response)
}

type supportedModel struct {
	Name               string   `json:"name"`
	DisplayName        string   `json:"displayName"`
	DisplayType        string   `json:"displayType"`
	Publisher          string   `json:"publisher"`
	Status             string   `json:"status"`
	ServerlessEndpoint string   `json:"serverlessEndpoint"`
	InputModalities    []string `json:"inputModalities"`
	OutputModalities   []string `json:"outputModalities"`
}

type supportedModelsPage struct {
	Data       []supportedModel `json:"data"`
	NextCursor string           `json:"next_cursor"`
}

// maxCatalogPages bounds the catalogue walk against a server that keeps returning a cursor.
const maxCatalogPages = 100

func (d *TogetherAIDriver) supportedModels(ctx context.Context, cursor, operation string) (supportedModelsPage, error) {
	query := map[string]string{"product": "PRODUCT_SERVERLESS"}
	if cursor != "" {
		query["after"] = cursor
	}
	var page supportedModelsPage
	err := d.DoAPIRequest(ctx, http.MethodGet, "../v2/supported-models", nil, &page, openai.APIRequestOptions{Operation: operation, Query: query})
	return page, err
}

// ListModels lists the serverless models of the Together catalogue, following its cursor, and leaves out
// embedding and hidden models. Models are listed by the name inference takes: the serverless endpoint name, or
// the catalogue name.
func (d *TogetherAIDriver) ListModels(ctx context.Context, _ *common.ModelSearchPayload) ([]common.AIModel, error) {
	var models []common.AIModel
	cursor := ""
	for range maxCatalogPages {
		page, err := d.supportedModels(ctx, cursor, "listModels")
		if err != nil {
			return nil, err
		}
		for _, model := range page.Data {
			modelType := togetherModelType(model.DisplayType)
			if modelType == wire.ModelTypeEmbedding || model.Status == "SUPPORTED_MODEL_STATUS_HIDDEN" {
				continue
			}
			id := model.ServerlessEndpoint
			if id == "" {
				id = model.Name
			}
			name := model.DisplayName
			if name == "" {
				name = id
			}
			aiModel := common.AIModel{
				ID:        id,
				Name:      name,
				Provider:  string(common.ProviderTogetherAI),
				Type:      &modelType,
				CanStream: ptr(true),
			}
			if model.Publisher != "" {
				aiModel.Owner = ptr(model.Publisher)
			}
			capability.ForModel(id, common.ProviderTogetherAI).Apply(&aiModel)
			models = append(models, aiModel)
		}
		if page.NextCursor == "" || len(page.Data) == 0 {
			break
		}
		cursor = page.NextCursor
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	return models, nil
}

// ValidateConnection checks the API key against the first page of the model catalogue.
func (d *TogetherAIDriver) ValidateConnection(ctx context.Context) error {
	_, err := d.supportedModels(ctx, "", "validateConnection")
	return err
}

func togetherModelType(displayType string) wire.ModelType {
	switch displayType {
	case "chat":
		return wire.ModelTypeChat
	case "code":
		return wire.ModelTypeCode
	case "image":
		return wire.ModelTypeImage
	case "embedding":
		return wire.ModelTypeEmbedding
	}
	return wire.ModelTypeText
}

func ptr[T any](v T) *T { return &v }

var _ common.Driver = (*TogetherAIDriver)(nil)
