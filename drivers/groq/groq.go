package groq

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

// DefaultEndpoint is the Groq API root; the driver sends requests under /openai/v1.
const DefaultEndpoint = "https://api.groq.com"

// GroqOptions configures a Groq driver.
type GroqOptions struct {
	common.DriverOptions
	// APIKey is the Groq API key. Required.
	APIKey string
	// Endpoint overrides the API root, as the TS client's endpoint_url does; the driver adds /openai/v1.
	// It defaults to DefaultEndpoint.
	Endpoint string
	// DefaultMaxTokens is sent as max_tokens when the request sets none.
	DefaultMaxTokens int
	// HTTPClient overrides the HTTP client.
	HTTPClient *http.Client
}

// GroqDriver implements Driver for Groq on its OpenAI-compatible Chat Completions API.
type GroqDriver struct {
	*openai.OpenAIChatCompletionsDriver
}

// NewGroqDriver creates a Groq driver.
func NewGroqDriver(options GroqOptions) (*GroqDriver, error) {
	if options.APIKey == "" {
		return nil, errors.New("groq api key is required")
	}
	endpoint := strings.TrimSuffix(strings.TrimRight(options.Endpoint, "/"), "/openai/v1")
	if endpoint == "" {
		endpoint = DefaultEndpoint
	}
	chat, err := openai.NewOpenAIChatCompletionsDriver(openai.OpenAIChatCompletionsOptions{
		OpenAICompatibleOptions: openai.OpenAICompatibleOptions{
			DriverOptions: options.DriverOptions,
			APIKey:        options.APIKey,
			Endpoint:      endpoint + "/openai/v1",
			HTTPClient:    options.HTTPClient,
			Provider:      common.ProviderGroq,
		},
		ChatCompletionsProtocolOptions: openai.ChatCompletionsProtocolOptions{
			Provider:         common.ProviderGroq,
			DefaultMaxTokens: options.DefaultMaxTokens,
			// The schema always goes in the prompt; preparePayload adds response_format for the models that
			// take Structured Outputs.
			ResultSchemaInPrompt:  true,
			CompatibleToolSchemas: true,
			PreparePayload:        preparePayload,
			NormalizeResponse:     normalizeResponse,
		},
	})
	if err != nil {
		return nil, err
	}
	return &GroqDriver{OpenAIChatCompletionsDriver: chat}, nil
}

// Provider returns ProviderGroq.
func (d *GroqDriver) Provider() common.Provider {
	return common.ProviderGroq
}

// preparePayload adapts a Chat Completions request to Groq: max_completion_tokens instead of max_tokens, system
// messages for developer ones, Structured Outputs where Groq supports them, and reasoning_format.
func preparePayload(payload map[string]any, options common.ExecutionOptions, _ bool) error {
	if chatcompat.HasAudio(payload) {
		return chatcompat.ErrAudioInput
	}
	if value, ok := payload["max_tokens"]; ok {
		delete(payload, "max_tokens")
		payload["max_completion_tokens"] = value
	}
	for _, message := range chatcompat.Messages(payload) {
		if message["role"] == "developer" {
			message["role"] = "system"
		}
	}
	if format := responseFormat(payload, options); format != nil {
		payload["response_format"] = format
	}
	if format, _ := options.ModelOptions["reasoning_format"].(string); format != "" {
		payload["reasoning_format"] = format
	}
	return nil
}

// responseFormat returns Groq Structured Outputs: json_schema on gpt-oss and Qwen 3.8 and later, strict
// (constrained decoding) except on gpt-oss-safeguard. Groq rejects response_format together with tools, so a
// request with tools keeps the prompt instruction alone, which the driver always sends.
func responseFormat(payload map[string]any, options common.ExecutionOptions) map[string]any {
	if options.ResultSchema == nil || chatcompat.HasTools(payload) {
		return nil
	}
	model := chatcompat.BaseModelID(options.Model)
	if !strings.HasPrefix(model, "gpt-oss-") && !chatcompat.IsQwenVersionGTE(model, 3, 8) {
		return nil
	}
	schema, strict := openai.StructuredOutputSchema(options.ResultSchema)
	return map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name":   "output",
			"strict": strict && !strings.Contains(model, "safeguard"),
			"schema": schema,
		},
	}
}

// normalizeResponse moves the usage Groq reports under x_groq in stream chunks to the chunk's usage.
func normalizeResponse(data []byte, _ common.ExecutionOptions) ([]byte, error) {
	if !strings.Contains(string(data), `"x_groq"`) {
		return data, nil
	}
	var chunk map[string]any
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil, err
	}
	extension, _ := chunk["x_groq"].(map[string]any)
	usage, ok := extension["usage"]
	if !ok || chunk["usage"] != nil {
		return data, nil
	}
	chunk["usage"] = usage
	return json.Marshal(chunk)
}

var (
	whisperPattern = regexp.MustCompile(`(^|[-_.:/])whisper([-_.:/]|$)`)
	orpheusPattern = regexp.MustCompile(`^canopylabs/orpheus([-_.:/]|$)`)
)

type groqModel struct {
	ID      string `json:"id"`
	OwnedBy string `json:"owned_by"`
}

// ListModels lists the Groq text models, leaving out embedding, Whisper transcription and Orpheus speech models.
func (d *GroqDriver) ListModels(ctx context.Context, _ *common.ModelSearchPayload) ([]common.AIModel, error) {
	var response struct {
		Data []groqModel `json:"data"`
	}
	if err := d.DoAPIRequest(ctx, http.MethodGet, "models", nil, &response, openai.APIRequestOptions{Operation: "listModels"}); err != nil {
		return nil, err
	}
	models := make([]common.AIModel, 0, len(response.Data))
	for _, model := range response.Data {
		id := strings.ToLower(model.ID)
		if capability.IsEmbeddingModel(capability.Listing{ID: model.ID}, common.ProviderGroq) ||
			whisperPattern.MatchString(id) || orpheusPattern.MatchString(id) {
			continue
		}
		owner := model.OwnedBy
		aiModel := common.AIModel{
			ID:        model.ID,
			Name:      model.ID,
			Provider:  string(common.ProviderGroq),
			Owner:     &owner,
			Type:      ptr(wire.ModelTypeText),
			CanStream: ptr(true),
		}
		capability.ForModel(model.ID, common.ProviderGroq).Apply(&aiModel)
		models = append(models, aiModel)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	return models, nil
}

// ValidateConnection checks the API key against the model listing.
func (d *GroqDriver) ValidateConnection(ctx context.Context) error {
	return d.DoAPIRequest(ctx, http.MethodGet, "models", nil, nil, openai.APIRequestOptions{Operation: "validateConnection"})
}

// GenerateEmbeddings is not available: Groq serves no embedding models.
func (d *GroqDriver) GenerateEmbeddings(context.Context, common.EmbeddingsOptions) (*common.EmbeddingsResult, error) {
	return nil, errors.New("groq does not expose an embeddings transport")
}

func ptr[T any](v T) *T { return &v }

var _ common.Driver = (*GroqDriver)(nil)
