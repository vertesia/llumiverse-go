package bedrockmantle

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/vertesia/llumiverse-go/capability"
	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
	"github.com/vertesia/llumiverse-go/drivers/openai"
)

// openAIPlaceholderKey satisfies the shared OpenAI drivers, which require a key; the transport replaces it with
// the Mantle bearer token before sending.
const openAIPlaceholderKey = "bedrock-mantle"

// BedrockMantleOptions configures a Bedrock Mantle driver.
type BedrockMantleOptions struct {
	DriverOptions
	// Region is the AWS region of the Mantle endpoint, https://bedrock-mantle.<region>.api.aws. Required.
	Region string
	// Config supplies the AWS credentials. When its Credentials are nil and APIKey is empty,
	// NewBedrockMantleDriver loads the default AWS configuration for Region.
	Config aws.Config
	// APIKey is a Bedrock API key. When set, every request sends it as the bearer token instead of credentials
	// derived from Config.
	APIKey string
	// Endpoint overrides the Mantle endpoint root (scheme and host); the driver adds /v1, /openai/v1 and
	// /anthropic. It defaults to https://bedrock-mantle.<Region>.api.aws.
	Endpoint string
	// HTTPClient overrides the HTTP client; its transport is kept.
	HTTPClient *http.Client
}

// BedrockMantleDriver implements Driver for the Amazon Bedrock Mantle endpoint, which serves each model family on
// its own protocol: OpenAI GPT, Grok and Gemma 4+ on Responses, Claude on the Anthropic Messages API, and the
// other families (GPT-OSS, Gemma 3, DeepSeek, Qwen, Mistral, ...) on Chat Completions.
type BedrockMantleDriver struct {
	options   BedrockMantleOptions
	v1        string
	anthropic string
	// client sends model listing on /v1; messages sends Claude requests.
	client   *http.Client
	messages *http.Client
	// responses speaks Responses on /openai/v1; chat speaks Chat Completions on /v1.
	responses *openai.OpenAICompatibleDriver
	chat      *openai.OpenAIChatCompletionsDriver
}

var regionPattern = regexp.MustCompile(`^[a-z]{2,8}(?:-[a-z0-9]+)+-\d+$`)

// NewBedrockMantleDriver creates a Bedrock Mantle driver. Without an APIKey or credentials in Config it loads the
// default AWS configuration for Region, as NewBedrockDriver does.
func NewBedrockMantleDriver(ctx context.Context, options BedrockMantleOptions) (*BedrockMantleDriver, error) {
	if options.APIKey == "" && options.Config.Credentials == nil {
		cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(options.Region))
		if err != nil {
			return nil, fmt.Errorf("bedrock mantle: load AWS config: %w", err)
		}
		options.Config = cfg
	}
	return NewBedrockMantleDriverWithCredentials(options)
}

// NewBedrockMantleDriverWithCredentials creates a Bedrock Mantle driver from the APIKey or the credentials in
// options.Config alone, without loading any AWS configuration.
func NewBedrockMantleDriverWithCredentials(options BedrockMantleOptions) (*BedrockMantleDriver, error) {
	region := strings.TrimSpace(options.Region)
	if region == "" {
		return nil, errors.New("bedrock mantle region is required")
	}
	if !regionPattern.MatchString(region) {
		return nil, fmt.Errorf("invalid bedrock mantle region %q", options.Region)
	}
	if options.APIKey == "" && options.Config.Credentials == nil {
		return nil, errors.New("bedrock mantle requires an APIKey or AWS credentials")
	}
	options.Region = region
	endpoint := strings.TrimRight(options.Endpoint, "/")
	if endpoint == "" {
		endpoint = "https://bedrock-mantle." + region + ".api.aws"
	}
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Host == "" {
		return nil, fmt.Errorf("invalid bedrock mantle endpoint %q", options.Endpoint)
	}
	base := options.HTTPClient
	if base == nil {
		base = &http.Client{}
	}
	base = newHTTPClient(base, options.HTTPTimeout)
	credentials := newAuth(options.APIKey, options.Config.Credentials, region)
	openAIClient := authenticatedClient(base, &transport{auth: credentials, host: parsed.Host})
	d := &BedrockMantleDriver{
		options:   options,
		v1:        endpoint + "/v1",
		anthropic: endpoint + "/anthropic",
		client:    openAIClient,
		messages:  authenticatedClient(base, &transport{auth: credentials, host: parsed.Host, messages: true}),
	}
	driverOptions := options.DriverOptions
	driverOptions.HTTPTimeout = HTTPTimeoutOptions{}
	if d.responses, err = openai.NewOpenAICompatibleDriver(openai.OpenAICompatibleOptions{
		DriverOptions: driverOptions,
		APIKey:        openAIPlaceholderKey,
		Endpoint:      endpoint + "/openai/v1",
		HTTPClient:    openAIClient,
		Provider:      ProviderBedrockMantle,
	}); err != nil {
		return nil, err
	}
	if d.chat, err = openai.NewOpenAIChatCompletionsDriver(openai.OpenAIChatCompletionsOptions{
		OpenAICompatibleOptions: openai.OpenAICompatibleOptions{
			DriverOptions: driverOptions,
			APIKey:        openAIPlaceholderKey,
			Endpoint:      d.v1,
			HTTPClient:    openAIClient,
			Provider:      ProviderBedrockMantle,
		},
		ChatCompletionsProtocolOptions: openai.ChatCompletionsProtocolOptions{
			Provider: ProviderBedrockMantle,
			// Mantle enforces response_format; Magistral follows it loosely, so it also gets the schema as an
			// instruction. Tool parameters are plain JSON Schema.
			IncludeResultSchemaInPromptForModel: func(model string) bool {
				return strings.Contains(strings.ToLower(model), "mistral.magistral-")
			},
			CompatibleToolSchemas: true,
		},
	}); err != nil {
		return nil, err
	}
	return d, nil
}

// Provider returns ProviderBedrockMantle.
func (d *BedrockMantleDriver) Provider() Provider {
	return ProviderBedrockMantle
}

func unsupportedModel(model string, operation string) error {
	return newLlumiverseError(fmt.Sprintf("[%s] unsupported Bedrock Mantle model: %s", ProviderBedrockMantle, model), boolPtr(false), LlumiverseErrorContext{
		Provider:  ProviderBedrockMantle,
		Model:     model,
		Operation: operation,
	}, nil, 0, "UnsupportedModel")
}

// CreatePrompt formats the prompt of the model's protocol: Responses input, Chat Completions messages, or a
// Claude Messages prompt.
func (d *BedrockMantleDriver) CreatePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (any, error) {
	protocol, ok := ModelProtocol(options.Model)
	if !ok {
		return nil, unsupportedModel(options.Model, "createPrompt")
	}
	switch protocol {
	case ProtocolResponses:
		return d.responses.CreatePrompt(ctx, segments, options)
	case ProtocolChatCompletions:
		return d.chat.CreatePrompt(ctx, segments, options)
	default:
		return internalclaude.FormatPrompt(ctx, segments, options)
	}
}

// FormatDebugPrompt returns a copy of a prompt with inline media truncated, for ExecutionResponse.Prompt.
func (d *BedrockMantleDriver) FormatDebugPrompt(prompt any) any {
	switch value := prompt.(type) {
	case internalclaude.Prompt:
		return internalclaude.FormatDebugPrompt(value)
	case openai.ChatCompletionsPrompt:
		return d.chat.FormatDebugPrompt(value)
	}
	return d.responses.FormatDebugPrompt(prompt)
}

// Execute runs a non-streaming completion on the model's protocol.
func (d *BedrockMantleDriver) Execute(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (*ExecutionResponse, error) {
	if options.Model == "" {
		return nil, errors.New("model is required")
	}
	protocol, ok := ModelProtocol(options.Model)
	if !ok {
		return nil, unsupportedModel(options.Model, "execute")
	}
	switch protocol {
	case ProtocolResponses:
		return d.responses.Execute(ctx, segments, options)
	case ProtocolChatCompletions:
		return d.chat.Execute(ctx, segments, options)
	default:
		return executeWithPrompt(ctx, d, segments, options, d.requestClaude)
	}
}

// Stream runs a streaming completion on the model's protocol.
func (d *BedrockMantleDriver) Stream(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (CompletionStream, error) {
	if options.Model == "" {
		return nil, errors.New("model is required")
	}
	protocol, ok := ModelProtocol(options.Model)
	if !ok {
		return nil, unsupportedModel(options.Model, "stream")
	}
	switch protocol {
	case ProtocolResponses:
		return d.responses.Stream(ctx, segments, options)
	case ProtocolChatCompletions:
		return d.chat.Stream(ctx, segments, options)
	}
	// The shared OpenAI drivers lease their own streams; the Claude stream is this driver's.
	stream, err := streamWithPrompt(ctx, d, segments, options, d.streamClaude)
	if err != nil {
		return nil, err
	}
	return leaseCompletionStream(stream, d.options.StreamStartTimeout), nil
}

// mantleModel is an entry of the /v1/models listing.
type mantleModel struct {
	ID string `json:"id"`
}

// ListModels lists the Mantle models the driver can route, sorted by ID, with the capability catalogue's modalities
// and tool support. Embedding models and models of families without a known protocol are left out.
func (d *BedrockMantleDriver) ListModels(ctx context.Context, _ *ModelSearchPayload) ([]AIModel, error) {
	endpoint, err := joinEndpoint(d.v1, "models")
	if err != nil {
		return nil, err
	}
	var response struct {
		Data []mantleModel `json:"data"`
	}
	if err := doJSON(ctx, d.client, http.MethodGet, endpoint, nil, nil, &response); err != nil {
		return nil, d.formatError(err, "", "listModels", "OpenAIError")
	}
	models := make([]AIModel, 0, len(response.Data))
	for _, model := range response.Data {
		if capability.IsEmbeddingModel(capability.Listing{ID: model.ID}, ProviderBedrockMantle) {
			continue
		}
		if _, ok := ModelProtocol(model.ID); !ok {
			continue
		}
		owner := modelOwner(model.ID)
		aiModel := AIModel{
			ID:        model.ID,
			Name:      modelDisplayName(model.ID, owner),
			Provider:  string(ProviderBedrockMantle),
			Owner:     ptr(owner),
			Type:      ptr(modelTypeText),
			CanStream: ptr(true),
		}
		capability.ResolveListingMetadata(model.ID, ProviderBedrockMantle, nil, nil).Apply(&aiModel)
		models = append(models, aiModel)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	return models, nil
}

// ValidateConnection checks the credentials against the model listing.
func (d *BedrockMantleDriver) ValidateConnection(ctx context.Context) error {
	endpoint, err := joinEndpoint(d.v1, "models")
	if err != nil {
		return err
	}
	if err := doJSON(ctx, d.client, http.MethodGet, endpoint, nil, nil, nil); err != nil {
		return d.formatError(err, "", "validateConnection", "OpenAIError")
	}
	return nil
}

// GenerateEmbeddings generates embeddings on the OpenAI-compatible endpoint, as the TS client does.
func (d *BedrockMantleDriver) GenerateEmbeddings(ctx context.Context, options EmbeddingsOptions) (*EmbeddingsResult, error) {
	return d.responses.GenerateEmbeddings(ctx, options)
}

// formatError wraps an HTTP error as a LlumiverseError with the provider, model and operation, the status, and a
// retryability derived from the status and message. LlumiverseErrors pass through unchanged.
func (d *BedrockMantleDriver) formatError(err error, model string, operation string, name string) error {
	var llumiErr *LlumiverseError
	if err == nil || errors.As(err, &llumiErr) {
		return err
	}
	code, typeName := errorStatusAndName(err)
	if code == 0 {
		name = typeName
	}
	return newLlumiverseError(fmt.Sprintf("[%s] %s", ProviderBedrockMantle, err.Error()), retryableFromStatusAndMessage(code, err.Error()), LlumiverseErrorContext{
		Provider:  ProviderBedrockMantle,
		Model:     model,
		Operation: operation,
	}, err, code, name)
}
