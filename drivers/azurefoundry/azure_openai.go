package azurefoundry

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/vertesia/llumiverse-go/capability"
	"github.com/vertesia/llumiverse-go/drivers/openai"
)

// AzureOpenAIOptions configures an Azure OpenAI driver on the version-less v1
// API (<resource>/openai/v1), where Azure ships the Responses API.
type AzureOpenAIOptions struct {
	DriverOptions
	// Endpoint is the resource endpoint, https://<resource>.openai.azure.com
	// (a trailing /openai or /openai/v1 is accepted). Required.
	Endpoint string
	// APIKey is the resource key, sent as the api-key header.
	APIKey string
	// TokenProvider returns Microsoft Entra ID tokens for the scope of the
	// endpoint's cloud (see CognitiveServicesScope). One of APIKey and
	// TokenProvider is required.
	TokenProvider TokenProvider
	// Deployment is the deployment of the deployment-scoped APIs (Chat
	// Completions, embeddings, images), sent as their model whatever the
	// request's model, as the legacy Azure OpenAI client did. ListModels needs it.
	Deployment string
	// SourceModel names the model of the configured deployment (or of any bare
	// deployment name when Deployment is unset) when its name does not say it.
	SourceModel string
	// HTTPClient overrides the HTTP client; its transport is kept.
	HTTPClient *http.Client
}

// AzureOpenAIDriver implements Driver for Azure OpenAI with the shared OpenAI
// Responses implementation.
type AzureOpenAIDriver struct {
	*openai.OpenAICompatibleDriver
	options AzureOpenAIOptions
	baseURL string
	creds   credentials
	client  *http.Client
}

// deploymentScopedPaths are the endpoints the legacy Azure OpenAI client sent
// to its configured deployment, whatever the request's model.
var deploymentScopedPaths = []string{
	"/completions",
	"/chat/completions",
	"/embeddings",
	"/audio/transcriptions",
	"/audio/translations",
	"/audio/speech",
	"/images/generations",
	"/images/edits",
}

// NewAzureOpenAIDriver creates an Azure OpenAI driver.
func NewAzureOpenAIDriver(options AzureOpenAIOptions) (*AzureOpenAIDriver, error) {
	if options.Endpoint == "" {
		return nil, errors.New("azure openai endpoint is required")
	}
	if options.APIKey == "" && options.TokenProvider == nil {
		return nil, errors.New("azure openai requires an APIKey or a TokenProvider")
	}
	baseURL := azureOpenAIV1BaseURL(options.Endpoint)
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Host == "" {
		return nil, fmt.Errorf("invalid azure openai endpoint %q", options.Endpoint)
	}
	creds := credentials{apiKey: options.APIKey, tokens: options.TokenProvider, scope: CognitiveServicesScope(options.Endpoint)}
	base := options.HTTPClient
	if base == nil {
		base = &http.Client{}
	}
	base = newHTTPClient(base, options.HTTPTimeout)
	client := noKeyRedirects(base, options.APIKey != "")
	client.Transport = &openAITransport{
		base:  base.Transport,
		creds: creds,
		host:  parsed.Host,
		// Image requests take the preview api-version on the v1 API, as in the TS client.
		query: func(path string) map[string]string {
			if strings.Contains(path, "/images/") {
				return map[string]string{"api-version": "preview"}
			}
			return nil
		},
		model: func(path string, model string) string {
			if options.Deployment != "" && isDeploymentScopedPath(path) {
				return options.Deployment
			}
			deployment, _ := parseModelID(model)
			return deployment
		},
	}
	driverOptions := options.DriverOptions
	driverOptions.HTTPTimeout = HTTPTimeoutOptions{}
	inner, err := newOpenAIDriver(driverOptions, baseURL, client, ProviderAzureOpenAI)
	if err != nil {
		return nil, err
	}
	return &AzureOpenAIDriver{
		OpenAICompatibleDriver: inner,
		options:                options,
		baseURL:                baseURL,
		creds:                  creds,
		client:                 noKeyRedirects(base, false),
	}, nil
}

func isDeploymentScopedPath(path string) bool {
	for _, scoped := range deploymentScopedPaths {
		if strings.HasSuffix(path, scoped) {
			return true
		}
	}
	return false
}

// Provider returns ProviderAzureOpenAI.
func (d *AzureOpenAIDriver) Provider() Provider {
	return ProviderAzureOpenAI
}

// azureOpenAIV1BaseURL returns the v1 base URL of an Azure OpenAI or Foundry
// resource endpoint, accepting an endpoint that already ends in /openai or /openai/v1.
func azureOpenAIV1BaseURL(endpoint string) string {
	trimmed := trimTrailingSlashes(endpoint)
	trimmed = strings.TrimSuffix(trimmed, "/openai/v1")
	trimmed = strings.TrimSuffix(trimmed, "/openai")
	return trimmed + "/openai/v1"
}

// requestModel qualifies a bare deployment name with the configured source
// model, so the shared driver's model-based decisions (image routing,
// reasoning options) see the model; only the deployment is sent. The source
// model applies to the configured deployment, or to any bare name when no
// deployment is configured.
func (d *AzureOpenAIDriver) requestModel(model string) string {
	if isCompositeModelID(model) || d.options.SourceModel == "" {
		return model
	}
	if d.options.Deployment != "" && model != d.options.Deployment {
		return model
	}
	return model + "::" + d.options.SourceModel
}

// CreatePrompt builds the Responses prompt.
func (d *AzureOpenAIDriver) CreatePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (any, error) {
	options.Model = d.requestModel(options.Model)
	return d.OpenAICompatibleDriver.CreatePrompt(ctx, segments, options)
}

// Execute runs a non-streaming completion.
func (d *AzureOpenAIDriver) Execute(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (*ExecutionResponse, error) {
	options.Model = d.requestModel(options.Model)
	return d.OpenAICompatibleDriver.Execute(ctx, segments, options)
}

// Stream runs a streaming completion. Image models do not stream: their result
// arrives as a single chunk.
func (d *AzureOpenAIDriver) Stream(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (CompletionStream, error) {
	options.Model = d.requestModel(options.Model)
	if isImageModel(options.Model, ProviderAzureOpenAI) {
		response, err := d.OpenAICompatibleDriver.Execute(ctx, segments, options)
		if err != nil {
			return nil, err
		}
		return singleChunkStream(response), nil
	}
	return d.OpenAICompatibleDriver.Stream(ctx, segments, options)
}

// ListModels describes the configured deployment: Azure OpenAI cannot list
// deployments. A one-token Chat Completions call reports the model a text
// deployment runs, as in the TS client; embedding and DALL-E deployments are
// not listed. Modalities and tool support come from the capability catalogue.
func (d *AzureOpenAIDriver) ListModels(ctx context.Context, _ *ModelSearchPayload) ([]AIModel, error) {
	deployment := d.options.Deployment
	if deployment == "" {
		return nil, errors.New("a specific deployment is not set: Azure OpenAI cannot list deployments. Set the Deployment option to the deployment name")
	}
	modelID := d.options.SourceModel
	if modelID == "" {
		modelID = deployment
	}
	if !isImageModel(modelID, ProviderAzureOpenAI) {
		if reported, err := d.deploymentModel(ctx, deployment); err == nil && reported != "" {
			modelID = reported
		}
	}
	if strings.Contains(strings.ToLower(modelID), "dall-e") ||
		capability.IsEmbeddingModel(capability.Listing{ID: modelID}, ProviderAzureOpenAI) {
		return []AIModel{}, nil
	}
	model := AIModel{
		ID:       modelID,
		Name:     deployment,
		Provider: string(ProviderAzureOpenAI),
		Owner:    ptr("openai"),
	}
	capability.ResolveListingMetadata(modelID, ProviderAzureOpenAI, nil, nil).Apply(&model)
	if isImageModel(modelID, ProviderAzureOpenAI) {
		model.ID = deployment + "::" + modelID
		model.Type = ptr(modelTypeImage)
	}
	return []AIModel{model}, nil
}

// deploymentModel asks a deployment which model it runs with a one-token completion.
func (d *AzureOpenAIDriver) deploymentModel(ctx context.Context, deployment string) (string, error) {
	endpoint, err := joinEndpoint(d.baseURL, "chat", "completions")
	if err != nil {
		return "", err
	}
	headers, err := d.creds.keyOrBearer(ctx, headerAPIKey)
	if err != nil {
		return "", err
	}
	payload := map[string]any{
		"model":      deployment,
		"messages":   []map[string]any{{"role": "user", "content": "Hi"}},
		"max_tokens": 1,
	}
	var response struct {
		Model string `json:"model"`
	}
	if err := doJSON(ctx, d.client, http.MethodPost, endpoint, headers, payload, &response); err != nil {
		return "", err
	}
	return response.Model, nil
}

// ValidateConnection checks the credentials against the v1 model list.
func (d *AzureOpenAIDriver) ValidateConnection(ctx context.Context) error {
	endpoint, err := joinEndpoint(d.baseURL, "models")
	if err != nil {
		return err
	}
	headers, err := d.creds.keyOrBearer(ctx, headerAPIKey)
	if err != nil {
		return err
	}
	if err := doJSON(ctx, d.client, http.MethodGet, endpoint, headers, nil, nil); err != nil {
		return formatAzureError(err, ProviderAzureOpenAI, "", "validateConnection", "AzureError")
	}
	return nil
}
