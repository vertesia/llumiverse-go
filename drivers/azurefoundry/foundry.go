package azurefoundry

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"

	"github.com/vertesia/llumiverse-go/core"
	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
	"github.com/vertesia/llumiverse-go/drivers/openai"
)

// foundryProjectAPIVersion is the api-version of the Foundry project API (deployments).
const foundryProjectAPIVersion = "v1"

// openAIPlaceholderKey satisfies the shared OpenAI driver, which requires a key;
// openAITransport replaces it with the Azure credentials before sending.
const openAIPlaceholderKey = "azure"

// AzureFoundryOptions configures a Microsoft Foundry driver.
type AzureFoundryOptions struct {
	DriverOptions
	// Endpoint is the Foundry project endpoint,
	// https://<resource>.services.ai.azure.com/api/projects/<project>. Required.
	Endpoint string
	// APIKey is a Foundry resource key. When set, inference (Responses, Chat
	// Completions, Claude Messages, images and embeddings) authenticates with it
	// on the resource endpoint instead of Microsoft Entra ID. The project API
	// (deployment discovery and ListModels) accepts Entra ID only, so set
	// TokenProvider too to resolve bare deployment names by their publisher.
	APIKey string
	// TokenProvider returns Microsoft Entra ID tokens for the scope of the
	// endpoint's cloud (see FoundryScope). One of APIKey and TokenProvider is required.
	TokenProvider TokenProvider
	// SourceModel names the model of bare deployment names whose names do not say
	// it (an opaque image deployment, for example). A "deployment::source-model"
	// ID takes precedence.
	SourceModel string
	// APIVersion, when set, is sent as the api-version query parameter of the
	// OpenAI v1 requests. The v1 API needs none.
	APIVersion string
	// HTTPClient overrides the HTTP client; its transport is kept.
	HTTPClient *http.Client
}

// AzureFoundryDriver implements Driver for Microsoft Foundry deployments.
type AzureFoundryDriver struct {
	options     AzureFoundryOptions
	endpoint    string
	creds       credentials
	client      *http.Client
	inferenceV1 string
	resourceV1  string
	anthropic   string
	// inference speaks the Responses API on the inference endpoint (the project
	// with Entra ID, the resource with a key); resource serves images on the resource endpoint.
	inference *openai.OpenAICompatibleDriver
	resource  *openai.OpenAICompatibleDriver

	mu        sync.Mutex
	protocols map[string]protocol
	sources   map[string]string
}

// NewAzureFoundryDriver creates a Microsoft Foundry driver.
func NewAzureFoundryDriver(options AzureFoundryOptions) (*AzureFoundryDriver, error) {
	if options.Endpoint == "" {
		return nil, errors.New("microsoft foundry endpoint is required")
	}
	if options.APIKey == "" && options.TokenProvider == nil {
		return nil, errors.New("microsoft foundry requires an APIKey or a TokenProvider")
	}
	endpoint := trimTrailingSlashes(options.Endpoint)
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Host == "" {
		return nil, fmt.Errorf("invalid microsoft foundry endpoint %q", options.Endpoint)
	}
	creds := credentials{apiKey: options.APIKey, tokens: options.TokenProvider, scope: FoundryScope(endpoint)}
	projectV1 := endpoint + "/openai/v1"
	resourceV1, err := resourceInferenceURL(projectV1)
	if err != nil {
		return nil, err
	}
	// API keys authenticate the resource endpoint, so key-based inference goes
	// there instead of the project.
	inferenceV1 := projectV1
	if options.APIKey != "" {
		inferenceV1 = resourceV1
	}
	anthropic, err := anthropicBaseURL(endpoint)
	if err != nil {
		return nil, err
	}
	base := options.HTTPClient
	if base == nil {
		base = &http.Client{}
	}
	base = newHTTPClient(base, options.HTTPTimeout)
	d := &AzureFoundryDriver{
		options:     options,
		endpoint:    endpoint,
		creds:       creds,
		client:      noKeyRedirects(base, false),
		inferenceV1: inferenceV1,
		resourceV1:  resourceV1,
		anthropic:   anthropic,
		protocols:   map[string]protocol{},
		sources:     map[string]string{},
	}
	openAIClient := noKeyRedirects(base, options.APIKey != "")
	openAIClient.Transport = &openAITransport{
		base:  base.Transport,
		creds: creds,
		host:  parsed.Host,
		query: func(string) map[string]string {
			if options.APIVersion == "" {
				return nil
			}
			return map[string]string{"api-version": options.APIVersion}
		},
		model: deploymentOf,
	}
	driverOptions := options.DriverOptions
	driverOptions.HTTPTimeout = HTTPTimeoutOptions{}
	if d.inference, err = newOpenAIDriver(driverOptions, inferenceV1, openAIClient, ProviderAzureFoundry); err != nil {
		return nil, err
	}
	if d.resource, err = newOpenAIDriver(driverOptions, resourceV1, openAIClient, ProviderAzureFoundry); err != nil {
		return nil, err
	}
	return d, nil
}

// newOpenAIDriver builds the shared OpenAI driver on an Azure v1 base URL.
func newOpenAIDriver(options DriverOptions, baseURL string, client *http.Client, provider Provider) (*openai.OpenAICompatibleDriver, error) {
	return openai.NewOpenAICompatibleDriver(openai.OpenAICompatibleOptions{
		DriverOptions: options,
		APIKey:        openAIPlaceholderKey,
		Endpoint:      baseURL,
		HTTPClient:    client,
		Provider:      provider,
	})
}

// Provider returns ProviderAzureFoundry.
func (d *AzureFoundryDriver) Provider() Provider {
	return ProviderAzureFoundry
}

// route is how a model ID is served: the protocol of its deployment, or the
// Images API for image models.
type route struct {
	protocol   protocol
	image      bool
	deployment string
	source     string
}

// openAIModel is the model the shared OpenAI driver receives: the deployment
// qualified with its source model, so the driver's model-based decisions see
// the model rather than the deployment alias. openAITransport sends the
// deployment alone.
func (r route) openAIModel() string {
	if r.source == "" || r.source == r.deployment {
		return r.deployment
	}
	return r.deployment + "::" + r.source
}

// apiKeyOnly reports whether inference uses a key and no Entra credential can
// reach the project API.
func (d *AzureFoundryDriver) apiKeyOnly() bool {
	return d.options.APIKey != "" && d.options.TokenProvider == nil
}

// sourceModel returns the model a model ID runs: the source half of a composite
// ID, else the SourceModel option, else the model the project API reported for
// the deployment, else the deployment name.
func (d *AzureFoundryDriver) sourceModel(model string) string {
	deployment, source := parseModelID(model)
	if isCompositeModelID(model) {
		return source
	}
	if d.options.SourceModel != "" {
		return d.options.SourceModel
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if discovered, ok := d.sources[deployment]; ok {
		return discovered
	}
	return source
}

// resolve returns the route of a model ID.
func (d *AzureFoundryDriver) resolve(ctx context.Context, model string) (route, error) {
	deployment, _ := parseModelID(model)
	// An explicit image source keeps the Images path independent of publisher discovery.
	explicit := isCompositeModelID(model) || d.options.SourceModel != ""
	if source := d.sourceModel(model); explicit && isImageModel(source) {
		return route{image: true, deployment: deployment, source: source}, nil
	}
	p, err := d.deploymentProtocol(ctx, model)
	if err != nil {
		return route{}, err
	}
	source := d.sourceModel(model)
	if isImageModel(source) {
		return route{image: true, deployment: deployment, source: source}, nil
	}
	return route{protocol: p, deployment: deployment, source: source}, nil
}

// deploymentProtocol returns the protocol of a model's deployment. Claude
// source models use Messages. Without an Entra credential the protocol follows
// the source model and is derived on every call, so a later composite ID
// overrides a guess from the bare name; otherwise the project API reports the
// deployment's publisher, which is cached. Failed lookups are not cached.
func (d *AzureFoundryDriver) deploymentProtocol(ctx context.Context, model string) (protocol, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	deployment, _ := parseModelID(model)
	source := d.sourceModel(model)
	if isClaudeModel(source) {
		d.setProtocol(deployment, protocolMessages, "")
		return protocolMessages, nil
	}
	if d.apiKeyOnly() {
		p := inferProtocol(source)
		d.setProtocol(deployment, p, "")
		return p, nil
	}
	d.mu.Lock()
	cached, ok := d.protocols[deployment]
	d.mu.Unlock()
	if ok {
		return cached, nil
	}
	info, err := d.getDeployment(ctx, deployment)
	if err != nil {
		return "", err
	}
	p := protocolForPublisher(info.ModelPublisher)
	d.setProtocol(deployment, p, info.ModelName)
	return p, nil
}

func (d *AzureFoundryDriver) setProtocol(deployment string, p protocol, source string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.protocols[deployment] = p
	if source != "" {
		d.sources[deployment] = source
	}
}

// CreatePrompt formats the prompt of the model's protocol: Claude Messages,
// Chat Completions messages, or the shared OpenAI driver's Responses input.
func (d *AzureFoundryDriver) CreatePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (any, error) {
	r, err := d.resolve(ctx, options.Model)
	if err != nil {
		return nil, d.formatError(err, options.Model, "createPrompt")
	}
	switch {
	case r.image:
		return d.resource.CreatePrompt(ctx, segments, d.openAIOptions(r, options))
	case r.protocol == protocolMessages:
		claudeOptions := options
		claudeOptions.Model = r.source
		return internalclaude.FormatPrompt(ctx, segments, claudeOptions)
	case r.protocol == protocolResponses:
		return d.inference.CreatePrompt(ctx, segments, d.openAIOptions(r, options))
	default:
		return createChatPrompt(ctx, segments, options)
	}
}

// openAIOptions are the options the shared OpenAI driver receives.
func (d *AzureFoundryDriver) openAIOptions(r route, options ExecutionOptions) ExecutionOptions {
	options.Model = r.openAIModel()
	return options
}

// Execute runs a non-streaming completion on the deployment's protocol.
func (d *AzureFoundryDriver) Execute(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (*ExecutionResponse, error) {
	if options.Model == "" {
		return nil, errors.New("model is required")
	}
	r, err := d.resolve(ctx, options.Model)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	switch {
	case r.image:
		return d.resource.Execute(ctx, segments, d.openAIOptions(r, options))
	case r.protocol == protocolResponses:
		return d.inference.Execute(ctx, segments, d.openAIOptions(r, options))
	}
	return executeWithPrompt(ctx, d, segments, options, d.requestTextCompletion)
}

// Stream runs a streaming completion on the deployment's protocol. Image models
// do not stream: their result arrives as a single chunk.
func (d *AzureFoundryDriver) Stream(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (CompletionStream, error) {
	if options.Model == "" {
		return nil, errors.New("model is required")
	}
	r, err := d.resolve(ctx, options.Model)
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	switch {
	case r.image:
		response, err := d.resource.Execute(ctx, segments, d.openAIOptions(r, options))
		if err != nil {
			return nil, err
		}
		return singleChunkStream(response), nil
	case r.protocol == protocolResponses:
		return d.inference.Stream(ctx, segments, d.openAIOptions(r, options))
	}
	return streamWithPrompt(ctx, d, segments, options, d.requestTextCompletionStream)
}

// requestTextCompletion serves the Claude and Chat Completions prompts.
func (d *AzureFoundryDriver) requestTextCompletion(ctx context.Context, prompt any, options ExecutionOptions) (*Completion, error) {
	r, err := d.resolve(ctx, options.Model)
	if err != nil {
		return nil, err
	}
	switch value := prompt.(type) {
	case internalclaude.Prompt:
		return d.requestClaude(ctx, value, r, options)
	case chatPrompt:
		return d.requestChat(ctx, value, r, options)
	}
	return nil, fmt.Errorf("unexpected microsoft foundry prompt type %T", prompt)
}

// requestTextCompletionStream streams the Claude and Chat Completions prompts.
func (d *AzureFoundryDriver) requestTextCompletionStream(ctx context.Context, prompt any, options ExecutionOptions) (CompletionStream, error) {
	r, err := d.resolve(ctx, options.Model)
	if err != nil {
		return nil, err
	}
	switch value := prompt.(type) {
	case internalclaude.Prompt:
		return d.streamClaude(ctx, value, r, options)
	case chatPrompt:
		return d.streamChat(ctx, value, r, options)
	}
	return nil, fmt.Errorf("unexpected microsoft foundry prompt type %T", prompt)
}

// singleChunkStream presents a finished response as a one-chunk stream.
func singleChunkStream(response *ExecutionResponse) CompletionStream {
	ch := make(chan streamItem, 1)
	ch <- streamItem{Chunk: CompletionChunk{
		Result:       response.Result,
		TokenUsage:   response.TokenUsage,
		FinishReason: response.FinishReason,
		ServiceTier:  response.ServiceTier,
	}}
	close(ch)
	completion := &ExecutionResponse{Prompt: response.Prompt}
	return newChannelStreamWithFinalizer(ch, completion, nil, func(resp *ExecutionResponse) {
		resp.ToolUse = response.ToolUse
		resp.Conversation = response.Conversation
		resp.OriginalResponse = response.OriginalResponse
	})
}

// v1Endpoint joins path elements to an OpenAI v1 base URL and adds the
// configured api-version.
func (d *AzureFoundryDriver) v1Endpoint(base string, elems ...string) (string, error) {
	endpoint, err := joinEndpoint(base, elems...)
	if err != nil || d.options.APIVersion == "" {
		return endpoint, err
	}
	return endpoint + "?api-version=" + url.QueryEscape(d.options.APIVersion), nil
}

// httpClient returns the client for one request, applying per-request timeouts.
func (d *AzureFoundryDriver) httpClient(timeout HTTPTimeoutOptions) *http.Client {
	if !hasHTTPTimeout(timeout) {
		return d.client
	}
	return newHTTPClient(d.client, timeout)
}

// foundryDeployment is a deployment of the Foundry project API.
type foundryDeployment struct {
	Type           string         `json:"type"`
	Name           string         `json:"name"`
	ModelName      string         `json:"modelName"`
	ModelVersion   string         `json:"modelVersion"`
	ModelPublisher string         `json:"modelPublisher"`
	Capabilities   map[string]any `json:"capabilities"`
}

// getDeployment reads one deployment from the project API.
func (d *AzureFoundryDriver) getDeployment(ctx context.Context, name string) (foundryDeployment, error) {
	endpoint, err := joinEndpoint(d.endpoint, "deployments", name)
	if err != nil {
		return foundryDeployment{}, err
	}
	headers, err := d.creds.entra(ctx)
	if err != nil {
		return foundryDeployment{}, err
	}
	var out foundryDeployment
	err = doJSON(ctx, d.client, http.MethodGet, endpoint+"?api-version="+foundryProjectAPIVersion, headers, nil, &out)
	return out, err
}

// listDeployments reads every page of the project's deployments, or only the
// first when firstPage is set.
func (d *AzureFoundryDriver) listDeployments(ctx context.Context, firstPage bool) ([]foundryDeployment, error) {
	endpoint, err := joinEndpoint(d.endpoint, "deployments")
	if err != nil {
		return nil, err
	}
	next := endpoint + "?api-version=" + foundryProjectAPIVersion
	var out []foundryDeployment
	for next != "" {
		headers, err := d.creds.entra(ctx)
		if err != nil {
			return nil, err
		}
		var page struct {
			Value    []foundryDeployment `json:"value"`
			NextLink string              `json:"nextLink"`
		}
		if err := doJSON(ctx, d.client, http.MethodGet, next, headers, nil, &page); err != nil {
			return nil, err
		}
		out = append(out, page.Value...)
		if firstPage {
			break
		}
		next, err = d.sameHostLink(page.NextLink)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// sameHostLink resolves a nextLink against the endpoint and refuses one on
// another host, which would receive the Entra token.
func (d *AzureFoundryDriver) sameHostLink(link string) (string, error) {
	if link == "" {
		return "", nil
	}
	base, err := url.Parse(d.endpoint + "/")
	if err != nil {
		return "", err
	}
	next, err := base.Parse(link)
	if err != nil {
		return "", err
	}
	if !strings.EqualFold(next.Host, base.Host) {
		return "", fmt.Errorf("refusing a deployments nextLink on another host: %s", next.Host)
	}
	if next.Query().Get("api-version") == "" {
		query := next.Query()
		query.Set("api-version", foundryProjectAPIVersion)
		next.RawQuery = query.Encode()
	}
	return next.String(), nil
}

// ListModels returns the project's inference deployments as
// "deployment::source-model" IDs. It needs Microsoft Entra ID: the project API
// does not accept resource keys.
func (d *AzureFoundryDriver) ListModels(ctx context.Context, _ *ModelSearchPayload) ([]AIModel, error) {
	if d.apiKeyOnly() {
		return nil, errors.New("listing Microsoft Foundry deployments requires Microsoft Entra ID: the project API accepts Entra tokens only. Set TokenProvider alongside APIKey, or address deployments by ID")
	}
	deployments, err := d.listDeployments(ctx, false)
	if err != nil {
		return nil, d.formatError(err, "", "listModels")
	}
	models := make([]AIModel, 0, len(deployments))
	for _, deployment := range deployments {
		if deployment.Type != "ModelDeployment" || !isStandardInferenceDeployment(deployment) {
			continue
		}
		modelType := modelTypeText
		if isImageModel(deployment.ModelName) {
			modelType = modelTypeImage
		}
		model := AIModel{
			ID:          deployment.Name + "::" + deployment.ModelName,
			Name:        deployment.Name,
			Description: ptr(deployment.ModelName + " - " + deployment.ModelVersion),
			Provider:    string(ProviderAzureFoundry),
			Type:        ptr(modelType),
		}
		if deployment.ModelVersion != "" {
			model.Version = ptr(deployment.ModelVersion)
		}
		if deployment.ModelPublisher != "" {
			model.Owner = ptr(deployment.ModelPublisher)
		}
		models = append(models, model)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	return models, nil
}

// ValidateConnection checks the credentials. A key is checked against the
// resource's v1 model list; an Entra credential against the project's deployments.
func (d *AzureFoundryDriver) ValidateConnection(ctx context.Context) error {
	if d.options.APIKey != "" {
		endpoint, err := d.v1Endpoint(d.resourceV1, "models")
		if err != nil {
			return err
		}
		headers := map[string]string{headerAPIKey: d.options.APIKey}
		if err := doJSON(ctx, d.client, http.MethodGet, endpoint, headers, nil, nil); err != nil {
			return d.formatError(err, "", "validateConnection")
		}
		if d.apiKeyOnly() {
			return nil
		}
	}
	if _, err := d.listDeployments(ctx, true); err != nil {
		return d.formatError(err, "", "validateConnection")
	}
	return nil
}

// FormatDebugPrompt returns prompt with inline base64 payloads truncated, for
// ExecutionResponse.Prompt. Responses prompts are formatted by the shared
// OpenAI driver when it supports debug prompts.
func (d *AzureFoundryDriver) FormatDebugPrompt(prompt any) any {
	switch value := prompt.(type) {
	case internalclaude.Prompt:
		return claudeDebugPrompt(value)
	case chatPrompt:
		return chatDebugPrompt(value)
	}
	if formatter, ok := any(d.inference).(core.DebugPromptFormatter); ok {
		return formatter.FormatDebugPrompt(prompt)
	}
	return prompt
}

// claudeDebugPrompt copies a Claude prompt with its base64 sources truncated.
func claudeDebugPrompt(prompt internalclaude.Prompt) internalclaude.Prompt {
	out := prompt
	out.System = claudeDebugBlocks(prompt.System)
	out.Messages = make([]internalclaude.Message, len(prompt.Messages))
	for i, message := range prompt.Messages {
		message.Content = claudeDebugBlocks(message.Content)
		out.Messages[i] = message
	}
	return out
}

func claudeDebugBlocks(blocks []internalclaude.Block) []internalclaude.Block {
	if blocks == nil {
		return nil
	}
	out := make([]internalclaude.Block, len(blocks))
	for i, block := range blocks {
		if block.Source != nil && block.Source.Type == "base64" {
			source := *block.Source
			source.Data = truncateBinaryForDebug(source.Data)
			block.Source = &source
		}
		block.Content = claudeDebugBlocks(block.Content)
		out[i] = block
	}
	return out
}
