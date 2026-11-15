package vertexai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"regexp"
	"strings"

	openaidriver "github.com/vertesia/llumiverse-go/drivers/openai"
)

// Open models served as Model-as-a-Service (MaaS) on the OpenAI-compatible
// endpoint (endpoints/openapi/chat/completions). Prompt formatting, payloads and
// response parsing come from the shared OpenAI Chat Completions protocol; this
// file only decides which models take this path, the region and API version,
// and the request model name.

// vertexMaaSModel is a catalogue entry. The catalogue keeps exact aliases and
// endpoint workarounds; it is not a routing allowlist (see vertexMaaSRoute).
type vertexMaaSModel struct {
	publisher        string
	model            string
	aliases          []string
	regions          []string
	apiVersion       string
	defaultMaxTokens int
	extraBody        map[string]any
}

// llamaSafetyExtraBody turns off the Llama Guard filter Vertex otherwise applies.
func llamaSafetyExtraBody() map[string]any {
	return map[string]any{"google": map[string]any{"model_safety_settings": map[string]any{"enabled": false, "llama_guard_settings": map[string]any{}}}}
}

var vertexMaaSModels = []vertexMaaSModel{
	{publisher: "meta", model: "llama-4-maverick-17b-128e-instruct-maas", aliases: []string{"llama-4-maverick-17b-128e"}, regions: []string{"us-east5"}, apiVersion: "v1beta1", extraBody: llamaSafetyExtraBody()},
	// Scout returns an empty completion when max_tokens is omitted; caller options still win.
	{publisher: "meta", model: "llama-4-scout-17b-16e-instruct-maas", aliases: []string{"llama-4-scout-17b-16e"}, regions: []string{"us-east5"}, apiVersion: "v1beta1", defaultMaxTokens: 8192, extraBody: llamaSafetyExtraBody()},
	{publisher: "meta", model: "llama-3.3-70b-instruct-maas", aliases: []string{"llama-3.3-70b"}, regions: []string{"us-central1"}, apiVersion: "v1beta1", extraBody: llamaSafetyExtraBody()},
	{publisher: "deepseek-ai", model: "deepseek-v4.1-flash-maas", regions: []string{"global"}},
	{publisher: "deepseek-ai", model: "deepseek-v3.2-maas", regions: []string{"global"}},
	{publisher: "deepseek-ai", model: "deepseek-r1-0528-maas", regions: []string{"us-central1"}},
	{publisher: "qwen", model: "qwen3-next-80b-a3b-instruct-maas", regions: []string{"global"}},
	{publisher: "qwen", model: "qwen3-next-80b-a3b-thinking-maas", regions: []string{"global"}},
	{publisher: "qwen", model: "qwen3-coder-480b-a35b-instruct-maas", regions: []string{"global"}},
	{publisher: "qwen", model: "qwen3-235b-a22b-instruct-2507-maas", regions: []string{"global", "us-south1"}},
	{publisher: "zai-org", model: "glm-5.2-maas", regions: []string{"global"}},
	{publisher: "zai-org", model: "glm-5-maas", regions: []string{"global"}},
	{publisher: "zai-org", model: "glm-4.7-maas", regions: []string{"global"}},
	{publisher: "moonshotai", model: "kimi-k2-thinking-maas", regions: []string{"global"}},
	{publisher: "minimaxai", model: "minimax-m2-maas", regions: []string{"global"}},
	// 120B can spend the provider default budget on reasoning alone; caller options still win.
	{publisher: "openai", model: "gpt-oss-120b-maas", regions: []string{"global", "us-central1"}, defaultMaxTokens: 16384},
	{publisher: "openai", model: "gpt-oss-20b-maas", regions: []string{"us-central1"}},
	{publisher: "google", model: "gemma-4-26b-a4b-it-maas", regions: []string{"global"}},
	{publisher: "xai", model: "grok-4.7", regions: []string{"global", "us"}},
	{publisher: "xai", model: "grok-4.6", regions: []string{"global", "us"}},
	{publisher: "xai", model: "grok-4.3", regions: []string{"global"}},
	{publisher: "xai", model: "grok-4.20-reasoning", regions: []string{"global"}},
	{publisher: "xai", model: "grok-4.20-non-reasoning", regions: []string{"global"}},
}

// vertexMaaSRequest is how one model is reached on the MaaS endpoint.
type vertexMaaSRequest struct {
	modelName        string
	region           string
	apiVersion       string
	defaultMaxTokens int
	extraBody        map[string]any
}

var modelFamilyPattern = regexp.MustCompile(`^[a-z]+`)

// vertexMaaSFamily is the model family used to route models that are not in
// the catalogue: the leading letters of the ID (glm, deepseek, qwen, llama...).
func vertexMaaSFamily(model string) string {
	return modelFamilyPattern.FindString(strings.ToLower(model))
}

// parseVertexModelPath splits "locations/{region}/publishers/{publisher}/models/{model}"
// (or a shorter form) into its parts, dropping any "@version" suffix.
func parseVertexModelPath(model string) (region, publisher, name string) {
	parts := strings.Split(model, "/")
	for i := 0; i+1 < len(parts); i++ {
		switch parts[i] {
		case "locations":
			region = parts[i+1]
		case "publishers":
			publisher = parts[i+1]
		}
	}
	name = parts[len(parts)-1]
	if i := strings.LastIndex(name, "@"); i >= 0 {
		name = name[:i]
	}
	return region, publisher, name
}

// vertexMaaSRoute reports whether a model is an open MaaS model and how to
// reach it. Catalogue entries match by name or alias; a new model from a
// catalogued publisher and family is routed the same way when its ID ends in
// -maas (or the publisher is xAI); any other xAI model goes to the global
// region. A model-path location overrides the catalogue region.
func vertexMaaSRoute(model string) (vertexMaaSRequest, bool) {
	region, publisher, name := parseVertexModelPath(model)
	if publisher == "zaiorg" {
		publisher = "zai-org"
	}
	if publisher == "" {
		return vertexMaaSRequest{}, false
	}
	route, ok := vertexMaaSCatalogRoute(publisher, name)
	if !ok {
		return vertexMaaSRequest{}, false
	}
	if region != "" {
		route.region = region
	}
	return route, true
}

func vertexMaaSCatalogRoute(publisher, name string) (vertexMaaSRequest, bool) {
	for _, entry := range vertexMaaSModels {
		if entry.publisher != publisher {
			continue
		}
		if entry.model == name || contains(entry.aliases, name) {
			return vertexMaaSRequest{
				modelName:        entry.publisher + "/" + entry.model,
				region:           entry.regions[0],
				apiVersion:       entry.apiVersion,
				defaultMaxTokens: entry.defaultMaxTokens,
				extraBody:        entry.extraBody,
			}, true
		}
	}
	if strings.HasSuffix(name, "-maas") || publisher == "xai" {
		family := vertexMaaSFamily(name)
		for _, entry := range vertexMaaSModels {
			if entry.publisher == publisher && vertexMaaSFamily(entry.model) == family {
				return vertexMaaSRequest{
					modelName:  entry.publisher + "/" + name,
					region:     entry.regions[0],
					apiVersion: entry.apiVersion,
					extraBody:  entry.extraBody,
				}, true
			}
		}
	}
	if publisher == "xai" {
		// xAI's publisher API also accepts models outside the catalogue.
		return vertexMaaSRequest{modelName: "xai/" + name, region: "global"}, true
	}
	return vertexMaaSRequest{}, false
}

func contains(values []string, value string) bool {
	for _, v := range values {
		if v == value {
			return true
		}
	}
	return false
}

// vertexMaaSProtocol returns the Chat Completions protocol for a MaaS route.
// MaaS endpoints reject OpenAI strict tool schemas, so tools use plain JSON Schema.
func vertexMaaSProtocol(route vertexMaaSRequest) openaidriver.ChatCompletionsProtocol {
	return openaidriver.ChatCompletionsProtocol{Options: openaidriver.ChatCompletionsProtocolOptions{
		Provider:              ProviderVertexAI,
		ModelName:             route.modelName,
		DefaultMaxTokens:      route.defaultMaxTokens,
		ExtraBody:             route.extraBody,
		CompatibleToolSchemas: true,
	}}
}

// createMaaSPrompt formats a prompt for an open MaaS model.
func (d *VertexAIDriver) createMaaSPrompt(ctx context.Context, route vertexMaaSRequest, segments []PromptSegment, options ExecutionOptions) (any, error) {
	return vertexMaaSProtocol(route).CreatePrompt(ctx, segments, options)
}

// requestMaaS runs a non-streaming request against an open MaaS model.
func (d *VertexAIDriver) requestMaaS(ctx context.Context, route vertexMaaSRequest, prompt any, options ExecutionOptions) (*Completion, error) {
	return vertexMaaSProtocol(route).Execute(ctx, vertexMaaSTransport{driver: d, route: route}, prompt, options)
}

// requestMaaSStream runs a streaming request against an open MaaS model.
func (d *VertexAIDriver) requestMaaSStream(ctx context.Context, route vertexMaaSRequest, prompt any, options ExecutionOptions) (CompletionStream, error) {
	return vertexMaaSProtocol(route).Stream(ctx, vertexMaaSTransport{driver: d, route: route}, prompt, options)
}

// vertexMaaSTransport posts Chat Completions requests to the MaaS endpoint of
// the route's region and API version. The payload keeps its extra_body wrapper,
// which the MaaS endpoint reads.
type vertexMaaSTransport struct {
	driver *VertexAIDriver
	route  vertexMaaSRequest
}

func (t vertexMaaSTransport) endpoint() (string, error) {
	return joinEndpoint(t.driver.maaSBaseURL(t.route.region, t.route.apiVersion), "projects", t.driver.options.Project, "locations", t.route.region, "endpoints", "openapi", "chat", "completions")
}

func (t vertexMaaSTransport) PostChatCompletion(ctx context.Context, payload map[string]any, options ExecutionOptions) (json.RawMessage, error) {
	endpoint, err := t.endpoint()
	if err != nil {
		return nil, err
	}
	headers, err := t.driver.authHeaders(ctx)
	if err != nil {
		return nil, err
	}
	var raw json.RawMessage
	if err := doJSON(ctx, t.driver.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, payload, &raw); err != nil {
		return nil, t.driver.formatError(err, options.Model, "execute")
	}
	return raw, nil
}

func (t vertexMaaSTransport) PostChatCompletionStream(ctx context.Context, payload map[string]any, options ExecutionOptions) (io.ReadCloser, error) {
	endpoint, err := t.endpoint()
	if err != nil {
		return nil, err
	}
	headers, err := t.driver.authHeaders(ctx)
	if err != nil {
		return nil, err
	}
	body, err := postSSE(ctx, t.driver.httpClient(options.HTTPTimeout), endpoint, headers, payload)
	if err != nil {
		return nil, t.driver.formatError(err, options.Model, "stream")
	}
	return body, nil
}

func (t vertexMaaSTransport) FormatChatCompletionsError(err error, model string, operation string) error {
	return t.driver.formatError(err, model, operation)
}

// maaSBaseURL returns the API base for a MaaS region and version: the global
// host for "global", the US multi-region host for "us", and the regional host
// otherwise. A configured BaseURL is kept, with its version swapped when the
// route needs another one.
func (d *VertexAIDriver) maaSBaseURL(region, apiVersion string) string {
	if apiVersion == "" {
		apiVersion = "v1"
	}
	if base := d.options.BaseURL; base != "" {
		if trimmed, ok := strings.CutSuffix(strings.TrimRight(base, "/"), "/v1"); ok {
			return trimmed + "/" + apiVersion
		}
		return base
	}
	switch region {
	case "global":
		return "https://aiplatform.googleapis.com/" + apiVersion
	case "us":
		return "https://aiplatform.us.rep.googleapis.com/" + apiVersion
	}
	return "https://" + region + "-aiplatform.googleapis.com/" + apiVersion
}
