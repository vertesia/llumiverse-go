package azurefoundry

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
)

func newKeyDriver(t *testing.T, f *fakeAzure, mutate func(*AzureFoundryOptions)) *AzureFoundryDriver {
	t.Helper()
	options := AzureFoundryOptions{Endpoint: f.project(), APIKey: "test-key"}
	if mutate != nil {
		mutate(&options)
	}
	driver, err := NewAzureFoundryDriver(options)
	if err != nil {
		t.Fatal(err)
	}
	return driver
}

func newEntraDriver(t *testing.T, f *fakeAzure, tokens *tokenRecorder, mutate func(*AzureFoundryOptions)) *AzureFoundryDriver {
	t.Helper()
	options := AzureFoundryOptions{Endpoint: f.project(), TokenProvider: tokens.provider()}
	if mutate != nil {
		mutate(&options)
	}
	driver, err := NewAzureFoundryDriver(options)
	if err != nil {
		t.Fatal(err)
	}
	return driver
}

func TestEntraScopeFollowsTheEndpointCloud(t *testing.T) {
	t.Parallel()
	cases := map[string][2]string{
		"https://acme.services.ai.azure.com/api/projects/p": {"https://ai.azure.com/.default", "https://cognitiveservices.azure.com/.default"},
		"https://acme.services.ai.azure.us/api/projects/p":  {"https://ai.azure.us/.default", "https://cognitiveservices.azure.us/.default"},
		"https://acme.openai.azure.us":                      {"https://ai.azure.us/.default", "https://cognitiveservices.azure.us/.default"},
		"not a url":                                         {"https://ai.azure.com/.default", "https://cognitiveservices.azure.com/.default"},
		"":                                                  {"https://ai.azure.com/.default", "https://cognitiveservices.azure.com/.default"},
	}
	for endpoint, want := range cases {
		if got := FoundryScope(endpoint); got != want[0] {
			t.Errorf("FoundryScope(%q) = %q, want %q", endpoint, got, want[0])
		}
		if got := CognitiveServicesScope(endpoint); got != want[1] {
			t.Errorf("CognitiveServicesScope(%q) = %q, want %q", endpoint, got, want[1])
		}
	}
}

func TestNewAzureFoundryDriverRequiresEndpointAndCredential(t *testing.T) {
	t.Parallel()
	if _, err := NewAzureFoundryDriver(AzureFoundryOptions{APIKey: "k"}); err == nil {
		t.Fatal("expected an error without endpoint")
	}
	if _, err := NewAzureFoundryDriver(AzureFoundryOptions{Endpoint: "https://x.services.ai.azure.com"}); err == nil {
		t.Fatal("expected an error without credentials")
	}
}

func TestAPIKeySendsResponsesToTheResourceEndpoint(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	driver := newKeyDriver(t, f, nil)
	resp, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "chat::gpt-6-luna"})
	if err != nil {
		t.Fatal(err)
	}
	requests := f.recorded()
	if len(requests) != 1 || requests[0].Path != "/openai/v1/responses" {
		t.Fatalf("requests = %+v", requests)
	}
	if requests[0].Body["model"] != "chat" {
		t.Fatalf("model = %v", requests[0].Body["model"])
	}
	expectAPIKeyHeaders(t, requests[0])
	if got := resultText(resp.Result); got != "Hello" {
		t.Fatalf("result = %q", got)
	}
}

func TestAPIKeySendsPartnerModelsToChatCompletions(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	driver := newKeyDriver(t, f, nil)
	if _, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "deepseek::DeepSeek-V4-Pro"}); err != nil {
		t.Fatal(err)
	}
	requests := f.recorded()
	if len(requests) != 1 || requests[0].Path != "/openai/v1/chat/completions" {
		t.Fatalf("requests = %+v", requests)
	}
	body := requests[0].Body
	if body["model"] != "deepseek" || body["stream"] != false || body["n"] != float64(1) {
		t.Fatalf("body = %v", body)
	}
	for _, unset := range []string{"temperature", "top_p", "max_tokens", "stream_options", "tools", "response_format"} {
		if _, ok := body[unset]; ok {
			t.Fatalf("payload sends %s the caller did not set: %v", unset, body)
		}
	}
	expectAPIKeyHeaders(t, requests[0])
}

func TestAPIKeySendsClaudeWithTheXAPIKeyHeader(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	driver := newKeyDriver(t, f, nil)
	resp, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "claude::claude-sonnet-5-5"})
	if err != nil {
		t.Fatal(err)
	}
	requests := f.recorded()
	if len(requests) != 1 || requests[0].Path != "/anthropic/v1/messages" {
		t.Fatalf("requests = %+v", requests)
	}
	if requests[0].Header.Get("x-api-key") != "test-key" || requests[0].Header.Get("Authorization") != "" {
		t.Fatalf("headers = %v", requests[0].Header)
	}
	if requests[0].Header.Get("anthropic-version") != "2023-06-01" {
		t.Fatalf("anthropic-version = %q", requests[0].Header.Get("anthropic-version"))
	}
	if requests[0].Body["model"] != "claude" {
		t.Fatalf("model = %v", requests[0].Body["model"])
	}
	if got := resultText(resp.Result); got != "Green" {
		t.Fatalf("result = %q", got)
	}
	if _, ok := resp.Prompt.(internalclaude.Prompt); !ok {
		t.Fatalf("prompt = %T", resp.Prompt)
	}
}

func TestAPIKeyRoutesBareDeploymentsByName(t *testing.T) {
	t.Parallel()
	cases := map[string]string{
		"gpt-6-luna":    "/openai/v1/responses",
		"gpt-5.3-codex": "/openai/v1/responses",
		"codex-mini":    "/openai/v1/responses",
		"gpt-oss-120b":  "/openai/v1/chat/completions",
		"grok-4.7":      "/openai/v1/chat/completions",
		"my-llama":      "/openai/v1/chat/completions",
	}
	for model, path := range cases {
		f := newFakeAzure(t)
		driver := newKeyDriver(t, f, nil)
		if _, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: model}); err != nil {
			t.Fatalf("%s: %v", model, err)
		}
		requests := f.recorded()
		if len(requests) != 1 || requests[0].Path != path {
			t.Fatalf("%s: requests = %+v, want %s", model, requests, path)
		}
		if requests[0].Body["model"] != model {
			t.Fatalf("%s: model = %v", model, requests[0].Body["model"])
		}
	}
}

func TestAPIKeyRoutesABareDeploymentBySourceModel(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	driver := newKeyDriver(t, f, func(o *AzureFoundryOptions) { o.SourceModel = "gpt-6-luna" })
	if _, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "production"}); err != nil {
		t.Fatal(err)
	}
	requests := f.recorded()
	if requests[0].Path != "/openai/v1/responses" || requests[0].Body["model"] != "production" {
		t.Fatalf("request = %+v", requests[0])
	}
}

func TestAPIKeyValidatesAgainstTheResourceModelList(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	driver := newKeyDriver(t, f, nil)
	if err := driver.ValidateConnection(context.Background()); err != nil {
		t.Fatal(err)
	}
	requests := f.recorded()
	if len(requests) != 1 || requests[0].Method != http.MethodGet || requests[0].Path != "/openai/v1/models" {
		t.Fatalf("requests = %+v", requests)
	}
	expectAPIKeyHeaders(t, requests[0])
	if _, err := driver.ListModels(context.Background(), nil); err == nil || !strings.Contains(err.Error(), "requires Microsoft Entra ID") {
		t.Fatalf("ListModels error = %v", err)
	}
}

func TestAPIKeyWithEntraUsesTheProjectAPIForDiscovery(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	f.deployments["chat"] = foundryDeployment{Type: "ModelDeployment", Name: "chat", ModelPublisher: "OpenAI", ModelName: "gpt-6-luna"}
	tokens := &tokenRecorder{}
	driver := newKeyDriver(t, f, func(o *AzureFoundryOptions) { o.TokenProvider = tokens.provider() })
	for range 2 {
		if _, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "chat"}); err != nil {
			t.Fatal(err)
		}
	}
	if f.deploymentLookups() != 1 {
		t.Fatalf("deployment lookups = %d, want 1 (memoized)", f.deploymentLookups())
	}
	lookup := f.recorded()[0]
	if lookup.Path != "/api/projects/project/deployments/chat" || lookup.Query.Get("api-version") != "v1" {
		t.Fatalf("lookup = %+v", lookup)
	}
	if lookup.Header.Get("Authorization") != "Bearer test-token" || lookup.Header.Get("api-key") != "" {
		t.Fatalf("lookup headers = %v", lookup.Header)
	}
	inference := f.inference()
	if inference[0].Path != "/openai/v1/responses" {
		t.Fatalf("inference = %+v", inference[0])
	}
	expectAPIKeyHeaders(t, inference[0])
	if scopes := tokens.calls(); len(scopes) != 1 || scopes[0] != "https://ai.azure.com/.default" {
		t.Fatalf("token scopes = %v", scopes)
	}
}

func TestAPIKeyRequestsDoNotFollowRedirects(t *testing.T) {
	t.Parallel()
	elsewhere := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("the redirect target received a request")
	}))
	defer elsewhere.Close()
	f := newFakeAzure(t)
	f.respond = func(w http.ResponseWriter, r recordedRequest) bool {
		http.Redirect(w, &http.Request{Method: r.Method}, elsewhere.URL+r.Path, http.StatusTemporaryRedirect)
		return true
	}
	driver := newKeyDriver(t, f, nil)
	for _, model := range []string{"chat::gpt-6-luna", "deepseek::DeepSeek-V4-Pro", "claude::claude-sonnet-5-5"} {
		if _, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: model}); err == nil {
			t.Fatalf("%s: expected the redirect to fail", model)
		}
	}
	if err := driver.ValidateConnection(context.Background()); err == nil {
		t.Fatal("expected the redirect to fail")
	}
}

func TestEntraTokensAreNotSentToARedirectTarget(t *testing.T) {
	t.Parallel()
	var leaked string
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		leaked = r.Header.Get("Authorization")
		writeRaw(w, responsesBody)
	}))
	defer elsewhere.Close()
	f := newFakeAzure(t)
	f.respond = func(w http.ResponseWriter, r recordedRequest) bool {
		http.Redirect(w, &http.Request{Method: r.Method}, elsewhere.URL+r.Path, http.StatusTemporaryRedirect)
		return true
	}
	driver := newEntraDriver(t, f, &tokenRecorder{}, nil)
	_, _ = driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "chat::gpt-6-luna"})
	if leaked != "" {
		t.Fatalf("redirect target received Authorization %q", leaked)
	}
}

func TestProjectURLWithATrailingSlash(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	driver := newKeyDriver(t, f, func(o *AzureFoundryOptions) { o.Endpoint += "/" })
	if _, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "chat::gpt-6-luna"}); err != nil {
		t.Fatal(err)
	}
	if err := driver.ValidateConnection(context.Background()); err != nil {
		t.Fatal(err)
	}
	var paths []string
	for _, request := range f.recorded() {
		paths = append(paths, request.Path)
	}
	if want := []string{"/openai/v1/responses", "/openai/v1/models"}; !reflect.DeepEqual(paths, want) {
		t.Fatalf("paths = %v, want %v", paths, want)
	}
}

func TestACompositeIDOverridesTheProtocolGuessedFromTheBareName(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	driver := newKeyDriver(t, f, nil)
	for _, model := range []string{"production", "production::gpt-6-luna"} {
		if _, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: model}); err != nil {
			t.Fatal(err)
		}
	}
	requests := f.recorded()
	if requests[0].Path != "/openai/v1/chat/completions" || requests[1].Path != "/openai/v1/responses" {
		t.Fatalf("paths = %s, %s", requests[0].Path, requests[1].Path)
	}
}

func TestEntraChatUsesTheProjectEndpointAndAPIVersion(t *testing.T) {
	t.Parallel()
	for _, apiVersion := range []string{"", "explicit-version"} {
		f := newFakeAzure(t)
		f.deployments["chat"] = foundryDeployment{Type: "ModelDeployment", Name: "chat", ModelPublisher: "Meta", ModelName: "Llama-5"}
		tokens := &tokenRecorder{}
		driver := newEntraDriver(t, f, tokens, func(o *AzureFoundryOptions) { o.APIVersion = apiVersion })
		resp, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "chat::Llama-5", IncludeOriginalResponse: true})
		if err != nil {
			t.Fatal(err)
		}
		request := f.inference()[0]
		if request.Path != "/api/projects/project/openai/v1/chat/completions" || request.Query.Get("api-version") != apiVersion {
			t.Fatalf("request = %s?%s", request.Path, request.Query.Encode())
		}
		if request.Body["model"] != "chat" || request.Header.Get("Authorization") != "Bearer test-token" {
			t.Fatalf("request = %+v", request)
		}
		if got := resultText(resp.Result); got != "Hello" {
			t.Fatalf("result = %q", got)
		}
		usage := resp.TokenUsage
		if common.Count(usage.Prompt) != 2 || common.Count(usage.PromptNew) != 2 || common.Count(usage.Result) != 1 || common.Count(usage.Total) != 3 {
			t.Fatalf("usage = %+v", usage)
		}
		if original, ok := resp.OriginalResponse.(map[string]any); !ok || original["id"] != "chat-1" {
			t.Fatalf("original = %v", resp.OriginalResponse)
		}
		if resp.FinishReason != "stop" {
			t.Fatalf("finish reason = %q", resp.FinishReason)
		}
		for _, scope := range tokens.calls() {
			if scope != "https://ai.azure.com/.default" {
				t.Fatalf("scope = %q", scope)
			}
		}
	}
}

func TestEntraResponsesUseTheProjectEndpoint(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	f.deployments["gpt-deployment"] = foundryDeployment{Type: "ModelDeployment", Name: "gpt-deployment", ModelPublisher: "OpenAI", ModelName: "gpt-5"}
	driver := newEntraDriver(t, f, &tokenRecorder{}, nil)
	options := ExecutionOptions{
		Model:        "gpt-deployment::gpt-5",
		ModelOptions: map[string]any{"effort": "high", "temperature": 0.3, "top_p": 0.7},
		Tools:        []ToolDefinition{{Name: "lookup", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}}},
	}
	for range 2 {
		if _, err := driver.Execute(context.Background(), helloSegments, options); err != nil {
			t.Fatal(err)
		}
	}
	if f.deploymentLookups() != 1 {
		t.Fatalf("deployment lookups = %d", f.deploymentLookups())
	}
	request := f.inference()[0]
	if request.Path != "/api/projects/project/openai/v1/responses" || request.Header.Get("Authorization") != "Bearer test-token" {
		t.Fatalf("request = %+v", request)
	}
	if request.Body["model"] != "gpt-deployment" {
		t.Fatalf("model = %v", request.Body["model"])
	}
	// The shared driver sees the source model, a reasoning model: no sampling knobs.
	if _, ok := request.Body["temperature"]; ok {
		t.Fatalf("temperature sent to a reasoning model: %v", request.Body)
	}
	if reasoning, _ := request.Body["reasoning"].(map[string]any); reasoning["effort"] != "high" {
		t.Fatalf("reasoning = %v", request.Body["reasoning"])
	}
}

func TestFailedDeploymentLookupsAreNotCached(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	driver := newEntraDriver(t, f, &tokenRecorder{}, nil)
	_, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "deployment::gpt-5"})
	var llumiErr *LlumiverseError
	if !errors.As(err, &llumiErr) || llumiErr.Code != http.StatusNotFound || llumiErr.Retryable == nil || *llumiErr.Retryable {
		t.Fatalf("error = %#v", err)
	}
	f.deployments["deployment"] = foundryDeployment{Type: "ModelDeployment", Name: "deployment", ModelPublisher: "OpenAI", ModelName: "gpt-5"}
	for range 2 {
		if _, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "deployment::gpt-5"}); err != nil {
			t.Fatal(err)
		}
	}
	if f.deploymentLookups() != 2 {
		t.Fatalf("deployment lookups = %d, want 2", f.deploymentLookups())
	}
}

func TestListModelsFiltersDedicatedEndpointDeployments(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	add := func(name, model, publisher string, capabilities map[string]any) {
		f.deployments[name] = foundryDeployment{Type: "ModelDeployment", Name: name, ModelName: model, ModelVersion: "1", ModelPublisher: publisher, Capabilities: capabilities}
	}
	add("future-chat", "Future-Chat-7", "Meta", map[string]any{})
	add("explicit-chat", "Llama-5", "Meta", map[string]any{"chat_completion": "true"})
	add("not-chat", "Llama-4", "Meta", map[string]any{"chat_completion": "false"})
	add("embedding", "text-embedding-4", "Meta", map[string]any{"chat_completion": "true"})
	add("flux", "FLUX.1-Kontext-pro", "Meta", map[string]any{})
	add("future-flux", "FLUX-9-pro", "Meta", map[string]any{"chat_completion": "true"})
	add("anthropic", "claude-future", "Anthropic", map[string]any{"chat_completion": "true"})
	add("speech", "gpt-4o-mini-tts", "Meta", map[string]any{"chat_completion": "true"})
	add("image", "gpt-image-2.5-flare", "OpenAI", map[string]any{"chat_completion": "false"})
	f.deployments["connection"] = foundryDeployment{Type: "Connection", Name: "connection"}
	driver := newEntraDriver(t, f, &tokenRecorder{}, nil)
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	want := []string{"anthropic::claude-future", "explicit-chat::Llama-5", "future-chat::Future-Chat-7", "image::gpt-image-2.5-flare"}
	if !reflect.DeepEqual(ids, want) {
		t.Fatalf("ids = %v, want %v", ids, want)
	}
	image := models[3]
	if image.Type == nil || *image.Type != modelTypeImage || *image.Owner != "OpenAI" || image.Name != "image" || *image.Description != "gpt-image-2.5-flare - 1" {
		t.Fatalf("image model = %+v", image)
	}
	// The capability catalogue describes each source model: Claude takes images and calls tools, an image model
	// answers with images, and Foundry keeps tools off for a partner model without a documented rule.
	claude, future := models[0], models[2]
	if strings.Join(*claude.InputModalities, ",") != "text,image" || strings.Join(*claude.OutputModalities, ",") != "text" || !*claude.ToolSupport {
		t.Fatalf("claude model = %+v", claude)
	}
	if strings.Join(*image.OutputModalities, ",") != "image" || *image.ToolSupport {
		t.Fatalf("image model = %+v", image)
	}
	if strings.Join(*future.InputModalities, ",") != "text" || *future.ToolSupport {
		t.Fatalf("future model = %+v", future)
	}
}

func TestListModelsFollowsNextLinkOnTheSameHostOnly(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	f.respond = func(w http.ResponseWriter, r recordedRequest) bool {
		if !strings.HasSuffix(r.Path, "/deployments") {
			return false
		}
		if r.Query.Get("page") == "" {
			writeJSON(w, map[string]any{
				"value":    []foundryDeployment{{Type: "ModelDeployment", Name: "a", ModelName: "Llama-5", ModelPublisher: "Meta"}},
				"nextLink": f.project() + "/deployments?api-version=v1&page=2",
			})
			return true
		}
		writeJSON(w, map[string]any{"value": []foundryDeployment{{Type: "ModelDeployment", Name: "b", ModelName: "Llama-5", ModelPublisher: "Meta"}}})
		return true
	}
	driver := newEntraDriver(t, f, &tokenRecorder{}, nil)
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 {
		t.Fatalf("models = %+v", models)
	}
	if _, err := driver.sameHostLink("https://elsewhere.example/deployments"); err == nil {
		t.Fatal("expected a nextLink on another host to be refused")
	}
}

func TestOpaqueImageDeploymentsUseTheImagesAPI(t *testing.T) {
	t.Parallel()
	for _, source := range []string{"", "gpt-image-2"} {
		f := newFakeAzure(t)
		f.deployments["gpt-replacement"] = foundryDeployment{Type: "ModelDeployment", Name: "gpt-replacement", ModelPublisher: "OpenAI", ModelName: "gpt-image-2"}
		driver := newEntraDriver(t, f, &tokenRecorder{}, func(o *AzureFoundryOptions) { o.SourceModel = source })
		resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "A garden"}}, ExecutionOptions{Model: "gpt-replacement"})
		if err != nil {
			t.Fatal(err)
		}
		request := f.inference()[0]
		if request.Path != "/openai/v1/images/generations" || request.Body["model"] != "gpt-replacement" {
			t.Fatalf("source %q: request = %+v", source, request)
		}
		if request.Header.Get("Authorization") != "Bearer test-token" {
			t.Fatalf("authorization = %q", request.Header.Get("Authorization"))
		}
		wantLookups := 1
		if source != "" {
			wantLookups = 0
		}
		if f.deploymentLookups() != wantLookups {
			t.Fatalf("source %q: lookups = %d", source, f.deploymentLookups())
		}
		if got := resultText(resp.Result); got != "data:image/png;base64,YQ==" {
			t.Fatalf("result = %q", got)
		}
		stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "A garden"}}, ExecutionOptions{Model: "gpt-replacement"})
		if err != nil {
			t.Fatal(err)
		}
		if chunks := drain(t, stream); len(chunks) != 1 || resultText(stream.Completion().Result) != "data:image/png;base64,YQ==" {
			t.Fatalf("stream chunks = %+v", chunks)
		}
	}
}

func TestEmbeddingsPreserveOrderAndUseDeploymentIDs(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	f.respond = func(w http.ResponseWriter, r recordedRequest) bool {
		if !strings.HasSuffix(r.Path, "/embeddings") {
			return false
		}
		if inputs, _ := r.Body["input"].([]any); len(inputs) == 2 {
			writeRaw(w, `{"object":"list","data":[{"index":1,"embedding":[2]},{"index":0,"embedding":[1]}],"usage":{"prompt_tokens":4,"total_tokens":4}}`)
			return true
		}
		writeRaw(w, `{"object":"list","data":[{"index":0,"embedding":[3]}],"usage":{"prompt_tokens":1,"total_tokens":1}}`)
		return true
	}
	driver := newEntraDriver(t, f, &tokenRecorder{}, nil)
	result, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{
		Model: "embedding::cohere-embed-v4",
		Inputs: []common.EmbeddingInput{
			{Type: embeddingInputText, Text: "first"},
			{Type: embeddingInputImage, Source: common.BytesDataSource{FileName: "reference.png", MIME: "image/png", Data: []byte("a")}},
			{Type: embeddingInputText, Text: "second"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	requests := f.recorded()
	wantBodies := []map[string]any{
		{"input": []any{"first", "second"}, "model": "embedding", "encoding_format": "float"},
		{"input": []any{"YQ=="}, "model": "embedding", "encoding_format": "float"},
	}
	for i, request := range requests {
		if request.Path != "/openai/v1/embeddings" || !reflect.DeepEqual(request.Body, wantBodies[i]) {
			t.Fatalf("request %d = %s %v", i, request.Path, request.Body)
		}
	}
	var values [][]float64
	for _, item := range result.Results {
		values = append(values, item.Outputs[0].Values)
	}
	if !reflect.DeepEqual(values, [][]float64{{1}, {3}, {2}}) {
		t.Fatalf("values = %v", values)
	}
	if *result.Results[1].Outputs[0].Modality != embeddingOutputModalityImage || common.Count(result.Usage.InputTokens) != 5 {
		t.Fatalf("result = %+v", result)
	}
}

func TestEmbeddingsRequireAModel(t *testing.T) {
	t.Parallel()
	driver := newKeyDriver(t, newFakeAzure(t), nil)
	if _, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{Inputs: []common.EmbeddingInput{{Text: "x"}}}); err == nil {
		t.Fatal("expected an error without a model")
	}
}

func TestErrorsKeepTheHTTPStatusAndRetryability(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	f.respond = func(w http.ResponseWriter, r recordedRequest) bool {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"code":"ServiceUnavailable","message":"Temporarily unavailable"}}`)
		return true
	}
	driver := newKeyDriver(t, f, nil)
	for _, model := range []string{"llama-deployment::llama", "claude::claude-opus-5-5"} {
		_, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: model})
		var llumiErr *LlumiverseError
		if !errors.As(err, &llumiErr) {
			t.Fatalf("%s: error = %#v", model, err)
		}
		if llumiErr.Code != http.StatusServiceUnavailable || llumiErr.Retryable == nil || !*llumiErr.Retryable || llumiErr.Context.Provider != ProviderAzureFoundry || llumiErr.Context.Model != model {
			t.Fatalf("%s: error = %+v", model, llumiErr)
		}
		if !strings.Contains(llumiErr.Message, "Temporarily unavailable") {
			t.Fatalf("message = %q", llumiErr.Message)
		}
	}
	_, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{Model: "embedding::model", Inputs: []common.EmbeddingInput{{Text: "Hello"}}})
	var llumiErr *LlumiverseError
	if !errors.As(err, &llumiErr) || llumiErr.Code != http.StatusServiceUnavailable || !*llumiErr.Retryable || llumiErr.Context.Operation != "embeddings" {
		t.Fatalf("embeddings error = %#v", err)
	}
}

func TestClaudeOnFoundryStreamsWithEntraAndTheSourceModel(t *testing.T) {
	t.Parallel()
	for _, source := range []string{"", "claude-opus-5-5"} {
		f := newFakeAzure(t)
		f.deployments["deployment"] = foundryDeployment{Type: "ModelDeployment", Name: "deployment", ModelPublisher: "Anthropic", ModelName: "claude-opus-5-5"}
		tokens := &tokenRecorder{}
		driver := newEntraDriver(t, f, tokens, func(o *AzureFoundryOptions) { o.SourceModel = source })
		segments := []PromptSegment{{Role: PromptRoleUser, Content: "What color is grass?", Files: []common.DataSource{
			common.BytesDataSource{FileName: "a.png", MIME: "image/png", Data: []byte(strings.Repeat("x", 300))},
		}}}
		stream, err := driver.Stream(context.Background(), segments, ExecutionOptions{Model: "deployment", ModelOptions: map[string]any{"temperature": 0.5}})
		if err != nil {
			t.Fatal(err)
		}
		drain(t, stream)
		completion := stream.Completion()
		if got := resultText(completion.Result); got != "Green" {
			t.Fatalf("result = %q", got)
		}
		request := f.inference()[0]
		if request.Path != "/anthropic/v1/messages" || request.Body["model"] != "deployment" || request.Body["stream"] != true {
			t.Fatalf("request = %+v", request)
		}
		if request.Header.Get("Authorization") != "Bearer test-token" || request.Header.Get("x-api-key") != "" {
			t.Fatalf("headers = %v", request.Header)
		}
		wantLookups := 1
		if source != "" {
			wantLookups = 0
		}
		if f.deploymentLookups() != wantLookups {
			t.Fatalf("lookups = %d", f.deploymentLookups())
		}
		prompt, ok := completion.Prompt.(internalclaude.Prompt)
		if !ok {
			t.Fatalf("prompt = %T", completion.Prompt)
		}
		data := prompt.Messages[0].Content[1].Source.Data
		if !strings.Contains(data, "truncated") || len(data) > 60 {
			t.Fatalf("debug prompt keeps the image payload: %q", data)
		}
		conversation, ok := completion.Conversation.(internalclaude.Prompt)
		if !ok || conversation.Messages[len(conversation.Messages)-1].Role != "assistant" {
			t.Fatalf("conversation = %#v", completion.Conversation)
		}
	}
}

func TestAnOpenAISourceOverridesAClaudeNamedDeployment(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	f.deployments["claude-replacement"] = foundryDeployment{Type: "ModelDeployment", Name: "claude-replacement", ModelPublisher: "OpenAI", ModelName: "gpt-6"}
	driver := newEntraDriver(t, f, &tokenRecorder{}, func(o *AzureFoundryOptions) { o.SourceModel = "gpt-6" })
	if _, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "claude-replacement"}); err != nil {
		t.Fatal(err)
	}
	if path := f.inference()[0].Path; path != "/api/projects/project/openai/v1/responses" {
		t.Fatalf("path = %s", path)
	}
}

func TestCanceledContextStopsBeforeDiscovery(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	driver := newEntraDriver(t, f, &tokenRecorder{}, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := driver.Execute(ctx, helloSegments, ExecutionOptions{Model: "deployment"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v", err)
	}
	if len(f.recorded()) != 0 {
		t.Fatalf("requests = %+v", f.recorded())
	}
}

func TestResourceInferenceURL(t *testing.T) {
	t.Parallel()
	cases := map[string]string{
		"https://r.services.ai.azure.com/api/projects/p/openai/v1":  "https://r.services.ai.azure.com/openai/v1",
		"https://r.services.ai.azure.com/api/projects/p//openai/v1": "https://r.services.ai.azure.com/openai/v1",
		"https://gateway.example.test/custom/openai/v1":             "https://gateway.example.test/custom/openai/v1",
		"https://r.services.ai.azure.com/openai/v1":                 "https://r.services.ai.azure.com/openai/v1",
	}
	for in, want := range cases {
		got, err := resourceInferenceURL(in)
		if err != nil || got != want {
			t.Errorf("resourceInferenceURL(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
}
