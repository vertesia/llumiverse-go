package azurefoundry

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

func newAzureOpenAI(t *testing.T, f *fakeAzure, mutate func(*AzureOpenAIOptions)) *AzureOpenAIDriver {
	t.Helper()
	options := AzureOpenAIOptions{Endpoint: f.server.URL, APIKey: "test-key"}
	if mutate != nil {
		mutate(&options)
	}
	driver, err := NewAzureOpenAIDriver(options)
	if err != nil {
		t.Fatal(err)
	}
	return driver
}

func TestAzureOpenAIUsesTheV1PathWithTheAPIKey(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	driver := newAzureOpenAI(t, f, nil)
	resp, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "chat"})
	if err != nil {
		t.Fatal(err)
	}
	request := f.recorded()[0]
	if request.Path != "/openai/v1/responses" || request.Body["model"] != "chat" || len(request.Query) != 0 {
		t.Fatalf("request = %+v", request)
	}
	expectAPIKeyHeaders(t, request)
	if resultText(resp.Result) != "Hello" || driver.Provider() != ProviderAzureOpenAI {
		t.Fatalf("result = %+v", resp.Result)
	}
}

func TestAzureOpenAIAcceptsEndpointsWithTheOpenAIPath(t *testing.T) {
	t.Parallel()
	for _, suffix := range []string{"/", "/openai", "/openai/v1", "/openai/v1/"} {
		if got := azureOpenAIV1BaseURL("https://example.openai.azure.com" + suffix); got != "https://example.openai.azure.com/openai/v1" {
			t.Fatalf("%s: %s", suffix, got)
		}
	}
}

func TestAzureOpenAISendsAnEntraTokenForTheCognitiveServicesScope(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	tokens := &tokenRecorder{}
	driver := newAzureOpenAI(t, f, func(o *AzureOpenAIOptions) {
		o.APIKey = ""
		o.TokenProvider = tokens.provider()
	})
	if _, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "chat"}); err != nil {
		t.Fatal(err)
	}
	request := f.recorded()[0]
	if request.Header.Get("Authorization") != "Bearer test-token" || request.Header.Get("api-key") != "" {
		t.Fatalf("headers = %v", request.Header)
	}
	if scopes := tokens.calls(); len(scopes) != 1 || scopes[0] != "https://cognitiveservices.azure.com/.default" {
		t.Fatalf("scopes = %v", scopes)
	}
}

func TestAzureOpenAISendsTheConfiguredDeploymentOnDeploymentScopedRequests(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	f.respond = func(w http.ResponseWriter, r recordedRequest) bool {
		switch {
		case strings.HasSuffix(r.Path, "/chat/completions"):
			writeRaw(w, `{"model":"gpt-6-luna-2026-09-22","choices":[{"message":{"role":"assistant","content":"H"}}]}`)
		case strings.HasSuffix(r.Path, "/embeddings"):
			writeRaw(w, `{"data":[{"index":0,"embedding":[1]}],"usage":{"prompt_tokens":1,"total_tokens":1}}`)
		default:
			return false
		}
		return true
	}
	driver := newAzureOpenAI(t, f, func(o *AzureOpenAIOptions) { o.Deployment = "production" })
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].ID != "gpt-6-luna-2026-09-22" || models[0].Name != "production" {
		t.Fatalf("models = %+v", models)
	}
	if _, err := driver.GenerateEmbeddings(context.Background(), EmbeddingsOptions{Model: "text-embedding-4", Inputs: []common.EmbeddingInput{{Text: "x"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "other::gpt-6"}); err != nil {
		t.Fatal(err)
	}
	var got [][2]any
	for _, request := range f.recorded() {
		got = append(got, [2]any{request.Path, request.Body["model"]})
	}
	want := [][2]any{
		{"/openai/v1/chat/completions", "production"},
		{"/openai/v1/embeddings", "production"},
		{"/openai/v1/responses", "other"},
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("requests = %v, want %v", got, want)
		}
	}
}

func TestAzureOpenAIImageDeploymentsUseTheImagesAPI(t *testing.T) {
	t.Parallel()
	f := newFakeAzure(t)
	driver := newAzureOpenAI(t, f, func(o *AzureOpenAIOptions) { o.SourceModel = "gpt-image-2" })
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "A garden"}}, ExecutionOptions{Model: "garden"})
	if err != nil {
		t.Fatal(err)
	}
	request := f.recorded()[0]
	if request.Path != "/openai/v1/images/generations" || request.Body["model"] != "garden" || request.Query.Get("api-version") != "preview" {
		t.Fatalf("request = %+v", request)
	}
	if resultText(resp.Result) != "data:image/png;base64,YQ==" {
		t.Fatalf("result = %+v", resp.Result)
	}
}

func TestAzureOpenAIListModelsNeedsADeployment(t *testing.T) {
	t.Parallel()
	driver := newAzureOpenAI(t, newFakeAzure(t), nil)
	if _, err := driver.ListModels(context.Background(), nil); err == nil {
		t.Fatal("expected an error without a deployment")
	}
}

func TestAzureOpenAIKeyRequestsDoNotFollowRedirects(t *testing.T) {
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
	driver := newAzureOpenAI(t, f, nil)
	if _, err := driver.Execute(context.Background(), helloSegments, ExecutionOptions{Model: "chat"}); err == nil {
		t.Fatal("expected the redirect to fail")
	}
	if err := driver.ValidateConnection(context.Background()); err == nil {
		t.Fatal("expected the redirect to fail")
	}
}
