package bedrockmantle

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/vertesia/llumiverse-go/common"
	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
	"github.com/vertesia/llumiverse-go/drivers/openai"
)

var testCredentials = aws.Credentials{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "secret", SessionToken: "session"}

var testTime = time.Date(2026, 11, 15, 12, 0, 0, 0, time.UTC)

type recordedRequest struct {
	Method string
	Path   string
	Header http.Header
	Raw    []byte
	Body   map[string]any
}

type fakeMantle struct {
	t        *testing.T
	server   *httptest.Server
	mu       sync.Mutex
	requests []recordedRequest
}

const responsesBody = `{
	"id":"response-1","object":"response","created_at":1,"model":"openai.gpt-5.5","status":"completed",
	"output":[{"id":"message-1","type":"message","role":"assistant","status":"completed",
		"content":[{"type":"output_text","text":"Hello","annotations":[]}]}],
	"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}
}`

const chatBody = `{
	"id":"chat-1","object":"chat.completion","created":1,"model":"deepseek.v3.2",
	"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"{\"answer\":\"ok\"}"}}],
	"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}
}`

const claudeBody = `{
	"id":"msg-test","type":"message","role":"assistant","model":"anthropic.claude-opus-4-7",
	"content":[{"type":"text","text":"Green"}],"stop_reason":"end_turn",
	"usage":{"input_tokens":2,"output_tokens":1}
}`

func claudeStreamBody() string {
	events := []string{
		`{"type":"message_start","message":{"id":"msg-1","type":"message","role":"assistant","model":"claude","content":[],"usage":{"input_tokens":2,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Green"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
		`{"type":"message_stop"}`,
	}
	var b strings.Builder
	for _, event := range events {
		var envelope struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(event), &envelope)
		fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", envelope.Type, event)
	}
	return b.String()
}

const modelsBody = `{"object":"list","data":[
	{"id":"unverified.model-1","object":"model","created":1,"owned_by":"system"},
	{"id":"openai.gpt-5.6","object":"model","created":1,"owned_by":"system"},
	{"id":"openai.gpt-5.5","object":"model","created":1,"owned_by":"system"},
	{"id":"openai.gpt-oss-120b","object":"model","created":1,"owned_by":"system"},
	{"id":"xai.grok-4.3","object":"model","created":1,"owned_by":"system"},
	{"id":"anthropic.claude-opus-4-7","object":"model","created":1,"owned_by":"system"},
	{"id":"google.gemma-3-4b-it","object":"model","created":1,"owned_by":"system"},
	{"id":"google.gemma-4-31b","object":"model","created":1,"owned_by":"system"},
	{"id":"zai.glm-5","object":"model","created":1,"owned_by":"system"},
	{"id":"deepseek.v3.2","object":"model","created":1,"owned_by":"system"},
	{"id":"amazon.titan-embed-text-v2:0","object":"model","created":1,"owned_by":"system"},
	{"id":"qwen.qwen3-embedding-8b","object":"model","created":1,"owned_by":"system"}
]}`

func newFakeMantle(t *testing.T) *fakeMantle {
	t.Helper()
	f := &fakeMantle{t: t}
	f.server = httptest.NewServer(http.HandlerFunc(f.handle))
	t.Cleanup(f.server.Close)
	return f
}

func (f *fakeMantle) handle(w http.ResponseWriter, r *http.Request) {
	data, _ := io.ReadAll(r.Body)
	request := recordedRequest{Method: r.Method, Path: r.URL.Path, Header: r.Header.Clone(), Raw: data}
	if len(data) > 0 {
		_ = json.Unmarshal(data, &request.Body)
	}
	f.mu.Lock()
	f.requests = append(f.requests, request)
	f.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	switch r.URL.Path {
	case "/anthropic/v1/messages":
		if request.Body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, claudeStreamBody())
			return
		}
		_, _ = io.WriteString(w, claudeBody)
	case "/openai/v1/responses":
		_, _ = io.WriteString(w, responsesBody)
	case "/v1/chat/completions":
		_, _ = io.WriteString(w, chatBody)
	case "/v1/models":
		_, _ = io.WriteString(w, modelsBody)
	case "/redirect/v1/models":
		http.Redirect(w, r, "/v1/models", http.StatusFound)
	default:
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, `{"error":{"message":"not found"}}`)
	}
}

func (f *fakeMantle) last() recordedRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.requests) == 0 {
		f.t.Fatal("no request recorded")
	}
	return f.requests[len(f.requests)-1]
}

func newTestDriver(t *testing.T, f *fakeMantle, apiKey string) *BedrockMantleDriver {
	t.Helper()
	options := BedrockMantleOptions{Region: "us-west-2", Endpoint: f.server.URL, APIKey: apiKey}
	if apiKey == "" {
		options.Config = aws.Config{Credentials: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
			return testCredentials, nil
		})}
	}
	d, err := NewBedrockMantleDriverWithCredentials(options)
	if err != nil {
		t.Fatal(err)
	}
	fixClock(d, testTime)
	return d
}

// fixClock pins the signing time of both transports.
func fixClock(d *BedrockMantleDriver, now time.Time) {
	for _, client := range []*http.Client{d.client, d.messages} {
		client.Transport.(*transport).auth.now = func() time.Time { return now }
	}
}

func userPrompt(text string) []PromptSegment {
	return []PromptSegment{{Role: common.PromptRoleUser, Content: text}}
}

// Mirrors the TS routing test.
func TestModelProtocol(t *testing.T) {
	t.Parallel()
	cases := map[string]Protocol{
		"openai.gpt-oss-120b":           ProtocolChatCompletions,
		"openai.gpt-oss-safeguard-120b": ProtocolChatCompletions,
		"openai.gpt-6.1":                ProtocolResponses,
		"openai.gpt-5.5":                ProtocolResponses,
		"xai.grok-4.3":                  ProtocolResponses,
		"anthropic.claude-opus-4-7":     ProtocolMessages,
		"us.anthropic.claude-opus-4-7":  ProtocolMessages,
		"google.gemma-3-27b-it":         ProtocolChatCompletions,
		"google.gemma-4-31b":            ProtocolResponses,
		"google.gemma-5-70b":            ProtocolResponses,
		"qwen.qwen4-coder":              ProtocolChatCompletions,
		"deepseek.v3.2":                 ProtocolChatCompletions,
		"mistral.magistral-small-2509":  ProtocolChatCompletions,
	}
	for model, want := range cases {
		if got, ok := ModelProtocol(model); !ok || got != want {
			t.Errorf("ModelProtocol(%q) = %q, %v; want %q", model, got, ok, want)
		}
	}
	for _, model := range []string{"openai.o7", "google.gemma-2-9b", "amazon.nova-pro-v1:0", "anthropic.titan", "unverified.model-1"} {
		if got, ok := ModelProtocol(model); ok {
			t.Errorf("ModelProtocol(%q) = %q, want unrouted", model, got)
		}
	}
}

func TestBearerTokenFormat(t *testing.T) {
	t.Parallel()
	a := newAuth("", aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) { return testCredentials, nil }), "us-west-2")
	a.now = func() time.Time { return testTime }
	token, err := a.bearerToken(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(token, tokenPrefix) {
		t.Fatalf("token = %q", token)
	}
	decoded, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(token, tokenPrefix))
	if err != nil {
		t.Fatal(err)
	}
	presigned := string(decoded)
	if !strings.HasPrefix(presigned, tokenHost+"/?") || !strings.HasSuffix(presigned, "&Version=1") {
		t.Fatalf("presigned = %q", presigned)
	}
	query, err := url.ParseQuery(strings.TrimSuffix(strings.TrimPrefix(presigned, tokenHost+"/?"), "&Version=1"))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"Action":               "CallWithBearerToken",
		"X-Amz-Algorithm":      "AWS4-HMAC-SHA256",
		"X-Amz-Credential":     "AKIDEXAMPLE/20261115/us-west-2/bedrock/aws4_request",
		"X-Amz-Date":           "20261115T120000Z",
		"X-Amz-Expires":        "43200",
		"X-Amz-SignedHeaders":  "host",
		"X-Amz-Security-Token": "session",
	}
	for key, value := range want {
		if got := query.Get(key); got != value {
			t.Errorf("%s = %q, want %q", key, got, value)
		}
	}
	if len(query.Get("X-Amz-Signature")) != 64 {
		t.Errorf("X-Amz-Signature = %q", query.Get("X-Amz-Signature"))
	}

	// An API key is sent as is.
	keyed := newAuth("bedrock-api-key-static", nil, "us-west-2")
	if token, _ := keyed.bearerToken(context.Background()); token != "bedrock-api-key-static" {
		t.Fatalf("api key token = %q", token)
	}
}

func TestResponsesUseOpenAIEndpointWithBearerToken(t *testing.T) {
	t.Parallel()
	f := newFakeMantle(t)
	d := newTestDriver(t, f, "")
	resp, err := d.Execute(context.Background(), userPrompt("hello"), ExecutionOptions{Model: "openai.gpt-5.5"})
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Result[0].String(); got != "Hello" {
		t.Fatalf("result = %q", got)
	}
	request := f.last()
	if request.Path != "/openai/v1/responses" || request.Body["model"] != "openai.gpt-5.5" {
		t.Fatalf("request = %s %v", request.Path, request.Body["model"])
	}
	authorization := request.Header.Get("Authorization")
	if !strings.HasPrefix(authorization, "Bearer "+tokenPrefix) {
		t.Fatalf("Authorization = %q", authorization)
	}
	prompt, err := d.CreatePrompt(context.Background(), userPrompt("hello"), ExecutionOptions{Model: "google.gemma-4-31b"})
	if err != nil {
		t.Fatal(err)
	}
	switch prompt.(type) {
	case internalclaude.Prompt, openai.ChatCompletionsPrompt:
		t.Fatalf("gemma 4 prompt = %T, want Responses input", prompt)
	}
}

func TestChatCompletionsUseV1WithResponseFormat(t *testing.T) {
	t.Parallel()
	f := newFakeMantle(t)
	d := newTestDriver(t, f, "")
	schema := map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}, "required": []any{"answer"}}
	resp, err := d.Execute(context.Background(), userPrompt("hello"), ExecutionOptions{Model: "deepseek.v3.2", ResultSchema: schema})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Error != nil || resp.Result[0].Type() != common.ResultTypeJSON {
		t.Fatalf("result = %v, error = %#v", resp.Result, resp.Error)
	}
	request := f.last()
	if request.Path != "/v1/chat/completions" || request.Body["model"] != "deepseek.v3.2" {
		t.Fatalf("request = %s %v", request.Path, request.Body["model"])
	}
	if format, _ := request.Body["response_format"].(map[string]any); format["type"] != "json_schema" {
		t.Fatalf("response_format = %#v", request.Body["response_format"])
	}
	if strings.Contains(string(request.Raw), "<response_schema>") {
		t.Fatalf("schema in the prompt: %s", request.Raw)
	}
	if !strings.HasPrefix(request.Header.Get("Authorization"), "Bearer "+tokenPrefix) {
		t.Fatalf("Authorization = %q", request.Header.Get("Authorization"))
	}

	// Magistral gets the schema as an instruction too.
	if _, err := d.Execute(context.Background(), userPrompt("hello"), ExecutionOptions{Model: "mistral.magistral-small-2509", ResultSchema: schema}); err != nil {
		t.Fatal(err)
	}
	magistral := f.last()
	if _, ok := magistral.Body["response_format"]; !ok || !strings.Contains(string(magistral.Raw), "<response_schema>") {
		t.Fatalf("magistral request = %s", magistral.Raw)
	}
}

// verifySigV4 re-signs a recorded Messages request with the test credentials and compares the signature.
func verifySigV4(t *testing.T, f *fakeMantle, request recordedRequest) {
	t.Helper()
	authorization := request.Header.Get("Authorization")
	if !strings.HasPrefix(authorization, "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261115/us-west-2/bedrock-mantle/aws4_request") {
		t.Fatalf("Authorization = %q", authorization)
	}
	if request.Header.Get("X-Amz-Security-Token") != "session" || request.Header.Get("X-Amz-Date") != "20261115T120000Z" {
		t.Fatalf("signing headers = %v", request.Header)
	}
	replay, err := http.NewRequest(request.Method, f.server.URL+request.Path, bytes.NewReader(request.Raw))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"Content-Type", "Accept", "Anthropic-Version", "Anthropic-Beta"} {
		if value := request.Header.Get(name); value != "" {
			replay.Header.Set(name, value)
		}
	}
	replay.ContentLength = int64(len(request.Raw))
	a := newAuth("", aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) { return testCredentials, nil }), "us-west-2")
	a.now = func() time.Time { return testTime }
	if err := a.signMessages(replay); err != nil {
		t.Fatal(err)
	}
	if got := replay.Header.Get("Authorization"); got != authorization {
		t.Fatalf("signature mismatch:\n got %s\nwant %s", authorization, got)
	}
}

func TestClaudeUsesSignedMessagesAPI(t *testing.T) {
	t.Parallel()
	f := newFakeMantle(t)
	d := newTestDriver(t, f, "")
	schema := map[string]any{"type": "object"}
	resp, err := d.Execute(context.Background(), userPrompt("hello"), ExecutionOptions{Model: "anthropic.claude-opus-4-7", ResultSchema: schema})
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Result[0].String(); got != "Green" {
		t.Fatalf("result = %q", got)
	}
	request := f.last()
	if request.Path != "/anthropic/v1/messages" || request.Body["model"] != "anthropic.claude-opus-4-7" {
		t.Fatalf("request = %s %v", request.Path, request.Body["model"])
	}
	if request.Header.Get("Anthropic-Version") != claudeAPIVersion {
		t.Fatalf("anthropic-version = %q", request.Header.Get("Anthropic-Version"))
	}
	// Mantle is not a native result-schema platform: the schema stays in the prompt.
	if _, ok := request.Body["output_config"]; ok {
		t.Fatalf("output_config sent: %s", request.Raw)
	}
	verifySigV4(t, f, request)

	stream, err := d.Stream(context.Background(), userPrompt("hello"), ExecutionOptions{Model: "anthropic.claude-opus-4-7"})
	if err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		for _, result := range chunk.Result {
			text.WriteString(result.String())
		}
	}
	_ = stream.Close()
	if text.String() != "Green" {
		t.Fatalf("streamed = %q", text.String())
	}
	streamed := f.last()
	if streamed.Body["stream"] != true {
		t.Fatalf("stream body = %s", streamed.Raw)
	}
	verifySigV4(t, f, streamed)
}

func TestAPIKeyIsSentAsBearerEverywhere(t *testing.T) {
	t.Parallel()
	f := newFakeMantle(t)
	d := newTestDriver(t, f, "bedrock-api-key-static")
	for _, model := range []string{"openai.gpt-5.5", "deepseek.v3.2", "anthropic.claude-opus-4-7"} {
		if _, err := d.Execute(context.Background(), userPrompt("hello"), ExecutionOptions{Model: model}); err != nil {
			t.Fatalf("%s: %v", model, err)
		}
		if got := f.last().Header.Get("Authorization"); got != "Bearer bedrock-api-key-static" {
			t.Fatalf("%s: Authorization = %q", model, got)
		}
	}
}

func TestListModelsKeepsRoutableModels(t *testing.T) {
	t.Parallel()
	f := newFakeMantle(t)
	d := newTestDriver(t, f, "")
	models, err := d.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if request := f.last(); request.Path != "/v1/models" || !strings.HasPrefix(request.Header.Get("Authorization"), "Bearer "+tokenPrefix) {
		t.Fatalf("request = %s %q", request.Path, request.Header.Get("Authorization"))
	}
	var ids []string
	names := map[string]string{}
	owners := map[string]string{}
	for _, model := range models {
		ids = append(ids, model.ID)
		names[model.ID] = model.Name
		owners[model.ID] = *model.Owner
		if model.Provider != string(ProviderBedrockMantle) || model.CanStream == nil || !*model.CanStream {
			t.Errorf("model %s = %#v", model.ID, model)
		}
	}
	want := []string{
		"anthropic.claude-opus-4-7", "deepseek.v3.2", "google.gemma-3-4b-it", "google.gemma-4-31b",
		"openai.gpt-5.5", "openai.gpt-5.6", "openai.gpt-oss-120b", "xai.grok-4.3", "zai.glm-5",
	}
	if strings.Join(ids, ",") != strings.Join(want, ",") {
		t.Fatalf("ids = %v", ids)
	}
	for id, name := range map[string]string{
		"anthropic.claude-opus-4-7": "claude-opus-4-7",
		"openai.gpt-oss-120b":       "GPT-oss-120b",
		"xai.grok-4.3":              "Grok 4.3",
		"deepseek.v3.2":             "DeepSeek V3.2",
	} {
		if names[id] != name {
			t.Errorf("name of %s = %q, want %q", id, names[id], name)
		}
	}
	if owners["zai.glm-5"] != "Z.AI" || owners["google.gemma-4-31b"] != "Google" {
		t.Errorf("owners = %v", owners)
	}
}

func TestUnsupportedModelIsNotRetryable(t *testing.T) {
	t.Parallel()
	d := newTestDriver(t, newFakeMantle(t), "")
	_, err := d.Execute(context.Background(), userPrompt("hello"), ExecutionOptions{Model: "openai.o7"})
	var llumiErr *LlumiverseError
	if !errors.As(err, &llumiErr) || llumiErr.Retryable == nil || *llumiErr.Retryable || llumiErr.Context.Provider != ProviderBedrockMantle {
		t.Fatalf("err = %#v", err)
	}
}

func TestTransportRefusesOtherHostsAndRedirects(t *testing.T) {
	t.Parallel()
	f := newFakeMantle(t)
	d := newTestDriver(t, f, "")
	req, _ := http.NewRequest(http.MethodGet, "https://example.test/v1/models", nil)
	if _, err := d.client.Do(req); !errors.Is(err, errForeignHost) {
		t.Fatalf("foreign host err = %v", err)
	}
	req, _ = http.NewRequest(http.MethodGet, f.server.URL+"/redirect/v1/models", nil)
	if _, err := d.client.Do(req); !errors.Is(err, errRedirect) {
		t.Fatalf("redirect err = %v", err)
	}
}

func TestConstructorValidation(t *testing.T) {
	t.Parallel()
	creds := aws.Config{Credentials: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) { return testCredentials, nil })}
	for name, options := range map[string]BedrockMantleOptions{
		"no region":      {Config: creds},
		"bad region":     {Region: "us_west", Config: creds},
		"no credentials": {Region: "us-west-2"},
	} {
		if _, err := NewBedrockMantleDriverWithCredentials(options); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	d, err := NewBedrockMantleDriverWithCredentials(BedrockMantleOptions{Region: "us-west-2", Config: creds})
	if err != nil {
		t.Fatal(err)
	}
	if d.v1 != "https://bedrock-mantle.us-west-2.api.aws/v1" || d.anthropic != "https://bedrock-mantle.us-west-2.api.aws/anthropic" {
		t.Fatalf("endpoints = %s %s", d.v1, d.anthropic)
	}
}
