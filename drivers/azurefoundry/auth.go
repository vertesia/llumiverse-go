package azurefoundry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// TokenProvider returns a Microsoft Entra ID access token for scope. The drivers
// call it before each Entra-authenticated request, so it should cache tokens
// until they expire, as the Azure SDK credentials do. With azidentity:
//
//	func(ctx context.Context, scope string) (string, error) {
//		token, err := credential.GetToken(ctx, policy.TokenRequestOptions{Scopes: []string{scope}})
//		return token.Token, err
//	}
type TokenProvider func(ctx context.Context, scope string) (string, error)

// StaticToken returns a TokenProvider that always returns token, whatever the
// scope: for tests and short-lived scripts that already hold a token.
func StaticToken(token string) TokenProvider {
	return func(context.Context, string) (string, error) { return token, nil }
}

// Entra scopes of the Azure AI services in the public cloud and in Azure Government.
const (
	foundryScopePublic             = "https://ai.azure.com/.default"
	foundryScopeGovernment         = "https://ai.azure.us/.default"
	cognitiveServicesScopePublic   = "https://cognitiveservices.azure.com/.default"
	cognitiveServicesScopeGovernmt = "https://cognitiveservices.azure.us/.default"
)

// isAzureGovernmentEndpoint reports whether endpoint is served by Azure
// Government (`*.azure.us`), which only accepts tokens issued for the matching
// `.us` resource. A missing or unparsable endpoint counts as the public cloud.
func isAzureGovernmentEndpoint(endpoint string) bool {
	if endpoint == "" {
		return false
	}
	u, err := url.Parse(endpoint)
	if err != nil {
		return false
	}
	return strings.HasSuffix(strings.ToLower(u.Hostname()), ".azure.us")
}

// FoundryScope returns the Entra scope of a Microsoft Foundry endpoint's cloud:
// https://ai.azure.us/.default for Azure Government, else https://ai.azure.com/.default.
func FoundryScope(endpoint string) string {
	if isAzureGovernmentEndpoint(endpoint) {
		return foundryScopeGovernment
	}
	return foundryScopePublic
}

// CognitiveServicesScope returns the Entra scope of an Azure OpenAI endpoint's
// cloud: https://cognitiveservices.azure.us/.default for Azure Government, else
// https://cognitiveservices.azure.com/.default.
func CognitiveServicesScope(endpoint string) string {
	if isAzureGovernmentEndpoint(endpoint) {
		return cognitiveServicesScopeGovernmt
	}
	return cognitiveServicesScopePublic
}

// Headers that carry an Azure resource key: api-key on the OpenAI APIs, x-api-key on Claude Messages.
const (
	headerAPIKey       = "api-key"
	headerAnthropicKey = "x-api-key"
)

// errRedirectWithKey is returned instead of following a redirect on a request
// that carries a resource key.
var errRedirectWithKey = errors.New("refusing to follow a redirect on a request that carries an Azure resource key")

// credentials authenticates requests with a resource key or an Entra token.
type credentials struct {
	apiKey string
	tokens TokenProvider
	scope  string
}

// bearer returns the Authorization header value of an Entra token.
func (c credentials) bearer(ctx context.Context) (string, error) {
	if c.tokens == nil {
		return "", errors.New("a Microsoft Entra ID TokenProvider is required")
	}
	token, err := c.tokens(ctx, c.scope)
	if err != nil {
		return "", err
	}
	if token == "" {
		return "", errors.New("the Microsoft Entra ID TokenProvider returned an empty token")
	}
	return "Bearer " + token, nil
}

// keyOrBearer returns the auth header: keyHeader with the resource key when one
// is set, else an Entra bearer token.
func (c credentials) keyOrBearer(ctx context.Context, keyHeader string) (map[string]string, error) {
	if c.apiKey != "" {
		return map[string]string{keyHeader: c.apiKey}, nil
	}
	return c.entra(ctx)
}

// entra returns the Authorization header of an Entra token, for APIs that do not
// accept the resource key (the Foundry project API).
func (c credentials) entra(ctx context.Context) (map[string]string, error) {
	bearer, err := c.bearer(ctx)
	if err != nil {
		return nil, err
	}
	return map[string]string{"Authorization": bearer}, nil
}

// errRedirectWithToken is returned instead of following a redirect to another
// host on a request that carries an Entra token.
var errRedirectWithToken = errors.New("refusing to follow a redirect to another host on a request that carries a Microsoft Entra ID token")

// noKeyRedirects returns a shallow copy of client that does not follow
// redirects of requests carrying a resource key (all of them when always is
// set), nor redirects to another host of requests carrying a token. net/http
// forwards custom headers such as api-key across hosts on a redirect, and keeps
// Authorization for the same host name on another port, so following one could
// send the credential elsewhere; the TS client makes the same choice for keys.
func noKeyRedirects(client *http.Client, always bool) *http.Client {
	next := *client
	previous := client.CheckRedirect
	next.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if always || (len(via) > 0 && (via[0].Header.Get(headerAPIKey) != "" || via[0].Header.Get(headerAnthropicKey) != "")) {
			return errRedirectWithKey
		}
		if len(via) > 0 && via[0].Header.Get("Authorization") != "" && !strings.EqualFold(req.URL.Host, via[0].URL.Host) {
			return errRedirectWithToken
		}
		if previous != nil {
			return previous(req, via)
		}
		if len(via) >= 10 {
			return errors.New("stopped after 10 redirects")
		}
		return nil
	}
	return &next
}

// openAITransport adapts the requests of the shared OpenAI driver to Azure: it
// replaces the SDK's placeholder bearer key with the Azure credentials (only
// for the configured host, so a redirect elsewhere never receives them), adds
// the configured query parameters, and rewrites the request model.
type openAITransport struct {
	base  http.RoundTripper
	creds credentials
	host  string
	// query returns the query parameters a request path needs (api-version), or nil.
	query func(path string) map[string]string
	// model maps the request body's model, or is nil to leave it unchanged.
	model func(path string, model string) string
}

// RoundTrip implements http.RoundTripper.
func (t *openAITransport) RoundTrip(req *http.Request) (*http.Response, error) {
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	out := req.Clone(req.Context())
	out.Header.Del("Authorization")
	if !strings.EqualFold(out.URL.Host, t.host) {
		return base.RoundTrip(out)
	}
	headers, err := t.creds.keyOrBearer(req.Context(), headerAPIKey)
	if err != nil {
		closeBody(req)
		return nil, err
	}
	for key, value := range headers {
		out.Header.Set(key, value)
	}
	if t.query != nil {
		if params := t.query(out.URL.Path); len(params) > 0 {
			values := out.URL.Query()
			for key, value := range params {
				if values.Get(key) == "" {
					values.Set(key, value)
				}
			}
			out.URL.RawQuery = values.Encode()
		}
	}
	if t.model != nil && out.Method == http.MethodPost && req.Body != nil && strings.Contains(out.Header.Get("Content-Type"), "application/json") {
		if err := rewriteRequestModel(req, out, t.model); err != nil {
			return nil, err
		}
	}
	return base.RoundTrip(out)
}

// rewriteRequestModel reads the JSON body of req and sets it on out with its
// "model" member mapped through model. A body that is not a JSON object, or has
// no string model, is forwarded unchanged.
func rewriteRequestModel(req *http.Request, out *http.Request, model func(path string, model string) string) error {
	data, err := io.ReadAll(req.Body)
	closeBody(req)
	if err != nil {
		return err
	}
	var body map[string]json.RawMessage
	if json.Unmarshal(data, &body) == nil {
		var current string
		if raw, ok := body["model"]; ok && json.Unmarshal(raw, &current) == nil {
			if next := model(out.URL.Path, current); next != current {
				encoded, _ := json.Marshal(next)
				body["model"] = encoded
				if rewritten, err := json.Marshal(body); err == nil {
					data = rewritten
				}
			}
		}
	}
	out.Body = io.NopCloser(bytes.NewReader(data))
	out.ContentLength = int64(len(data))
	out.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(data)), nil }
	return nil
}

func closeBody(req *http.Request) {
	if req.Body != nil {
		_ = req.Body.Close()
	}
}

// deploymentOf returns the deployment half of a "deployment::source-model" ID,
// or the ID itself.
func deploymentOf(_ string, model string) string {
	deployment, _ := parseModelID(model)
	return deployment
}
