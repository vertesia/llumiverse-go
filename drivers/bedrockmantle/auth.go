package bedrockmantle

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
)

// Bedrock API key tokens, as @aws/bedrock-token-generator creates them: a SigV4-presigned
// https://bedrock.amazonaws.com/?Action=CallWithBearerToken request for the bedrock service, without its scheme,
// with "&Version=1" appended, base64-encoded behind the "bedrock-api-key-" prefix.
const (
	tokenHost           = "bedrock.amazonaws.com"
	tokenService        = "bedrock"
	tokenPrefix         = "bedrock-api-key-"
	tokenExpiresSeconds = 43200 // 12 hours, the generator's default and maximum
	// messagesService is the SigV4 service of the Mantle Messages API.
	messagesService = "bedrock-mantle"
	// emptyPayloadHash is the SHA-256 of an empty body.
	emptyPayloadHash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)

// auth signs Mantle requests: with a Bedrock API key when one is configured, else with AWS credentials (bearer
// tokens derived from them on the OpenAI endpoints, SigV4 on the Messages API).
type auth struct {
	apiKey      string
	credentials aws.CredentialsProvider
	region      string
	signer      *v4.Signer
	now         func() time.Time
}

func newAuth(apiKey string, credentials aws.CredentialsProvider, region string) *auth {
	return &auth{apiKey: apiKey, credentials: credentials, region: region, signer: v4.NewSigner(), now: time.Now}
}

// bearerToken returns the bearer token of a request: the API key, or a token derived from the AWS credentials. A
// token is derived for every request, as the TS token provider does; it is an HMAC and needs no network call
// beyond what the credential provider itself does.
func (a *auth) bearerToken(ctx context.Context) (string, error) {
	if a.apiKey != "" {
		return a.apiKey, nil
	}
	return a.createToken(ctx)
}

func (a *auth) createToken(ctx context.Context) (string, error) {
	creds, err := a.retrieve(ctx)
	if err != nil {
		return "", err
	}
	query := url.Values{}
	query.Set("Action", "CallWithBearerToken")
	query.Set("X-Amz-Expires", strconv.Itoa(tokenExpiresSeconds))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://"+tokenHost+"/?"+query.Encode(), nil)
	if err != nil {
		return "", err
	}
	presigned, _, err := a.signer.PresignHTTP(ctx, creds, req, emptyPayloadHash, tokenService, a.region, a.now())
	if err != nil {
		return "", fmt.Errorf("bedrock mantle: create bearer token: %w", err)
	}
	unprefixed := strings.TrimPrefix(presigned, "https://") + "&Version=1"
	return tokenPrefix + base64.StdEncoding.EncodeToString([]byte(unprefixed)), nil
}

func (a *auth) retrieve(ctx context.Context) (aws.Credentials, error) {
	if a.credentials == nil {
		return aws.Credentials{}, errors.New("bedrock mantle: no AWS credentials or API key configured")
	}
	creds, err := a.credentials.Retrieve(ctx)
	if err != nil {
		return aws.Credentials{}, fmt.Errorf("bedrock mantle: retrieve AWS credentials: %w", err)
	}
	return creds, nil
}

// signMessages authenticates a Messages API request: a bearer API key, or SigV4 for the bedrock-mantle service
// over the request as sent.
func (a *auth) signMessages(req *http.Request) error {
	if a.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+a.apiKey)
		return nil
	}
	creds, err := a.retrieve(req.Context())
	if err != nil {
		return err
	}
	payloadHash := emptyPayloadHash
	if req.Body != nil && req.Body != http.NoBody {
		body, err := io.ReadAll(req.Body)
		_ = req.Body.Close()
		if err != nil {
			return err
		}
		sum := sha256.Sum256(body)
		payloadHash = hex.EncodeToString(sum[:])
		req.Body = io.NopCloser(bytes.NewReader(body))
		req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
		req.ContentLength = int64(len(body))
	}
	req.Header.Del("Authorization")
	return a.signer.SignHTTP(req.Context(), creds, req, payloadHash, messagesService, a.region, a.now())
}

// errForeignHost is returned for a request to a host other than the Mantle endpoint.
var errForeignHost = errors.New("bedrock mantle: refusing to authenticate a request to another host")

// errRedirect is returned instead of following a redirect, which could send the credential elsewhere.
var errRedirect = errors.New("bedrock mantle: refusing to follow a redirect")

// transport authenticates the requests of the shared OpenAI drivers (bearer token) or of the Messages API
// (SigV4), only towards the Mantle endpoint host.
type transport struct {
	base     http.RoundTripper
	auth     *auth
	host     string
	messages bool
}

func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	if !strings.EqualFold(req.URL.Host, t.host) {
		return nil, errForeignHost
	}
	next := req.Clone(req.Context())
	if t.messages {
		if err := t.auth.signMessages(next); err != nil {
			return nil, err
		}
	} else {
		token, err := t.auth.bearerToken(req.Context())
		if err != nil {
			return nil, err
		}
		next.Header.Set("Authorization", "Bearer "+token)
	}
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(next)
}

// authenticatedClient returns a copy of client whose requests t authenticates and which does not follow redirects.
func authenticatedClient(client *http.Client, t *transport) *http.Client {
	next := *client
	t.base = client.Transport
	next.Transport = t
	next.CheckRedirect = func(*http.Request, []*http.Request) error { return errRedirect }
	return &next
}
