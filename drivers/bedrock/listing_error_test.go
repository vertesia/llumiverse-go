package bedrock

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	awshttp "github.com/aws/aws-sdk-go-v2/aws/transport/http"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
	"github.com/aws/smithy-go"
	smithyhttp "github.com/aws/smithy-go/transport/http"
	"github.com/vertesia/llumiverse-go/common"
)

// fakeBedrockFailingModels fails the foundation model listing.
type fakeBedrockFailingModels struct {
	fakeBedrockModels
	err error
}

func (f *fakeBedrockFailingModels) ListFoundationModels(context.Context, *bedrock.ListFoundationModelsInput, ...func(*bedrock.Options)) (*bedrock.ListFoundationModelsOutput, error) {
	return nil, f.err
}

// Mirrors the TS model-listing.test.ts "rethrows a foundation listing failure
// with the provider status instead of a generic error".
func TestBedrockListModelsSurfacesTheProviderError(t *testing.T) {
	rejection := &smithy.OperationError{ServiceID: "Bedrock", OperationName: "ListFoundationModels", Err: &awshttp.ResponseError{
		ResponseError: &smithyhttp.ResponseError{
			Response: &smithyhttp.Response{Response: &http.Response{StatusCode: 400}},
			Err: &smithy.GenericAPIError{
				Code:    "InvalidIdentityTokenException",
				Message: "The web identity token provided could not be validated.",
				Fault:   smithy.FaultClient,
			},
		},
		RequestID: "req-1",
	}}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-2"}, &fakeBedrockRuntime{}, &fakeBedrockFailingModels{err: rejection})
	_, err := driver.ListModels(context.Background(), nil)
	var llumiErr *common.LlumiverseError
	if !errors.As(err, &llumiErr) || llumiErr.Code != 400 || llumiErr.Name != "InvalidIdentityTokenException" || llumiErr.Retryable == nil || *llumiErr.Retryable {
		t.Fatalf("error = %#v", err)
	}
	want := "[bedrock] InvalidIdentityTokenException: [400] The web identity token provided could not be validated. (Request ID: req-1)"
	if llumiErr.Message != want {
		t.Fatalf("message = %q", llumiErr.Message)
	}
	if !strings.Contains(err.Error(), "InvalidIdentityTokenException") {
		t.Fatalf("error text = %q", err.Error())
	}
}
