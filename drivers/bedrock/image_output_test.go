package bedrock

import (
	"context"
	"reflect"
	"testing"
)

func bedrockCanvasResponse(t *testing.T, body string) *ExecutionResponse {
	t.Helper()
	runtime := &fakeBedrockRuntime{invokeBody: []byte(body)}
	driver := NewBedrockDriverWithClient(BedrockOptions{Region: "us-east-1"}, runtime, nil)
	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "Draw a tree"}}, ExecutionOptions{
		Model:        "amazon.nova-canvas-v1:0",
		ModelOptions: map[string]any{"taskType": "TEXT_IMAGE"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

// Mirrors the TS service-tier.test.ts "reports a Nova Canvas failure without image output %j".
func TestBedrockImageGenerationReportsMissingOutput(t *testing.T) {
	for _, c := range []struct {
		body, code, message string
	}{
		{`{"error":"Image generation was blocked"}`, "content_policy_violation", "Image generation was blocked"},
		{`{"images":[]}`, "validation_error", "No images returned by Nova Canvas"},
		{`{"images":["","   ",null]}`, "validation_error", "No images returned by Nova Canvas"},
		{`{}`, "validation_error", "No images returned by Nova Canvas"},
	} {
		resp := bedrockCanvasResponse(t, c.body)
		if resp.Error == nil || resp.Error.Code != c.code || resp.Error.Message != c.message || len(resp.Result) != 0 {
			t.Errorf("%s: error = %#v, result = %#v", c.body, resp.Error, resp.Result)
		}
	}
}

// Mirrors the TS "retains usable Nova Canvas images alongside provider error %s".
func TestBedrockImageGenerationKeepsUsableImages(t *testing.T) {
	want := []CompletionResult{newImageResult("data:image/png;base64,YQ==")}
	resp := bedrockCanvasResponse(t, `{"images":["","YQ==",null,"   "]}`)
	if resp.Error != nil || !reflect.DeepEqual(resp.Result, want) {
		t.Fatalf("error = %#v, result = %#v", resp.Error, resp.Result)
	}
	resp = bedrockCanvasResponse(t, `{"images":["","YQ==",null,"   "],"error":"Filtered image"}`)
	if resp.Error == nil || resp.Error.Code != "content_policy_violation" || resp.Error.Message != "Filtered image" || !reflect.DeepEqual(resp.Result, want) {
		t.Fatalf("error = %#v, result = %#v", resp.Error, resp.Result)
	}
}
