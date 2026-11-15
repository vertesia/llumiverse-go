package xai

import (
	"context"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
	"github.com/vertesia/llumiverse-go/core"
	"github.com/vertesia/llumiverse-go/drivers/internal/fakehttp"
)

const responsesBody = `{
	"id":"response-1","object":"response","created_at":1,"model":"grok-5","status":"completed",
	"output":[{"id":"message-1","type":"message","role":"assistant","status":"completed",
		"content":[{"type":"output_text","text":"Hello","annotations":[]}]}],
	"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}
}`

func newTestDriver(t *testing.T, routes map[string]fakehttp.Response) (*XAIDriver, *fakehttp.Server) {
	t.Helper()
	server := fakehttp.New(t, routes)
	driver, err := NewXAIDriver(XAIOptions{APIKey: "test-key", Endpoint: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	return driver, server
}

func TestXAILanguageModelsUseResponses(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{"/v1/responses": {Body: responsesBody}})
	resp, err := driver.Execute(context.Background(), []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Hi"}}, common.ExecutionOptions{Model: "grok-5"})
	if err != nil {
		t.Fatal(err)
	}
	if core.CompletionResultsToText(resp.Result) != "Hello" || driver.Provider() != common.ProviderXAI {
		t.Fatalf("resp = %+v", resp.Completion)
	}
	if got := server.Last(t, "/v1/responses").Header.Get("Authorization"); got != "Bearer test-key" {
		t.Fatalf("authorization = %q", got)
	}
}

func TestXAIImageGeneration(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{
		"/v1/images/generations": {Body: `{"data":[{"b64_json":"QUJD","mime_type":"image/png"},{"url":"https://images.x.ai/b.jpg"},null],
			"usage":{"cost_in_usd_ticks":700000000}}`},
	})
	resp, err := driver.Execute(context.Background(), []common.PromptSegment{
		{Role: common.PromptRoleSystem, Content: "Photographic style."},
		{Role: common.PromptRoleUser, Content: "A lighthouse at dusk."},
	}, common.ExecutionOptions{
		Model:        "grok-imagine-image",
		ModelOptions: map[string]any{"aspect_ratio": "16:9", "n": 2, "quality": "high"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Error != nil || len(resp.Result) != 2 {
		t.Fatalf("resp = %+v", resp.Completion)
	}
	if got := resp.Result[0].String(); got != "data:image/png;base64,QUJD" {
		t.Fatalf("first image = %q", got)
	}
	if resp.Result[1].String() != "https://images.x.ai/b.jpg" {
		t.Fatalf("second image = %q", resp.Result[1].String())
	}
	if resp.TokenUsage == nil || *resp.TokenUsage.ProviderCostUsd != 0.07 {
		t.Fatalf("usage = %+v", resp.TokenUsage)
	}
	body := server.Last(t, "/v1/images/generations").Body
	if body["prompt"] != "Photographic style.\nA lighthouse at dusk." || body["aspect_ratio"] != "16:9" || body["n"] != float64(2) || body["quality"] != "high" {
		t.Fatalf("body = %v", body)
	}
	if _, ok := body["image"]; ok {
		t.Fatalf("image sent for a generation: %v", body)
	}
}

func TestXAIImageEditsAndStream(t *testing.T) {
	driver, server := newTestDriver(t, map[string]fakehttp.Response{
		"/v1/images/edits": {Body: `{"data":[{"b64_json":"REVG"}]}`},
	})
	png := &common.BytesDataSource{FileName: "in.png", MIME: "image/png", Data: []byte("PNG")}
	segments := []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Make it night.", Files: []common.DataSource{png}}}
	stream, err := driver.Stream(context.Background(), segments, common.ExecutionOptions{Model: "xai/grok-2-image"})
	if err != nil {
		t.Fatal(err)
	}
	chunks, resp := fakehttp.Drain(t, stream)
	if len(chunks) != 1 || resp.Result[0].String() != "data:image/jpeg;base64,REVG" {
		t.Fatalf("chunks = %+v, resp = %+v", chunks, resp.Completion)
	}
	image, _ := server.Last(t, "/v1/images/edits").Body["image"].(map[string]any)
	if image["type"] != "image_url" || !strings.HasPrefix(image["url"].(string), "data:image/png;base64,") {
		t.Fatalf("image = %v", image)
	}
	debug := driver.FormatDebugPrompt(resp.Prompt).(ImagePrompt)
	if len(debug.Images) != 1 || debug.Prompt != "Make it night." {
		t.Fatalf("debug prompt = %+v", debug)
	}
}

func TestXAIImageRejectionIsTheCompletionError(t *testing.T) {
	driver, _ := newTestDriver(t, map[string]fakehttp.Response{
		"/v1/images/generations": {Status: 400, Body: `{"error":{"code":"content_policy_violation","message":"Rejected by the content policy"}}`},
	})
	resp, err := driver.Execute(context.Background(), []common.PromptSegment{{Role: common.PromptRoleUser, Content: "Something"}}, common.ExecutionOptions{Model: "grok-imagine-image"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Error == nil || resp.Error.Code != "content_policy_violation" || len(resp.Result) != 0 {
		t.Fatalf("resp = %+v", resp.Completion)
	}
}

func TestXAIListModels(t *testing.T) {
	driver, _ := newTestDriver(t, map[string]fakehttp.Response{
		"/v1/language-models":         {Body: `{"models":[{"id":"grok-5","owned_by":"xai"},{"id":"grok-embed","owned_by":"xai"}]}`},
		"/v1/image-generation-models": {Body: `{"models":[{"id":"grok-imagine-image","owned_by":"xai","version":"1.0"}]}`},
	})
	models, err := driver.ListModels(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "grok-5" || models[1].ID != "grok-imagine-image" {
		t.Fatalf("models = %+v", models)
	}
	image := models[1]
	if *image.Type != "image" || *image.CanStream || *image.Version != "1.0" || *image.Description != "grok-imagine-image by xai" {
		t.Fatalf("image model = %+v", image)
	}
	// Modalities and tool support come from the capability catalogue.
	if grok := models[0]; strings.Join(*grok.Tags, ",") != "i:text,i:image,o:text" || !*grok.ToolSupport || !*grok.IsMultimodal {
		t.Fatalf("grok = %+v", grok)
	}
	if strings.Join(*image.Tags, ",") != "i:text,i:image,o:image" || *image.ToolSupport || !*image.IsMultimodal {
		t.Fatalf("image tags = %v", *image.Tags)
	}

	// One listing failing still lists the other.
	onlyLanguage, _ := newTestDriver(t, map[string]fakehttp.Response{
		"/v1/language-models": {Body: `{"models":[{"id":"grok-5","owned_by":"xai"}]}`},
	})
	if models, err := onlyLanguage.ListModels(context.Background(), nil); err != nil || len(models) != 1 {
		t.Fatalf("models = %+v, err = %v", models, err)
	}
	none, _ := newTestDriver(t, nil)
	if _, err := none.ListModels(context.Background(), nil); err == nil {
		t.Fatal("both listings failing is not an error")
	}
}

func TestIsImageModel(t *testing.T) {
	for model, want := range map[string]bool{"grok-imagine-image": true, "xai/grok-2-image-1212": true, "grok-5": false, "gpt-image-2": false} {
		if IsImageModel(model) != want {
			t.Errorf("%s: want %v", model, want)
		}
	}
}

func TestXAIImageWithoutUsableImagesIsAnError(t *testing.T) {
	driver, _ := newTestDriver(t, map[string]fakehttp.Response{"/v1/images/generations": {Body: `{"data":[{"b64_json":"  "},{"url":""}]}`}})
	resp, err := driver.Execute(context.Background(), []common.PromptSegment{{Role: common.PromptRoleUser, Content: "A cat"}}, common.ExecutionOptions{Model: "grok-imagine-image"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Error == nil || resp.Error.Code != "validation_error" || len(resp.Result) != 0 {
		t.Fatalf("resp = %+v", resp.Completion)
	}
}
