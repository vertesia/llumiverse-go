package vertexai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

// imagenRequest runs an Imagen execution against a server answering with
// predictions and returns the request instance and parameters.
func imagenRequest(t *testing.T, modelOptions map[string]any, roles []common.PromptRole, predictions string) (map[string]any, map[string]any, *ExecutionResponse) {
	t.Helper()
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"predictions":` + predictions + `}`))
	}))
	defer server.Close()
	driver := newTestVertexDriver(t, server.URL)
	segments := make([]PromptSegment, len(roles))
	for i, role := range roles {
		segments[i] = PromptSegment{Role: role, Content: "Edit the image", Files: []DataSource{BytesDataSource{FileName: "image.png", MIME: "image/png", Data: []byte{1, 2, 3}}}}
	}
	response, err := driver.Execute(context.Background(), segments, ExecutionOptions{Model: "imagen-3.0-capability-001", ModelOptions: modelOptions})
	if err != nil {
		t.Fatal(err)
	}
	instance := body["instances"].([]any)[0].(map[string]any)
	return instance, body["parameters"].(map[string]any), response
}

func referenceImages(instance map[string]any) []map[string]any {
	var out []map[string]any
	for _, ref := range instance["referenceImages"].([]any) {
		out = append(out, ref.(map[string]any))
	}
	return out
}

// TestImagenPersonGeneration mirrors "Imagen person generation options".
func TestImagenPersonGeneration(t *testing.T) {
	t.Parallel()

	for _, value := range []string{"allow_adult", "allow_adults"} {
		_, parameters, _ := imagenRequest(t, map[string]any{"person_generation": value}, []common.PromptRole{PromptRoleUser}, `[]`)
		if parameters["personGeneration"] != "allow_adult" {
			t.Errorf("%s: personGeneration = %v", value, parameters["personGeneration"])
		}
	}
}

// TestImagenOptionSerialization mirrors "Imagen option serialization" of
// imagen-options.test.ts.
func TestImagenOptionSerialization(t *testing.T) {
	t.Parallel()

	for _, mode := range []string{imagenEditInpaintRemoval, imagenEditInpaintInsertion, imagenEditBackgroundSwap, imagenEditOutpaint} {
		instance, parameters, _ := imagenRequest(t, map[string]any{
			"edit_mode": mode, "mask_mode": imagenMaskModeSemantic, "mask_class": []any{175, 176},
			"mask_dilation": 0.01, "edit_steps": 35, "guidance_scale": 60,
		}, []common.PromptRole{PromptRoleUser}, `[]`)
		refs := referenceImages(instance)
		if len(refs) != 2 || refs[0]["referenceType"] != "REFERENCE_TYPE_RAW" || refs[1]["referenceType"] != "REFERENCE_TYPE_MASK" ||
			!reflect.DeepEqual(refs[1]["maskImageConfig"], map[string]any{"maskMode": imagenMaskModeSemantic, "maskClasses": []any{175.0, 176.0}, "dilation": 0.01}) {
			t.Errorf("%s: references = %#v", mode, refs)
		}
		if parameters["editMode"] != mode || !reflect.DeepEqual(parameters["editConfig"], map[string]any{"baseSteps": 35.0}) || parameters["guidanceScale"] != 60.0 {
			t.Errorf("%s: parameters = %#v", mode, parameters)
		}
	}

	for _, maskMode := range []string{"MASK_MODE_BACKGROUND", "MASK_MODE_FOREGROUND", imagenMaskModeUserProvided} {
		roles := []common.PromptRole{PromptRoleUser}
		if maskMode == imagenMaskModeUserProvided {
			roles = append(roles, PromptRoleMask)
		}
		instance, _, _ := imagenRequest(t, map[string]any{"edit_mode": imagenEditInpaintInsertion, "mask_mode": maskMode, "mask_class": []any{175}, "mask_dilation": 0.01}, roles, `[]`)
		refs := referenceImages(instance)
		if len(refs) != 2 || refs[1]["referenceType"] != "REFERENCE_TYPE_MASK" || !reflect.DeepEqual(refs[1]["maskImageConfig"], map[string]any{"maskMode": maskMode, "dilation": 0.01}) {
			t.Errorf("%s: references = %#v", maskMode, refs)
		}
		if data, _ := json.Marshal(instance); strings.Contains(string(data), "maskClasses") {
			t.Errorf("%s: stale semantic classes sent: %s", maskMode, data)
		}
	}

	for _, mode := range []string{imagenTextImage, imagenEditInpaintInsertion} {
		options := map[string]any{
			"edit_mode": mode, "number_of_images": 2, "seed": 0, "person_generation": "allow_adults",
			"safety_setting": "block_medium_and_above", "image_file_type": "image/jpeg", "jpeg_compression_quality": 0, "guidance_scale": 0,
		}
		want := map[string]any{
			"sampleCount": 2.0, "seed": 0.0, "personGeneration": "allow_adult", "safetySetting": "block_medium_and_above",
			"guidanceScale": 0.0, "outputOptions": map[string]any{"mimeType": "image/jpeg", "compressionQuality": 0.0},
		}
		if mode == imagenTextImage {
			options["add_watermark"], options["enhance_prompt"], options["aspect_ratio"] = false, false, "4:3"
			want["addWatermark"], want["enhancePrompt"], want["aspectRatio"] = false, false, "4:3"
		}
		_, parameters, _ := imagenRequest(t, options, []common.PromptRole{PromptRoleUser}, `[]`)
		for key, value := range want {
			if !reflect.DeepEqual(parameters[key], value) {
				t.Errorf("%s: %s = %#v, want %#v", mode, key, parameters[key], value)
			}
		}
	}

	instance, parameters, _ := imagenRequest(t, map[string]any{"edit_mode": imagenEditInpaintInsertion}, []common.PromptRole{PromptRoleUser, PromptRoleMask}, `[]`)
	refs := referenceImages(instance)
	if len(refs) != 2 || refs[1]["referenceType"] != "REFERENCE_TYPE_MASK" || refs[1]["referenceImage"].(map[string]any)["bytesBase64Encoded"] != "AQID" ||
		!reflect.DeepEqual(refs[1]["maskImageConfig"], map[string]any{"maskMode": imagenMaskModeUserProvided}) {
		t.Fatalf("default mask references = %#v", refs)
	}
	if _, ok := parameters["editConfig"]; ok {
		t.Fatalf("unspecified editConfig sent: %#v", parameters)
	}

	_, parameters, _ = imagenRequest(t, nil, []common.PromptRole{PromptRoleUser}, `[]`)
	for _, key := range []string{"outputOptions", "guidanceScale"} {
		if _, ok := parameters[key]; ok {
			t.Fatalf("unspecified %s sent: %#v", key, parameters)
		}
	}

	for _, mode := range []string{imagenCustomizationSubject, imagenCustomizationControl} {
		instance, _, _ := imagenRequest(t, map[string]any{"edit_mode": mode, "controlType": "CONTROL_TYPE_SCRIBBLE", "controlImageComputation": false}, []common.PromptRole{PromptRoleUser}, `[]`)
		refs := referenceImages(instance)
		if len(refs) != 1 || refs[0]["referenceType"] != "REFERENCE_TYPE_CONTROL" ||
			!reflect.DeepEqual(refs[0]["controlImageConfig"], map[string]any{"controlType": "CONTROL_TYPE_SCRIBBLE", "enableControlImageComputation": false}) {
			t.Errorf("%s: references = %#v", mode, refs)
		}
	}
}

// TestImagenOutputMetadata mirrors "Imagen output MIME metadata" of
// imagen-options.test.ts.
func TestImagenOutputMetadata(t *testing.T) {
	t.Parallel()

	results := func(response *ExecutionResponse) []string {
		var out []string
		for _, result := range response.Result {
			out = append(out, result.String())
		}
		return out
	}
	for requested, want := range map[string]string{"": "image/png", "image/jpeg": "image/jpeg"} {
		options := map[string]any{}
		if requested != "" {
			options["image_file_type"] = requested
		}
		_, _, response := imagenRequest(t, options, nil, `[{"bytesBase64Encoded":"YQ=="}]`)
		if got := results(response); !reflect.DeepEqual(got, []string{"data:" + want + ";base64,YQ=="}) {
			t.Errorf("requested %q: results = %v", requested, got)
		}
	}

	_, _, partial := imagenRequest(t, nil, nil, `[{"bytesBase64Encoded":"YQ==","mimeType":"image/png"},{"raiFilteredReason":"Filtered image"},{"bytesBase64Encoded":"   "}]`)
	if partial.Error != nil || !reflect.DeepEqual(results(partial), []string{"data:image/png;base64,YQ=="}) {
		t.Fatalf("partial response = %#v", partial.Completion)
	}

	_, _, filtered := imagenRequest(t, nil, nil, `[{"raiFilteredReason":"All images were filtered"}]`)
	if filtered.Error == nil || filtered.Error.Code != "content_policy_violation" || filtered.Error.Message != "All images were filtered" || len(filtered.Result) != 0 {
		t.Fatalf("filtered response = %#v", filtered.Completion)
	}

	for _, predictions := range []string{`[]`, `[{"bytesBase64Encoded":"   "}]`} {
		_, _, empty := imagenRequest(t, nil, nil, predictions)
		if empty.Error == nil || empty.Error.Code != "validation_error" || empty.Error.Message != "No images returned by Imagen" {
			t.Errorf("%s: response = %#v", predictions, empty.Completion)
		}
	}

	for _, mime := range []string{"image/png", "image/jpeg"} {
		_, _, response := imagenRequest(t, map[string]any{"image_file_type": "image/png"}, nil, `[{"bytesBase64Encoded":"YQ==","mimeType":"`+mime+`"},{"bytesBase64Encoded":"Yg==","mimeType":"`+mime+`"}]`)
		if got := results(response); !reflect.DeepEqual(got, []string{"data:" + mime + ";base64,YQ==", "data:" + mime + ";base64,Yg=="}) {
			t.Errorf("%s: results = %v", mime, got)
		}
	}
}
