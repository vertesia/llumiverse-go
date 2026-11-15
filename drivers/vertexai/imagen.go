package vertexai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// Imagen edit modes (model option edit_mode). Without one, Imagen generates from
// text.
const (
	imagenTextImage              = "TEXT_IMAGE"
	imagenEditInpaintRemoval     = "EDIT_MODE_INPAINT_REMOVAL"
	imagenEditInpaintInsertion   = "EDIT_MODE_INPAINT_INSERTION"
	imagenEditBackgroundSwap     = "EDIT_MODE_BGSWAP"
	imagenEditOutpaint           = "EDIT_MODE_OUTPAINT"
	imagenCustomizationSubject   = "CUSTOMIZATION_SUBJECT"
	imagenCustomizationControl   = "CUSTOMIZATION_CONTROLLED"
	imagenCustomizationInstruct  = "CUSTOMIZATION_INSTRUCT"
	imagenCustomizationStyle     = "CUSTOMIZATION_STYLE"
	imagenMaskModeUserProvided   = "MASK_MODE_USER_PROVIDED"
	imagenMaskModeSemantic       = "MASK_MODE_SEMANTIC"
	imagenDefaultControlType     = "CONTROL_TYPE_CANNY"
	imagenDefaultSubjectType     = "SUBJECT_TYPE_DEFAULT"
	imagenDefaultOutputMIMEType  = "image/png"
	imagenNoImagesMessage        = "No images returned by Imagen"
	imagenContentPolicyErrorCode = "content_policy_violation"
	imagenValidationErrorCode    = "validation_error"
)

// imagenMaskImageConfig builds the maskImageConfig of a mask reference: the mask
// mode (user provided by default), the semantic classes only in semantic mode,
// and the dilation when set.
func imagenMaskImageConfig(modelOptions map[string]any, maskMode string) map[string]any {
	config := map[string]any{"maskMode": maskMode}
	if maskMode == imagenMaskModeSemantic {
		if classes, ok := modelOptions["mask_class"]; ok && classes != nil {
			config["maskClasses"] = classes
		}
	}
	if dilation := optionFloat(modelOptions, "mask_dilation"); dilation != nil {
		config["dilation"] = *dilation
	}
	return config
}

// imagenControlImageConfig builds the controlImageConfig of a control reference.
func imagenControlImageConfig(modelOptions map[string]any) map[string]any {
	controlType := optionString(modelOptions, "controlType")
	if controlType == "" {
		controlType = imagenDefaultControlType
	}
	config := map[string]any{"controlType": controlType}
	if computation, ok := modelOptions["controlImageComputation"].(bool); ok {
		config["enableControlImageComputation"] = computation
	}
	return config
}

// formatVertexImagenPrompt assembles an Imagen request as the TS client does:
// system, user and safety text are joined into the prompt, negative segments
// become the negativePrompt, and each segment's images become references of one
// reference ID shaped by edit_mode. Edit modes send the image as RAW (plus an
// automatic mask reference unless the mask is user provided); the customization
// modes send control, subject, style or raw references; a mask segment adds the
// user-provided mask.
func formatVertexImagenPrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (vertexImagenPrompt, error) {
	var system, user, safety, negative []string
	prompt := vertexImagenPrompt{}
	modelOptions := options.ModelOptions
	editMode := optionString(modelOptions, "edit_mode")
	maskMode := optionString(modelOptions, "mask_mode")
	if maskMode == "" {
		maskMode = imagenMaskModeUserProvided
	}
	for _, segment := range segments {
		switch segment.Role {
		case PromptRoleSafety:
			safety = append(safety, segment.Content)
		case PromptRoleSystem:
			system = append(system, segment.Content)
		case PromptRoleNegative:
			negative = append(negative, segment.Content)
		default:
			user = append(user, segment.Content)
		}
		refID := len(prompt.ReferenceImages) + 1
		for _, file := range segment.Files {
			if !strings.Contains(file.MIMEType(), "image") {
				continue
			}
			encoded, err := dataSourceToBase64(ctx, file)
			if err != nil {
				return vertexImagenPrompt{}, err
			}
			image := map[string]any{"bytesBase64Encoded": encoded}
			if segment.Role == PromptRoleMask {
				if maskMode == imagenMaskModeUserProvided {
					prompt.ReferenceImages = append(prompt.ReferenceImages, map[string]any{
						"referenceType":   "REFERENCE_TYPE_MASK",
						"referenceId":     refID,
						"referenceImage":  image,
						"maskImageConfig": imagenMaskImageConfig(modelOptions, maskMode),
					})
				}
				continue
			}
			switch {
			case strings.Contains(editMode, "EDIT_MODE"):
				prompt.ReferenceImages = append(prompt.ReferenceImages, map[string]any{
					"referenceType":  "REFERENCE_TYPE_RAW",
					"referenceId":    refID,
					"referenceImage": image,
				})
				if maskMode != imagenMaskModeUserProvided {
					prompt.ReferenceImages = append(prompt.ReferenceImages, map[string]any{
						"referenceType":   "REFERENCE_TYPE_MASK",
						"referenceId":     refID,
						"maskImageConfig": imagenMaskImageConfig(modelOptions, maskMode),
					})
				}
			case editMode == imagenCustomizationSubject && refID == 1, editMode == imagenCustomizationControl:
				// In subject customization the first reference is the control image.
				prompt.ReferenceImages = append(prompt.ReferenceImages, map[string]any{
					"referenceType":      "REFERENCE_TYPE_CONTROL",
					"referenceId":        refID,
					"referenceImage":     image,
					"controlImageConfig": imagenControlImageConfig(modelOptions),
				})
			case editMode == imagenCustomizationSubject:
				subjectType := optionString(modelOptions, "subjectType")
				if subjectType == "" {
					subjectType = imagenDefaultSubjectType
				}
				prompt.ReferenceImages = append(prompt.ReferenceImages, map[string]any{
					"referenceType":      "REFERENCE_TYPE_SUBJECT",
					"referenceId":        refID,
					"referenceImage":     image,
					"subjectImageConfig": map[string]any{"subjectDescription": segment.Content, "subjectType": subjectType},
				})
			case editMode == imagenCustomizationStyle:
				prompt.ReferenceImages = append(prompt.ReferenceImages, map[string]any{
					"referenceType":    "REFERENCE_TYPE_STYLE",
					"referenceId":      refID,
					"referenceImage":   image,
					"styleImageConfig": map[string]any{"styleDescription": segment.Content},
				})
			case editMode == imagenCustomizationInstruct:
				prompt.ReferenceImages = append(prompt.ReferenceImages, map[string]any{
					"referenceType":  "REFERENCE_TYPE_RAW",
					"referenceId":    refID,
					"referenceImage": image,
				})
			}
		}
	}
	prompt.Prompt = strings.TrimSpace(strings.Join([]string{
		strings.Join(system, "\n\n"),
		strings.Join(user, "\n\n"),
		strings.Join(safety, "\n\n"),
	}, "\n\n"))
	if len(negative) > 0 {
		prompt.NegativePrompt = strings.Join(negative, ", ")
	}
	return prompt, nil
}

// vertexImagenParameters builds the Imagen "parameters" object for the edit
// mode: common sampling, safety and output options, editMode / editConfig for the
// edit modes, and watermark, aspect ratio and prompt enhancement for text to
// image. Unset options are omitted, while explicit zero and false values are kept.
func vertexImagenParameters(prompt vertexImagenPrompt, options ExecutionOptions) (map[string]any, error) {
	modelOptions := options.ModelOptions
	taskType := optionString(modelOptions, "edit_mode")
	if taskType == "" {
		taskType = imagenTextImage
	}
	personGeneration := optionString(modelOptions, "person_generation")
	if personGeneration == "allow_adults" {
		personGeneration = "allow_adult"
	}
	out := map[string]any{
		"sampleCount":      optionIntValue(modelOptions, "number_of_images"),
		"seed":             optionIntValue(modelOptions, "seed"),
		"safetySetting":    nilIfEmpty(optionString(modelOptions, "safety_setting")),
		"personGeneration": nilIfEmpty(personGeneration),
		"guidanceScale":    optionFloatValue(modelOptions, "guidance_scale"),
		"negativePrompt":   nilIfEmpty(prompt.NegativePrompt),
	}
	outputOptions := map[string]any{}
	if mimeType := optionString(modelOptions, "image_file_type"); mimeType != "" {
		outputOptions["mimeType"] = mimeType
	}
	if quality := optionInt(modelOptions, "jpeg_compression_quality"); quality != nil {
		outputOptions["compressionQuality"] = *quality
	}
	if len(outputOptions) > 0 {
		out["outputOptions"] = outputOptions
	}
	switch taskType {
	case imagenEditInpaintRemoval, imagenEditInpaintInsertion, imagenEditBackgroundSwap, imagenEditOutpaint:
		out["editMode"] = taskType
		if steps := optionInt(modelOptions, "edit_steps"); steps != nil {
			out["editConfig"] = map[string]any{"baseSteps": *steps}
		}
	case imagenTextImage:
		// A seed and a watermark cannot be used together; the provider validates it.
		out["addWatermark"] = optionBoolPtrValue(modelOptions, "add_watermark")
		out["aspectRatio"] = nilIfEmpty(optionString(modelOptions, "aspect_ratio"))
		out["enhancePrompt"] = optionBoolPtrValue(modelOptions, "enhance_prompt")
	case imagenCustomizationSubject, imagenCustomizationControl, imagenCustomizationInstruct, imagenCustomizationStyle:
	default:
		return nil, fmt.Errorf("imagen task type %q is not supported", taskType)
	}
	return removeNilMapValues(out), nil
}

// nilIfEmpty returns nil for an empty string, so the option is omitted.
func nilIfEmpty(value string) any {
	if value == "" {
		return nil
	}
	return value
}

// vertexImagenPrediction is one Imagen prediction: an image, or the reason
// responsible-AI filtering removed it.
type vertexImagenPrediction struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded"`
	MIMEType           string `json:"mimeType"`
	RAIFilteredReason  string `json:"raiFilteredReason"`
}

// vertexImagenCompletion converts Imagen predictions into image results, each
// with the MIME type Imagen returned (else the requested type, else PNG). Blank
// and filtered predictions are skipped; when none is left the completion carries
// a content_policy_violation (the filtering reason) or validation_error instead.
func vertexImagenCompletion(predictions []vertexImagenPrediction, options ExecutionOptions) *Completion {
	requested := optionString(options.ModelOptions, "image_file_type")
	results := []CompletionResult{}
	reason := ""
	for _, prediction := range predictions {
		if reason == "" {
			reason = prediction.RAIFilteredReason
		}
		if strings.TrimSpace(prediction.BytesBase64Encoded) == "" {
			continue
		}
		mimeType := prediction.MIMEType
		if mimeType == "" {
			mimeType = requested
		}
		if mimeType == "" {
			mimeType = imagenDefaultOutputMIMEType
		}
		results = append(results, newImageResult("data:"+mimeType+";base64,"+prediction.BytesBase64Encoded))
	}
	completion := &Completion{Result: results}
	if len(results) == 0 {
		completion.Error = &ResultValidationError{Code: imagenValidationErrorCode, Message: imagenNoImagesMessage}
		if reason != "" {
			completion.Error = &ResultValidationError{Code: imagenContentPolicyErrorCode, Message: reason}
		}
	}
	return completion
}

// requestVertexImagen POSTs an Imagen predict request and returns the generated
// images.
func (d *VertexAIDriver) requestVertexImagen(ctx context.Context, prompt vertexImagenPrompt, options ExecutionOptions) (*Completion, error) {
	model := shortResourceName(options.Model)
	base, err := d.vertexBaseEndpoint("publishers", "google", "models", model+":predict")
	if err != nil {
		return nil, err
	}
	parameters, err := vertexImagenParameters(prompt, options)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	payload := map[string]any{
		"instances":  []vertexImagenPrompt{prompt},
		"parameters": parameters,
	}
	var response struct {
		Predictions *[]vertexImagenPrediction `json:"predictions"`
	}
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, base, headers, payload, &response); err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	if response.Predictions == nil {
		return nil, d.formatError(errors.New("no predictions found"), options.Model, "execute")
	}
	return vertexImagenCompletion(*response.Predictions, options), nil
}
