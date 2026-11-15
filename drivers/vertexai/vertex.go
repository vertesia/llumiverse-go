package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
)

const vertexScopeCloudPlatform = "https://www.googleapis.com/auth/cloud-platform"

// Gemini Enterprise Agent Platform embeddings default to gemini-embedding-2 to match the TypeScript client.
const defaultVertexEmbeddingModel = "gemini-embedding-2"

// multimodalembedding@001 still uses the legacy predict API instead of embedContent.
const vertexLegacyMultimodalEmbeddingModel = "multimodalembedding@001"

// VertexAIOptions configures authentication and routing for Gemini Enterprise Agent Platform APIs.
type VertexAIOptions struct {
	DriverOptions
	// Project is the Google Cloud project ID.
	Project string
	// Region is the Gemini Enterprise Agent Platform location, such as us-central1.
	Region string
	// BaseURL overrides the regional Google Cloud AI Platform endpoint.
	BaseURL string
	// TokenSource supplies OAuth2 tokens. If nil, Application Default Credentials are used.
	TokenSource oauth2.TokenSource
	// HTTPClient overrides the default HTTP client.
	HTTPClient *http.Client
}

// VertexAIDriver implements Driver for Gemini Enterprise Agent Platform Gemini,
// Imagen, embedding, and Claude partner models while keeping the vertexai
// provider ID for compatibility.
type VertexAIDriver struct {
	options VertexAIOptions
	client  *http.Client
}

// NewVertexAIDriver creates a Gemini Enterprise Agent Platform driver using the supplied token source or ADC.
func NewVertexAIDriver(options VertexAIOptions) (*VertexAIDriver, error) {
	if options.Project == "" {
		return nil, errors.New("project is required")
	}
	if options.Region == "" {
		return nil, errors.New("region is required")
	}
	client := options.HTTPClient
	if client == nil {
		client = newHTTPClient(nil, options.HTTPTimeout)
	} else if hasHTTPTimeout(options.HTTPTimeout) {
		client = newHTTPClient(client, options.HTTPTimeout)
	}
	return &VertexAIDriver{options: options, client: client}, nil
}

// Provider returns ProviderVertexAI, the stable provider ID retained for the
// Gemini Enterprise Agent Platform driver.
func (d *VertexAIDriver) Provider() Provider {
	return ProviderVertexAI
}

// geminiPrompt is the generateContent request body shape: an ordered list of
// turns plus an optional systemInstruction. Turn is internal-only bookkeeping.
type geminiPrompt struct {
	Contents []geminiContent `json:"contents"`
	System   *geminiContent  `json:"systemInstruction,omitempty"`
	Turn     int             `json:"-"`
}

// geminiContent is a single conversation turn: a role ("user" or "model") and
// its ordered parts. Meta is internal-only and never serialized.
type geminiContent struct {
	Role  string         `json:"role,omitempty"`
	Parts []geminiPart   `json:"parts"`
	Meta  map[string]any `json:"-"`
}

// geminiPart is one piece of a turn. Exactly one of Text, InlineData, FileData,
// FunctionCall, or FunctionResponse is normally set. ThoughtSignature carries the
// opaque reasoning token that thinking models require to be echoed back.
type geminiPart struct {
	Text             string                `json:"text,omitempty"`
	InlineData       *geminiInlineData     `json:"inlineData,omitempty"`
	FileData         *geminiFileData       `json:"fileData,omitempty"`
	FunctionCall     *geminiFunctionCall   `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResult `json:"functionResponse,omitempty"`
	ThoughtSignature string                `json:"thoughtSignature,omitempty"`
	// Thought marks a thought-summary part; it is sent back unchanged so the
	// model's reasoning chain stays intact.
	Thought bool `json:"thought,omitempty"`
}

// MarshalJSON always emits a text field on a part that carries no other data, so
// an empty signed terminal part (text "" plus a thought signature) keeps the
// data field Gemini requires.
func (p geminiPart) MarshalJSON() ([]byte, error) {
	type plain geminiPart
	if p.InlineData != nil || p.FileData != nil || p.FunctionCall != nil || p.FunctionResponse != nil || p.Text != "" {
		return json.Marshal(plain(p))
	}
	return json.Marshal(struct {
		Text string `json:"text"`
		plain
	}{Text: p.Text, plain: plain(p)})
}

// geminiInlineData carries base64-encoded media embedded directly in the request.
type geminiInlineData struct {
	MIMEType string `json:"mimeType"`
	Data     string `json:"data"`
}

// geminiFileData references media by URI (e.g. a GCS object) instead of inlining it.
type geminiFileData struct {
	MIMEType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

// geminiFunctionCall is a tool invocation emitted by the model. ID is unique within
// the conversation (see assignGeminiFunctionCallIDs).
type geminiFunctionCall struct {
	ID   string `json:"id,omitempty"`
	Name string `json:"name"`
	Args any    `json:"args,omitempty"`
}

// geminiFunctionResult is the caller-supplied result for a prior function call. ID
// echoes the call id when the call has one; Parts carries media attached to the
// tool result.
type geminiFunctionResult struct {
	ID       string                    `json:"id,omitempty"`
	Name     string                    `json:"name"`
	Response map[string]any            `json:"response"`
	Parts    []geminiFunctionMediaPart `json:"parts,omitempty"`
}

// geminiFunctionMediaPart is a media attachment of a function response, passed by
// URI for Cloud Storage objects and inlined otherwise.
type geminiFunctionMediaPart struct {
	InlineData *geminiInlineData `json:"inlineData,omitempty"`
	FileData   *geminiFileData   `json:"fileData,omitempty"`
}

// vertexImagenPrompt is the Imagen predict instance: a text prompt plus optional
// reference/mask images and a negative prompt.
type vertexImagenPrompt struct {
	Prompt          string           `json:"prompt"`
	ReferenceImages []map[string]any `json:"referenceImages,omitempty"`
	NegativePrompt  string           `json:"negativePrompt,omitempty"`
}

// CreatePrompt builds a provider-specific request body, routing by model: Imagen
// models get an Imagen predict prompt, Claude partner models reuse the shared
// Claude prompt format, and everything else is formatted as a Gemini prompt.
func (d *VertexAIDriver) CreatePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (any, error) {
	if isVertexImagenModel(options.Model) {
		return formatVertexImagenPrompt(ctx, segments, options)
	}
	if isClaudeModel(options.Model) {
		return internalclaude.FormatPrompt(ctx, segments, options)
	}
	return d.formatGeminiPrompt(ctx, segments, options)
}

// formatGeminiPrompt converts neutral prompt segments into a geminiPrompt:
// system text becomes systemInstruction, tool results become functionResponse
// parts (echoing the thoughtSignature for thinking models), assistant maps to the
// "model" role, safety segments are appended last, and a ResultSchema either adds
// a terse JSON-output instruction (the schema travels in generationConfig) or, when
// tools are present, the full schema text. Adjacent same-role turns are merged.
func (d *VertexAIDriver) formatGeminiPrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (geminiPrompt, error) {
	prompt := geminiPrompt{System: &geminiContent{Role: "user"}}
	var safety []geminiContent
	callNamesByID := geminiFunctionCallNamesByID(geminiStoredContents(options.Conversation))
	budget := newGeminiInlineBudget(options.Conversation)
	for _, segment := range segments {
		switch segment.Role {
		case PromptRoleNegative, PromptRoleMask:
			continue
		case PromptRoleSystem:
			if len(segment.Files) > 0 {
				return geminiPrompt{}, errors.New("gemini system messages support text only")
			}
			if segment.Content != "" {
				prompt.System.Parts = append(prompt.System.Parts, geminiPart{Text: segment.Content})
			}
		case PromptRoleTool:
			if segment.ToolUseID == "" {
				return geminiPrompt{}, errors.New("tool prompt segment requires ToolUseID")
			}
			// Gemini thinking models require the thoughtSignature that arrived
			// with the tool call to be echoed on the matching functionResponse.
			// Media attached to a tool result (an image the tool rendered, for
			// example) travels as functionResponse parts, so the model sees it.
			var responseParts []geminiFunctionMediaPart
			for _, file := range segment.Files {
				part, err := geminiMediaPart(ctx, file, budget)
				if err != nil {
					return geminiPrompt{}, err
				}
				responseParts = append(responseParts, part)
			}
			id, name := resolveGeminiFunctionResponseTarget(segment.ToolUseID, callNamesByID)
			prompt.Contents = append(prompt.Contents, geminiContent{
				Role: "user",
				Parts: []geminiPart{{
					FunctionResponse: &geminiFunctionResult{
						ID:       id,
						Name:     name,
						Response: formatGeminiFunctionResponse(segment.Content),
						Parts:    responseParts,
					},
					ThoughtSignature: segment.ThoughtSignature,
				}},
			})
		default:
			parts := []geminiPart{}
			if segment.Content != "" {
				parts = append(parts, geminiPart{Text: segment.Content})
			}
			for _, file := range segment.Files {
				part, err := d.geminiFilePart(ctx, file, budget)
				if err != nil {
					return geminiPrompt{}, err
				}
				parts = append(parts, part)
			}
			if len(parts) == 0 {
				continue
			}
			content := geminiContent{Role: "user", Parts: parts}
			if segment.Role == PromptRoleAssistant {
				content.Role = "model"
			}
			if segment.Role == PromptRoleSafety {
				safety = append(safety, content)
			} else {
				prompt.Contents = append(prompt.Contents, content)
			}
		}
	}
	if options.ResultSchema != nil {
		schema, _ := json.Marshal(options.ResultSchema)
		if len(options.Tools) > 0 {
			prompt.System.Parts = append(prompt.System.Parts, geminiPart{Text: "When not calling tools, the output must be a JSON object using the following JSON Schema:\n" + string(schema)})
		} else {
			// Gemini structured output already carries the schema in generationConfig.
			// Keep prompt text sparse and avoid duplicating the full JSON schema.
			prompt.System.Parts = append(prompt.System.Parts, geminiPart{Text: "Fill all appropriate fields in the JSON output."})
		}
	}
	if len(prompt.System.Parts) == 0 {
		prompt.System = nil
	}
	// Safety messages are appended at the end. Segment boundaries are kept, as in the
	// TS driver: they are explicit-cache breakpoints and the caller's turn shape.
	// Split tool results are recombined when the request is built.
	prompt.Contents = append(prompt.Contents, safety...)
	return prompt, nil
}

// geminiFilePart turns a file into a part: a GCS URL is passed through as
// fileData (kept in storage), otherwise the bytes are inlined as base64
// inlineData within the request's inline budget.
func (d *VertexAIDriver) geminiFilePart(ctx context.Context, file DataSource, budget *geminiInlineBudget) (geminiPart, error) {
	media, err := geminiMediaPart(ctx, file, budget)
	if err != nil {
		return geminiPart{}, err
	}
	return geminiPart{InlineData: media.InlineData, FileData: media.FileData}, nil
}

// isVertexGCSURL reports whether a URI points at Google Cloud Storage and can be
// referenced by URI rather than re-encoded inline.
func isVertexGCSURL(uri string) bool {
	// Gemini Enterprise Agent Platform APIs accept GCS URIs directly. Avoid re-encoding those files so
	// large media can stay in Google Cloud Storage.
	return strings.HasPrefix(uri, "gs://") ||
		strings.HasPrefix(uri, "https://storage.googleapis.com/") ||
		strings.HasPrefix(uri, "https://storage.cloud.google.com/")
}

// formatGeminiFunctionResponse normalizes a tool result string into the object
// Gemini expects: a JSON object is passed through parsed, anything else is wrapped
// under an "output" key.
func formatGeminiFunctionResponse(response string) map[string]any {
	response = strings.TrimSpace(response)
	if response != "" && strings.HasPrefix(response, "{") && strings.HasSuffix(response, "}") {
		var parsed map[string]any
		if json.Unmarshal([]byte(response), &parsed) == nil {
			return parsed
		}
	}
	return map[string]any{"output": response}
}

func optionFloatValue(options map[string]any, key string) any {
	if v := optionFloat(options, key); v != nil {
		return *v
	}
	return nil
}

// Execute runs a single non-streaming completion against Gemini, Imagen, or a
// Claude partner model depending on the requested model.
func (d *VertexAIDriver) Execute(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (*ExecutionResponse, error) {
	return executeWithPrompt(ctx, vertexPromptDriver{d, "execute"}, segments, options, d.requestTextCompletion)
}

// Stream runs a streaming completion against Gemini or a Claude partner model
// (Imagen does not support streaming). A stream left unconsumed is released
// after DriverOptions.StreamStartTimeout.
func (d *VertexAIDriver) Stream(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (CompletionStream, error) {
	stream, err := streamWithPrompt(ctx, vertexPromptDriver{d, "stream"}, segments, options, d.requestTextCompletionStream)
	if err != nil {
		return nil, err
	}
	return leaseCompletionStream(stream, d.options.StreamStartTimeout), nil
}

// requestTextCompletion routes a non-streaming request by model to the Imagen,
// Claude, or Gemini handler, type-asserting the prompt produced by CreatePrompt.
func (d *VertexAIDriver) requestTextCompletion(ctx context.Context, prompt any, options ExecutionOptions) (*Completion, error) {
	if isVertexImagenModel(options.Model) {
		return d.requestVertexImagen(ctx, prompt.(vertexImagenPrompt), options)
	}
	if isClaudeModel(options.Model) {
		return d.requestVertexClaude(ctx, prompt.(internalclaude.Prompt), options)
	}
	return d.requestGemini(ctx, prompt.(geminiPrompt), options)
}

// requestTextCompletionStream opens an SSE stream. Claude partner models are
// handed to requestVertexClaudeStream; Gemini models use streamGenerateContent
// with alt=sse, chunks are decoded, and a finalizer rebuilds the conversation.
func (d *VertexAIDriver) requestTextCompletionStream(ctx context.Context, prompt any, options ExecutionOptions) (CompletionStream, error) {
	if isVertexImagenModel(options.Model) {
		return nil, errors.New("gemini enterprise agent platform imagen models do not support streaming")
	}
	if isClaudeModel(options.Model) {
		return d.requestVertexClaudeStream(ctx, prompt.(internalclaude.Prompt), options)
	}
	return d.requestGeminiStream(ctx, prompt.(geminiPrompt), options)
}

func optionIntValue(options map[string]any, key string) any {
	if v := optionInt(options, key); v != nil {
		return *v
	}
	return nil
}

func optionBoolPtrValue(options map[string]any, key string) any {
	if options == nil {
		return nil
	}
	if v, ok := options[key].(bool); ok {
		return v
	}
	return nil
}

// geminiPayload assembles the generateContent request body: sampling/limit options
// map into generationConfig, a ResultSchema without tools is sent via Gemini's
// structured-output transport (responseJsonSchema + JSON mime type) rather than the
// prompt, thinkingConfig is added only when the caller sets a thinking control,
// image models (Nano Banana included) enable IMAGE output with imageConfig and no
// tools, and tools add functionDeclarations with the requested calling mode.
// When tools are disabled but the history still has function parts, they are
// converted to text so the request stays valid without toolConfig.
func (d *VertexAIDriver) geminiPayload(prompt geminiPrompt, options ExecutionOptions) map[string]any {
	lowerModel := strings.ToLower(options.Model)
	isNanoBanana := strings.Contains(lowerModel, "gemini-nano-banana")
	isImageModel := isNanoBanana || strings.Contains(lowerModel, "image")
	config := map[string]any{}
	// Retain the single-candidate contract only where the model accepts it; caller
	// sampling options still pass through for provider validation.
	if isImageModel || !hasGeminiSamplingRestriction(options.Model) {
		config["candidateCount"] = 1
	}
	// Nano Banana does not accept sampling parameters.
	if !isNanoBanana {
		if v := optionFloat(options.ModelOptions, "temperature"); v != nil {
			config["temperature"] = *v
		}
		if v := optionFloat(options.ModelOptions, "top_p"); v != nil {
			config["topP"] = *v
		}
	}
	// Image generation requests omit the other generation controls.
	if !isImageModel {
		if v := optionInt(options.ModelOptions, "top_k"); v != nil {
			config["topK"] = *v
		}
		if v := optionFloat(options.ModelOptions, "presence_penalty"); v != nil {
			config["presencePenalty"] = *v
		}
		if v := optionFloat(options.ModelOptions, "frequency_penalty"); v != nil {
			config["frequencyPenalty"] = *v
		}
		if v := optionInt(options.ModelOptions, "seed"); v != nil {
			config["seed"] = *v
		}
	}
	if v := optionInt(options.ModelOptions, "max_tokens"); v != nil {
		config["maxOutputTokens"] = *v
	}
	if stops := optionStringSlice(options.ModelOptions, "stop_sequence"); len(stops) > 0 {
		config["stopSequences"] = stops
	}
	if options.ResultSchema != nil && len(options.Tools) == 0 && !isImageModel {
		// Use Gemini's structured-output transport instead of placing the schema in
		// the prompt; this avoids competing instructions and provider token bloat.
		config["responseMimeType"] = "application/json"
		config["responseJsonSchema"] = options.ResultSchema
	}
	if thinking := geminiThinkingConfig(options); thinking != nil {
		config["thinkingConfig"] = thinking
	}
	if isImageModel {
		// TEXT alone is an error and IMAGE alone returns blank responses.
		config["responseModalities"] = []string{"TEXT", "IMAGE"}
		imageConfig := removeNilMapValues(map[string]any{
			"aspectRatio":              optionString(options.ModelOptions, "image_aspect_ratio"),
			"imageSize":                optionString(options.ModelOptions, "image_size"),
			"personGeneration":         optionString(options.ModelOptions, "person_generation"),
			"prominentPeople":          optionString(options.ModelOptions, "prominent_people"),
			"outputMimeType":           optionString(options.ModelOptions, "output_mime_type"),
			"outputCompressionQuality": optionIntValue(options.ModelOptions, "output_compression_quality"),
		})
		if len(imageConfig) > 0 {
			config["imageConfig"] = imageConfig
		}
	}
	// Image generation requests declare no tools.
	hasTools := len(options.Tools) > 0 && !isImageModel
	contents := geminiRequestContents(prompt.Contents, hasTools)
	payload := map[string]any{
		"contents":         contents,
		"generationConfig": config,
	}
	if len(options.Labels) > 0 {
		payload["labels"] = options.Labels
	}
	if prompt.System != nil {
		payload["systemInstruction"] = prompt.System
	}
	if hasTools {
		// Gemini expects one tool entry that contains all function declarations.
		payload["tools"] = []map[string]any{{
			"functionDeclarations": geminiTools(options.Tools),
		}}
		payload["toolConfig"] = map[string]any{
			"functionCallingConfig": geminiFunctionCallingConfig(options.ModelOptions),
		}
	}
	return payload
}

// geminiTools converts neutral tool definitions into Gemini functionDeclarations.
func geminiTools(tools []ToolDefinition) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		// parametersJsonSchema is standard JSON Schema and is mutually exclusive
		// with the older provider-specific parameters field.
		out = append(out, map[string]any{
			"name":                 tool.Name,
			"description":          stringValue(tool.Description),
			"parametersJsonSchema": tool.InputSchema,
		})
	}
	return out
}

// geminiStoredConversation decodes a prior conversation in any of its stored
// forms: the geminiPrompt the driver returns (or its JSON object), a bare content
// array, or the TS client's wrapper ({"_arrayConversation": [...],
// "_llumiverse_system": {...}}). ok is false for a value that is none of these.
func geminiStoredConversation(conversation any) (out geminiPrompt, ok bool) {
	switch value := conversation.(type) {
	case nil:
		return out, true
	case geminiPrompt:
		return value, true
	case *geminiPrompt:
		if value != nil {
			out = *value
		}
		return out, true
	case []geminiContent:
		out.Contents = value
		return out, true
	case map[string]any:
		if wrapped, isWrapped := value["_arrayConversation"].([]any); isWrapped {
			if !decodeStoredConversation(wrapped, &out.Contents) {
				return out, false
			}
			if system, hasSystem := value["_llumiverse_system"].(map[string]any); hasSystem {
				var decoded geminiContent
				if decodeStoredConversation(system, &decoded) {
					out.System = &decoded
				}
			}
			return out, true
		}
		if _, hasContents := value["contents"]; !hasContents {
			return out, false
		}
		return out, decodeStoredConversation(value, &out)
	case []any:
		return out, decodeStoredConversation(value, &out.Contents)
	default:
		return out, false
	}
}

// geminiStoredContents returns the contents of a prior conversation, or nil when
// it has none or is not a Gemini conversation.
func geminiStoredContents(conversation any) []geminiContent {
	stored, _ := geminiStoredConversation(conversation)
	return stored.Contents
}

// geminiConversationInput appends the new prompt to a prior conversation, carrying
// over the system instruction. The stored conversation is copied, never modified.
func geminiConversationInput(conversation any, prompt geminiPrompt) geminiPrompt {
	out, _ := geminiStoredConversation(conversation)
	if prompt.System != nil {
		out.System = prompt.System
	}
	out.Contents = append(append(make([]geminiContent, 0, len(out.Contents)+len(prompt.Contents)), out.Contents...), prompt.Contents...)
	return out
}

func decodeStoredConversation[T any](conversation any, target *T) bool {
	data, err := json.Marshal(conversation)
	if err != nil {
		return false
	}
	return json.Unmarshal(data, target) == nil
}

// appendGeminiResponseToConversation appends the model's response turn to the
// conversation so the next request includes it.
func appendGeminiResponseToConversation(prompt geminiPrompt, response geminiResponse) geminiPrompt {
	if len(response.Candidates) == 0 {
		return prompt
	}
	content := response.Candidates[0].Content
	if len(content.Parts) > 0 {
		if content.Role == "" {
			content.Role = "model"
		}
		prompt.Contents = append(prompt.Contents, content)
	}
	return prompt
}

// geminiContentsContainToolParts reports whether any turn carries a functionCall
// or functionResponse part.
func geminiContentsContainToolParts(contents []geminiContent) bool {
	for _, content := range contents {
		for _, part := range content.Parts {
			if part.FunctionCall != nil || part.FunctionResponse != nil {
				return true
			}
		}
	}
	return false
}

// convertGeminiFunctionPartsToText rewrites functionCall/functionResponse parts as
// plain text placeholders for replaying history without tools enabled.
func convertGeminiFunctionPartsToText(contents []geminiContent) []geminiContent {
	// Gemini rejects function parts when no tools/toolConfig are present. Preserve
	// the history as plain text so old checkpoints remain usable with tool-free runs.
	out := make([]geminiContent, 0, len(contents))
	for _, content := range contents {
		next := content
		next.Parts = make([]geminiPart, 0, len(content.Parts))
		for _, part := range content.Parts {
			switch {
			case part.FunctionCall != nil:
				next.Parts = append(next.Parts, geminiPart{Text: fmt.Sprintf("[Tool call: %s(%s)]", part.FunctionCall.Name, truncateForConversation(toolInputString(part.FunctionCall.Args), 500))})
			case part.FunctionResponse != nil:
				next.Parts = append(next.Parts, geminiPart{Text: fmt.Sprintf("[Tool result for %s: %s]", part.FunctionResponse.Name, truncateForConversation(toolInputString(part.FunctionResponse.Response), 500))})
			default:
				next.Parts = append(next.Parts, part)
			}
		}
		out = append(out, next)
	}
	return out
}

// ListModels lists published Google and Anthropic models for the project/region,
// optionally filtering by name substring, and returns them sorted by ID.
func (d *VertexAIDriver) ListModels(ctx context.Context, params *ModelSearchPayload) ([]AIModel, error) {
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, err
	}
	endpoints := []struct {
		publisher string
		path      string
	}{
		{publisher: "google", path: "publishers/google/models"},
		{publisher: "anthropic", path: "publishers/anthropic/models"},
	}
	var models []AIModel
	for _, item := range endpoints {
		endpoint, err := d.vertexBaseEndpoint(item.path)
		if err != nil {
			return nil, err
		}
		var response struct {
			PublisherModels []struct {
				Name        string `json:"name"`
				DisplayName string `json:"displayName"`
				VersionID   string `json:"versionId"`
			} `json:"publisherModels"`
			Models []struct {
				Name        string `json:"name"`
				DisplayName string `json:"displayName"`
				VersionID   string `json:"versionId"`
			} `json:"models"`
		}
		if err := doJSON(ctx, d.client, http.MethodGet, endpoint, headers, nil, &response); err != nil {
			continue
		}
		for _, model := range append(response.PublisherModels, response.Models...) {
			if params != nil && stringValue(params.Text) != "" && !strings.Contains(strings.ToLower(model.Name), strings.ToLower(*params.Text)) {
				continue
			}
			// Only families with a request path are listed; embedding, Live and other
			// special-purpose models are left out.
			if !isListedVertexPublisherModel(item.publisher, model.Name) {
				continue
			}
			short := shortResourceName(model.Name)
			name := model.DisplayName
			if name == "" {
				name = short
			}
			listed := AIModel{
				ID:          model.Name,
				Name:        name,
				Provider:    string(ProviderVertexAI),
				Owner:       ptr(item.publisher),
				Type:        ptr(ModelType("text")),
				CanStream:   ptr(true),
				ToolSupport: ptr(true),
			}
			if item.publisher == "google" && isGeminiWithoutRegionalEndpoints(short) {
				// Models without regional endpoints are listed once, under global.
				listed.ID, listed.Name = "locations/global/"+model.Name, "Global "+short
				models = append(models, listed)
				continue
			}
			models = append(models, listed)
			if hasVertexGlobalListing(item.publisher, model.Name) {
				global := listed
				global.ID, global.Name = "locations/global/"+model.Name, "Global "+short
				models = append(models, global)
			}
		}
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	return models, nil
}

// ValidateConnection verifies that OAuth credentials can be resolved.
func (d *VertexAIDriver) ValidateConnection(ctx context.Context) error {
	_, err := d.authHeaders(ctx)
	return err
}

func (d *VertexAIDriver) httpClient(timeout HTTPTimeoutOptions) *http.Client {
	if !hasHTTPTimeout(timeout) {
		return d.client
	}
	return newHTTPClient(d.client, timeout)
}

// GenerateEmbeddings embeds text or media through Gemini Enterprise Agent Platform models.
func (d *VertexAIDriver) GenerateEmbeddings(ctx context.Context, options EmbeddingsOptions) (*EmbeddingsResult, error) {
	normalized, err := normalizeEmbeddingsOptions(options)
	if err != nil {
		return nil, err
	}
	model := normalized.Model
	if model == "" {
		model = defaultVertexEmbeddingModel
	}
	normalized.Model = model
	if shortResourceName(model) == vertexLegacyMultimodalEmbeddingModel {
		return d.generateVertexLegacyMultimodalEmbeddings(ctx, normalized, model)
	}
	if vertexEmbeddingUsesTextPredict(model) {
		return d.generateVertexTextPredictEmbeddings(ctx, normalized, model)
	}
	return d.generateVertexEmbedContentEmbeddings(ctx, normalized, model)
}

// generateVertexTextPredictEmbeddings embeds text through the publisher-model
// predict API used by Google's older text embedding models.
func (d *VertexAIDriver) generateVertexTextPredictEmbeddings(ctx context.Context, options EmbeddingsOptions, model string) (*EmbeddingsResult, error) {
	// The @google/genai TS driver exposes this as embedContent, but the Vertex
	// REST API for text-embedding-005/text-multilingual-embedding-002 and
	// gemini-embedding-001 is predict: {"instances":[{"content":...}]}.
	for _, input := range options.Inputs {
		if input.Type != embeddingInputText {
			return nil, fmt.Errorf("gemini enterprise agent platform text embedding model '%s' only supports text input", model)
		}
	}
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, d.formatError(err, model, "embeddings")
	}
	endpoint, err := d.vertexEndpoint(model, "predict")
	if err != nil {
		return nil, err
	}
	usage := &EmbeddingsTokenUsage{}
	usageSet := false
	items := make([]EmbeddingResultItem, len(options.Inputs))

	for _, batch := range vertexTextPredictBatches(options.Inputs, model) {
		instances := make([]map[string]any, 0, len(batch))
		for _, entry := range batch {
			input := entry.input
			instance := map[string]any{"content": input.Text}
			if taskType := vertexEmbeddingTaskType(input.TaskType); taskType != "" {
				instance["task_type"] = taskType
			}
			if input.Title != "" {
				instance["title"] = input.Title
			}
			instances = append(instances, instance)
		}
		payload := map[string]any{"instances": instances}
		if options.Dimensions > 0 {
			payload["parameters"] = map[string]any{"outputDimensionality": options.Dimensions}
		}

		var response vertexTextPredictResponse
		if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, payload, &response); err != nil {
			return nil, d.formatError(err, model, "embeddings")
		}
		if len(response.Predictions) != len(batch) {
			return nil, fmt.Errorf("gemini enterprise agent platform predict returned %d embeddings for %d inputs (model %s)", len(response.Predictions), len(batch), model)
		}
		for i, prediction := range response.Predictions {
			values := prediction.Embeddings.Values
			if len(values) == 0 {
				return nil, fmt.Errorf("gemini enterprise agent platform predict returned an empty embedding for input %d (model %s)", i, model)
			}
			tokenCount := prediction.Embeddings.Statistics.TokenCount
			items[batch[i].index] = EmbeddingResultItem{
				Outputs:     []EmbeddingOutput{{Values: values, Modality: embeddingModality(embeddingInputText)}},
				InputTokens: reportedTokens(tokenCount),
			}
			if tokenCount > 0 {
				usageSet = true
				addTokens(&usage.InputTokens, tokenCount)
				addTokens(&usage.InputTextTokens, tokenCount)
			}
		}
	}
	if !usageSet {
		usage = nil
	}
	return &EmbeddingsResult{Model: model, Results: items, Usage: usage}, nil
}

type vertexTextPredictBatchEntry struct {
	index int
	input EmbeddingInput
}

type vertexTextPredictResponse struct {
	Predictions []struct {
		Embeddings struct {
			Values     []float64 `json:"values"`
			Statistics struct {
				TokenCount int `json:"token_count"`
			} `json:"statistics"`
		} `json:"embeddings"`
	} `json:"predictions"`
}

// vertexTextPredictBatches keeps gemini-embedding-001 to one input per request;
// the other text embedding predict models accept small batches.
func vertexTextPredictBatches(inputs []EmbeddingInput, model string) [][]vertexTextPredictBatchEntry {
	entries := make([]vertexTextPredictBatchEntry, 0, len(inputs))
	for i, input := range inputs {
		entries = append(entries, vertexTextPredictBatchEntry{index: i, input: input})
	}
	if shortResourceName(model) != "gemini-embedding-001" {
		return [][]vertexTextPredictBatchEntry{entries}
	}
	batches := make([][]vertexTextPredictBatchEntry, 0, len(entries))
	for _, entry := range entries {
		batches = append(batches, []vertexTextPredictBatchEntry{entry})
	}
	return batches
}

// generateVertexEmbedContentEmbeddings embeds inputs via the modern embedContent API.
func (d *VertexAIDriver) generateVertexEmbedContentEmbeddings(ctx context.Context, options EmbeddingsOptions, model string) (*EmbeddingsResult, error) {
	// Modern Gemini Enterprise Agent Platform embedding models use embedContent.
	// The REST API accepts one content per call, so issue one request per input
	// and preserve input order. Text uses a Content text part with task/title in
	// config or prefix; media uses inlineData or fileData.
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, d.formatError(err, model, "embeddings")
	}
	endpoint, err := d.vertexEndpointForRegion(model, "embedContent", vertexEmbeddingRegion(model))
	if err != nil {
		return nil, err
	}
	items := make([]EmbeddingResultItem, 0, len(options.Inputs))
	usage := &EmbeddingsTokenUsage{}
	usageSet := false
	viaPrefix := vertexEmbeddingUsesTaskPrefix(model)
	for _, input := range options.Inputs {
		payload, err := vertexEmbedContentPayload(ctx, input, options, viaPrefix)
		if err != nil {
			return nil, err
		}
		var response vertexEmbedContentResponse
		if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, payload, &response); err != nil {
			return nil, d.formatError(err, model, "embeddings")
		}
		if len(response.Embedding.Values) == 0 {
			return nil, fmt.Errorf("gemini enterprise agent platform embedContent returned an empty embedding for input %d (model %s)", len(items), model)
		}
		tokenCount := response.UsageMetadata.PromptTokenCount
		items = append(items, EmbeddingResultItem{
			Outputs:     []EmbeddingOutput{{Values: response.Embedding.Values, Modality: embeddingModality(input.Type)}},
			InputTokens: reportedTokens(tokenCount),
		})
		if tokenCount > 0 {
			usageSet = true
			addVertexEmbeddingUsage(usage, input.Type, response.UsageMetadata)
		}
	}
	if !usageSet {
		usage = nil
	}
	return &EmbeddingsResult{Model: model, Results: items, Usage: usage}, nil
}

// vertexEmbedContentResponse is the embedContent response: one embedding vector and
// token usage.
type vertexEmbedContentResponse struct {
	Embedding struct {
		Values []float64 `json:"values"`
	} `json:"embedding"`
	UsageMetadata vertexEmbeddingUsageMetadata `json:"usageMetadata"`
}

// vertexEmbeddingUsageMetadata carries the aggregate prompt token count and, on
// newer models, a per-modality breakdown.
type vertexEmbeddingUsageMetadata struct {
	PromptTokenCount   int `json:"promptTokenCount"`
	PromptTokensDetail []struct {
		Modality   string `json:"modality"`
		TokenCount int    `json:"tokenCount"`
	} `json:"promptTokensDetails"`
}

// vertexEmbedContentPayload builds one embedContent request body for an input,
// adding taskType/title and outputDimensionality as appropriate.
func vertexEmbedContentPayload(ctx context.Context, input EmbeddingInput, options EmbeddingsOptions, viaPrefix bool) (map[string]any, error) {
	// Some Gemini embedding models reject taskType as an API field; those use a
	// documented text prefix instead. Other models receive taskType normally.
	content, err := vertexEmbeddingContent(ctx, input, viaPrefix)
	if err != nil {
		return nil, err
	}
	payload := map[string]any{"content": content}
	if input.Type == embeddingInputText {
		if !viaPrefix {
			if taskType := vertexEmbeddingTaskType(input.TaskType); taskType != "" {
				payload["taskType"] = taskType
			}
		}
		if input.Title != "" {
			payload["title"] = input.Title
		}
	}
	if options.Dimensions > 0 {
		payload["outputDimensionality"] = options.Dimensions
	}
	return payload, nil
}

// vertexEmbeddingContent builds the Content object for an input: a text part
// (optionally task-prefixed) for text, or a media data part for other modalities.
func vertexEmbeddingContent(ctx context.Context, input EmbeddingInput, viaPrefix bool) (map[string]any, error) {
	if input.Type == embeddingInputText {
		text := input.Text
		if viaPrefix {
			text = vertexEmbeddingPrefixText(input)
		}
		return map[string]any{
			"role":  "user",
			"parts": []map[string]any{{"text": text}},
		}, nil
	}
	if input.Source == nil {
		return nil, fmt.Errorf("gemini enterprise agent platform embeddings require a source for '%s' input", input.Type)
	}
	part, err := vertexEmbeddingDataPart(ctx, input.Source)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"role":  "user",
		"parts": []map[string]any{part},
	}, nil
}

// vertexEmbeddingDataPart turns a media source into an embedContent part.
func vertexEmbeddingDataPart(ctx context.Context, source DataSource) (map[string]any, error) {
	// Prefer fileData for GCS objects; inlineData is the portable fallback.
	if uri, err := source.URL(ctx); err == nil && isVertexGCSURL(uri) {
		return map[string]any{"fileData": map[string]any{"fileUri": uri, "mimeType": source.MIMEType()}}, nil
	}
	encoded, err := dataSourceToBase64(ctx, source)
	if err != nil {
		return nil, err
	}
	return map[string]any{"inlineData": map[string]any{"data": encoded, "mimeType": source.MIMEType()}}, nil
}

// vertexEmbeddingPrefixText builds the task-prefixed text for models that signal
// task type via a documented prefix rather than the taskType field.
func vertexEmbeddingPrefixText(input EmbeddingInput) string {
	// gemini-embedding-2 uses documented task prefixes instead of taskType:
	// query -> "task: search result | query: {text}" and document ->
	// "title: {title} | text: {text}", with "none" for missing titles.
	switch input.TaskType {
	case EmbeddingTaskQuery:
		return "task: search result | query: " + input.Text
	case EmbeddingTaskDocument:
		title := input.Title
		if title == "" {
			title = "none"
		}
		return "title: " + title + " | text: " + input.Text
	default:
		return input.Text
	}
}

// vertexEmbeddingTaskType maps a neutral task type to Google's taskType enum.
func vertexEmbeddingTaskType(taskType EmbeddingTaskType) string {
	// Older Gemini embedding models still accept Google's taskType enum.
	switch taskType {
	case EmbeddingTaskQuery:
		return "RETRIEVAL_QUERY"
	case EmbeddingTaskDocument:
		return "RETRIEVAL_DOCUMENT"
	default:
		return ""
	}
}

// vertexEmbeddingUsesTaskPrefix reports whether the model signals task type via a
// text prefix instead of the taskType field.
func vertexEmbeddingUsesTaskPrefix(model string) bool {
	return shortResourceName(model) == "gemini-embedding-2"
}

// vertexEmbeddingUsesTextPredict reports whether a text embedding model uses the
// Vertex publisher-model predict API instead of embedContent.
func vertexEmbeddingUsesTextPredict(model string) bool {
	model = strings.ToLower(shortResourceName(model))
	return model == "gemini-embedding-001" ||
		model == "embedding-001" ||
		strings.HasPrefix(model, "text-embedding-") ||
		strings.HasPrefix(model, "text-multilingual-embedding-")
}

// vertexEmbeddingRegion returns the region override for a model (global-only for
// gemini-embedding-2), or "" to use the driver's configured region.
func vertexEmbeddingRegion(model string) string {
	// gemini-embedding-2 is global-only on Gemini Enterprise Agent Platform.
	if vertexEmbeddingUsesTaskPrefix(model) {
		return "global"
	}
	return ""
}

// addVertexEmbeddingUsage accumulates per-input token usage into the running total,
// splitting by modality when details are present or by the input type otherwise.
func addVertexEmbeddingUsage(usage *EmbeddingsTokenUsage, inputType string, metadata vertexEmbeddingUsageMetadata) {
	// Gemini Enterprise Agent Platform reports aggregate prompt tokens and, on newer models, modality
	// details. Fall back to the input modality when details are absent.
	addTokens(&usage.InputTokens, metadata.PromptTokenCount)
	if len(metadata.PromptTokensDetail) == 0 {
		switch inputType {
		case embeddingInputImage:
			addTokens(&usage.InputImageTokens, metadata.PromptTokenCount)
		default:
			addTokens(&usage.InputTextTokens, metadata.PromptTokenCount)
		}
		return
	}
	for _, detail := range metadata.PromptTokensDetail {
		switch strings.ToUpper(detail.Modality) {
		case "IMAGE":
			addTokens(&usage.InputImageTokens, detail.TokenCount)
		default:
			addTokens(&usage.InputTextTokens, detail.TokenCount)
		}
	}
}

// generateVertexLegacyMultimodalEmbeddings embeds inputs via the legacy predict API
// used by multimodalembedding@001.
func (d *VertexAIDriver) generateVertexLegacyMultimodalEmbeddings(ctx context.Context, options EmbeddingsOptions, model string) (*EmbeddingsResult, error) {
	// multimodalembedding@001 predates embedContent and returns different
	// fields for text, image, and segmented video/audio predictions. Predict
	// returns one prediction per input, while video/audio predictions may contain
	// multiple segment embeddings inside that prediction.
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return nil, d.formatError(err, model, "embeddings")
	}
	endpoint, err := d.vertexEndpoint(model, "predict")
	if err != nil {
		return nil, err
	}
	instances := make([]map[string]any, 0, len(options.Inputs))
	modalities := make([]string, 0, len(options.Inputs))
	for _, input := range options.Inputs {
		instance, modality, err := vertexLegacyMultimodalInstance(ctx, input)
		if err != nil {
			return nil, err
		}
		instances = append(instances, instance)
		modalities = append(modalities, modality)
	}
	payload := map[string]any{"instances": instances}
	if options.Dimensions > 0 {
		payload["parameters"] = map[string]any{"dimension": options.Dimensions}
	}
	var response struct {
		Predictions []vertexLegacyMultimodalPrediction `json:"predictions"`
	}
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, payload, &response); err != nil {
		return nil, d.formatError(err, model, "embeddings")
	}
	if len(response.Predictions) != len(instances) {
		return nil, fmt.Errorf("gemini enterprise agent platform predict returned %d predictions for %d instances (model %s)", len(response.Predictions), len(instances), model)
	}
	items := make([]EmbeddingResultItem, 0, len(response.Predictions))
	for i, prediction := range response.Predictions {
		outputs, err := vertexLegacyMultimodalOutputs(prediction, modalities[i])
		if err != nil {
			return nil, err
		}
		items = append(items, EmbeddingResultItem{Outputs: outputs})
	}
	return &EmbeddingsResult{Model: model, Results: items}, nil
}

// vertexLegacyMultimodalPrediction is one prediction from the legacy predict API;
// the field set varies by input modality (text, image, or segmented video/audio).
type vertexLegacyMultimodalPrediction struct {
	TextEmbedding   []float64                    `json:"textEmbedding"`
	ImageEmbedding  []float64                    `json:"imageEmbedding"`
	VideoEmbeddings []vertexLegacyVideoEmbedding `json:"videoEmbeddings"`
}

// vertexLegacyVideoEmbedding is one time-segmented embedding for video/audio input.
type vertexLegacyVideoEmbedding struct {
	Embedding      []float64 `json:"embedding"`
	StartOffsetSec *float64  `json:"startOffsetSec"`
	EndOffsetSec   *float64  `json:"endOffsetSec"`
}

// vertexLegacyMultimodalInstance builds one predict instance for an input and
// returns the modality label to associate with its prediction.
func vertexLegacyMultimodalInstance(ctx context.Context, input EmbeddingInput) (map[string]any, string, error) {
	// The legacy API represents audio inputs as video instances but callers
	// should still receive audio modality labels in the normalized output.
	switch input.Type {
	case embeddingInputText:
		return map[string]any{"text": input.Text}, embeddingInputText, nil
	case embeddingInputImage:
		image, err := vertexLegacyImagePart(ctx, input.Source)
		if err != nil {
			return nil, "", err
		}
		return map[string]any{"image": image}, embeddingInputImage, nil
	case embeddingInputVideo:
		video, err := vertexLegacyVideoPart(ctx, input)
		if err != nil {
			return nil, "", err
		}
		return map[string]any{"video": video}, embeddingInputVideo, nil
	case embeddingInputAudio:
		video, err := vertexLegacyVideoPart(ctx, input)
		if err != nil {
			return nil, "", err
		}
		return map[string]any{"video": video}, embeddingInputAudio, nil
	default:
		return nil, "", fmt.Errorf("gemini enterprise agent platform multimodal embeddings do not support '%s' input", input.Type)
	}
}

// vertexLegacyImagePart builds the image instance field, using gcsUri for GCS
// objects and base64 bytes otherwise.
func vertexLegacyImagePart(ctx context.Context, source DataSource) (map[string]any, error) {
	if source == nil {
		return nil, errors.New("gemini enterprise agent platform image embeddings require a source")
	}
	if uri, err := source.URL(ctx); err == nil && isVertexGCSURL(uri) {
		return map[string]any{"gcsUri": uri, "mimeType": source.MIMEType()}, nil
	}
	encoded, err := dataSourceToBase64(ctx, source)
	if err != nil {
		return nil, err
	}
	return map[string]any{"bytesBase64Encoded": encoded, "mimeType": source.MIMEType()}, nil
}

// vertexLegacyVideoPart builds the video instance field (also used for audio),
// including the segment config; offsets use endOffsetSec rather than lengthSec.
func vertexLegacyVideoPart(ctx context.Context, input EmbeddingInput) (map[string]any, error) {
	if input.Source == nil {
		return nil, fmt.Errorf("gemini enterprise agent platform %s embeddings require a source", input.Type)
	}
	part := map[string]any{}
	if uri, err := input.Source.URL(ctx); err == nil && isVertexGCSURL(uri) {
		part["gcsUri"] = uri
	} else {
		encoded, err := dataSourceToBase64(ctx, input.Source)
		if err != nil {
			return nil, err
		}
		part["bytesBase64Encoded"] = encoded
	}
	segmentConfig := map[string]any{}
	// Gemini Enterprise Agent Platform expects endOffsetSec rather than lengthSec.
	if input.StartSec != nil {
		segmentConfig["startOffsetSec"] = *input.StartSec
	}
	if input.StartSec != nil && input.LengthSec != nil {
		segmentConfig["endOffsetSec"] = *input.StartSec + *input.LengthSec
	}
	if input.IntervalSec != nil {
		segmentConfig["intervalSec"] = *input.IntervalSec
	}
	if len(segmentConfig) > 0 {
		part["videoSegmentConfig"] = segmentConfig
	}
	return part, nil
}

// vertexLegacyMultimodalOutputs extracts embedding outputs from a prediction,
// preferring the expected modality and falling back to whatever field is populated.
func vertexLegacyMultimodalOutputs(prediction vertexLegacyMultimodalPrediction, modality string) ([]EmbeddingOutput, error) {
	// Prefer the expected modality but fall back to whichever embedding field the
	// legacy API returned, matching the TS driver's tolerant normalization.
	switch {
	case modality == embeddingInputText && len(prediction.TextEmbedding) > 0:
		return []EmbeddingOutput{{Values: prediction.TextEmbedding, Modality: embeddingModality(embeddingInputText)}}, nil
	case modality == embeddingInputImage && len(prediction.ImageEmbedding) > 0:
		return []EmbeddingOutput{{Values: prediction.ImageEmbedding, Modality: embeddingModality(embeddingInputImage)}}, nil
	case (modality == embeddingInputVideo || modality == embeddingInputAudio) && len(prediction.VideoEmbeddings) > 0:
		outputs := make([]EmbeddingOutput, 0, len(prediction.VideoEmbeddings))
		for _, segment := range prediction.VideoEmbeddings {
			outputs = append(outputs, EmbeddingOutput{
				Values:   segment.Embedding,
				Modality: embeddingModality(modality),
				StartSec: segment.StartOffsetSec,
				EndSec:   segment.EndOffsetSec,
			})
		}
		return outputs, nil
	case len(prediction.TextEmbedding) > 0:
		return []EmbeddingOutput{{Values: prediction.TextEmbedding, Modality: embeddingModality(embeddingInputText)}}, nil
	case len(prediction.ImageEmbedding) > 0:
		return []EmbeddingOutput{{Values: prediction.ImageEmbedding, Modality: embeddingModality(embeddingInputImage)}}, nil
	case len(prediction.VideoEmbeddings) > 0:
		outputs := make([]EmbeddingOutput, 0, len(prediction.VideoEmbeddings))
		for _, segment := range prediction.VideoEmbeddings {
			outputs = append(outputs, EmbeddingOutput{
				Values:   segment.Embedding,
				Modality: embeddingModality(embeddingInputVideo),
				StartSec: segment.StartOffsetSec,
				EndSec:   segment.EndOffsetSec,
			})
		}
		return outputs, nil
	default:
		return nil, errors.New("gemini enterprise agent platform multimodal prediction returned no embedding values")
	}
}

// vertexEndpoint builds the full URL for a model action (e.g. ":generateContent")
// in the driver's configured region.
func (d *VertexAIDriver) vertexEndpoint(model string, method string) (string, error) {
	base, modelPath := d.vertexModelPath(model)
	return joinEndpoint(base, modelPath+":"+method)
}

// vertexBaseEndpoint builds a project/location URL with the given path elements in
// the driver's configured region.
func (d *VertexAIDriver) vertexBaseEndpoint(elems ...string) (string, error) {
	return d.vertexBaseEndpointForRegion(d.options.Region, elems...)
}

// vertexEndpointForRegion is vertexEndpoint with an explicit region override.
func (d *VertexAIDriver) vertexEndpointForRegion(model string, method string, region string) (string, error) {
	// Some models are location-specific while gemini-embedding-2 is global.
	// Keep the region override local to endpoint construction.
	base, modelPath := d.vertexModelPathForRegion(model, region)
	return joinEndpoint(base, modelPath+":"+method)
}

// vertexBaseEndpointForRegion builds projects/{project}/locations/{region}/... on
// the configured or region-default base URL.
func (d *VertexAIDriver) vertexBaseEndpointForRegion(region string, elems ...string) (string, error) {
	if region == "" {
		region = d.options.Region
	}
	base := d.options.BaseURL
	if base == "" {
		base = vertexDefaultBaseURL(region)
	}
	parts := []string{"projects", d.options.Project, "locations", region}
	parts = append(parts, elems...)
	return joinEndpoint(base, parts...)
}

// vertexModelPath returns the base URL and model resource path for the driver's
// configured region.
func (d *VertexAIDriver) vertexModelPath(model string) (base string, modelPath string) {
	return d.vertexModelPathForRegion(model, "")
}

// vertexModelPathForRegion resolves the base URL and publisher model path for a
// model. A "locations/..." prefix overrides the region; a bare model name is
// expanded to publishers/{google|anthropic}/models/{name}, choosing the anthropic
// publisher for Claude models.
func (d *VertexAIDriver) vertexModelPathForRegion(model string, region string) (base string, modelPath string) {
	if region == "" {
		region = d.options.Region
	}
	modelPath = model
	if strings.HasPrefix(model, "locations/") {
		parts := strings.Split(model, "/")
		if len(parts) >= 2 {
			region = parts[1]
			modelPath = strings.Join(parts[2:], "/")
		}
	}
	if !strings.Contains(modelPath, "publishers/") {
		publisher := "google"
		if isClaudeModel(modelPath) {
			publisher = "anthropic"
		}
		modelPath = "publishers/" + publisher + "/models/" + shortResourceName(modelPath)
	}
	base = d.options.BaseURL
	if base == "" {
		base = vertexDefaultBaseURL(region)
	}
	base, _ = joinEndpoint(base, "projects", d.options.Project, "locations", region)
	return base, modelPath
}

// vertexDefaultBaseURL returns the AI Platform base URL for a region, using the
// global host for region "global" and the regional host otherwise.
func vertexDefaultBaseURL(region string) string {
	if region == "global" {
		return "https://aiplatform.googleapis.com/v1"
	}
	return "https://" + region + "-aiplatform.googleapis.com/v1"
}

// authHeaders builds the Bearer Authorization header from a fresh OAuth2 token.
func (d *VertexAIDriver) authHeaders(ctx context.Context) (map[string]string, error) {
	ts, err := d.tokenSource(ctx)
	if err != nil {
		return nil, err
	}
	token, err := ts.Token()
	if err != nil {
		return nil, err
	}
	return map[string]string{"Authorization": "Bearer " + token.AccessToken}, nil
}

// tokenSource returns the configured token source, falling back to Application
// Default Credentials scoped to cloud-platform.
func (d *VertexAIDriver) tokenSource(ctx context.Context) (oauth2.TokenSource, error) {
	if d.options.TokenSource != nil {
		return d.options.TokenSource, nil
	}
	creds, err := google.FindDefaultCredentials(ctx, vertexScopeCloudPlatform)
	if err != nil {
		return nil, err
	}
	return creds.TokenSource, nil
}

// formatError wraps a provider error in a LlumiverseError. Gemini, Imagen and
// embedding models use the Gemini classification (formatGeminiError), as the TS
// driver routes every non-Claude model to the Gemini error handler. Claude partner
// models keep the status-based classification, passing errors without an HTTP
// status through. Either way a 403 naming Cloud Storage gets the bucket hint.
func (d *VertexAIDriver) formatError(err error, model string, operation string) error {
	if !isClaudeModel(model) {
		return formatGeminiError(err, model, operation)
	}
	code, _ := errorStatusAndName(err)
	if code != 0 {
		return withVertexStorageHint(newLlumiverseError(err.Error(), retryableFromStatusAndMessage(code, err.Error()), LlumiverseErrorContext{
			Provider:  ProviderVertexAI,
			Model:     model,
			Operation: operation,
		}, err, code, vertexErrorName))
	}
	return err
}

// isVertexImagenModel reports whether a model ID refers to an Imagen image model.
func isVertexImagenModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "imagen")
}

// isGeminiVersionGTE reports whether the Gemini version embedded in the model ID is
// greater than or equal to the given major.minor version.
func isGeminiVersionGTE(model string, version string) bool {
	model = strings.ToLower(model)
	idx := strings.Index(model, "gemini-")
	if idx == -1 {
		return false
	}
	tail := model[idx+len("gemini-"):]
	var major, minor int
	if _, err := fmt.Sscanf(tail, "%d.%d", &major, &minor); err != nil {
		if _, err := fmt.Sscanf(tail, "%d", &major); err != nil {
			return false
		}
	}
	var targetMajor, targetMinor int
	if _, err := fmt.Sscanf(version, "%d.%d", &targetMajor, &targetMinor); err != nil {
		if _, err := fmt.Sscanf(version, "%d", &targetMajor); err != nil {
			return false
		}
	}
	if major != targetMajor {
		return major > targetMajor
	}
	return minor >= targetMinor
}

// withShortModel returns a copy of options with Model reduced to its short name.
func withShortModel(options ExecutionOptions) ExecutionOptions {
	options.Model = shortResourceName(options.Model)
	return options
}
