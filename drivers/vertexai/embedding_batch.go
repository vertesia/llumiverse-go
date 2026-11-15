package vertexai

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"

	"github.com/vertesia/llumiverse-go/core"
)

// Batch embeddings run as Vertex batchPredictionJobs over JSONL files in Cloud
// Storage. Gemini embedding models read "gemini" rows (an embedContent request
// per row, keyed); text-embedding models read "legacy" rows (the predict
// instance), whose output echoes only the model input, so their results are
// correlated by a hash of that input.

var _ core.EmbeddingBatchAdapter = (*VertexAIDriver)(nil)

const (
	vertexEmbeddingBatchSchemaGemini = "gemini"
	vertexEmbeddingBatchSchemaLegacy = "legacy"
	vertexEmbeddingBatchOperation    = "embedding_batch"
)

// vertexTextEmbeddingVersion reads the version of a text-embedding-NNN model.
var vertexTextEmbeddingVersion = regexp.MustCompile(`^text-embedding-(\d{3})$`)

// vertexEmbeddingBatchProfile is how one model runs batch embeddings.
type vertexEmbeddingBatchProfile struct {
	model     string
	schema    string
	maxRows   int
	global    bool
	viaPrefix bool
	images    bool
}

// vertexEmbeddingBatchProfileFor returns the batch profile of model, by family:
// Gemini embedding models read Gemini rows (the multimodal, global, task-prefixed
// generation also takes images), and text-embedding-004 and later read legacy
// rows. Other models, multimodalembedding included, cannot batch.
func vertexEmbeddingBatchProfileFor(model string, modality core.EmbeddingBatchModality) (vertexEmbeddingBatchProfile, bool) {
	short := shortResourceName(model)
	var profile vertexEmbeddingBatchProfile
	switch {
	case strings.HasPrefix(short, "gemini-embedding-"):
		multimodal := vertexEmbeddingUsesTaskPrefix(short)
		profile = vertexEmbeddingBatchProfile{model: short, schema: vertexEmbeddingBatchSchemaGemini, maxRows: 1_000_000, global: multimodal, viaPrefix: multimodal, images: multimodal}
	case vertexTextEmbeddingVersion.MatchString(short):
		version, _ := strconv.Atoi(vertexTextEmbeddingVersion.FindStringSubmatch(short)[1])
		if version < 4 {
			return vertexEmbeddingBatchProfile{}, false
		}
		profile = vertexEmbeddingBatchProfile{model: short, schema: vertexEmbeddingBatchSchemaLegacy, maxRows: 30_000}
	default:
		return vertexEmbeddingBatchProfile{}, false
	}
	switch modality {
	case core.EmbeddingBatchModalityText:
		return profile, true
	case core.EmbeddingBatchModalityImage:
		return profile, profile.images
	default:
		return vertexEmbeddingBatchProfile{}, false
	}
}

func requireVertexEmbeddingBatchProfile(model string, modality core.EmbeddingBatchModality) (vertexEmbeddingBatchProfile, error) {
	profile, ok := vertexEmbeddingBatchProfileFor(model, modality)
	if !ok {
		return profile, fmt.Errorf("vertex embedding model %s is not batch capable", model)
	}
	return profile, nil
}

// EmbeddingBatchCapability reports how model batches inputs of modality: the row
// format, the row limit, and the location, which is global for models served
// only there and location otherwise.
func (d *VertexAIDriver) EmbeddingBatchCapability(model string, modality core.EmbeddingBatchModality, location string) (core.EmbeddingBatchCapability, bool) {
	profile, ok := vertexEmbeddingBatchProfileFor(model, modality)
	if !ok {
		return core.EmbeddingBatchCapability{}, false
	}
	if profile.global {
		location = "global"
	}
	return core.EmbeddingBatchCapability{Model: profile.model, InputFormat: profile.schema, MaxRows: profile.maxRows, Location: location}, true
}

// FormatEmbeddingBatchRow formats one input as a JSONL row. Images must be Cloud
// Storage objects, referenced in place and never read.
func (d *VertexAIDriver) FormatEmbeddingBatchRow(ctx context.Context, options core.EmbeddingBatchRowOptions) (map[string]any, string, error) {
	if options.InputFormat != vertexEmbeddingBatchSchemaGemini && options.InputFormat != vertexEmbeddingBatchSchemaLegacy {
		return nil, "", fmt.Errorf("unsupported Vertex embedding batch input format %s", options.InputFormat)
	}
	row, err := formatVertexEmbeddingBatchRow(ctx, options)
	if err != nil {
		return nil, "", err
	}
	key, ok := vertexEmbeddingBatchCorrelationKey(row)
	if !ok {
		return nil, "", errors.New("embedding batch input cannot be correlated with its result")
	}
	return row, key, nil
}

func formatVertexEmbeddingBatchRow(ctx context.Context, options core.EmbeddingBatchRowOptions) (map[string]any, error) {
	input := options.Input
	if input.Type == "" {
		input.Type = embeddingInputText
	}
	modality := core.EmbeddingBatchModalityImage
	if input.Type == embeddingInputText {
		modality = core.EmbeddingBatchModalityText
	}
	profile, ok := vertexEmbeddingBatchProfileFor(options.Model, modality)
	if !ok {
		return nil, fmt.Errorf("vertex embedding model %s does not support batch %s inputs", options.Model, modality)
	}
	if options.InputFormat != "" && options.InputFormat != profile.schema {
		return nil, fmt.Errorf("vertex embedding model %s uses %s batch rows, not %s", options.Model, profile.schema, options.InputFormat)
	}
	if input.Type != embeddingInputText {
		if input.Source == nil {
			return nil, fmt.Errorf("vertex embedding batch %s inputs require a source", input.Type)
		}
		// Batch media is referenced in place: never inline or download the source.
		if uri, err := input.Source.URL(ctx); err != nil || !strings.HasPrefix(uri, "gs://") {
			return nil, errors.New("vertex embedding batch images require a gs:// source URI")
		}
	}
	if profile.schema == vertexEmbeddingBatchSchemaLegacy {
		row := map[string]any{"key": options.Key, "content": input.Text}
		if taskType := vertexEmbeddingTaskType(input.TaskType); taskType != "" {
			row["task_type"] = taskType
		}
		if input.Title != "" {
			row["title"] = input.Title
		}
		return row, nil
	}
	content, err := vertexEmbeddingContent(ctx, input, profile.viaPrefix)
	if err != nil {
		return nil, err
	}
	config := map[string]any{"output_dimensionality": options.Dimensions}
	if input.Type == embeddingInputText && !profile.viaPrefix {
		if taskType := vertexEmbeddingTaskType(input.TaskType); taskType != "" {
			config["task_type"] = taskType
		}
		if input.Title != "" {
			config["title"] = input.Title
		}
	}
	return map[string]any{
		"key": options.Key,
		"request": map[string]any{
			"content":              map[string]any{"parts": content["parts"]},
			"embed_content_config": config,
		},
	}, nil
}

// vertexEmbeddingBatchCorrelationKey returns the key a row's result carries.
// Legacy output echoes only the model input fields, dropping custom keys, so a
// legacy row is identified by a hash of its content, task type and title.
func vertexEmbeddingBatchCorrelationKey(row map[string]any) (string, bool) {
	content, isLegacy := row["content"].(string)
	if !isLegacy {
		key, ok := row["key"].(string)
		return key, ok
	}
	fields := []any{content, nil, nil}
	for i, name := range []string{"task_type", "title"} {
		value, present := row[name]
		if !present || value == nil {
			continue
		}
		text, ok := value.(string)
		if !ok {
			return "", false
		}
		fields[i+1] = text
	}
	sum := sha256.Sum256(jsStringify(fields))
	return "input:" + hex.EncodeToString(sum[:]), true
}

// jsStringify encodes value as JavaScript's JSON.stringify does for strings and
// nulls, so a key derived here matches one derived by the TS client.
func jsStringify(value any) []byte {
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(value)
	out := bytes.TrimSuffix(buf.Bytes(), []byte("\n"))
	out = bytes.ReplaceAll(out, []byte("\\u2028"), []byte("\u2028"))
	return bytes.ReplaceAll(out, []byte("\\u2029"), []byte("\u2029"))
}

// ParseEmbeddingBatchResult normalizes a Gemini or legacy output row, keeping
// neither input content nor provider error messages.
func (d *VertexAIDriver) ParseEmbeddingBatchResult(record map[string]any) core.ParsedEmbeddingBatchResult {
	// Gemini JSONL encodes google.rpc.Status as a JSON string; legacy rows may use an object.
	status := record["status"]
	if text, ok := status.(string); ok && strings.TrimSpace(text) != "" {
		var parsed any
		if json.Unmarshal([]byte(text), &parsed) == nil {
			status = parsed
		} else {
			status = map[string]any{"code": float64(-1)}
		}
	}
	var code any
	if object, ok := status.(map[string]any); ok {
		code = object["code"]
	}
	statusError := code != nil && code != float64(0)
	response, _ := record["response"].(map[string]any)
	var embeddings any
	if response != nil {
		embeddings = response["embedding"]
		if embeddings == nil {
			embeddings = response["embeddings"]
		}
	}
	if embeddings == nil {
		if predictions, ok := record["predictions"].([]any); ok && len(predictions) > 0 {
			if prediction, ok := predictions[0].(map[string]any); ok {
				embeddings = prediction["embeddings"]
			}
		}
	}
	if list, ok := embeddings.([]any); ok {
		embeddings = nil
		if len(list) > 0 {
			embeddings = list[0]
		}
	}
	var vector []float64
	if object, ok := embeddings.(map[string]any); ok {
		if values, ok := object["values"].([]any); ok {
			vector = make([]float64, 0, len(values))
			for _, value := range values {
				number, _ := value.(float64)
				vector = append(vector, number)
			}
		}
	}
	result := core.ParsedEmbeddingBatchResult{
		Vector:        vector,
		ProviderError: statusError || record["error"] != nil || (response != nil && response["error"] != nil),
	}
	if key, ok := record["key"].(string); ok {
		result.Key = key
	} else if record["key"] == nil {
		if instance, ok := record["instance"].(map[string]any); ok {
			result.Key, _ = vertexEmbeddingBatchCorrelationKey(instance)
		}
	}
	if statusError {
		result.FailureCategory = "provider_row_error"
		if code == float64(7) {
			result.FailureCategory = "provider_permission_denied"
		}
	}
	return result
}

// vertexBatchJob is the batchPredictionJobs resource fields the driver reads.
type vertexBatchJob struct {
	Name        string `json:"name"`
	DisplayName string `json:"displayName"`
	State       string `json:"state"`
	Model       string `json:"model"`
	InputConfig struct {
		GCSSource struct {
			URIs []string `json:"uris"`
		} `json:"gcsSource"`
	} `json:"inputConfig"`
	OutputConfig struct {
		GCSDestination struct {
			OutputURIPrefix string `json:"outputUriPrefix"`
		} `json:"gcsDestination"`
	} `json:"outputConfig"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (j vertexBatchJob) normalize() (*core.EmbeddingBatchJob, error) {
	if j.Name == "" {
		return nil, errors.New("vertex AI batch job response did not contain a resource name")
	}
	job := &core.EmbeddingBatchJob{
		Name:        j.Name,
		DisplayName: j.DisplayName,
		State:       vertexBatchJobState(j.State),
		Model:       j.Model,
		OutputURI:   j.OutputConfig.GCSDestination.OutputURIPrefix,
	}
	if len(j.InputConfig.GCSSource.URIs) > 0 {
		job.InputURI = j.InputConfig.GCSSource.URIs[0]
	}
	if j.Error != nil {
		job.Error = j.Error.Message
	}
	return job, nil
}

// vertexBatchJobState normalizes a Vertex job state. A partial success is
// terminal so its successful rows can be applied.
func vertexBatchJobState(state string) core.EmbeddingBatchState {
	switch state {
	case "JOB_STATE_SUCCEEDED":
		return core.EmbeddingBatchStateSucceeded
	case "JOB_STATE_FAILED", "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED":
		return core.EmbeddingBatchStateFailed
	case "JOB_STATE_CANCELLED":
		return core.EmbeddingBatchStateCancelled
	case "JOB_STATE_PAUSED":
		return core.EmbeddingBatchStatePaused
	case "JOB_STATE_RUNNING", "JOB_STATE_UPDATING", "JOB_STATE_CANCELLING":
		return core.EmbeddingBatchStateRunning
	default:
		return core.EmbeddingBatchStatePending
	}
}

func assertVertexBatchModel(job *core.EmbeddingBatchJob, model string) (*core.EmbeddingBatchJob, error) {
	if job.Model != "" && shortResourceName(job.Model) != shortResourceName(model) {
		return nil, fmt.Errorf("vertex AI batch job %s belongs to model %s, not %s", job.Name, job.Model, model)
	}
	return job, nil
}

// vertexBatchJobsURL returns the batchPredictionJobs URL of the location the
// profile runs in, or the URL of one job.
func (d *VertexAIDriver) vertexBatchJobsURL(profile vertexEmbeddingBatchProfile, name string) string {
	location := d.options.Region
	if profile.global {
		location = "global"
	}
	base := d.options.BaseURL
	if base == "" {
		base = vertexDefaultBaseURL(location)
	}
	base = strings.TrimSuffix(base, "/") + "/"
	if strings.HasPrefix(name, "projects/") {
		return base + name
	}
	jobs := base + "projects/" + d.options.Project + "/locations/" + location + "/batchPredictionJobs"
	if name == "" {
		return jobs
	}
	return jobs + "/" + name[strings.LastIndex(name, "/")+1:]
}

func (d *VertexAIDriver) vertexBatchRequest(ctx context.Context, method, endpoint string, body any, out any, model string) error {
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return d.formatError(err, model, vertexEmbeddingBatchOperation)
	}
	if err := doJSON(ctx, d.client, method, endpoint, headers, body, out); err != nil {
		return d.formatError(err, model, vertexEmbeddingBatchOperation)
	}
	return nil
}

// CreateEmbeddingBatch submits a batch job, or returns the job a previous
// attempt created with the same display name, model, input and output.
func (d *VertexAIDriver) CreateEmbeddingBatch(ctx context.Context, options core.EmbeddingBatchCreateOptions, _ core.EmbeddingBatchArtifactStore) (*core.EmbeddingBatchJob, error) {
	profile, err := requireVertexEmbeddingBatchProfile(options.Model, options.Modality)
	if err != nil {
		return nil, err
	}
	jobsURL := d.vertexBatchJobsURL(profile, "")
	filter, _ := json.Marshal(options.DisplayName)
	query := url.Values{"filter": {"displayName=" + string(filter)}, "pageSize": {"10"}}
	for {
		var page struct {
			BatchPredictionJobs []vertexBatchJob `json:"batchPredictionJobs"`
			NextPageToken       string           `json:"nextPageToken"`
		}
		if err := d.vertexBatchRequest(ctx, http.MethodGet, jobsURL+"?"+query.Encode(), nil, &page, options.Model); err != nil {
			return nil, err
		}
		for _, candidate := range page.BatchPredictionJobs {
			job, err := candidate.normalize()
			if err != nil {
				return nil, err
			}
			if job.DisplayName == options.DisplayName && shortResourceName(job.Model) == shortResourceName(options.Model) &&
				job.InputURI == options.InputURI && job.OutputURI == options.OutputURI {
				return job, nil
			}
		}
		if page.NextPageToken == "" {
			break
		}
		query.Set("pageToken", page.NextPageToken)
	}
	body := map[string]any{
		"displayName":  options.DisplayName,
		"model":        "publishers/google/models/" + profile.model,
		"inputConfig":  map[string]any{"instancesFormat": "jsonl", "gcsSource": map[string]any{"uris": []string{options.InputURI}}},
		"outputConfig": map[string]any{"predictionsFormat": "jsonl", "gcsDestination": map[string]any{"outputUriPrefix": options.OutputURI}},
	}
	if profile.schema == vertexEmbeddingBatchSchemaLegacy && options.Dimensions > 0 {
		// The legacy batch backend reads numeric JSON parameters as FLOAT64, not INT64.
		body["modelParameters"] = map[string]any{"outputDimensionality": strconv.Itoa(options.Dimensions)}
	}
	var created vertexBatchJob
	if err := d.vertexBatchRequest(ctx, http.MethodPost, jobsURL, body, &created, options.Model); err != nil {
		return nil, err
	}
	job, err := created.normalize()
	if err != nil {
		return nil, err
	}
	return assertVertexBatchModel(job, options.Model)
}

// GetEmbeddingBatch returns the current state of a batch job of options.Model.
func (d *VertexAIDriver) GetEmbeddingBatch(ctx context.Context, options core.EmbeddingBatchJobOptions) (*core.EmbeddingBatchJob, error) {
	profile, err := requireVertexEmbeddingBatchProfile(options.Model, options.Modality)
	if err != nil {
		return nil, err
	}
	var got vertexBatchJob
	if err := d.vertexBatchRequest(ctx, http.MethodGet, d.vertexBatchJobsURL(profile, options.Name), nil, &got, options.Model); err != nil {
		return nil, err
	}
	job, err := got.normalize()
	if err != nil {
		return nil, err
	}
	return assertVertexBatchModel(job, options.Model)
}

// CancelEmbeddingBatch cancels a batch job and returns its state afterwards.
func (d *VertexAIDriver) CancelEmbeddingBatch(ctx context.Context, options core.EmbeddingBatchJobOptions) (*core.EmbeddingBatchJob, error) {
	profile, err := requireVertexEmbeddingBatchProfile(options.Model, options.Modality)
	if err != nil {
		return nil, err
	}
	if err := d.vertexBatchRequest(ctx, http.MethodPost, d.vertexBatchJobsURL(profile, options.Name)+":cancel", map[string]any{}, nil, options.Model); err != nil {
		return nil, err
	}
	return d.GetEmbeddingBatch(ctx, options)
}

// DeleteEmbeddingBatch deletes a batch job.
func (d *VertexAIDriver) DeleteEmbeddingBatch(ctx context.Context, options core.EmbeddingBatchJobOptions) (*core.EmbeddingBatchJob, error) {
	profile, err := requireVertexEmbeddingBatchProfile(options.Model, options.Modality)
	if err != nil {
		return nil, err
	}
	if err := d.vertexBatchRequest(ctx, http.MethodDelete, d.vertexBatchJobsURL(profile, options.Name), nil, nil, options.Model); err != nil {
		return nil, err
	}
	return &core.EmbeddingBatchJob{Name: options.Name, State: core.EmbeddingBatchStateCancelled}, nil
}

// EmbeddingBatchOutputArtifacts lists the result shards Vertex wrote under the
// job's output prefix, in every terminal state. Vertex writes to application
// storage directly, so nothing is copied.
func (d *VertexAIDriver) EmbeddingBatchOutputArtifacts(ctx context.Context, job core.EmbeddingBatchJob, artifacts core.EmbeddingBatchArtifactStore) ([]string, error) {
	if job.OutputURI == "" {
		return nil, errors.New("vertex embedding batch omitted its output prefix")
	}
	return artifacts.List(ctx, job.OutputURI)
}
