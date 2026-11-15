package core

import (
	"context"
	"io"
)

// EmbeddingBatchModality is the input modality of an embedding batch job.
type EmbeddingBatchModality string

// EmbeddingBatchModality values.
const (
	EmbeddingBatchModalityText  EmbeddingBatchModality = "text"
	EmbeddingBatchModalityImage EmbeddingBatchModality = "image"
)

// EmbeddingBatchState is the normalized state of an embedding batch job.
type EmbeddingBatchState string

// EmbeddingBatchState values.
const (
	EmbeddingBatchStatePending   EmbeddingBatchState = "pending"
	EmbeddingBatchStateRunning   EmbeddingBatchState = "running"
	EmbeddingBatchStateSucceeded EmbeddingBatchState = "succeeded"
	EmbeddingBatchStateFailed    EmbeddingBatchState = "failed"
	EmbeddingBatchStateCancelled EmbeddingBatchState = "cancelled"
	EmbeddingBatchStatePaused    EmbeddingBatchState = "paused"
)

// EmbeddingBatchCapability describes how a provider runs batch embeddings for one model.
type EmbeddingBatchCapability struct {
	// Model is the embedding model the capability applies to.
	Model string
	// InputFormat names the provider row format FormatEmbeddingBatchRow writes.
	InputFormat string
	// MaxRows is the largest number of rows one job accepts.
	MaxRows int
	// Location is the provider region the job must run in, when the model requires one.
	Location string
}

// EmbeddingBatchArtifactStore reads and writes durable application artifacts (object-store URIs),
// never provider file IDs or signed URLs. Implementations enforce tenant scope.
type EmbeddingBatchArtifactStore interface {
	// List returns the artifact URIs under prefix.
	List(ctx context.Context, prefix string) ([]string, error)
	// Read opens the artifact at uri. The caller closes the reader.
	Read(ctx context.Context, uri string) (io.ReadCloser, error)
	// Write stores content at uri, consuming it completely.
	Write(ctx context.Context, uri string, content io.Reader) error
}

// EmbeddingBatchJob is the normalized state of a provider batch job.
type EmbeddingBatchJob struct {
	// Name is the opaque provider job identity.
	Name string
	// DisplayName is the job's display name, when the provider keeps one.
	DisplayName string
	// State is the normalized job state.
	State EmbeddingBatchState
	// Model is the embedding model the job runs.
	Model string
	// InputURI and OutputURI are the original application artifact locations, kept for ownership
	// checks even when the provider works on uploaded files.
	InputURI  string
	OutputURI string
	// Error is a sanitized failure description, when the job failed.
	Error string
}

// EmbeddingBatchJobOptions identifies an existing batch job.
type EmbeddingBatchJobOptions struct {
	// Model is the embedding model the job is expected to run; adapters reject a mismatch.
	Model string
	// Modality is the job's input modality.
	Modality EmbeddingBatchModality
	// Name is the opaque provider job identity.
	Name string
}

// EmbeddingBatchCreateOptions describes a batch job to create.
type EmbeddingBatchCreateOptions struct {
	// Model is the embedding model to run.
	Model string
	// Modality is the input modality of every row.
	Modality EmbeddingBatchModality
	// Dimensions requests an output dimensionality when the model supports it. Zero means the
	// model default.
	Dimensions int
	// DisplayName identifies the submission; a retried create recovers the matching job.
	DisplayName string
	// InputURI is the artifact holding the formatted rows.
	InputURI string
	// OutputURI is the artifact prefix the provider writes results under.
	OutputURI string
}

// EmbeddingBatchRowOptions describes one input row to format.
type EmbeddingBatchRowOptions struct {
	// Key correlates the row with its result.
	Key string
	// Model is the embedding model the job runs.
	Model string
	// Input is the item to embed.
	Input EmbeddingInput
	// Dimensions is the requested output dimensionality, or zero for the model default.
	Dimensions int
	// InputFormat is the row format from EmbeddingBatchCapability.InputFormat.
	InputFormat string
}

// ParsedEmbeddingBatchResult is one parsed result row.
type ParsedEmbeddingBatchResult struct {
	// Key is the result key FormatEmbeddingBatchRow returned for the row, when the row is identified.
	Key string
	// Vector is the embedding, when the row succeeded.
	Vector []float64
	// ProviderError reports that the provider failed this row.
	ProviderError bool
	// FailureCategory is a sanitized failure category, never input content or a native error payload.
	FailureCategory string
}

// EmbeddingBatchAdapter is optional provider support for batch embedding jobs, the TS client's
// EmbeddingBatchAdapter. A driver opts in by implementing it; EmbeddingBatchAdapterFor finds it.
// Native payloads and file handles stay behind the interface. Lifecycle methods validate the
// requested model and keep the original artifact URIs for ownership checks; a retried create
// recovers the submission with the same display name, model, input and output instead of creating
// a duplicate; result parsing returns sanitized failure categories, never input content or native
// error payloads.
type EmbeddingBatchAdapter interface {
	// EmbeddingBatchCapability returns how the provider batches model for modality, preferring
	// location when the model allows it, or false when it cannot batch that model.
	EmbeddingBatchCapability(model string, modality EmbeddingBatchModality, location string) (EmbeddingBatchCapability, bool)
	// FormatEmbeddingBatchRow formats one input as a provider row and returns the key its result
	// will carry.
	FormatEmbeddingBatchRow(ctx context.Context, options EmbeddingBatchRowOptions) (row map[string]any, resultKey string, err error)
	// ParseEmbeddingBatchResult parses one provider result record.
	ParseEmbeddingBatchResult(record map[string]any) ParsedEmbeddingBatchResult
	// CreateEmbeddingBatch submits a job, or recovers the matching one on a retry.
	CreateEmbeddingBatch(ctx context.Context, options EmbeddingBatchCreateOptions, artifacts EmbeddingBatchArtifactStore) (*EmbeddingBatchJob, error)
	// GetEmbeddingBatch returns a job's current state.
	GetEmbeddingBatch(ctx context.Context, options EmbeddingBatchJobOptions) (*EmbeddingBatchJob, error)
	// CancelEmbeddingBatch cancels a job.
	CancelEmbeddingBatch(ctx context.Context, options EmbeddingBatchJobOptions) (*EmbeddingBatchJob, error)
	// DeleteEmbeddingBatch deletes a job.
	DeleteEmbeddingBatch(ctx context.Context, options EmbeddingBatchJobOptions) (*EmbeddingBatchJob, error)
	// EmbeddingBatchOutputArtifacts returns the result artifacts of a job whose ownership the caller
	// has validated, including partial successes and errors in every terminal state: existing
	// object-store artifacts, or provider files materialized idempotently under job.OutputURI.
	EmbeddingBatchOutputArtifacts(ctx context.Context, job EmbeddingBatchJob, artifacts EmbeddingBatchArtifactStore) ([]string, error)
}

// EmbeddingBatchAdapterFor returns driver's batch embedding support, or false when the driver has
// none (the TS client's getEmbeddingBatchAdapter).
func EmbeddingBatchAdapterFor(driver Driver) (EmbeddingBatchAdapter, bool) {
	adapter, ok := driver.(EmbeddingBatchAdapter)
	return adapter, ok
}
