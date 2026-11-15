package core

import (
	"context"
	"testing"
)

// batchDriver opts in to batch embeddings, as a provider driver would.
type batchDriver struct {
	fakeDriver
}

func (d *batchDriver) EmbeddingBatchCapability(model string, modality EmbeddingBatchModality, location string) (EmbeddingBatchCapability, bool) {
	if model != "batch-model" {
		return EmbeddingBatchCapability{}, false
	}
	return EmbeddingBatchCapability{Model: model, InputFormat: "rows", MaxRows: 10, Location: location}, true
}

func (d *batchDriver) FormatEmbeddingBatchRow(_ context.Context, options EmbeddingBatchRowOptions) (map[string]any, string, error) {
	return map[string]any{"key": options.Key, "text": options.Input.Text}, options.Key, nil
}

func (d *batchDriver) ParseEmbeddingBatchResult(record map[string]any) ParsedEmbeddingBatchResult {
	key, _ := record["key"].(string)
	return ParsedEmbeddingBatchResult{Key: key, Vector: []float64{1, 2}}
}

func (d *batchDriver) CreateEmbeddingBatch(_ context.Context, options EmbeddingBatchCreateOptions, _ EmbeddingBatchArtifactStore) (*EmbeddingBatchJob, error) {
	return &EmbeddingBatchJob{Name: "jobs/1", State: EmbeddingBatchStatePending, Model: options.Model, InputURI: options.InputURI, OutputURI: options.OutputURI}, nil
}

func (d *batchDriver) GetEmbeddingBatch(_ context.Context, options EmbeddingBatchJobOptions) (*EmbeddingBatchJob, error) {
	return &EmbeddingBatchJob{Name: options.Name, State: EmbeddingBatchStateRunning, Model: options.Model}, nil
}

func (d *batchDriver) CancelEmbeddingBatch(_ context.Context, options EmbeddingBatchJobOptions) (*EmbeddingBatchJob, error) {
	return &EmbeddingBatchJob{Name: options.Name, State: EmbeddingBatchStateCancelled, Model: options.Model}, nil
}

func (d *batchDriver) DeleteEmbeddingBatch(_ context.Context, options EmbeddingBatchJobOptions) (*EmbeddingBatchJob, error) {
	return &EmbeddingBatchJob{Name: options.Name, State: EmbeddingBatchStateCancelled, Model: options.Model}, nil
}

func (d *batchDriver) EmbeddingBatchOutputArtifacts(ctx context.Context, job EmbeddingBatchJob, artifacts EmbeddingBatchArtifactStore) ([]string, error) {
	return artifacts.List(ctx, job.OutputURI)
}

// Mirrors the TS client's "opts in explicitly" check: only a driver that implements the adapter
// has batch support.
func TestEmbeddingBatchAdapterFor(t *testing.T) {
	t.Parallel()

	if _, ok := EmbeddingBatchAdapterFor(&fakeDriver{}); ok {
		t.Fatal("a driver without batch support must not resolve an adapter")
	}
	adapter, ok := EmbeddingBatchAdapterFor(&batchDriver{})
	if !ok {
		t.Fatal("batch driver did not resolve an adapter")
	}
	capability, ok := adapter.EmbeddingBatchCapability("batch-model", EmbeddingBatchModalityText, "us-central1")
	if !ok || capability.InputFormat != "rows" || capability.MaxRows != 10 {
		t.Fatalf("capability = %#v, %v", capability, ok)
	}
	if _, ok := adapter.EmbeddingBatchCapability("other", EmbeddingBatchModalityImage, ""); ok {
		t.Fatal("unsupported model resolved a capability")
	}
	row, key, err := adapter.FormatEmbeddingBatchRow(context.Background(), EmbeddingBatchRowOptions{Key: "k", Input: EmbeddingInput{Text: "hello"}})
	if err != nil {
		t.Fatal(err)
	}
	if parsed := adapter.ParseEmbeddingBatchResult(row); parsed.Key != key || len(parsed.Vector) != 2 {
		t.Fatalf("parsed = %#v", parsed)
	}
}
