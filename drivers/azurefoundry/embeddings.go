package azurefoundry

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"
)

// embeddingResponse is the OpenAI v1 embeddings response.
type embeddingResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Usage *embeddingUsage `json:"usage"`
}

// embeddingUsage is the token usage of an embeddings response.
type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// GenerateEmbeddings embeds text and image inputs with the embeddings API of
// the resource endpoint. Text and images go in one batch each (images as
// base64), and the vectors are returned in input order. The model is required:
// Foundry has no default embedding deployment.
func (d *AzureFoundryDriver) GenerateEmbeddings(ctx context.Context, options EmbeddingsOptions) (*EmbeddingsResult, error) {
	options, err := normalizeEmbeddingsOptions(options)
	if err != nil {
		return nil, err
	}
	if options.Model == "" {
		return nil, errors.New("default embedding model selection is not supported for Microsoft Foundry: specify a model")
	}
	var textIndexes, imageIndexes []int
	var texts, images []string
	for i, input := range options.Inputs {
		switch input.Type {
		case embeddingInputText:
			textIndexes = append(textIndexes, i)
			texts = append(texts, input.Text)
		case embeddingInputImage:
			if input.Source == nil {
				return nil, errors.New("image embedding input requires a source")
			}
			encoded, err := dataSourceToBase64(ctx, input.Source)
			if err != nil {
				return nil, err
			}
			imageIndexes = append(imageIndexes, i)
			images = append(images, encoded)
		default:
			return nil, fmt.Errorf("provider %q does not support %q embeddings", ProviderAzureFoundry, input.Type)
		}
	}
	items := make([]EmbeddingResultItem, len(options.Inputs))
	var inputTokens int
	var usageReported bool
	batches := []struct {
		indexes  []int
		inputs   []string
		modality embeddingOutputModality
		kind     string
	}{
		{textIndexes, texts, embeddingOutputModalityText, "text"},
		{imageIndexes, images, embeddingOutputModalityImage, "image"},
	}
	for _, batch := range batches {
		if len(batch.inputs) == 0 {
			continue
		}
		vectors, usage, err := d.callEmbeddings(ctx, batch.inputs, options, batch.kind)
		if err != nil {
			return nil, err
		}
		for i, index := range batch.indexes {
			items[index] = EmbeddingResultItem{Outputs: []EmbeddingOutput{{Values: vectors[i], Modality: ptr(batch.modality)}}}
		}
		if usage != nil {
			usageReported = true
			inputTokens += usage.PromptTokens
		}
	}
	result := &EmbeddingsResult{Model: options.Model, Results: items}
	if usageReported {
		result.Usage = &EmbeddingsTokenUsage{InputTokens: tokens(inputTokens)}
	}
	return result, nil
}

// callEmbeddings embeds one batch and returns its vectors in input order.
func (d *AzureFoundryDriver) callEmbeddings(ctx context.Context, inputs []string, options EmbeddingsOptions, kind string) ([][]float64, *embeddingUsage, error) {
	deployment, _ := parseModelID(options.Model)
	endpoint, err := d.v1Endpoint(d.resourceV1, "embeddings")
	if err != nil {
		return nil, nil, err
	}
	headers, err := d.creds.keyOrBearer(ctx, headerAPIKey)
	if err != nil {
		return nil, nil, err
	}
	payload := map[string]any{"input": inputs, "model": deployment, "encoding_format": "float"}
	if options.Dimensions > 0 {
		payload["dimensions"] = options.Dimensions
	}
	var response embeddingResponse
	if err := doJSON(ctx, d.httpClient(options.HTTPTimeout), http.MethodPost, endpoint, headers, payload, &response); err != nil {
		return nil, nil, d.formatError(err, options.Model, "embeddings")
	}
	if len(response.Data) != len(inputs) {
		return nil, nil, d.formatError(fmt.Errorf("expected %d embeddings in the microsoft foundry %s response, got %d", len(inputs), kind, len(response.Data)), options.Model, "embeddings")
	}
	sort.SliceStable(response.Data, func(i, j int) bool { return response.Data[i].Index < response.Data[j].Index })
	vectors := make([][]float64, len(response.Data))
	for i, entry := range response.Data {
		if len(entry.Embedding) == 0 {
			return nil, nil, d.formatError(fmt.Errorf("empty embedding in the microsoft foundry %s response", kind), options.Model, "embeddings")
		}
		vectors[i] = entry.Embedding
	}
	return vectors, response.Usage, nil
}
