// Package xai implements the llumiverse driver for xAI (https://api.x.ai/v1).
//
// Grok language models run on the Responses API through the shared drivers/openai implementation. Grok image
// models (IDs naming both grok and image, see IsImageModel) run on the Images API: the text of every prompt
// segment is the image prompt, attached images are edited (/images/edits) instead of generated, and the
// ModelOptions aspect_ratio, resolution, quality, response_format and n are passed through. A rejected image
// request is returned as the completion's error rather than as an error, and the reported cost becomes
// ExecutionTokenUsage.ProviderCostUsd.
package xai
