package azurefoundry

import (
	"encoding/json"
	"errors"
	"strings"
)

// formatError wraps an Azure error as a LlumiverseError with the provider,
// model and operation, the HTTP status, and a retryability derived from the
// status and the Azure error code (nil when neither says). LlumiverseErrors
// pass through unchanged.
func (d *AzureFoundryDriver) formatError(err error, model string, operation string) error {
	return formatAzureError(err, ProviderAzureFoundry, model, operation, "AzureError")
}

// formatClaudeError wraps a Claude Messages error, named as the TS client's
// Anthropic errors are.
func (d *AzureFoundryDriver) formatClaudeError(err error, model string, operation string) error {
	return formatAzureError(err, ProviderAzureFoundry, model, operation, "AnthropicError")
}

func formatAzureError(err error, provider Provider, model string, operation string, name string) error {
	if err == nil {
		return nil
	}
	var llumiErr *LlumiverseError
	if errors.As(err, &llumiErr) {
		return err
	}
	code, typeName := errorStatusAndName(err)
	if code == 0 {
		name = typeName
	}
	message, errorCode := azureErrorMessage(err.Error())
	return newLlumiverseError(message, azureRetryable(code, errorCode, err.Error()), LlumiverseErrorContext{
		Provider:  provider,
		Model:     model,
		Operation: operation,
	}, err, code, name)
}

// azureErrorMessage extracts the message and code of an Azure or Anthropic
// error body ("HTTP 400: {"error":{"code":...,"message":...}}"), falling back to
// the raw error text.
func azureErrorMessage(text string) (string, string) {
	start := strings.Index(text, "{")
	if start < 0 {
		return text, ""
	}
	var body struct {
		Error struct {
			Code    any    `json:"code"`
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
		Message string `json:"message"`
	}
	if json.Unmarshal([]byte(text[start:]), &body) != nil {
		return text, ""
	}
	code, _ := body.Error.Code.(string)
	if code == "" {
		code = body.Error.Type
	}
	message := body.Error.Message
	if message == "" {
		message = body.Message
	}
	if message == "" {
		return text, code
	}
	return strings.TrimSpace(text[:start]) + " " + message, code
}

// azureRetryable classifies an error: known Azure and Anthropic error codes
// first, then the HTTP status and message; nil when unknown.
func azureRetryable(status int, code string, message string) *bool {
	switch strings.ToLower(code) {
	case "ratelimitexceeded", "rate_limit_exceeded", "rate_limit_error", "toomanyrequests", "serviceunavailable",
		"server_error", "internalservererror", "timeout", "overloaded_error", "api_error":
		return boolPtr(true)
	case "contentfilter", "content_filter", "invalid_request_error", "authentication_error", "permission_error",
		"not_found_error", "deploymentnotfound", "invalidapiversion", "unauthorized", "badrequest":
		return boolPtr(false)
	}
	return retryableFromStatusAndMessage(status, message)
}
