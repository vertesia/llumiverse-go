package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrockruntime "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"
)

// Claude on Bedrock runs on the Anthropic Messages API over InvokeModel and InvokeModelWithResponseStream, sharing
// the prompt, conversation, payload and stream handling of the Anthropic and Vertex AI drivers. Conversations
// started on the Converse API continue there, because their stored turns are Converse-shaped.

// bedrockAnthropicVersion is the anthropic_version Bedrock expects in a Claude Messages body.
const bedrockAnthropicVersion = "bedrock-2023-05-31"

// bedrockClaudePrompt is a Claude Messages prompt; its type routes Execute and Stream to the Messages API.
type bedrockClaudePrompt = internalclaude.Prompt

// InvokeModelStream is the subset of the InvokeModelWithResponseStream event-stream reader the driver relies on,
// with the service tier Bedrock applied.
type InvokeModelStream interface {
	// Events delivers the response stream events.
	Events() <-chan brtypes.ResponseStream
	// Close closes the stream.
	Close() error
	// Err is the error that ended the stream, if any.
	Err() error
	// ServiceTier is the service tier Bedrock applied to the request, or "".
	ServiceTier() string
}

// InvokeModelStreamAPI is implemented by runtime clients that can stream InvokeModel responses. Claude runs on the
// Messages API only on such a runtime; a RuntimeAPI without it keeps Claude on Converse.
type InvokeModelStreamAPI interface {
	// InvokeModelWithResponseStream opens an InvokeModel response stream.
	InvokeModelWithResponseStream(ctx context.Context, input *bedrockruntime.InvokeModelWithResponseStreamInput, optFns ...func(*bedrockruntime.Options)) (InvokeModelStream, error)
}

// bedrockInvokeModelStream adapts the AWS event stream to InvokeModelStream.
type bedrockInvokeModelStream struct {
	*bedrockruntime.InvokeModelWithResponseStreamEventStream
	tier string
}

// ServiceTier is the service tier from the response headers.
func (s bedrockInvokeModelStream) ServiceTier() string {
	return s.tier
}

// InvokeModelWithResponseStream unwraps the AWS output's event stream and returns an error when it is nil.
func (c *bedrockRuntimeClient) InvokeModelWithResponseStream(ctx context.Context, input *bedrockruntime.InvokeModelWithResponseStreamInput, optFns ...func(*bedrockruntime.Options)) (InvokeModelStream, error) {
	output, err := c.client.InvokeModelWithResponseStream(ctx, input, optFns...)
	if err != nil {
		return nil, err
	}
	if output.GetStream() == nil {
		return nil, errors.New("bedrock invoke model stream is nil")
	}
	return bedrockInvokeModelStream{InvokeModelWithResponseStreamEventStream: output.GetStream(), tier: string(output.ServiceTier)}, nil
}

// namedBedrockModel is the model a Bedrock identifier names without a lookup: a foundation-model ID or ARN, or a
// system inference profile (us.anthropic.claude-...), whose names AWS defines. It is empty for application
// inference profiles, custom models and provisioned throughput, whose ARNs carry caller-chosen names.
func namedBedrockModel(model string) string {
	if !strings.HasPrefix(model, "arn:") {
		return model
	}
	for _, resource := range []string{":foundation-model/", ":inference-profile/"} {
		if index := strings.Index(model, resource); index >= 0 {
			return model[index+len(resource):]
		}
	}
	return ""
}

// bedrockClaudeModel is the model Claude's capability checks use: the named model, or the identifier itself when
// the name says nothing.
func bedrockClaudeModel(model string) string {
	if named := namedBedrockModel(model); named != "" {
		return named
	}
	return model
}

// isBedrockClaudeModel reports whether a Bedrock identifier names a Claude model.
func isBedrockClaudeModel(model string) bool {
	return strings.Contains(strings.ToLower(bedrockClaudeModel(model)), "claude")
}

// isConverseConversation reports whether a stored conversation was built for the Converse API: its content blocks
// have no type field ({"text": ...}, {"Value": ...}) unlike Messages API blocks. The first message with content
// decides, then the system blocks.
func isConverseConversation(conversation any) bool {
	switch value := conversation.(type) {
	case bedrockPrompt, *bedrockPrompt:
		return true
	case map[string]any:
		isConverseBlock := func(block any) bool {
			fields, ok := block.(map[string]any)
			if !ok {
				return false
			}
			_, typed := fields["type"]
			return !typed
		}
		if raw, ok := bedrockMapLookup(value, "messages", "Messages"); ok {
			messages, _ := raw.([]any)
			for _, message := range messages {
				fields, _ := message.(map[string]any)
				content, _ := bedrockValueFromMap(fields, "content", "Content").([]any)
				if len(content) > 0 {
					for _, block := range content {
						if isConverseBlock(block) {
							return true
						}
					}
					return false
				}
			}
		}
		system, _ := bedrockValueFromMap(value, "system", "System").([]any)
		for _, block := range system {
			if isConverseBlock(block) {
				return true
			}
		}
	}
	return false
}

// isMessagesAPIConversation reports whether a stored conversation is in Anthropic Messages format, which cannot
// continue on a Converse model.
func isMessagesAPIConversation(conversation any) bool {
	switch value := conversation.(type) {
	case bedrockClaudePrompt:
		return len(value.Messages) > 0
	case *bedrockClaudePrompt:
		return value != nil && len(value.Messages) > 0
	case map[string]any:
		messages, _ := bedrockValueFromMap(value, "messages", "Messages").([]any)
		return len(messages) > 0 && !isConverseConversation(value)
	}
	return false
}

// usesClaudeMessagesAPI reports whether a request runs on the Claude Messages API: a Claude model, a runtime that
// can stream InvokeModel responses, and no stored Converse conversation.
func (d *BedrockDriver) usesClaudeMessagesAPI(options ExecutionOptions) bool {
	_, streams := d.runtime.(InvokeModelStreamAPI)
	return streams && isBedrockClaudeModel(options.Model) && !isConverseConversation(options.Conversation)
}

// createClaudeMessagesPrompt is the CreatePrompt hook for Claude. It formats the prompt of a request on the Claude
// Messages API (handled true), with the result schema left out when Bedrock enforces it natively. A Messages API
// conversation continuing on a Converse model fails clearly instead of sending a malformed request.
func (d *BedrockDriver) createClaudeMessagesPrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (any, bool, error) {
	if !d.usesClaudeMessagesAPI(options) {
		if isMessagesAPIConversation(options.Conversation) {
			message := fmt.Sprintf("The stored conversation is not in Converse format, so it cannot continue on %s. Start a new conversation.", options.Model)
			return nil, true, newLlumiverseError(message, boolPtr(false), LlumiverseErrorContext{
				Provider:  ProviderBedrock,
				Model:     options.Model,
				Operation: "execute",
			}, errors.New(message), 400, "ConversationFormatError")
		}
		return nil, false, nil
	}
	claudeOptions := bedrockClaudeOptions(options)
	prompt, err := internalclaude.FormatPromptWithNativeSchema(ctx, segments, claudeOptions, internalclaude.UsesNativeResultSchema(claudeOptions))
	return prompt, true, err
}

// bedrockClaudeMaxTokens is the max_tokens of a Bedrock Claude request: the caller's value clamped to the model's
// output limit (a value written for a larger-output model is otherwise rejected), or the limit itself.
func bedrockClaudeMaxTokens(options ExecutionOptions) int {
	limit := internalclaude.BedrockMaxTokensLimit(options.Model)
	if v := optionInt(options.ModelOptions, "max_tokens"); v != nil {
		if limit > 0 && *v > limit {
			return limit
		}
		return *v
	}
	if limit == 0 {
		limit = 8192
	}
	if strings.Contains(options.Model, "claude-3-7-sonnet") {
		// Claude 3.7 reaches 128K with a beta, but without an explicit max_tokens it defaults to 64K.
		if budget := optionInt(options.ModelOptions, "thinking_budget_tokens"); budget == nil || *budget < 48000 {
			limit = 64000
		}
	}
	return limit
}

// bedrockClaudeOptions are the execution options of the Claude Messages route: the named model for Claude's
// capability checks (the request still names the caller's identifier) and the clamped max_tokens. The caller's
// options are not modified.
func bedrockClaudeOptions(options ExecutionOptions) ExecutionOptions {
	out := options
	out.Model = bedrockClaudeModel(options.Model)
	modelOptions := make(map[string]any, len(options.ModelOptions)+1)
	for key, value := range options.ModelOptions {
		modelOptions[key] = value
	}
	modelOptions["max_tokens"] = bedrockClaudeMaxTokens(out)
	out.ModelOptions = modelOptions
	return out
}

// bedrockServiceTier is the service_tier model option as the InvokeModel field.
func bedrockServiceTier(options ExecutionOptions) brtypes.ServiceTierType {
	return brtypes.ServiceTierType(optionString(options.ModelOptions, "service_tier"))
}

// claudeMessagesRequest prepares a Claude request and its InvokeModel body: Bedrock selects the model from the
// model ID, and takes anthropic_version and the betas (as anthropic_beta) in the body.
func (d *BedrockDriver) claudeMessagesRequest(prompt bedrockClaudePrompt, options ExecutionOptions, operation string) (*internalclaude.Request, []byte, error) {
	request, err := internalclaude.NewRequest(prompt, bedrockClaudeOptions(options), internalclaude.RequestSettings{
		Provider:                internalclaude.ProviderNameBedrock,
		Operation:               operation,
		AllowNativeResultSchema: true,
	})
	if err != nil {
		return nil, nil, err
	}
	body := make(map[string]any, len(request.Body)+2)
	for key, value := range request.Body {
		body[key] = value
	}
	body["anthropic_version"] = bedrockAnthropicVersion
	if len(request.Betas) > 0 {
		body["anthropic_beta"] = request.Betas
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, nil, err
	}
	return request, data, nil
}

// requestClaudeMessages is the Execute hook for Claude: one InvokeModel call on the Messages API.
func (d *BedrockDriver) requestClaudeMessages(ctx context.Context, prompt bedrockClaudePrompt, options ExecutionOptions) (*Completion, error) {
	request, body, err := d.claudeMessagesRequest(prompt, options, "execute")
	if err != nil {
		return nil, err
	}
	output, err := d.runtime.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(options.Model),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		ServiceTier: bedrockServiceTier(options),
	}, d.runtimeOptions(options.HTTPTimeout)...)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	var response internalclaude.Response
	if err := json.Unmarshal(output.Body, &response); err != nil {
		return nil, fmt.Errorf("decode Bedrock Claude response: %w", err)
	}
	request.ServiceTier = string(output.ServiceTier)
	return request.Complete(response), nil
}

// requestClaudeMessagesStream is the Stream hook for Claude: an InvokeModelWithResponseStream call whose chunks are
// Messages API stream events.
func (d *BedrockDriver) requestClaudeMessagesStream(ctx context.Context, prompt bedrockClaudePrompt, options ExecutionOptions) (CompletionStream, error) {
	runtime, ok := d.runtime.(InvokeModelStreamAPI)
	if !ok {
		return nil, errors.New("bedrock runtime client cannot stream InvokeModel responses")
	}
	request, body, err := d.claudeMessagesRequest(prompt, options, "stream")
	if err != nil {
		return nil, err
	}
	stream, err := runtime.InvokeModelWithResponseStream(ctx, &bedrockruntime.InvokeModelWithResponseStreamInput{
		ModelId:     aws.String(options.Model),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		ServiceTier: bedrockServiceTier(options),
	}, d.runtimeOptions(options.HTTPTimeout)...)
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	request.ServiceTier = stream.ServiceTier()
	formatError := func(err error) error { return d.formatError(err, options.Model, "stream") }
	events := func(yield func([]byte) error) error {
		for event := range stream.Events() {
			chunk, ok := event.(*brtypes.ResponseStreamMemberChunk)
			if !ok || len(chunk.Value.Bytes) == 0 {
				continue
			}
			if err := yield(chunk.Value.Bytes); err != nil {
				return err
			}
		}
		if err := stream.Err(); err != nil {
			// A Bedrock exception raised mid-stream keeps the Bedrock retry rules.
			return formatError(err)
		}
		return nil
	}
	return request.Stream(prompt, events, stream.Close, formatError), nil
}
