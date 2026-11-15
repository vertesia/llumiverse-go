package bedrock

import (
	"errors"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// toolChoiceConfigurationErrorName names the error a forced tool turn fails
// with when it cannot be enforced, as in the TS client.
const toolChoiceConfigurationErrorName = "ToolChoiceConfigurationError"

// bedrockToolChoicePlan is how a request's tool_choice / required_tool_name
// model options map onto Converse.
type bedrockToolChoicePlan struct {
	// choice is the Converse toolChoice, nil for the provider's automatic choice.
	choice brtypes.ToolChoice
	// disableClaudeThinking is set when a forced Claude turn must run without
	// thinking: Bedrock permits only auto/none tool choice while thinking is on.
	disableClaudeThinking bool
}

// bedrockForcedToolRequested reports whether the caller requires a tool call.
func bedrockForcedToolRequested(modelOptions map[string]any) bool {
	choice := optionString(modelOptions, "tool_choice")
	return optionString(modelOptions, "required_tool_name") != "" || choice == "required" || choice == "any"
}

// bedrockSupportsForcedToolChoice reports whether a Converse model family can
// enforce a tool choice (Claude and Nova).
func bedrockSupportsForcedToolChoice(model string) bool {
	model = strings.ToLower(model)
	return strings.Contains(model, "claude") || strings.Contains(model, "amazon.nova")
}

// bedrockClaudeOnlyAdaptiveThinking reports whether a Claude family accepts only
// adaptive thinking (Fable and Mythos), so its thinking cannot be disabled.
func bedrockClaudeOnlyAdaptiveThinking(model string) bool {
	model = strings.ToLower(model)
	return strings.Contains(model, "claude-fable") || strings.Contains(model, "claude-mythos")
}

// bedrockToolChoice maps the tool_choice ("auto", "none", "any", "required") and
// required_tool_name model options onto a Converse tool choice. A forced tool
// turn fails with a non-retryable 400 when there are no tools or the model
// family cannot enforce it. Adaptive-only Claude families keep thinking and use
// the automatic choice; the caller's required-tool check stays authoritative.
func (d *BedrockDriver) bedrockToolChoice(options ExecutionOptions, operation string) (bedrockToolChoicePlan, error) {
	if !bedrockForcedToolRequested(options.ModelOptions) {
		return bedrockToolChoicePlan{}, nil
	}
	context := LlumiverseErrorContext{Provider: ProviderBedrock, Model: options.Model, Operation: operation}
	if len(options.Tools) == 0 {
		return bedrockToolChoicePlan{}, toolChoiceConfigurationError("A forced Bedrock tool turn requires at least one tool definition.", context)
	}
	if !bedrockSupportsForcedToolChoice(options.Model) {
		return bedrockToolChoicePlan{}, toolChoiceConfigurationError("Bedrock model "+options.Model+" cannot enforce the requested tool choice; use a tool-choice-capable model.", context)
	}
	claude := strings.Contains(strings.ToLower(options.Model), "claude")
	if claude && bedrockClaudeOnlyAdaptiveThinking(options.Model) {
		return bedrockToolChoicePlan{}, nil
	}
	plan := bedrockToolChoicePlan{disableClaudeThinking: claude && isClaudeVersionGTE(options.Model, 3, 7)}
	if name := optionString(options.ModelOptions, "required_tool_name"); name != "" {
		plan.choice = &brtypes.ToolChoiceMemberTool{Value: brtypes.SpecificToolChoice{Name: aws.String(name)}}
	} else {
		plan.choice = &brtypes.ToolChoiceMemberAny{Value: brtypes.AnyToolChoice{}}
	}
	return plan, nil
}

// toolChoiceConfigurationError is the typed contract failure for a tool choice
// the request cannot satisfy.
func toolChoiceConfigurationError(message string, context LlumiverseErrorContext) error {
	return newLlumiverseError(message, boolPtr(false), context, errors.New(message), 400, toolChoiceConfigurationErrorName)
}
