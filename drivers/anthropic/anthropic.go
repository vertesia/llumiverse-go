package anthropic

import internalclaude "github.com/vertesia/llumiverse-go/drivers/internal/claude"

type AnthropicOptions = internalclaude.AnthropicOptions
type AnthropicDriver = internalclaude.AnthropicDriver

// NewAnthropicDriver creates the direct Anthropic Claude driver.
func NewAnthropicDriver(options AnthropicOptions) (*AnthropicDriver, error) {
	return internalclaude.NewAnthropicDriver(options)
}
