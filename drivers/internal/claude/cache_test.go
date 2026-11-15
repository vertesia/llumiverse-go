package claude

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// stripCacheControl removes every cache_control field from a decoded JSON value.
func stripCacheControl(value any) any {
	switch v := value.(type) {
	case map[string]any:
		out := map[string]any{}
		for key, child := range v {
			if key != "cache_control" {
				out[key] = stripCacheControl(child)
			}
		}
		return out
	case []any:
		out := make([]any, len(v))
		for i, child := range v {
			out[i] = stripCacheControl(child)
		}
		return out
	default:
		return value
	}
}

// cacheMarkers lists the cache breakpoints as [message, block]; -1 for a system block or a tool.
func cacheMarkers(body map[string]any) [][2]int {
	var found [][2]int
	for _, key := range []string{"system", "tools"} {
		items, _ := body[key].([]any)
		for _, item := range items {
			if item.(map[string]any)["cache_control"] != nil {
				found = append(found, [2]int{-1, -1})
			}
		}
	}
	for m, message := range body["messages"].([]any) {
		for b, block := range message.(map[string]any)["content"].([]any) {
			if block.(map[string]any)["cache_control"] != nil {
				found = append(found, [2]int{m, b})
			}
		}
	}
	return found
}

// isPlainUserTurnJSON reports a decoded user turn without tool results.
func isPlainUserTurnJSON(message map[string]any) bool {
	if message["role"] != "user" {
		return false
	}
	for _, block := range message["content"].([]any) {
		if block.(map[string]any)["type"] == "tool_result" {
			return false
		}
	}
	return true
}

// prng is a small deterministic generator, so a generated run is reproducible from its seed.
func prng(seed int) func(limit int) int {
	state := seed
	return func(limit int) int {
		state = (state*1103515245 + 12345) % (1 << 31)
		return (state >> 16) % limit
	}
}

// decodeBlocks decodes response blocks as the API returns them.
func decodeBlocks(t *testing.T, data string) []claudeBlock {
	t.Helper()
	var blocks []claudeBlock
	if err := json.Unmarshal([]byte(data), &blocks); err != nil {
		t.Fatal(err)
	}
	return blocks
}

// runClaudeAgent drives a generated agent run through NewRequest and Complete and returns every request body. Each
// user turn runs a tool loop of up to 25 iterations with one to three parallel calls, then answers. The stored
// conversation goes through JSON between requests, as a caller persists it.
func runClaudeAgent(t *testing.T, model, provider string, seed, requests int) ([]map[string]any, []string) {
	t.Helper()
	random := prng(seed)
	options := ExecutionOptions{
		Model:          model,
		PromptCacheKey: "agent-generated-run",
		ModelOptions:   map[string]any{"effort": "low"},
		Tools: []ToolDefinition{
			{Name: "lookup", Description: ptr("Look a value up."), InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}},
			{Name: "write", Description: ptr("Write a value."), InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}},
		},
		StripHeartbeatsAfterTurns: ptr(1),
		StripTextMaxTokens:        8000,
	}
	prompt := claudePrompt{
		System:   []claudeBlock{{Type: "text", Text: "You are a test agent."}},
		Messages: []claudeMessage{{Role: "user", Content: []claudeBlock{{Type: "text", Text: "Start the task."}}}},
	}
	var conversation any
	var payloads []map[string]any
	var steps []string
	loopLeft := random(26)
	for request := 0; request < requests; request++ {
		step := "answer"
		if loopLeft > 0 {
			step = "tools"
		}
		calls := 1 + random(3)
		var content string
		stop := "end_turn"
		if step == "tools" {
			stop = "tool_use"
			parts := []string{fmt.Sprintf(`{"type":"thinking","thinking":"plan %d","signature":"sig-%d"}`, request, request)}
			for call := 0; call < calls; call++ {
				name := "lookup"
				if call%2 == 1 {
					name = "write"
				}
				parts = append(parts, fmt.Sprintf(`{"type":"tool_use","id":"call-%d-%d","name":"%s","input":{"n":%d}}`, request, call, name, request))
			}
			content = "[" + strings.Join(parts, ",") + "]"
		} else {
			content = fmt.Sprintf(`[{"type":"thinking","thinking":"answer %d","signature":"sig-%d"},{"type":"text","text":"Answer %d.","citations":null}]`, request, request, request)
		}
		requestOptions := options
		requestOptions.Conversation = conversation
		prepared, err := NewRequest(prompt, requestOptions, RequestSettings{Provider: provider, Operation: "execute"})
		if err != nil {
			t.Fatal(err)
		}
		payloads = append(payloads, jsonValue(t, prepared.Body).(map[string]any))
		steps = append(steps, step)
		completion := prepared.Complete(claudeResponse{Role: "assistant", Content: decodeBlocks(t, content), StopReason: stop, Usage: claudeUsage{InputTokens: 1, OutputTokens: 1}})
		conversation = jsonValue(t, completion.Conversation)

		if step == "tools" {
			loopLeft--
			results := make([]claudeBlock, calls)
			for call := range results {
				text := fmt.Sprintf("result %d-%d %s", request, call, strings.Repeat("x", random(3)*400))
				results[call] = claudeBlock{Type: "tool_result", ToolUseID: fmt.Sprintf("call-%d-%d", request, call), ContentText: &text}
			}
			prompt = claudePrompt{Messages: []claudeMessage{{Role: "user", Content: results}}}
		} else {
			loopLeft = random(26)
			text := fmt.Sprintf("Next question %d.", request)
			if random(2) == 0 {
				text = fmt.Sprintf("<heartbeat>[System] Status heartbeat %d</heartbeat>", request)
			}
			prompt = userPrompt(text)
		}
	}
	return payloads, steps
}

func TestClaudeAgentRunKeepsEveryRequestAPrefixOfTheNext(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"claude-sonnet-4-5", "claude-sonnet-4-6", "claude-sonnet-5-5"} {
		for _, provider := range []string{ProviderNameAnthropic, ProviderNameVertexAI, ProviderNameBedrock} {
			for _, seed := range []int{1, 7, 42} {
				t.Run(fmt.Sprintf("%s/%s/%d", model, provider, seed), func(t *testing.T) {
					payloads, steps := runClaudeAgent(t, model, provider, seed, 150)
					for index := 1; index < len(payloads); index++ {
						previous := stripCacheControl(payloads[index-1]).(map[string]any)
						current := stripCacheControl(payloads[index]).(map[string]any)
						if !reflect.DeepEqual(current["tools"], previous["tools"]) || !reflect.DeepEqual(current["system"], previous["system"]) {
							t.Fatalf("request %d changed the static prefix", index)
						}
						previousMessages := previous["messages"].([]any)
						currentMessages := current["messages"].([]any)
						// Models that keep earlier turns' thinking only ever append; for the others, turn boundaries
						// prune thinking, but never before the earliest marked user turn.
						appendOnly := keepsClaudePriorTurnThinking(model) ||
							(steps[index-1] == "tools" && !isPlainUserTurnJSON(previousMessages[len(previousMessages)-1].(map[string]any)))
						unchanged := len(previousMessages)
						if !appendOnly {
							unchanged = 0
							for _, marker := range cacheMarkers(payloads[index]) {
								if marker[0] >= 0 {
									unchanged = marker[0] + 1
									break
								}
							}
						}
						for m := 0; m < unchanged; m++ {
							a, _ := json.Marshal(previousMessages[m])
							b, _ := json.Marshal(currentMessages[m])
							if string(a) != string(b) {
								t.Fatalf("request %d rewrote message %d:\n%s\n%s", index, m, a, b)
							}
						}
					}
				})
			}
		}
	}
}

func TestClaudeAgentCacheMarkers(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"claude-sonnet-4-5", "claude-sonnet-4-6", "claude-sonnet-5-5"} {
		payloads, _ := runClaudeAgent(t, model, ProviderNameAnthropic, 7, 60)
		for i, body := range payloads {
			found := cacheMarkers(body)
			prefix := 0
			for _, marker := range found {
				if marker[0] < 0 {
					prefix++
				}
			}
			if len(found) > 4 || prefix != 1 {
				t.Fatalf("%s request %d: markers = %v", model, i, found)
			}
			messages := body["messages"].([]any)
			last := -1
			for m := len(messages) - 1; m >= 0; m-- {
				if messages[m].(map[string]any)["role"] != "system" {
					last = m
					break
				}
			}
			blocks := messages[last].(map[string]any)["content"].([]any)
			for b := len(blocks) - 1; b >= 0; b-- {
				block := blocks[b].(map[string]any)
				if block["type"] == "thinking" {
					continue
				}
				if block["cache_control"] == nil {
					t.Fatalf("%s request %d: the tail is not marked: %#v", model, i, blocks)
				}
				break
			}
			var want []int
			if !keepsClaudePriorTurnThinking(model) {
				for m := 0; m < last; m++ {
					if isPlainUserTurnJSON(messages[m].(map[string]any)) {
						want = append(want, m)
					}
				}
				if len(want) > 2 {
					want = want[len(want)-2:]
				}
			}
			var got []int
			for _, marker := range found {
				if marker[0] >= 0 && marker[0] != last {
					got = append(got, marker[0])
				}
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%s request %d: user-turn markers = %v, want %v", model, i, got, want)
			}
		}
	}
}

func TestClaudeCachedConversationKeepsHeartbeats(t *testing.T) {
	t.Parallel()

	payloads, _ := runClaudeAgent(t, "claude-sonnet-4-6", ProviderNameAnthropic, 3, 150)
	final := claudeCompactJSON(payloads[len(payloads)-1])
	if !strings.Contains(final, "<heartbeat>") || strings.Contains(final, "[Heartbeat removed from conversation history]") {
		t.Fatal("a cached conversation must keep its heartbeats")
	}
	if keep := claudeHeartbeatKeepTurns(ExecutionOptions{StripHeartbeatsAfterTurns: ptr(2)}); keep == nil || *keep != 2 {
		t.Fatalf("keep = %v", keep)
	}
	if keep := claudeHeartbeatKeepTurns(ExecutionOptions{PromptCacheKey: "agent-x", StripHeartbeatsAfterTurns: ptr(2)}); keep == nil || *keep < 1<<30 {
		t.Fatalf("cached keep = %v", keep)
	}
}

func TestClaudeAgentCacheMarksTheLastToolWithoutSystem(t *testing.T) {
	t.Parallel()

	request, err := NewRequest(userPrompt("hi"), ExecutionOptions{
		Model:          "claude-sonnet-4-6",
		PromptCacheKey: "agent-no-system",
		ModelOptions:   map[string]any{"cache_ttl": "1h"},
		Tools:          []ToolDefinition{{Name: "lookup", Description: ptr("Look up."), InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}}},
	}, RequestSettings{Provider: ProviderNameAnthropic})
	if err != nil {
		t.Fatal(err)
	}
	body := jsonValue(t, request.Body).(map[string]any)
	ttl := map[string]any{"type": "ephemeral", "ttl": "1h"}
	tools := body["tools"].([]any)
	if !reflect.DeepEqual(tools[len(tools)-1].(map[string]any)["cache_control"], ttl) {
		t.Fatalf("tools = %#v", tools)
	}
	if !reflect.DeepEqual(body["messages"], []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "hi", "cache_control": ttl}}}}) {
		t.Fatalf("messages = %#v", body["messages"])
	}
}

func TestClaudeRoutedCacheLayout(t *testing.T) {
	t.Parallel()

	ephemeral := map[string]any{"type": "ephemeral"}
	prompt := claudePrompt{
		System: []claudeBlock{{Type: "text", Text: "system"}},
		Messages: []claudeMessage{{Role: "user", Content: []claudeBlock{
			{Type: "text", Text: "stable context"},
			{Type: "text", Text: "dynamic task"},
		}}},
	}
	tools := []ToolDefinition{{Name: "a", InputSchema: map[string]any{"type": "object"}}, {Name: "b", InputSchema: map[string]any{"type": "object"}}}
	body, _ := payloadFor(t, ExecutionOptions{Model: "claude-sonnet-4-5", PromptCacheKey: "route-1", Tools: tools}, prompt, ProviderNameAnthropic)
	system := body["system"].([]any)
	if !reflect.DeepEqual(system[0].(map[string]any)["cache_control"], ephemeral) {
		t.Fatalf("system = %#v", system)
	}
	toolList := body["tools"].([]any)
	if toolList[0].(map[string]any)["cache_control"] != nil || !reflect.DeepEqual(toolList[1].(map[string]any)["cache_control"], ephemeral) {
		t.Fatalf("tools = %#v", toolList)
	}
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if !reflect.DeepEqual(content[0].(map[string]any)["cache_control"], ephemeral) || content[1].(map[string]any)["cache_control"] != nil {
		t.Fatalf("the stable block before the dynamic one is marked: %#v", content)
	}

	// Without a key, cache_enabled marks the end of the turn before the last once there are four turns.
	long := claudePrompt{Messages: []claudeMessage{
		{Role: "user", Content: []claudeBlock{{Type: "text", Text: "q1"}}},
		{Role: "assistant", Content: []claudeBlock{{Type: "text", Text: "a1"}}},
		{Role: "user", Content: []claudeBlock{{Type: "text", Text: "q2"}}},
		{Role: "assistant", Content: []claudeBlock{{Type: "text", Text: "a2"}}},
		{Role: "user", Content: []claudeBlock{{Type: "text", Text: "q3"}}},
	}}
	body, _ = payloadFor(t, ExecutionOptions{Model: "claude-sonnet-4-5", ModelOptions: map[string]any{"cache_enabled": true}}, long, ProviderNameAnthropic)
	if markers := cacheMarkers(body); !reflect.DeepEqual(markers, [][2]int{{3, 0}}) {
		t.Fatalf("markers = %v", markers)
	}
	body, _ = payloadFor(t, ExecutionOptions{Model: "claude-sonnet-4-5"}, long, ProviderNameAnthropic)
	if markers := cacheMarkers(body); len(markers) != 0 {
		t.Fatalf("caching is off: %v", markers)
	}
}
