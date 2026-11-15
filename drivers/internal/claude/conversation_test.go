package claude

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/vertesia/llumiverse-go/core"
)

// text is a text block.
func text(value string) claudeBlock {
	return claudeBlock{Type: "text", Text: value}
}

// toolUse is a tool_use block.
func toolUse(id, name string) claudeBlock {
	return claudeBlock{Type: "tool_use", ID: id, Name: name, Input: map[string]any{}}
}

// toolResult is a tool_result block with text content.
func toolResult(id, value string) claudeBlock {
	return claudeBlock{Type: "tool_result", ToolUseID: id, Content: []claudeBlock{text(value)}}
}

func TestClaudeFixOrphanedToolResults(t *testing.T) {
	t.Parallel()

	messages := []claudeMessage{
		{Role: "user", Content: []claudeBlock{text("start")}},
		{Role: "assistant", Content: []claudeBlock{toolUse("t1", "lookup")}},
		{Role: "user", Content: []claudeBlock{toolResult("t1", "ok"), toolResult("gone", "stale")}},
		{Role: "assistant", Content: []claudeBlock{text("done")}},
		{Role: "user", Content: []claudeBlock{toolResult("trimmed", "orphan")}},
	}
	fixed := fixClaudeOrphanedToolResults(messages)
	if len(fixed) != 4 {
		t.Fatalf("a turn of orphaned results is dropped: %#v", fixed)
	}
	if got := fixed[2].Content; len(got) != 1 || got[0].ToolUseID != "t1" {
		t.Fatalf("orphaned result kept: %#v", got)
	}
	if len(messages[2].Content) != 2 {
		t.Fatal("the input was modified")
	}
}

func TestClaudeFixOrphanedToolUse(t *testing.T) {
	t.Parallel()

	messages := []claudeMessage{
		{Role: "assistant", Content: []claudeBlock{toolUse("t1", "lookup"), toolUse("t2", "write")}},
		{Role: "user", Content: []claudeBlock{toolResult("t2", "ok"), text("next request")}},
	}
	repaired := fixClaudeOrphanedToolUse(messages)
	first := repaired[1].Content[0]
	if first.Type != "tool_result" || first.ToolUseID != "t1" || first.ContentText == nil || *first.ContentText != `[Tool interrupted: The user stopped the operation before "lookup" could execute.]` {
		t.Fatalf("repaired = %#v", repaired[1].Content)
	}
	if len(repaired[1].Content) != 3 || len(messages[1].Content) != 2 {
		t.Fatalf("synthetic results go into a copy: %#v", messages[1].Content)
	}
}

func TestClaudeMergeKeepsToolResultsFirst(t *testing.T) {
	t.Parallel()

	merged := mergeConsecutiveClaudeUserMessages([]claudeMessage{
		{Role: "assistant", Content: []claudeBlock{toolUse("t1", "a"), toolUse("t2", "b")}},
		{Role: "user", Content: []claudeBlock{toolResult("t1", "one")}},
		{Role: "user", Content: []claudeBlock{text("Continue."), toolResult("t2", "two")}},
	})
	if len(merged) != 2 {
		t.Fatalf("merged = %#v", merged)
	}
	types := []string{}
	for _, block := range merged[1].Content {
		types = append(types, block.Type)
	}
	if !reflect.DeepEqual(types, []string{"tool_result", "tool_result", "text"}) {
		t.Fatalf("types = %v", types)
	}
}

func TestClaudeConversationRoundTripsThroughJSON(t *testing.T) {
	t.Parallel()

	isError := true
	result := "failed"
	stored := claudePrompt{
		System: []claudeBlock{text("stored system")},
		Messages: []claudeMessage{
			{Role: "user", Content: []claudeBlock{text("old question")}},
			{Role: "assistant", Content: []claudeBlock{
				{Type: "thinking", Thinking: "", Signature: "sig"},
				{Type: "redacted_thinking", Data: "sealed"},
				{Type: "tool_use", ID: "t1", Name: "lookup", Input: json.RawMessage(`{"q":"kyoto","n":1}`)},
			}},
			{Role: "user", Content: []claudeBlock{{Type: "tool_result", ToolUseID: "t1", ContentText: &result, IsError: &isError}}},
			{Role: "assistant", Content: decodeBlocks(t, `[{"type":"compaction","content":"summary","encrypted_content":"opaque"},{"type":"server_tool_use","id":"s1","name":"web_search","input":{}}]`)},
		},
	}
	stored = stored.withMeta(3, &claudeToolState{Declared: []claudeDeclaredTool{{Definition: map[string]any{"name": "lookup", "input_schema": map[string]any{"type": "object"}}}}, Visible: []string{"lookup"}})
	first, err := json.Marshal(stored)
	if err != nil {
		t.Fatal(err)
	}
	restored := decodeClaudeConversation(json.RawMessage(first))
	second, err := json.Marshal(restored)
	if err != nil {
		t.Fatal(err)
	}
	if string(first) != string(second) {
		t.Fatalf("round trip changed the conversation:\n%s\n%s", first, second)
	}
	for _, want := range []string{`"thinking":""`, `"_llumiverse_meta":{"turnNumber":3`, `"is_error":true`, `"content":"failed"`, `"encrypted_content":"opaque"`} {
		if !strings.Contains(string(second), want) {
			t.Fatalf("%s missing from %s", want, second)
		}
	}
	// A conversation decoded into generic values (sorted keys) is stable from then on.
	generic := decodeClaudeConversation(jsonValue(t, stored))
	once, _ := json.Marshal(generic)
	twice, _ := json.Marshal(decodeClaudeConversation(jsonValue(t, generic)))
	if string(once) != string(twice) {
		t.Fatalf("generic round trip is not stable:\n%s\n%s", once, twice)
	}
	// A message list and string content are accepted too.
	messages := decodeClaudeConversation(jsonValue(t, []any{map[string]any{"role": "user", "content": "plain"}}))
	if len(messages.Messages) != 1 || messages.Messages[0].Content[0].Text != "plain" {
		t.Fatalf("messages = %#v", messages)
	}
	if got := decodeClaudeConversation("not a conversation"); len(got.Messages) != 0 {
		t.Fatalf("an unrecognized conversation is ignored: %#v", got)
	}
}

func TestClaudeUpdateConversationKeepsOneSchemaBlock(t *testing.T) {
	t.Parallel()

	schema := text(claudeJSONSchemaInstructionPrefix + "\n{}")
	conversation := claudePrompt{System: []claudeBlock{text("system"), schema, schema}}
	updated := updateClaudeConversation(conversation, claudePrompt{System: []claudeBlock{schema}, Messages: []claudeMessage{{Role: "user", Content: []claudeBlock{text("q")}}}})
	if len(updated.System) != 2 || updated.System[0].Text != "system" {
		t.Fatalf("system = %#v", updated.System)
	}
	plain := updateClaudeConversation(claudePrompt{System: []claudeBlock{text("a")}}, claudePrompt{System: []claudeBlock{text("b")}})
	if len(plain.System) != 2 {
		t.Fatalf("plain system = %#v", plain.System)
	}
}

func TestClaudePrepareStoresTheContinuationTurn(t *testing.T) {
	t.Parallel()

	stored := claudePrompt{Messages: []claudeMessage{
		{Role: "user", Content: []claudeBlock{text("go")}},
		{Role: "assistant", Content: []claudeBlock{text("partial")}},
	}}
	prepared := prepareClaudeConversation(ExecutionOptions{Model: "claude-sonnet-4-6", Conversation: stored}, claudePrompt{}, ProviderNameAnthropic)
	if last := prepared.Messages[len(prepared.Messages)-1]; last.Role != "user" || last.Content[0].Text != "Continue." {
		t.Fatalf("messages = %#v", prepared.Messages)
	}
	older := prepareClaudeConversation(ExecutionOptions{Model: "claude-sonnet-4-5", Conversation: stored}, claudePrompt{}, ProviderNameAnthropic)
	if older.Messages[len(older.Messages)-1].Role != "assistant" {
		t.Fatalf("older models keep the prefill: %#v", older.Messages)
	}
}

func TestClaudeFinalizePrunesThinkingOutsideTheActiveLoop(t *testing.T) {
	t.Parallel()

	thinking := claudeBlock{Type: "thinking", Thinking: "plan", Signature: "sig"}
	conversation := claudePrompt{Messages: []claudeMessage{
		{Role: "user", Content: []claudeBlock{text("q1")}},
		{Role: "assistant", Content: []claudeBlock{thinking, text("a1")}},
		{Role: "user", Content: []claudeBlock{text("q2")}},
		{Role: "assistant", Content: []claudeBlock{thinking, toolUse("t1", "lookup")}},
		{Role: "user", Content: []claudeBlock{toolResult("t1", "ok")}},
	}}
	response := claudeMessage{Role: "assistant", Content: []claudeBlock{thinking, toolUse("t2", "lookup")}}
	pruned := finalizeClaudeConversation(conversation, response, ExecutionOptions{Model: "claude-sonnet-4-5"})
	if pruned.Messages[1].Content[0].Type == "thinking" {
		t.Fatalf("finished turn kept its thinking: %#v", pruned.Messages[1])
	}
	if pruned.Messages[3].Content[0].Type != "thinking" || pruned.Messages[5].Content[0].Type != "thinking" {
		t.Fatalf("the active tool loop must keep its thinking: %#v", pruned.Messages)
	}
	if pruned.turn() != 1 {
		t.Fatalf("turn = %d", pruned.turn())
	}
	kept := finalizeClaudeConversation(conversation, response, ExecutionOptions{Model: "claude-sonnet-4-6"})
	if kept.Messages[1].Content[0].Type != "thinking" {
		t.Fatalf("4.6+ keeps earlier thinking: %#v", kept.Messages[1])
	}
}

func TestClaudeFinalizeRetention(t *testing.T) {
	t.Parallel()

	long := strings.Repeat("é", 50)
	conversation := claudePrompt{
		System: []claudeBlock{text(long)},
		Messages: []claudeMessage{
			{Role: "user", Content: []claudeBlock{
				text("<heartbeat>working</heartbeat>"),
				{Type: "image", Source: &claudeSource{Type: "base64", MediaType: "image/png", Data: "AAAA"}},
				{Type: "document", Source: &claudeSource{Type: "base64", MediaType: "application/pdf", Data: "AAAA"}},
				{Type: "document", Title: "notes", Source: &claudeSource{Type: "text", MediaType: "text/plain", Data: "notes"}},
			}},
			{Role: "assistant", Content: []claudeBlock{{Type: "thinking", Thinking: long, Signature: "sig"}, text(long)}},
		},
	}
	strip := 0
	options := ExecutionOptions{Model: "claude-sonnet-4-6", StripImagesAfterTurns: &strip, StripTextMaxTokens: 2}
	finalized := finalizeClaudeConversation(conversation, claudeMessage{Content: []claudeBlock{text("answer")}}, options)
	user := finalized.Messages[0].Content
	if user[0].Text != core.ConversationHeartbeatPlaceholder || user[1].Text != core.ConversationImagePlaceholder || user[2].Text != core.ConversationDocumentPlaceholder || user[3].Type != "document" {
		t.Fatalf("user = %#v", user)
	}
	assistant := finalized.Messages[1].Content
	if assistant[0].Thinking != long {
		t.Fatal("thinking is sent back verbatim")
	}
	if !strings.HasSuffix(assistant[1].Text, core.ConversationTextTruncatedMarker) || !strings.HasPrefix(assistant[1].Text, "éééé") {
		t.Fatalf("text = %q", assistant[1].Text)
	}
	// Prompt caching keeps the history byte-stable: no truncation, no heartbeat stripping.
	options.PromptCacheKey = "agent-run"
	cached := finalizeClaudeConversation(conversation, claudeMessage{Content: []claudeBlock{text("answer")}}, options)
	if cached.Messages[0].Content[0].Text != "<heartbeat>working</heartbeat>" || cached.Messages[1].Content[1].Text != long {
		t.Fatalf("cached = %#v", cached.Messages)
	}
	if conversation.Messages[0].Content[0].Text != "<heartbeat>working</heartbeat>" {
		t.Fatal("the input was modified")
	}
}

func TestClaudeFinalizeTrimsToTheLatestCompaction(t *testing.T) {
	t.Parallel()

	conversation := claudePrompt{Messages: []claudeMessage{
		{Role: "user", Content: []claudeBlock{text("old")}},
		{Role: "assistant", Content: decodeBlocks(t, `[{"type":"compaction","content":"first summary"},{"type":"text","text":"a1"}]`)},
		{Role: "user", Content: []claudeBlock{text("q2")}},
	}}
	response := claudeMessage{Role: "assistant", Content: decodeBlocks(t, `[{"type":"compaction","content":null},{"type":"compaction","content":"second summary","encrypted_content":"x"},{"type":"text","text":"a2"}]`)}
	finalized := finalizeClaudeConversation(conversation, response, ExecutionOptions{Model: "claude-opus-4-8"})
	if len(finalized.Messages) != 1 || len(finalized.Messages[0].Content) != 2 {
		t.Fatalf("messages = %#v", finalized.Messages)
	}
	if summary, ok := claudeCompactionSummary(finalized.Messages[0].Content[0]); !ok || summary != "second summary" {
		t.Fatalf("first block = %#v", finalized.Messages[0].Content[0])
	}
	unchanged := trimClaudeConversationToCompaction(finalized)
	if !reflect.DeepEqual(unchanged, finalized) {
		t.Fatal("a conversation that starts with its summary is unchanged")
	}
}

func TestClaudeStreamingFallbackKeepsRecentMessagesWhole(t *testing.T) {
	t.Parallel()

	long := strings.Repeat("x", 20000)
	var messages []claudeMessage
	for i := 0; i < 14; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		messages = append(messages, claudeMessage{Role: role, Content: []claudeBlock{text(long)}})
	}
	completion := &Completion{Result: []CompletionResult{newThoughtsResult("hidden"), newTextResult("answer")}}
	built := buildClaudeStreamingConversation(claudePrompt{System: []claudeBlock{text(long)}, Messages: messages}, completion, ExecutionOptions{StripTextMaxTokens: 10000})
	if len(built.Messages) != 15 || built.Messages[14].Content[0].Text != "answer" {
		t.Fatalf("messages = %d, last = %#v", len(built.Messages), built.Messages[len(built.Messages)-1])
	}
	cutoff := core.ConversationTruncationCutoff(15, claudeKeepRecentMessages)
	for i, message := range built.Messages {
		got := message.Content[0].Text
		if i < cutoff {
			if got != core.TruncateConversationText(long, claudeOldMessageTextMaxTokens) {
				t.Fatalf("message %d is not clamped to 2000 tokens: %d chars", i, len(got))
			}
		} else if i < 14 && got != long {
			t.Fatalf("recent message %d was truncated", i)
		}
	}
	if built.System[0].Text != long {
		t.Fatal("turn-aware truncation leaves the system prompt alone")
	}
	short := buildClaudeStreamingConversation(claudePrompt{Messages: messages[:3]}, completion, ExecutionOptions{StripTextMaxTokens: 10})
	if short.Messages[0].Content[0].Text != long {
		t.Fatal("a short conversation stays whole")
	}
	cached := buildClaudeStreamingConversation(claudePrompt{Messages: messages}, completion, ExecutionOptions{StripTextMaxTokens: 10, PromptCacheKey: "k"})
	if cached.Messages[0].Content[0].Text != long {
		t.Fatal("a cached conversation is never truncated")
	}
}
