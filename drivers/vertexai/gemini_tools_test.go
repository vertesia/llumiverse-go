package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func fnCall(id, name string) geminiPart {
	return geminiPart{FunctionCall: &geminiFunctionCall{ID: id, Name: name, Args: map[string]any{}}}
}

func fnResponse(id, name string) geminiPart {
	return geminiPart{FunctionResponse: &geminiFunctionResult{ID: id, Name: name, Response: map[string]any{}}}
}

// TestFixOrphanedGeminiToolResults mirrors gemini-orphaned-tool-result.test.ts.
func TestFixOrphanedGeminiToolResults(t *testing.T) {
	t.Parallel()

	if got := fixOrphanedGeminiToolResults(nil); len(got) != 0 {
		t.Fatalf("empty = %#v", got)
	}
	matched := []geminiContent{
		{Role: "model", Parts: []geminiPart{fnCall("", "a"), fnCall("", "b")}},
		{Role: "user", Parts: []geminiPart{fnResponse("", "a"), fnResponse("", "b")}},
	}
	if got := fixOrphanedGeminiToolResults(matched); !reflect.DeepEqual(got, matched) {
		t.Fatalf("matched batch = %#v", got)
	}

	dropped := fixOrphanedGeminiToolResults([]geminiContent{
		{Role: "model", Parts: []geminiPart{fnCall("", "a")}},
		{Role: "user", Parts: []geminiPart{fnResponse("", "a"), fnResponse("", "gone")}},
	})
	if len(dropped[1].Parts) != 1 || dropped[1].Parts[0].FunctionResponse.Name != "a" {
		t.Fatalf("orphan not dropped: %#v", dropped)
	}

	emptied := fixOrphanedGeminiToolResults([]geminiContent{
		{Role: "user", Parts: []geminiPart{{Text: "[summary of prior work]"}}},
		{Role: "user", Parts: []geminiPart{fnResponse("", "gone")}},
	})
	if len(emptied) != 1 || emptied[0].Parts[0].Text != "[summary of prior work]" {
		t.Fatalf("emptied content kept: %#v", emptied)
	}

	mixed := fixOrphanedGeminiToolResults([]geminiContent{
		{Role: "model", Parts: []geminiPart{{Text: "thinking"}}},
		{Role: "user", Parts: []geminiPart{fnResponse("", "gone"), {Text: "continue please"}}},
	})
	if len(mixed[1].Parts) != 1 || mixed[1].Parts[0].Text != "continue please" {
		t.Fatalf("text part not preserved: %#v", mixed)
	}

	// Split parallel results stay paired with the model turn that called them.
	split := []geminiContent{
		{Role: "model", Parts: []geminiPart{fnCall("", "a"), fnCall("", "b")}},
		{Role: "user", Parts: []geminiPart{fnResponse("", "a")}},
		{Role: "user", Parts: []geminiPart{fnResponse("", "b")}},
	}
	if got := fixOrphanedGeminiToolResults(split); !reflect.DeepEqual(got, split) {
		t.Fatalf("split batch = %#v", got)
	}

	// A response with a call id must match that id, not just the name.
	byID := fixOrphanedGeminiToolResults([]geminiContent{
		{Role: "model", Parts: []geminiPart{fnCall("call_1", "a"), fnCall("call_2", "a")}},
		{Role: "user", Parts: []geminiPart{fnResponse("call_1", "a"), fnResponse("call_2", "a"), fnResponse("call_0", "a")}},
	})
	var ids []string
	for _, part := range byID[1].Parts {
		ids = append(ids, part.FunctionResponse.ID)
	}
	if !reflect.DeepEqual(ids, []string{"call_1", "call_2"}) {
		t.Fatalf("paired ids = %v", ids)
	}
	byName := []geminiContent{
		{Role: "model", Parts: []geminiPart{fnCall("call_1", "a")}},
		{Role: "user", Parts: []geminiPart{fnResponse("", "a")}},
	}
	if got := fixOrphanedGeminiToolResults(byName); !reflect.DeepEqual(got, byName) {
		t.Fatalf("name pairing = %#v", got)
	}
}

func TestGeminiRequestContentsRecombinesAndStripsGeneratedIDs(t *testing.T) {
	t.Parallel()

	generatedA := geminiGeneratedCallIDPrefix + "a"
	generatedB := geminiGeneratedCallIDPrefix + "b"
	stored := []geminiContent{
		{Role: "user", Parts: []geminiPart{{Text: "stable"}}},
		{Role: "user", Parts: []geminiPart{{Text: "task"}}},
		{Role: "model", Parts: []geminiPart{
			{FunctionCall: &geminiFunctionCall{ID: generatedA, Name: "a"}, ThoughtSignature: "sig"},
			fnCall(generatedB, "a"),
			fnCall("call_3", "a"),
		}},
		{Role: "user", Parts: []geminiPart{fnResponse(generatedA, "a")}},
		{Role: "user", Parts: []geminiPart{fnResponse(generatedB, "a")}},
		{Role: "user", Parts: []geminiPart{fnResponse("call_3", "a")}},
	}
	before, _ := json.Marshal(stored)
	got := geminiRequestContents(stored, true)
	after, _ := json.Marshal(stored)
	if string(before) != string(after) {
		t.Fatal("request preparation modified the stored conversation")
	}
	// Text contents keep their boundaries; function responses are one turn.
	if len(got) != 4 || got[0].Parts[0].Text != "stable" || got[1].Parts[0].Text != "task" {
		t.Fatalf("contents = %#v", got)
	}
	calls, responses := got[2].Parts, got[3].Parts
	if len(calls) != 3 || len(responses) != 3 {
		t.Fatalf("calls=%d responses=%d", len(calls), len(responses))
	}
	if calls[0].FunctionCall.ID != "" || calls[0].ThoughtSignature != "sig" || calls[1].FunctionCall.ID != "" || calls[2].FunctionCall.ID != "call_3" {
		t.Fatalf("calls = %#v %#v %#v", calls[0].FunctionCall, calls[1].FunctionCall, calls[2].FunctionCall)
	}
	if responses[0].FunctionResponse.ID != "" || responses[1].FunctionResponse.ID != "" || responses[2].FunctionResponse.ID != "call_3" {
		t.Fatal("generated ids reached the request")
	}
}

func TestAssignGeminiFunctionCallIDs(t *testing.T) {
	t.Parallel()

	used := map[string]bool{"call_0": true}
	parts := []geminiPart{fnCall("", "a"), fnCall("", "a"), fnCall("call_1", "a"), fnCall("call_1", "a"), fnCall("call_0", "a"), {Text: "x"}}
	assignGeminiFunctionCallIDs(parts, used)
	seen := map[string]bool{}
	for i, part := range parts[:5] {
		id := part.FunctionCall.ID
		if id == "" || seen[id] {
			t.Fatalf("call %d id %q is not unique", i, id)
		}
		seen[id] = true
	}
	if parts[2].FunctionCall.ID != "call_1" {
		t.Fatalf("Vertex id not kept: %q", parts[2].FunctionCall.ID)
	}
	for _, i := range []int{0, 1, 3, 4} {
		if !isGeminiGeneratedCallID(parts[i].FunctionCall.ID) {
			t.Fatalf("call %d id %q is not generated", i, parts[i].FunctionCall.ID)
		}
	}
}

var parallelDocumentNames = []string{"alpha", "beta", "gamma", "delta"}

// parallelCreateDocumentResponse is a Gemini turn calling one tool four times.
func parallelCreateDocumentResponse(ids []string) string {
	parts := make([]string, 0, len(parallelDocumentNames))
	for i, name := range parallelDocumentNames {
		id := ""
		if ids != nil {
			id = fmt.Sprintf(`"id":%q,`, ids[i])
		}
		signature := ""
		if i == 0 {
			signature = `,"thoughtSignature":"first-signature"`
		}
		parts = append(parts, fmt.Sprintf(`{"functionCall":{%s"name":"create_document","args":{"name":%q}}%s}`, id, name, signature))
	}
	return `{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[` + strings.Join(parts, ",") + `]}}]}`
}

// recordingGeminiServer answers each request with the next canned body (SSE for
// streams) and records the decoded request payloads.
type recordingGeminiServer struct {
	mu       sync.Mutex
	requests []map[string]any
	bodies   []string
}

func (s *recordingGeminiServer) handler(t *testing.T) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		s.mu.Lock()
		s.requests = append(s.requests, payload)
		body := `{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"text":"done"}]}}]}`
		if len(s.bodies) > 0 {
			body, s.bodies = s.bodies[0], s.bodies[1:]
		}
		s.mu.Unlock()
		if strings.Contains(r.URL.Path, "streamGenerateContent") {
			_, _ = w.Write([]byte("data: " + body + "\n\n"))
			return
		}
		_, _ = w.Write([]byte(body))
	}
}

func (s *recordingGeminiServer) last() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requests[len(s.requests)-1]
}

func drainStream(t *testing.T, stream CompletionStream) *ExecutionResponse {
	t.Helper()
	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	return stream.Completion()
}

func requestTurn(payload map[string]any, index int) map[string]any {
	contents := payload["contents"].([]any)
	if index < 0 {
		index += len(contents)
	}
	return contents[index].(map[string]any)
}

func turnParts(turn map[string]any, key string) []map[string]any {
	var out []map[string]any
	for _, part := range turn["parts"].([]any) {
		if value, ok := part.(map[string]any)[key]; ok {
			entry := map[string]any{key: value}
			if signature, ok := part.(map[string]any)["thoughtSignature"]; ok {
				entry["thoughtSignature"] = signature
			}
			out = append(out, entry)
		}
	}
	return out
}

var createDocumentTool = []ToolDefinition{{Name: "create_document", InputSchema: map[string]any{"type": "object"}}}

// sendParallelResults answers one call per tool-use id and streams the follow-up request.
func sendParallelResults(t *testing.T, driver *VertexAIDriver, conversation any, toolUseIDs []string) {
	t.Helper()
	segments := make([]PromptSegment, len(toolUseIDs))
	for i, id := range toolUseIDs {
		segments[i] = PromptSegment{Role: PromptRoleTool, ToolUseID: id, Content: fmt.Sprintf(`{"created":%q}`, parallelDocumentNames[i])}
		if i == 0 {
			segments[i].ThoughtSignature = "first-signature"
		}
	}
	stream, err := driver.Stream(context.Background(), segments, ExecutionOptions{Model: "gemini-2.5-flash", Conversation: conversation, Tools: createDocumentTool})
	if err != nil {
		t.Fatal(err)
	}
	drainStream(t, stream)
}

// TestGeminiParallelCallsToOneToolStayDistinct mirrors gemini-parallel-same-tool-calls.test.ts.
func TestGeminiParallelCallsToOneToolStayDistinct(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name      string
		vertexIDs []string
	}{
		{name: "generated ids", vertexIDs: nil},
		{name: "vertex ids", vertexIDs: []string{"call_1", "call_2", "call_3", "call_4"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			recorder := &recordingGeminiServer{bodies: []string{parallelCreateDocumentResponse(tc.vertexIDs)}}
			server := httptest.NewServer(recorder.handler(t))
			defer server.Close()
			driver := newTestVertexDriver(t, server.URL)

			stream, err := driver.Stream(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "create four documents"}}, ExecutionOptions{Model: "gemini-2.5-flash", Tools: createDocumentTool})
			if err != nil {
				t.Fatal(err)
			}
			resp := drainStream(t, stream)
			if len(resp.ToolUse) != 4 {
				t.Fatalf("tool uses = %#v", resp.ToolUse)
			}
			ids := make([]string, 0, 4)
			for i, tool := range resp.ToolUse {
				if tool.ToolName != "create_document" || (*tool.ToolInput)["name"] != parallelDocumentNames[i] {
					t.Fatalf("tool %d = %#v", i, tool)
				}
				ids = append(ids, tool.ID)
			}
			if tc.vertexIDs != nil && !reflect.DeepEqual(ids, tc.vertexIDs) {
				t.Fatalf("ids = %v, want Vertex ids", ids)
			}
			stored := resp.Conversation.(geminiPrompt).Contents
			var storedIDs []string
			for _, part := range stored[len(stored)-1].Parts {
				storedIDs = append(storedIDs, part.FunctionCall.ID)
			}
			if !reflect.DeepEqual(storedIDs, ids) {
				t.Fatalf("stored call ids = %v, tool-use ids = %v", storedIDs, ids)
			}

			sendParallelResults(t, driver, resp.Conversation, ids)
			request := recorder.last()
			if requestTurn(request, -2)["role"] != "model" || requestTurn(request, -1)["role"] != "user" {
				t.Fatalf("request = %#v", request["contents"])
			}
			calls := turnParts(requestTurn(request, -2), "functionCall")
			responses := turnParts(requestTurn(request, -1), "functionResponse")
			if len(calls) != 4 || len(responses) != 4 {
				t.Fatalf("calls=%d responses=%d, want one user turn answering all four", len(calls), len(responses))
			}
			for i := range calls {
				call := calls[i]["functionCall"].(map[string]any)
				response := responses[i]["functionResponse"].(map[string]any)
				wantResponse := map[string]any{"name": "create_document", "response": map[string]any{"created": parallelDocumentNames[i]}}
				if tc.vertexIDs == nil {
					// Vertex never sees an id it did not issue.
					if _, has := call["id"]; has {
						t.Fatalf("generated id sent on call %d", i)
					}
				} else {
					wantResponse["id"] = tc.vertexIDs[i]
					if call["id"] != tc.vertexIDs[i] {
						t.Fatalf("call %d id = %v", i, call["id"])
					}
				}
				if !reflect.DeepEqual(response, wantResponse) {
					t.Fatalf("response %d = %#v, want %#v", i, response, wantResponse)
				}
			}
			if responses[0]["thoughtSignature"] != "first-signature" || responses[1]["thoughtSignature"] != nil {
				t.Fatalf("thought signatures = %#v", responses)
			}
		})
	}
}

func TestGeminiAnswersConversationStoredWithNameIDs(t *testing.T) {
	t.Parallel()

	recorder := &recordingGeminiServer{}
	server := httptest.NewServer(recorder.handler(t))
	defer server.Close()
	driver := newTestVertexDriver(t, server.URL)

	calls := make([]any, 0, 4)
	for _, name := range parallelDocumentNames {
		calls = append(calls, map[string]any{"functionCall": map[string]any{"name": "create_document", "args": map[string]any{"name": name}}})
	}
	// Stored by an earlier driver version: calls without ids, results keyed by the name.
	conversation := map[string]any{"contents": []any{
		map[string]any{"role": "user", "parts": []any{map[string]any{"text": "create four documents"}}},
		map[string]any{"role": "model", "parts": calls},
	}}
	sendParallelResults(t, driver, conversation, []string{"create_document", "create_document", "create_document", "create_document"})
	responses := turnParts(requestTurn(recorder.last(), -1), "functionResponse")
	if len(responses) != 4 {
		t.Fatalf("responses = %#v", responses)
	}
	for i, response := range responses {
		want := map[string]any{"name": "create_document", "response": map[string]any{"created": parallelDocumentNames[i]}}
		if !reflect.DeepEqual(response["functionResponse"], want) {
			t.Fatalf("response %d = %#v", i, response["functionResponse"])
		}
	}
}

func TestGeminiNonStreamedParallelCallsGetOwnIDs(t *testing.T) {
	t.Parallel()

	recorder := &recordingGeminiServer{bodies: []string{parallelCreateDocumentResponse(nil)}}
	server := httptest.NewServer(recorder.handler(t))
	defer server.Close()
	driver := newTestVertexDriver(t, server.URL)

	resp, err := driver.Execute(context.Background(), []PromptSegment{{Role: PromptRoleUser, Content: "create four documents"}}, ExecutionOptions{Model: "gemini-2.5-flash", Tools: createDocumentTool})
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, tool := range resp.ToolUse {
		seen[tool.ID] = true
	}
	if len(seen) != 4 || resp.FinishReason != "tool_use" {
		t.Fatalf("tool uses = %#v", resp.ToolUse)
	}
	stored := resp.Conversation.(geminiPrompt).Contents
	for i, part := range stored[len(stored)-1].Parts {
		if part.FunctionCall.ID != resp.ToolUse[i].ID {
			t.Fatalf("stored call %d id = %q, tool-use id = %q", i, part.FunctionCall.ID, resp.ToolUse[i].ID)
		}
	}
}

// TestGeminiOrphanedToolResultDroppedFromPayload mirrors the getGeminiPayload orphan cases.
func TestGeminiOrphanedToolResultDroppedFromPayload(t *testing.T) {
	t.Parallel()

	recorder := &recordingGeminiServer{}
	server := httptest.NewServer(recorder.handler(t))
	defer server.Close()
	driver := newTestVertexDriver(t, server.URL)

	// A compacted conversation lost the model call that this result answers.
	conversation := geminiPrompt{Contents: []geminiContent{{Role: "user", Parts: []geminiPart{{Text: "[summary of prior work]"}}}}}
	_, err := driver.Execute(context.Background(), []PromptSegment{
		{Role: PromptRoleTool, ToolUseID: "search", Content: "result"},
		{Role: PromptRoleUser, Content: "continue"},
	}, ExecutionOptions{Model: "gemini-2.5-flash", Conversation: conversation, Tools: createDocumentTool})
	if err != nil {
		t.Fatal(err)
	}
	contents := recorder.last()["contents"].([]any)
	if len(contents) != 2 || strings.Contains(fmt.Sprint(contents), "functionResponse") {
		t.Fatalf("contents = %#v", contents)
	}
}
