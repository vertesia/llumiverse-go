package core

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"
	"sync"
	"testing"
)

// The tests below mirror the TS client's validation.test.ts.

func resultValues(t *testing.T, results []CompletionResult) []map[string]any {
	t.Helper()
	out := make([]map[string]any, len(results))
	for i, result := range results {
		data, err := json.Marshal(result)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, &out[i]); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

func requireResults(t *testing.T, got []CompletionResult, want ...CompletionResult) {
	t.Helper()
	if g, w := resultValues(t, got), resultValues(t, want); !reflect.DeepEqual(g, w) {
		t.Fatalf("results = %v, want %v", g, w)
	}
}

func TestValidateResultKeepsThoughtsAroundValidatedJSON(t *testing.T) {
	t.Parallel()
	results, err := ValidateResult([]CompletionResult{
		NewThoughtsResult("first thought"),
		NewTextResult(`{"answer":"ok"}`),
		NewThoughtsResult("second thought"),
	}, map[string]any{"type": "object"})
	if err != nil {
		t.Fatal(err)
	}
	requireResults(t, results, NewThoughtsResult("first thought"), NewJSONResult(map[string]any{"answer": "ok"}), NewThoughtsResult("second thought"))
}

func TestValidateResultIgnoresUnregisteredFormats(t *testing.T) {
	t.Parallel()
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"answer": map[string]any{"type": "string", "format": "not-a-registered-format"}},
		"required":   []any{"answer"},
	}
	results, err := ValidateResult([]CompletionResult{NewTextResult(`{"answer":"ok"}`)}, schema)
	if err != nil {
		t.Fatal(err)
	}
	requireResults(t, results, NewJSONResult(map[string]any{"answer": "ok"}))
}

func TestValidateResultRepeatedAndEditedSchemaWithSameID(t *testing.T) {
	t.Parallel()
	id := "https://example.test/schemas/photo-observation.json"
	input := []CompletionResult{NewTextResult(`{"answer":"ok"}`)}
	for attempt := 0; attempt < 3; attempt++ {
		schema := map[string]any{"$id": id, "type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}, "required": []any{"answer"}}
		if _, err := ValidateResult(input, schema); err != nil {
			t.Fatalf("attempt %d: %v", attempt, err)
		}
	}
	edited := "https://example.test/schemas/edited.json"
	if _, err := ValidateResult(input, map[string]any{"$id": edited, "type": "object"}); err != nil {
		t.Fatal(err)
	}
	// Same $id, stricter definition: the new requirement is enforced, not the cached one.
	_, err := ValidateResult(input, map[string]any{"$id": edited, "type": "object", "required": []any{"missing"}})
	if err == nil || !strings.Contains(err.Message, "must have required property") {
		t.Fatalf("err = %#v", err)
	}
}

func invoiceSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"total":    map[string]any{"type": "number"},
			"currency": map[string]any{"type": "string", "default": "EUR"},
			"lines": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type":                 "object",
					"properties":           map[string]any{"sku": map[string]any{"type": "string"}},
					"required":             []any{"sku"},
					"additionalProperties": false,
				},
			},
		},
		"required": []any{"total"},
	}
}

func TestValidateResultCoercesDefaultsAndPrunes(t *testing.T) {
	t.Parallel()
	input := `{"total":"12.5","lines":[{"sku":"A","extra":true}]}`
	want := NewJSONResult(map[string]any{"total": 12.5, "currency": "EUR", "lines": []any{map[string]any{"sku": "A"}}})
	for attempt := 0; attempt < 3; attempt++ {
		results, err := ValidateResult([]CompletionResult{NewTextResult(input)}, invoiceSchema())
		if err != nil {
			t.Fatal(err)
		}
		requireResults(t, results, want)
	}
}

func TestValidateResultReportsAJVStyleErrors(t *testing.T) {
	t.Parallel()
	input := []CompletionResult{NewTextResult(`{"total":1,"lines":[{"sku":"A"},{"name":"no sku"}]}`)}
	for attempt := 0; attempt < 3; attempt++ {
		results, err := ValidateResult(input, invoiceSchema())
		if err == nil {
			t.Fatal("expected a validation error")
		}
		if err.Code != ResultValidationCodeSchema || err.Message != "/lines/1: must have required property 'sku'\n{\"missingProperty\":\"sku\"}" {
			t.Fatalf("err = %#v", err)
		}
		// The results are returned unchanged, and the error carries them.
		requireResults(t, results, input...)
		requireResults(t, err.Data, input...)
	}
}

func TestValidateResultAppliesEditedSchemaImmediately(t *testing.T) {
	t.Parallel()
	input := []CompletionResult{NewTextResult(`{"total":1}`)}
	results, err := ValidateResult(input, invoiceSchema())
	if err != nil {
		t.Fatal(err)
	}
	requireResults(t, results, NewJSONResult(map[string]any{"total": 1, "currency": "EUR"}))
	stricter := invoiceSchema()
	stricter["required"] = []any{"total", "lines"}
	if _, err := ValidateResult(input, stricter); err == nil || !strings.Contains(err.Message, "must have required property 'lines'") {
		t.Fatalf("err = %#v", err)
	}
}

func TestValidateResultConcurrentCallsStaySeparate(t *testing.T) {
	t.Parallel()
	var wg sync.WaitGroup
	failures := make([]string, 40)
	for i := range 40 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			input := fmt.Sprintf(`{"total":"%d"}`, i)
			if i%2 == 1 {
				input = fmt.Sprintf(`{"lines":[{"sku":%d}],"total":"not a number %d"}`, i, i)
			}
			results, err := ValidateResult([]CompletionResult{NewTextResult(input)}, invoiceSchema())
			switch {
			case i%2 == 0 && err != nil:
				failures[i] = err.Message
			case i%2 == 0:
				got := resultValues(t, results)
				want := resultValues(t, []CompletionResult{NewJSONResult(map[string]any{"total": i, "currency": "EUR"})})
				if !reflect.DeepEqual(got, want) {
					failures[i] = fmt.Sprintf("results = %v", got)
				}
			case err == nil || err.Message != "/total: must be number\n{\"type\":\"number\"}":
				failures[i] = fmt.Sprintf("err = %#v", err)
			}
		}(i)
	}
	wg.Wait()
	for i, failure := range failures {
		if failure != "" {
			t.Errorf("call %d: %s", i, failure)
		}
	}
}

// Go-specific coverage of the rules validateResult applies.

func TestValidateResultJSONErrors(t *testing.T) {
	t.Parallel()
	_, err := ValidateResult([]CompletionResult{NewTextResult("no json here")}, map[string]any{"type": "object"})
	if err == nil || err.Code != ResultValidationCodeJSON {
		t.Fatalf("text err = %#v", err)
	}
	_, err = ValidateResult(nil, map[string]any{"type": "object"})
	if err == nil || err.Code != ResultValidationCodeJSON || err.Message != "no JSON compatible response found in completion result" {
		t.Fatalf("empty err = %#v", err)
	}
}

func TestValidateResultRepairsAndExtractsJSON(t *testing.T) {
	t.Parallel()
	for _, text := range []string{
		"Here it is:\n```json\n{\"answer\": \"ok\"}\n```",
		"The answer is {\"answer\": \"ok\"} as requested.",
		"{'answer': 'ok',}",
	} {
		results, err := ValidateResult([]CompletionResult{NewTextResult(text)}, map[string]any{"type": "object"})
		if err != nil {
			t.Fatalf("%q: %v", text, err)
		}
		requireResults(t, results, NewJSONResult(map[string]any{"answer": "ok"}))
	}
}

func TestValidateResultUsesJSONResultWithoutChangingIt(t *testing.T) {
	t.Parallel()
	original := map[string]any{"total": "3"}
	input := []CompletionResult{NewTextResult("ignored"), NewJSONResult(original)}
	results, err := ValidateResult(input, invoiceSchema())
	if err != nil {
		t.Fatal(err)
	}
	requireResults(t, results, NewJSONResult(map[string]any{"total": 3, "currency": "EUR"}))
	if original["total"] != "3" {
		t.Fatal("caller's value modified")
	}
}

func TestValidateResultCoercion(t *testing.T) {
	t.Parallel()
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"count":    map[string]any{"type": "integer"},
			"label":    map[string]any{"type": "string"},
			"done":     map[string]any{"type": "boolean"},
			"tags":     map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			"single":   map[string]any{"type": "string"},
			"optional": map[string]any{"type": []any{"number", "null"}},
			"ref":      map[string]any{"$ref": "#/$defs/amount"},
			"extra":    map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "number"}},
		},
		"$defs": map[string]any{"amount": map[string]any{"type": "number"}},
	}
	input := `{"count":"7","label":42,"done":"true","tags":"one","single":["only"],"optional":"2.5","ref":"9","extra":{"a":"1","b":"x"}}`
	results, err := ValidateResult([]CompletionResult{NewTextResult(input)}, schema)
	if err != nil {
		t.Fatal(err)
	}
	requireResults(t, results, NewJSONResult(map[string]any{
		"count": 7, "label": "42", "done": true, "tags": []any{"one"}, "single": "only", "optional": 2.5, "ref": 9,
		"extra": map[string]any{"a": 1},
	}))
	// A string that is not a whole number stays a string, and fails.
	_, verr := ValidateResult([]CompletionResult{NewTextResult(`{"count":"7.5"}`)}, schema)
	if verr == nil || verr.Message != "/count: must be integer\n{\"type\":\"integer\"}" {
		t.Fatalf("err = %#v", verr)
	}
}

func TestValidateResultIgnoresEmptyOptionalDates(t *testing.T) {
	t.Parallel()
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"due":     map[string]any{"type": "string", "format": "date"},
			"created": map[string]any{"type": "string", "format": "date-time"},
		},
		"required": []any{"created"},
	}
	if _, err := ValidateResult([]CompletionResult{NewTextResult(`{"due":"","created":"2026-01-02T03:04:05Z"}`)}, schema); err != nil {
		t.Fatalf("empty optional date: %v", err)
	}
	_, err := ValidateResult([]CompletionResult{NewTextResult(`{"due":"","created":""}`)}, schema)
	if err == nil || err.Message != "/created: must match format \"date-time\"\n{\"format\":\"date-time\"}" {
		t.Fatalf("empty required date err = %#v", err)
	}
	_, err = ValidateResult([]CompletionResult{NewTextResult(`{"due":"soon","created":"2026-01-02T03:04:05Z"}`)}, schema)
	if err == nil || err.Message != "/due: must match format \"date\"\n{\"format\":\"date\"}" {
		t.Fatalf("bad date err = %#v", err)
	}
}

func TestValidateResultInvalidSchema(t *testing.T) {
	t.Parallel()
	_, err := ValidateResult([]CompletionResult{NewTextResult(`{}`)}, map[string]any{"type": 12})
	if err == nil || err.Code != ResultValidationCodeSchema || !strings.HasPrefix(err.Message, "invalid result schema") {
		t.Fatalf("err = %#v", err)
	}
}

func TestValidatorCacheBounds(t *testing.T) {
	t.Parallel()
	cache := newValidatorCache(2, 100)
	schema := func(n int) map[string]any { return map[string]any{"type": "object", "title": strings.Repeat("x", n)} }
	for _, n := range []int{1, 2, 3} {
		if _, err := cache.get(schema(n)); err != nil {
			t.Fatal(err)
		}
	}
	if got := cache.len(); got != 2 {
		t.Fatalf("entries = %d, want 2", got)
	}
	// A schema over the character budget evicts every other entry and is cached alone.
	if _, err := cache.get(schema(100)); err != nil {
		t.Fatal(err)
	}
	if got := cache.len(); got != 1 {
		t.Fatalf("entries = %d, want 1", got)
	}
	first, _ := cache.get(schema(100))
	second, _ := cache.get(schema(100))
	if first != second {
		t.Fatal("validator not reused")
	}
}

func TestExecuteWithPromptReportsValidationError(t *testing.T) {
	t.Parallel()
	driver := &fakeDriver{provider: ProviderOpenAI, prompt: "prompt"}
	input := []CompletionResult{NewTextResult(`{"total":"many"}`)}
	resp, err := ExecuteWithPrompt(context.Background(), driver, nil, ExecutionOptions{Model: "m", ResultSchema: invoiceSchema()},
		func(context.Context, any, ExecutionOptions) (*Completion, error) {
			return &Completion{Result: input}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Error == nil || resp.Error.Code != ResultValidationCodeSchema || resp.Error.Message != "/total: must be number\n{\"type\":\"number\"}" {
		t.Fatalf("Error = %#v", resp.Error)
	}
	requireResults(t, resp.Result, input...)

	// Tool calls, and media-only outputs, are not validated.
	for _, completion := range []*Completion{
		{Result: input, ToolUse: []ToolUse{{ID: "t", ToolName: "lookup"}}},
		{Result: []CompletionResult{NewImageResult("https://example.test/image.png")}},
	} {
		resp, err := ExecuteWithPrompt(context.Background(), driver, nil, ExecutionOptions{Model: "m", ResultSchema: invoiceSchema()},
			func(context.Context, any, ExecutionOptions) (*Completion, error) { return completion, nil })
		if err != nil || resp.Error != nil {
			t.Fatalf("err = %v, Error = %#v", err, resp.Error)
		}
	}
}

func TestStreamWithPromptValidatesTheFinalCompletion(t *testing.T) {
	t.Parallel()
	driver := &fakeDriver{provider: ProviderOpenAI, prompt: "prompt"}
	run := func(chunks ...string) *ExecutionResponse {
		stream, err := StreamWithPrompt(context.Background(), driver, nil, ExecutionOptions{Model: "m", ResultSchema: invoiceSchema()},
			func(context.Context, any, ExecutionOptions) (CompletionStream, error) {
				ch := make(chan StreamItem, len(chunks))
				for _, chunk := range chunks {
					ch <- StreamItem{Chunk: CompletionChunk{Result: []CompletionResult{NewTextResult(chunk)}}}
				}
				close(ch)
				return NewChannelStream(ch, &ExecutionResponse{}, nil), nil
			})
		if err != nil {
			t.Fatal(err)
		}
		for {
			if _, err := stream.Recv(); err == io.EOF {
				break
			} else if err != nil {
				t.Fatal(err)
			}
		}
		return stream.Completion()
	}
	ok := run(`{"total":`, `"4"}`)
	if ok.Error != nil {
		t.Fatalf("Error = %#v", ok.Error)
	}
	requireResults(t, ok.Result, NewJSONResult(map[string]any{"total": 4, "currency": "EUR"}))
	bad := run(`{"lines":[]}`)
	// The root instance path is empty, as in AJV.
	if bad.Error == nil || bad.Error.Message != ": must have required property 'total'\n{\"missingProperty\":\"total\"}" {
		t.Fatalf("Error = %#v", bad.Error)
	}
}
