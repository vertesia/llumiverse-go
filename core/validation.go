package core

import (
	"container/list"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/kaptinlin/jsonrepair"
	"github.com/santhosh-tekuri/jsonschema/v6"
	"github.com/santhosh-tekuri/jsonschema/v6/kind"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

var englishPrinter = message.NewPrinter(language.English)

// Result validation error codes, set on ResultValidationError.Code.
const (
	// ResultValidationCodeJSON means no result could be parsed as JSON.
	ResultValidationCodeJSON = "json_error"
	// ResultValidationCodeSchema means the JSON result does not match the result schema.
	ResultValidationCodeSchema = "validation_error"
)

// ValidateResult checks completion results against a JSON Schema, as the TS client's validateResult does with AJV.
//
// The JSON is the first JSON result or, failing that, the first text result that parses as JSON (fenced or
// surrounded by prose, and repaired when it is malformed). Before it is validated the JSON is prepared the way AJV's
// coerceTypes "array", useDefaults and removeAdditional "failing" options prepare it:
//   - a value whose type the schema does not allow is coerced to one it allows when the conversion is lossless
//     ("12.5" to 12.5, 1 to "1", "true" to true, a scalar to a one-item array and back);
//   - a missing property with a schema default gets the default;
//   - a property that additionalProperties rejects is removed.
//
// These steps follow properties, patternProperties, additionalProperties, items, allOf and local $ref; they are not
// applied inside anyOf, oneOf, not or if/then/else. An empty string that fails a date or date-time format is
// accepted when its property is not required.
//
// On success the content is replaced by one JSON result and thoughts results are kept in place. On failure the
// results are returned unchanged with a ResultValidationError whose Code is ResultValidationCodeJSON or
// ResultValidationCodeSchema; Data holds the results that failed.
func ValidateResult(results []CompletionResult, schema map[string]any) ([]CompletionResult, *ResultValidationError) {
	value, err := resultJSON(results)
	if err != nil {
		return results, &ResultValidationError{Code: ResultValidationCodeJSON, Message: err.Error(), Data: results}
	}
	validator, err := resultValidators.get(schema)
	if err != nil {
		return results, &ResultValidationError{Code: ResultValidationCodeSchema, Message: err.Error(), Data: results}
	}
	value, err = validator.validate(value)
	if err != nil {
		return results, &ResultValidationError{Code: ResultValidationCodeSchema, Message: err.Error(), Data: results}
	}
	validated := NewJSONResult(value)
	first := -1
	for i, result := range results {
		if result.Type() != ResultTypeThoughts {
			first = i
			break
		}
	}
	if first == -1 {
		return []CompletionResult{validated}, nil
	}
	out := make([]CompletionResult, 0, len(results))
	for i, result := range results {
		switch {
		case result.Type() == ResultTypeThoughts:
			out = append(out, result)
		case i == first:
			out = append(out, validated)
		}
	}
	return out, nil
}

// validateCompletion validates a completion against options.ResultSchema, setting completion.Error when it does
// not match. A completion with tool calls or an error is not validated, nor one without text or JSON content (an
// image, video or audio output), as the TS client validates only the text pathway.
func validateCompletion(completion *Completion, options ExecutionOptions) {
	if completion == nil || options.ResultSchema == nil || completion.Error != nil || len(completion.ToolUse) > 0 {
		return
	}
	if !hasTextOrJSONResult(completion.Result) && len(completion.Result) > 0 {
		return
	}
	results, validationErr := ValidateResult(completion.Result, options.ResultSchema)
	if validationErr != nil {
		completion.Error = validationErr
		return
	}
	completion.Result = results
}

func hasTextOrJSONResult(results []CompletionResult) bool {
	for _, result := range results {
		switch result.Type() {
		case ResultTypeText, ResultTypeJSON, ResultTypeThoughts:
			return true
		}
	}
	return false
}

// resultJSON returns a copy of the first JSON result, or the first text result parsed as JSON.
func resultJSON(results []CompletionResult) (any, error) {
	for _, result := range results {
		if result.Type() == ResultTypeJSON {
			parsed, err := result.AsJSONResult()
			if err != nil {
				return nil, err
			}
			return cloneJSON(parsed.Value)
		}
	}
	var lastErr error
	for _, result := range results {
		if result.Type() != ResultTypeText {
			continue
		}
		value, err := parseResultJSON(result.String())
		if err == nil {
			return value, nil
		}
		lastErr = err
	}
	if lastErr == nil {
		lastErr = errors.New("no JSON compatible response found in completion result")
	}
	return nil, lastErr
}

// parseResultJSON parses JSON out of model text: the whole text, a fenced block, or the outermost object or array,
// repairing it when it does not parse.
func parseResultJSON(text string) (any, error) {
	candidates := jsonResultCandidates(text)
	if len(candidates) == 0 {
		return nil, errors.New("unexpected end of JSON input")
	}
	var firstErr error
	for _, candidate := range candidates {
		var value any
		err := json.Unmarshal([]byte(candidate), &value)
		if err == nil {
			return value, nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	for _, candidate := range candidates {
		// Only an object or array is repaired, as the TS client repairs the text from the first "{" to the last
		// "}": prose would otherwise be repaired into a JSON string.
		if !strings.HasPrefix(candidate, "{") && !strings.HasPrefix(candidate, "[") {
			continue
		}
		repaired, err := jsonrepair.Repair(candidate)
		if err != nil {
			continue
		}
		var value any
		if json.Unmarshal([]byte(repaired), &value) == nil {
			return value, nil
		}
	}
	return nil, firstErr
}

func cloneJSON(value any) (any, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var out any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// resultValidators holds compiled validators keyed by schema content. It is bounded like the TS cache: at most
// 1,000 schemas and 500,000 characters of schema JSON, a larger schema being cached alone.
var resultValidators = newValidatorCache(1000, 500_000)

type validatorCache struct {
	mu        sync.Mutex
	maxItems  int
	maxChars  int
	chars     int
	entries   map[string]*list.Element
	lru       *list.List
	compileFn func(key string) (*resultValidator, error)
}

type validatorCacheEntry struct {
	key       string
	validator *resultValidator
	size      int
}

func newValidatorCache(maxItems, maxChars int) *validatorCache {
	return &validatorCache{maxItems: maxItems, maxChars: maxChars, entries: map[string]*list.Element{}, lru: list.New(), compileFn: compileResultValidator}
}

// get returns the validator of a schema. The key is the schema's JSON, so an edited schema always compiles anew, and
// the validator is compiled from that JSON, never from the caller's map, which the caller may change later.
func (c *validatorCache) get(schema map[string]any) (*resultValidator, error) {
	data, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("invalid result schema: %w", err)
	}
	key := string(data)
	c.mu.Lock()
	if element, ok := c.entries[key]; ok {
		c.lru.MoveToFront(element)
		c.mu.Unlock()
		return element.Value.(*validatorCacheEntry).validator, nil
	}
	c.mu.Unlock()
	validator, err := c.compileFn(key)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if element, ok := c.entries[key]; ok {
		c.lru.MoveToFront(element)
		return element.Value.(*validatorCacheEntry).validator, nil
	}
	size := min(len(key), c.maxChars)
	for c.lru.Len() > 0 && (c.lru.Len() >= c.maxItems || c.chars+size > c.maxChars) {
		oldest := c.lru.Back()
		entry := oldest.Value.(*validatorCacheEntry)
		c.lru.Remove(oldest)
		delete(c.entries, entry.key)
		c.chars -= entry.size
	}
	c.entries[key] = c.lru.PushFront(&validatorCacheEntry{key: key, validator: validator, size: size})
	c.chars += size
	return validator, nil
}

func (c *validatorCache) len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lru.Len()
}

const resultSchemaURL = "llumiverse:result-schema.json"

// resultValidator is one compiled result schema. Each has its own compiler, so a schema edited under the same $id
// compiles without clashing with the previous version.
type resultValidator struct {
	doc      any
	schema   *jsonschema.Schema
	mu       sync.Mutex
	compiler *jsonschema.Compiler
	subs     map[string]*jsonschema.Schema
}

func compileResultValidator(key string) (*resultValidator, error) {
	doc, err := jsonschema.UnmarshalJSON(strings.NewReader(key))
	if err != nil {
		return nil, fmt.Errorf("invalid result schema: %w", err)
	}
	compiler := jsonschema.NewCompiler()
	// AJV's default draft, with format assertions as ajv-formats adds them. Unknown formats are ignored.
	compiler.DefaultDraft(jsonschema.Draft7)
	compiler.AssertFormat()
	if err := compiler.AddResource(resultSchemaURL, doc); err != nil {
		return nil, fmt.Errorf("invalid result schema: %w", err)
	}
	schema, err := compiler.Compile(resultSchemaURL)
	if err != nil {
		return nil, fmt.Errorf("invalid result schema: %w", err)
	}
	return &resultValidator{doc: plainJSON(doc), schema: schema, compiler: compiler, subs: map[string]*jsonschema.Schema{}}, nil
}

// plainJSON converts the json.Number values of a document decoded by jsonschema.UnmarshalJSON to float64, as
// encoding/json decodes them, for the preparation pass.
func plainJSON(value any) any {
	switch v := value.(type) {
	case json.Number:
		f, _ := v.Float64()
		return f
	case map[string]any:
		out := make(map[string]any, len(v))
		for key, item := range v {
			out[key] = plainJSON(item)
		}
		return out
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = plainJSON(item)
		}
		return out
	}
	return value
}

func (v *resultValidator) validate(value any) (any, error) {
	value = v.prepare(value, v.doc, "", 0)
	err := v.schema.Validate(value)
	if err == nil {
		return value, nil
	}
	var validationErr *jsonschema.ValidationError
	if !errors.As(err, &validationErr) {
		return nil, err
	}
	failures := v.failures(validationErr, value)
	if len(failures) == 0 {
		return value, nil
	}
	return nil, errors.New(strings.Join(failures, ",\n\n"))
}

// subschema compiles the schema at a JSON pointer of the document, to test one value against it.
func (v *resultValidator) subschema(pointer string) *jsonschema.Schema {
	v.mu.Lock()
	defer v.mu.Unlock()
	if schema, ok := v.subs[pointer]; ok {
		return schema
	}
	schema, err := v.compiler.Compile(resultSchemaURL + "#" + pointer)
	if err != nil {
		schema = nil
	}
	v.subs[pointer] = schema
	return schema
}

const maxPrepareDepth = 64

// prepare applies AJV's coercion, defaults and removal of failing additional properties to value for the schema
// node at pointer, returning the possibly replaced value.
func (v *resultValidator) prepare(value any, node any, pointer string, depth int) any {
	schema, ok := node.(map[string]any)
	if !ok || depth > maxPrepareDepth {
		return value
	}
	if ref, ok := schema["$ref"].(string); ok && strings.HasPrefix(ref, "#") {
		if target, targetPointer, ok := resolveLocalRef(v.doc, ref); ok {
			value = v.prepare(value, target, targetPointer, depth+1)
		}
	}
	if types := schemaTypes(schema["type"]); len(types) > 0 {
		value = coerceValue(value, types)
	}
	if allOf, ok := schema["allOf"].([]any); ok {
		for i, sub := range allOf {
			value = v.prepare(value, sub, pointer+"/allOf/"+strconv.Itoa(i), depth+1)
		}
	}
	switch data := value.(type) {
	case map[string]any:
		v.prepareObject(data, schema, pointer, depth)
	case []any:
		switch items := schema["items"].(type) {
		case map[string]any:
			for i := range data {
				data[i] = v.prepare(data[i], items, pointer+"/items", depth+1)
			}
		case []any:
			for i, sub := range items {
				itemPointer := pointer + "/items/" + strconv.Itoa(i)
				if i < len(data) {
					data[i] = v.prepare(data[i], sub, itemPointer, depth+1)
				} else if def, ok := sub.(map[string]any)["default"]; ok && i == len(data) {
					// AJV fills tuple defaults only for the items that follow the last given one.
					copied, _ := cloneJSON(def)
					data = append(data, copied)
					value = data
				}
			}
		}
	}
	return value
}

func (v *resultValidator) prepareObject(data map[string]any, schema map[string]any, pointer string, depth int) {
	properties, _ := schema["properties"].(map[string]any)
	for _, name := range sortedKeys(properties) {
		sub := properties[name]
		subPointer := pointer + "/properties/" + escapePointer(name)
		if item, ok := data[name]; ok {
			data[name] = v.prepare(item, sub, subPointer, depth+1)
		} else if subSchema, ok := sub.(map[string]any); ok {
			if def, ok := subSchema["default"]; ok {
				copied, _ := cloneJSON(def)
				data[name] = copied
			}
		}
	}
	patterns := map[*regexp.Regexp]string{}
	if patternProperties, ok := schema["patternProperties"].(map[string]any); ok {
		for _, pattern := range sortedKeys(patternProperties) {
			re, err := regexp.Compile(pattern)
			if err != nil {
				continue
			}
			patterns[re] = pattern
			for key, item := range data {
				if re.MatchString(key) {
					data[key] = v.prepare(item, patternProperties[pattern], pointer+"/patternProperties/"+escapePointer(pattern), depth+1)
				}
			}
		}
	}
	additional, ok := schema["additionalProperties"]
	if !ok {
		return
	}
	for key, item := range data {
		if _, declared := properties[key]; declared {
			continue
		}
		matched := false
		for re := range patterns {
			if re.MatchString(key) {
				matched = true
				break
			}
		}
		if matched {
			continue
		}
		switch rule := additional.(type) {
		case bool:
			if !rule {
				delete(data, key)
			}
		case map[string]any:
			subPointer := pointer + "/additionalProperties"
			item = v.prepare(item, rule, subPointer, depth+1)
			if sub := v.subschema(subPointer); sub != nil && sub.Validate(item) != nil {
				delete(data, key)
			} else {
				data[key] = item
			}
		}
	}
}

func resolveLocalRef(doc any, ref string) (any, string, bool) {
	pointer := strings.TrimPrefix(ref, "#")
	node := doc
	if pointer == "" {
		return node, "", true
	}
	if !strings.HasPrefix(pointer, "/") {
		return nil, "", false
	}
	for _, token := range strings.Split(pointer[1:], "/") {
		token = strings.ReplaceAll(strings.ReplaceAll(token, "~1", "/"), "~0", "~")
		switch current := node.(type) {
		case map[string]any:
			next, ok := current[token]
			if !ok {
				return nil, "", false
			}
			node = next
		case []any:
			index, err := strconv.Atoi(token)
			if err != nil || index < 0 || index >= len(current) {
				return nil, "", false
			}
			node = current[index]
		default:
			return nil, "", false
		}
	}
	return node, pointer, true
}

func schemaTypes(value any) []string {
	switch v := value.(type) {
	case string:
		return []string{v}
	case []any:
		types := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				types = append(types, s)
			}
		}
		return types
	}
	return nil
}

func jsonType(value any) string {
	switch v := value.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case float64:
		if v == math.Trunc(v) && !math.IsInf(v, 0) {
			return "integer"
		}
		return "number"
	case string:
		return "string"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	}
	return ""
}

func typeAllowed(value any, types []string) bool {
	actual := jsonType(value)
	for _, t := range types {
		if t == actual || (t == "number" && actual == "integer") {
			return true
		}
	}
	return false
}

// coerceValue converts value to the first of types it converts to, as AJV's coerceTypes "array" does, when its own
// type is not allowed.
func coerceValue(value any, types []string) any {
	if typeAllowed(value, types) {
		return value
	}
	wantsArray := false
	for _, t := range types {
		if t == "array" {
			wantsArray = true
		}
	}
	if items, ok := value.([]any); ok && len(items) == 1 && !wantsArray {
		value = items[0]
		if typeAllowed(value, types) {
			return value
		}
	}
	for _, t := range types {
		if coerced, ok := coerceScalar(value, t); ok {
			return coerced
		}
	}
	if wantsArray {
		switch value.(type) {
		case []any, map[string]any:
		default:
			return []any{value}
		}
	}
	return value
}

func coerceScalar(value any, target string) (any, bool) {
	switch target {
	case "number", "integer":
		var number float64
		switch v := value.(type) {
		case string:
			trimmed := strings.TrimSpace(v)
			if trimmed == "" {
				return nil, false
			}
			f, err := strconv.ParseFloat(trimmed, 64)
			if err != nil || math.IsInf(f, 0) || math.IsNaN(f) {
				return nil, false
			}
			number = f
		case bool:
			if v {
				number = 1
			}
		case nil:
			number = 0
		default:
			return nil, false
		}
		if target == "integer" && number != math.Trunc(number) {
			return nil, false
		}
		return number, true
	case "string":
		switch v := value.(type) {
		case float64:
			return strconv.FormatFloat(v, 'f', -1, 64), true
		case bool:
			return strconv.FormatBool(v), true
		case nil:
			return "", true
		}
	case "boolean":
		switch v := value.(type) {
		case string:
			if v == "true" || v == "false" {
				return v == "true", true
			}
		case float64:
			if v == 0 || v == 1 {
				return v == 1, true
			}
		case nil:
			return false, true
		}
	case "null":
		switch v := value.(type) {
		case string:
			if v == "" {
				return nil, true
			}
		case float64:
			if v == 0 {
				return nil, true
			}
		case bool:
			if !v {
				return nil, true
			}
		}
	}
	return nil, false
}

// failures formats the leaf errors of a validation error as AJV does ("path: message\nparams"), sorted by instance
// path. An empty string that fails a date or date-time format of a property that is not required is ignored.
func (v *resultValidator) failures(root *jsonschema.ValidationError, value any) []string {
	type failure struct {
		path, text string
	}
	var out []failure
	var walk func(*jsonschema.ValidationError)
	walk = func(e *jsonschema.ValidationError) {
		switch e.ErrorKind.(type) {
		case *kind.Group, *kind.Schema, *kind.Reference, *kind.AllOf:
			if len(e.Causes) > 0 {
				for _, cause := range e.Causes {
					walk(cause)
				}
				return
			}
		}
		if v.ignorableDateError(e, value) {
			return
		}
		path := instancePath(e.InstanceLocation)
		message, params := ajvMessage(e.ErrorKind)
		data, _ := json.Marshal(params)
		out = append(out, failure{path: path, text: path + ": " + message + "\n" + string(data)})
	}
	walk(root)
	sort.SliceStable(out, func(i, j int) bool { return out[i].path < out[j].path })
	texts := make([]string, len(out))
	for i, f := range out {
		texts[i] = f.text
	}
	return texts
}

func (v *resultValidator) ignorableDateError(e *jsonschema.ValidationError, value any) bool {
	format, ok := e.ErrorKind.(*kind.Format)
	if !ok || (format.Want != "date" && format.Want != "date-time") || len(e.InstanceLocation) == 0 {
		return false
	}
	if got, ok := format.Got.(string); !ok || got != "" {
		return false
	}
	_, pointer, found := strings.Cut(e.SchemaURL, "#")
	if !found {
		return false
	}
	tokens := strings.Split(strings.TrimPrefix(pointer, "/"), "/")
	if len(tokens) < 2 || tokens[len(tokens)-2] != "properties" {
		return false
	}
	parentPointer := "#/" + strings.Join(tokens[:len(tokens)-2], "/")
	if len(tokens) == 2 {
		parentPointer = "#"
	}
	parent, _, ok := resolveLocalRef(v.doc, parentPointer)
	if !ok {
		return false
	}
	name := e.InstanceLocation[len(e.InstanceLocation)-1]
	if parentSchema, ok := parent.(map[string]any); ok {
		if required, ok := parentSchema["required"].([]any); ok {
			for _, item := range required {
				if item == name {
					return false
				}
			}
		}
	}
	return true
}

func instancePath(location []string) string {
	var b strings.Builder
	for _, token := range location {
		b.WriteByte('/')
		b.WriteString(escapePointer(token))
	}
	return b.String()
}

func escapePointer(token string) string {
	return strings.ReplaceAll(strings.ReplaceAll(token, "~", "~0"), "/", "~1")
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// ajvMessage returns AJV's message and params for an error kind, or the library's message for kinds AJV words
// differently.
func ajvMessage(errorKind jsonschema.ErrorKind) (string, map[string]any) {
	switch k := errorKind.(type) {
	case *kind.Required:
		missing := ""
		if len(k.Missing) > 0 {
			missing = k.Missing[0]
		}
		return fmt.Sprintf("must have required property '%s'", missing), map[string]any{"missingProperty": missing}
	case *kind.Type:
		want := strings.Join(k.Want, ",")
		return "must be " + want, map[string]any{"type": want}
	case *kind.Enum:
		return "must be equal to one of the allowed values", map[string]any{"allowedValues": k.Want}
	case *kind.Const:
		return "must be equal to constant", map[string]any{"allowedValue": k.Want}
	case *kind.Format:
		return fmt.Sprintf("must match format %q", k.Want), map[string]any{"format": k.Want}
	case *kind.Pattern:
		return fmt.Sprintf("must match pattern %q", k.Want), map[string]any{"pattern": k.Want}
	case *kind.AdditionalProperties:
		property := ""
		if len(k.Properties) > 0 {
			property = k.Properties[0]
		}
		return "must NOT have additional properties", map[string]any{"additionalProperty": property}
	case *kind.MinLength:
		return fmt.Sprintf("must NOT have fewer than %d characters", k.Want), map[string]any{"limit": k.Want}
	case *kind.MaxLength:
		return fmt.Sprintf("must NOT have more than %d characters", k.Want), map[string]any{"limit": k.Want}
	case *kind.MinItems:
		return fmt.Sprintf("must NOT have fewer than %d items", k.Want), map[string]any{"limit": k.Want}
	case *kind.MaxItems:
		return fmt.Sprintf("must NOT have more than %d items", k.Want), map[string]any{"limit": k.Want}
	case *kind.MinProperties:
		return fmt.Sprintf("must NOT have fewer than %d properties", k.Want), map[string]any{"limit": k.Want}
	case *kind.MaxProperties:
		return fmt.Sprintf("must NOT have more than %d properties", k.Want), map[string]any{"limit": k.Want}
	case *kind.Minimum:
		return limitMessage(">=", k.Want)
	case *kind.Maximum:
		return limitMessage("<=", k.Want)
	case *kind.ExclusiveMinimum:
		return limitMessage(">", k.Want)
	case *kind.ExclusiveMaximum:
		return limitMessage("<", k.Want)
	case *kind.MultipleOf:
		limit := ratNumber(k.Want)
		return fmt.Sprintf("must be multiple of %v", limit), map[string]any{"multipleOf": limit}
	case *kind.UniqueItems:
		return fmt.Sprintf("must NOT have duplicate items (items ## %d and %d are identical)", k.Duplicates[1], k.Duplicates[0]), map[string]any{"i": k.Duplicates[1], "j": k.Duplicates[0]}
	case *kind.AnyOf:
		return "must match a schema in anyOf", map[string]any{}
	case *kind.OneOf:
		return "must match exactly one schema in oneOf", map[string]any{"passingSchemas": k.Subschemas}
	case *kind.Not:
		return "must NOT be valid", map[string]any{}
	case *kind.FalseSchema:
		return "boolean schema is false", map[string]any{}
	}
	return errorKind.LocalizedString(englishPrinter), map[string]any{}
}

func limitMessage(comparison string, want *big.Rat) (string, map[string]any) {
	limit := ratNumber(want)
	return fmt.Sprintf("must be %s %v", comparison, limit), map[string]any{"comparison": comparison, "limit": limit}
}

func ratNumber(r *big.Rat) any {
	if r == nil {
		return nil
	}
	if r.IsInt() {
		return r.Num().Int64()
	}
	f, _ := r.Float64()
	return f
}
