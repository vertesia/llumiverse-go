package openai

import "encoding/json"

// optionString returns the first option value found under the given keys as a
// string, or "" when none is set.
func optionString(options map[string]any, keys ...string) string {
	if options == nil {
		return ""
	}
	for _, key := range keys {
		if value, ok := options[key].(string); ok && value != "" {
			return value
		}
	}
	return ""
}

// optionBoolPtr reads a tri-state boolean option: nil when the caller did not set it.
func optionBoolPtr(options map[string]any, key string) *bool {
	if options == nil {
		return nil
	}
	if value, ok := options[key].(bool); ok {
		return &value
	}
	return nil
}

// optionMap reads an object-valued option such as extra_body, accepting any
// value that encodes to a JSON object (a map or a struct).
func optionMap(options map[string]any, key string) map[string]any {
	if options == nil {
		return nil
	}
	switch value := options[key].(type) {
	case nil:
		return nil
	case map[string]any:
		return value
	default:
		data, err := json.Marshal(value)
		if err != nil {
			return nil
		}
		var out map[string]any
		if json.Unmarshal(data, &out) != nil {
			return nil
		}
		return out
	}
}

// optionStrings reads a string-list option, accepting a single string too.
func optionStrings(options map[string]any, key string) []string {
	if options == nil {
		return nil
	}
	switch value := options[key].(type) {
	case string:
		if value == "" {
			return nil
		}
		return []string{value}
	case []string:
		return value
	case []any:
		out := make([]string, 0, len(value))
		for _, item := range value {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

// optionNumber returns a numeric option as it should be sent (int or float), or nil when unset.
func optionNumber(options map[string]any, key string) any {
	if v := optionInt(options, key); v != nil {
		if f := optionFloat(options, key); f != nil && *f != float64(*v) {
			return *f
		}
		return *v
	}
	if f := optionFloat(options, key); f != nil {
		return *f
	}
	return nil
}
