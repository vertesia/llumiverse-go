package capability

import "testing"

// Ported from common/src/options/version-parsing.test.ts, for the helpers this package has. Claude parsing and the
// Claude option helpers live outside this package.

func TestParseGptVersion(t *testing.T) {
	for _, tc := range []struct {
		model        string
		major, minor int
	}{
		{"gpt-5", 5, 0},
		{"gpt-5.6-sol", 5, 6},
		{"models/gpt-5.4-2026-03-05", 5, 4},
		{"gpt-5.7-pro-20270101", 5, 7},
		{"us.openai.gpt-5.6-sol-v1:0", 5, 6},
		{"gpt-deployment::gpt-5", 5, 0},
		{"gpt-6", 6, 0},
		{"gpt-6-astra", 6, 0},
		{"openai.gpt-daybreak-blue-5.6-sol", 5, 6},
		{"global.openai.gpt-daybreak-blue-5.6-sol", 5, 6},
		{"gpt-daybreak-red-6", 6, 0},
	} {
		major, minor, ok := parseGptVersion(tc.model)
		if !ok || major != tc.major || minor != tc.minor {
			t.Errorf("parseGptVersion(%q) = %d, %d, %v; want %d, %d, true", tc.model, major, minor, ok, tc.major, tc.minor)
		}
	}
}

func TestGptVersionNumericComparison(t *testing.T) {
	checkBool(t, "gpt-5.10 >= 5.6", gptVersionGTE("gpt-5.10", 5, 6), true)
	checkBool(t, "gpt-5.2 >= 5.6", gptVersionGTE("gpt-5.2", 5, 6), false)
	if _, _, ok := parseGptVersion("notgpt-6"); ok {
		t.Error("parseGptVersion(notgpt-6) parsed a version")
	}
}

func TestOpenAIReasoningEffortLevels(t *testing.T) {
	// The TypeScript record maps labels to values; the values, in order, are what this package returns.
	current := []string{"none", "low", "medium", "high", "xhigh", "max"}
	for _, tc := range []struct {
		model string
		want  []string
	}{
		{"gpt-5", []string{"minimal", "low", "medium", "high"}},
		{"gpt-5.1", []string{"none", "low", "medium", "high"}},
		{"gpt-5.4", []string{"none", "low", "medium", "high", "xhigh"}},
		{"gpt-5.6-sol", current},
		{"gpt-5-pro", []string{"high"}},
		{"gpt-5.4-pro", []string{"medium", "high", "xhigh"}},
		{"gpt-6-astra", []string{"low", "medium", "high", "xhigh", "max"}},
		{"gpt-6-sol", current},
		{"gpt-6-luna", current},
	} {
		checkLevels(t, tc.model, openAIReasoningEffortLevels(tc.model), tc.want)
	}
}

func TestGeminiVersionParsing(t *testing.T) {
	// getGeminiModelVersion returns the version string; this package only compares versions, so a parsed X.Y is
	// checked as at least X.Y and below X.(Y+1).
	for _, tc := range []struct {
		model        string
		major, minor int
	}{
		{"gemini-3.5-flash", 3, 5},
		{"publishers/google/models/gemini-3.1-pro-preview", 3, 1},
		{"gemini-4-flash", 4, 0},
	} {
		if !geminiVersionGTE(tc.model, tc.major, tc.minor) || geminiVersionGTE(tc.model, tc.major, tc.minor+1) {
			t.Errorf("%s does not parse as %d.%d", tc.model, tc.major, tc.minor)
		}
	}
}

func TestGeminiVersionNumericComparison(t *testing.T) {
	checkBool(t, "gemini-3.10-flash >= 3.5", geminiVersionGTE("gemini-3.10-flash", 3, 5), true)
	checkBool(t, "gemini-3.1-pro >= 3.5", geminiVersionGTE("gemini-3.1-pro", 3, 5), false)
	checkBool(t, "gemini-4-pro >= 3.5", geminiVersionGTE("gemini-4-pro", 3, 5), true)
}

func TestGeminiOmniVersions(t *testing.T) {
	for _, tc := range []struct {
		model        string
		major, minor int
		ok           bool
	}{
		{"gemini-omni-flash-preview", 1, 0, true},
		{"gemini-omni-1.1-flash-preview", 1, 1, true},
		{"locations/global/publishers/google/models/gemini-omni-1.2-flash", 1, 2, true},
		{"gemini-omni-1.3-flash", 1, 3, true},
		{"gemini-omni-2-flash", 2, 0, true},
		{"gemini-3.5-flash", 0, 0, false},
		{"gemini-omnia-flash", 0, 0, false},
	} {
		major, minor, ok := geminiOmniVersion(tc.model)
		if ok != tc.ok || (ok && (major != tc.major || minor != tc.minor)) {
			t.Errorf("geminiOmniVersion(%q) = %d, %d, %v; want %d, %d, %v", tc.model, major, minor, ok,
				tc.major, tc.minor, tc.ok)
		}
		checkBool(t, tc.model+" is Omni video", IsGeminiOmniVideoModel(tc.model), tc.ok)
	}
}

func TestGeminiOmniVersionComparison(t *testing.T) {
	checkBool(t, "gemini-omni-1.2-flash >= 1.2", geminiOmniVersionGTE("gemini-omni-1.2-flash", 1, 2), true)
	checkBool(t, "gemini-omni-2-flash >= 1.2", geminiOmniVersionGTE("gemini-omni-2-flash", 1, 2), true)
	checkBool(t, "gemini-omni-1.1-flash-preview >= 1.2", geminiOmniVersionGTE("gemini-omni-1.1-flash-preview", 1, 2), false)
	checkBool(t, "gemini-omni-flash-preview >= 1.1", geminiOmniVersionGTE("gemini-omni-flash-preview", 1, 1), false)
	checkBool(t, "gemini-3.5-flash >= 1.0", geminiOmniVersionGTE("gemini-3.5-flash", 1, 0), false)
}

func TestGeminiXHighThinking(t *testing.T) {
	for _, tc := range []struct {
		model string
		want  bool
	}{
		{"gemini-4-hydrogen", true},
		{"publishers/google/models/gemini-4-hydrogen", true},
		{"google/gemini-4-hydrogen:batch", true},
		{"gemini-4.5-hydrogen-preview", true},
		{"gemini-4-argon", false},
		{"gemini-4-carbon", false},
		{"gemini-4-flash", false},
		{"gemini-4-hydrogenx", false},
	} {
		checkBool(t, tc.model, supportsGeminiXHighThinking(tc.model), tc.want)
	}
}

func TestGeminiLongDecode(t *testing.T) {
	for _, tc := range []struct {
		model string
		want  bool
	}{
		{"gemini-4-argon", true},
		{"gemini-4-argon-cyber-permissive", true},
		{"gemini-4-flash", true},
		{"locations/global/publishers/google/models/gemini-5-pro", true},
		{"gemini-3.8-flash", false},
		{"gemini-4-flash-image", false},
		{"gemini-4-flash-tts", false},
		{"gemini-4-flash-live", false},
		{"gemini-4-flash-native-audio", false},
		{"gemini-4-flash-transcribe", false},
		{"gemini-4-robotics", false},
		{"gemini-embedding-4", false},
	} {
		checkBool(t, tc.model, isGeminiLongDecodeModel(tc.model), tc.want)
	}
}

func TestQwenVersionComparison(t *testing.T) {
	for _, tc := range []struct {
		model        string
		major, minor int
		want         bool
	}{
		{"Qwen/Qwen3.5-9B", 3, 5, true},
		{"qwen3.8-flash", 3, 8, true},
		{"Qwen/Qwen4-Plus", 3, 8, true},
		{"Qwen/Qwen3.7-Max", 3, 8, false},
		// A dash starts the model size, not a minor version.
		{"Qwen/Qwen3-235B-A22B-Instruct-2507-tput", 3, 5, false},
		{"qwen/qwen3-32b", 3, 8, false},
		{"llama-3.3-70b", 3, 0, false},
	} {
		checkBool(t, tc.model, qwenVersionGTE(tc.model, tc.major, tc.minor), tc.want)
	}
}
