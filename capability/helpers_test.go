package capability

import (
	"fmt"
	"reflect"
	"testing"
)

// Tri-state expectations: nil is TypeScript's undefined.
var (
	T = boolPtr(true)
	F = boolPtr(false)
)

func fmtFlag(v *bool) string {
	if v == nil {
		return "undefined"
	}
	return fmt.Sprint(*v)
}

func sameFlag(a, b *bool) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

// checkFlag asserts a tri-state flag, distinguishing undefined from false.
func checkFlag(t *testing.T, label string, got, want *bool) {
	t.Helper()
	if !sameFlag(got, want) {
		t.Errorf("%s = %s, want %s", label, fmtFlag(got), fmtFlag(want))
	}
}

func modality(m Modalities, name string) *bool {
	switch name {
	case "text":
		return m.Text
	case "image":
		return m.Image
	case "video":
		return m.Video
	case "audio":
		return m.Audio
	case "embed":
		return m.Embed
	}
	panic("unknown modality " + name)
}

// mods is a partial modality expectation, as toMatchObject checks it.
type mods map[string]bool

func checkMods(t *testing.T, label string, got Modalities, want mods) {
	t.Helper()
	for name, value := range want {
		checkFlag(t, label+"."+name, modality(got, name), boolPtr(value))
	}
}

// capsMatch is a partial Capabilities expectation, as toMatchObject checks it. Nil tool fields are not checked.
type capsMatch struct {
	in, out         mods
	tool, streaming *bool
}

func checkCaps(t *testing.T, label string, got Capabilities, want capsMatch) {
	t.Helper()
	checkMods(t, label+".input", got.Input, want.in)
	checkMods(t, label+".output", got.Output, want.out)
	if want.tool != nil {
		checkFlag(t, label+".tool_support", got.ToolSupport, want.tool)
	}
	if want.streaming != nil {
		checkFlag(t, label+".tool_support_streaming", got.ToolSupportStreaming, want.streaming)
	}
}

// checkLevels asserts reasoning effort levels, distinguishing nil (undefined) from an empty list.
func checkLevels(t *testing.T, label string, got, want []string) {
	t.Helper()
	if (got == nil) != (want == nil) || !reflect.DeepEqual(append([]string{}, got...), append([]string{}, want...)) {
		t.Errorf("%s = %#v, want %#v", label, got, want)
	}
}

func checkInt(t *testing.T, label string, got, want int) {
	t.Helper()
	if got != want {
		t.Errorf("%s = %d, want %d", label, got, want)
	}
}

func checkString(t *testing.T, label, got, want string) {
	t.Helper()
	if got != want {
		t.Errorf("%s = %q, want %q", label, got, want)
	}
}

func checkBool(t *testing.T, label string, got, want bool) {
	t.Helper()
	if got != want {
		t.Errorf("%s = %v, want %v", label, got, want)
	}
}

func fmtCaps(c Capabilities) string {
	m := func(x Modalities) string {
		return fmt.Sprintf("{text:%s image:%s video:%s audio:%s embed:%s}", fmtFlag(x.Text), fmtFlag(x.Image),
			fmtFlag(x.Video), fmtFlag(x.Audio), fmtFlag(x.Embed))
	}
	return fmt.Sprintf("input:%s output:%s tool_support:%s tool_support_streaming:%s", m(c.Input), m(c.Output),
		fmtFlag(c.ToolSupport), fmtFlag(c.ToolSupportStreaming))
}

// checkCapsEqual asserts full equality, as toEqual checks it.
func checkCapsEqual(t *testing.T, label string, got, want Capabilities) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Errorf("%s:\n got  %s\n want %s", label, fmtCaps(got), fmtCaps(want))
	}
}
