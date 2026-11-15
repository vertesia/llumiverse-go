package capability

import (
	"reflect"
	"testing"

	"github.com/vertesia/llumiverse-go/common"
)

// Ported from drivers/src/shared/model-listing.test.ts.

func checkStrings(t *testing.T, label string, got, want []string) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Errorf("%s = %#v, want %#v", label, got, want)
	}
}

func TestListingKeepsCatalogMetadataOverRuntime(t *testing.T) {
	got := ResolveListingMetadata("nvidia.nemotron-nano-12b-v2-vl-bf16", common.ProviderBedrock,
		[]string{"TEXT"}, []string{"TEXT", "IMAGE"})
	checkStrings(t, "input_modalities", got.InputModalities, []string{"text", "image"})
	checkStrings(t, "output_modalities", got.OutputModalities, []string{"text"})
}

func TestListingKeepsKnownFamilyMetadata(t *testing.T) {
	for _, provider := range []common.Provider{common.ProviderOpenAI, common.ProviderAzureFoundry, common.ProviderVertexAI} {
		got := ResolveListingMetadata("gpt-5.6-sol", provider, []string{"TEXT"}, []string{"AUDIO"})
		label := string(provider)
		checkStrings(t, label+" input_modalities", got.InputModalities, []string{"text", "image"})
		checkStrings(t, label+" output_modalities", got.OutputModalities, []string{"text"})
		checkFlag(t, label+" tool_support", got.ToolSupport, T)
	}
}

func TestListingUsesRuntimeModalitiesForUnknownModels(t *testing.T) {
	got := ResolveListingMetadata("nvidia.future-foundation-model", common.ProviderBedrock,
		[]string{"TEXT", "IMAGE", "FUTURE_MODALITY"}, []string{"SPEECH"})
	want := ListingMetadata{
		InputModalities:  []string{"text", "image", "future_modality"},
		OutputModalities: []string{"audio"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %#v, want %#v", got, want)
	}
}

func TestListingDefaultsUnknownModelToText(t *testing.T) {
	got := ResolveListingMetadata("future-provider/new-model", common.ProviderOpenAICompatible, nil, nil)
	want := ListingMetadata{InputModalities: []string{"text"}, OutputModalities: []string{"text"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %#v, want %#v", got, want)
	}
}
