package vertexai

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/vertesia/llumiverse-go/wire"
)

// Explicit Vertex context caching for Gemini.
//
// ExecutionOptions.PromptCacheKey turns on a cachedContents resource holding the
// stable prefix of the prompt; every read of that prefix is then billed at the
// cached-input rate. Without a key the request is unchanged.
//
// The prefix is the system instruction plus the leading contents that hold only
// static text, stopping before the first content with any other part (media, a
// tool call, a signed thought) and never including the final content, which is
// the turn that changes from call to call. Trailing model contents stay out, as
// Vertex rejects a cache that ends on a model turn.
//
// A cache is identified by a hash of what goes into it (model, system, prefix
// contents, tools, tool config), never by the caller's key, so callers that shard
// their key share one resource for identical content. The driver keeps a local
// memo with one lookup per content in flight; a cold memo recovers a resource by
// listing on its deterministic display name before creating one.
//
// In "auto" mode any cache failure sends the full uncached request; in
// "required" mode it fails the execution. A prefix Vertex refuses for its size is
// not tried again, and a model Vertex reports without explicit caching is skipped
// for an hour.

const (
	defaultGeminiContextCacheTTLSeconds = 30 * 60
	geminiCacheRefreshMargin            = 5 * time.Minute
	geminiCacheRegistryMaxEntries       = 512
	geminiCacheDisplayNamePrefix        = "llmv-cache"
	geminiCacheDisplayNameMaxLength     = 128
	geminiCacheDisplayNameModelLength   = 64
	geminiCacheDisplayNameHashLength    = 16
	geminiCacheListPageSize             = 100
	geminiCacheListMaxScanned           = 1000
	geminiCacheCreatePermitWait         = 1500 * time.Millisecond
	geminiCacheMaxConcurrentCreates     = 2
	geminiCacheQuotaCooldown            = 30 * time.Second
	geminiCacheUnsupportedModelRecheck  = time.Hour
)

var (
	geminiMinimumTokenPattern     = regexp.MustCompile(`(?i)minimum token|min_?total_?token|too (?:small|few).*token|at least \d+ tokens`)
	geminiCacheUnsupportedPattern = regexp.MustCompile(`(?i)does not support explicit (?:context )?cach|explicit (?:context )?caching is not supported`)
	geminiCacheMentionPattern     = regexp.MustCompile(`(?i)cached ?content|cache`)
)

// geminiContextCacheEntry is a live cachedContents resource.
type geminiContextCacheEntry struct {
	name      string
	expiresAt time.Time
}

// geminiContextCacheResolution is the outcome of finding or creating a cache.
type geminiContextCacheResolution struct {
	entry          *geminiContextCacheEntry
	path           wire.PromptCachePath
	waitLatency    *float64
	providerStatus *int
}

type geminiCacheCall struct {
	done   chan struct{}
	result geminiContextCacheResolution
}

// geminiContextCacheManager is the per-driver memo of cachedContents resources,
// keyed by location and content hash.
type geminiContextCacheManager struct {
	mu                 sync.Mutex
	entries            map[string]geminiContextCacheEntry
	entryOrder         []string
	uncacheable        map[string]bool
	uncacheableOrder   []string
	unsupportedUntil   map[string]time.Time
	pending            map[string]*geminiCacheCall
	cooldownUntil      time.Time
	activeCreates      int
	defaultTTLSeconds  int
	now                func() time.Time
	createPermitPoll   time.Duration
	createPermitWindow time.Duration
}

func newGeminiContextCacheManager(ttlSeconds int) *geminiContextCacheManager {
	if ttlSeconds <= 0 {
		ttlSeconds = defaultGeminiContextCacheTTLSeconds
	}
	return &geminiContextCacheManager{
		entries:            map[string]geminiContextCacheEntry{},
		uncacheable:        map[string]bool{},
		unsupportedUntil:   map[string]time.Time{},
		pending:            map[string]*geminiCacheCall{},
		defaultTTLSeconds:  ttlSeconds,
		now:                time.Now,
		createPermitPoll:   25 * time.Millisecond,
		createPermitWindow: geminiCacheCreatePermitWait,
	}
}

func (m *geminiContextCacheManager) get(key string) (geminiContextCacheEntry, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	entry, ok := m.entries[key]
	return entry, ok
}

func (m *geminiContextCacheManager) set(key string, entry geminiContextCacheEntry) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.entries[key]; !ok {
		m.entryOrder = append(m.entryOrder, key)
	}
	m.entries[key] = entry
	for len(m.entries) > geminiCacheRegistryMaxEntries && len(m.entryOrder) > 0 {
		delete(m.entries, m.entryOrder[0])
		m.entryOrder = m.entryOrder[1:]
	}
}

func (m *geminiContextCacheManager) invalidate(key string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.entries, key)
}

func (m *geminiContextCacheManager) isUncacheable(key string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.uncacheable[key]
}

func (m *geminiContextCacheManager) markUncacheable(key string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.entries, key)
	if !m.uncacheable[key] {
		m.uncacheable[key] = true
		m.uncacheableOrder = append(m.uncacheableOrder, key)
	}
	for len(m.uncacheable) > geminiCacheRegistryMaxEntries && len(m.uncacheableOrder) > 0 {
		delete(m.uncacheable, m.uncacheableOrder[0])
		m.uncacheableOrder = m.uncacheableOrder[1:]
	}
}

func (m *geminiContextCacheManager) markModelUnsupported(location, model string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.unsupportedUntil[location+":"+model] = m.now().Add(geminiCacheUnsupportedModelRecheck)
}

func (m *geminiContextCacheManager) isModelUnsupported(location, model string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	until, ok := m.unsupportedUntil[location+":"+model]
	if !ok {
		return false
	}
	if until.After(m.now()) {
		return true
	}
	delete(m.unsupportedUntil, location+":"+model)
	return false
}

func (m *geminiContextCacheManager) localCooldown() (time.Time, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.cooldownUntil, m.cooldownUntil.After(m.now())
}

func (m *geminiContextCacheManager) setLocalCooldown(until time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if until.After(m.cooldownUntil) {
		m.cooldownUntil = until
	}
}

func (m *geminiContextCacheManager) acquireCreatePermit(ctx context.Context) bool {
	deadline := time.Now().Add(m.createPermitWindow)
	for {
		m.mu.Lock()
		if m.activeCreates < geminiCacheMaxConcurrentCreates {
			m.activeCreates++
			m.mu.Unlock()
			return true
		}
		m.mu.Unlock()
		if time.Now().After(deadline) {
			return false
		}
		select {
		case <-ctx.Done():
			return false
		case <-time.After(m.createPermitPoll):
		}
	}
}

func (m *geminiContextCacheManager) releaseCreatePermit() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.activeCreates > 0 {
		m.activeCreates--
	}
}

// resolve returns the memoized entry, or runs load on a miss or near expiry.
// Concurrent callers for the same content share one load.
func (m *geminiContextCacheManager) resolve(key string, load func() geminiContextCacheResolution) geminiContextCacheResolution {
	m.mu.Lock()
	if existing, ok := m.entries[key]; ok && existing.expiresAt.Sub(m.now()) > geminiCacheRefreshMargin {
		m.mu.Unlock()
		return geminiContextCacheResolution{entry: &existing, path: wire.PromptCachePathLocalMemoHit}
	}
	if call, ok := m.pending[key]; ok {
		m.mu.Unlock()
		started := time.Now()
		<-call.done
		result := call.result
		if result.entry != nil {
			result.path = wire.PromptCachePathWaitedForCreator
			result.waitLatency = ptr(float64(time.Since(started).Milliseconds()))
		}
		return result
	}
	call := &geminiCacheCall{done: make(chan struct{})}
	m.pending[key] = call
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		delete(m.pending, key)
		m.mu.Unlock()
		close(call.done)
	}()
	call.result = load()
	return call.result
}

// geminiCachePrefix is the cacheable prefix of a prompt.
type geminiCachePrefix struct {
	system    *geminiContent
	contents  []geminiContent
	partCount int
}

// isStaticTextPart reports a part that carries non-empty text and nothing else.
func isStaticTextPart(part geminiPart) bool {
	return part.Text != "" && part.isTextOnly() && part.ThoughtSignature == "" && !part.Thought
}

// deriveGeminiCachePrefix returns the cacheable prefix of a prompt, or false when
// there is nothing worth caching.
func deriveGeminiCachePrefix(prompt geminiPrompt) (geminiCachePrefix, bool) {
	prefix := geminiCachePrefix{system: prompt.System}
	for i := 0; i < len(prompt.Contents)-1; i++ {
		parts := prompt.Contents[i].Parts
		static := len(parts) > 0
		for _, part := range parts {
			if !isStaticTextPart(part) {
				static = false
				break
			}
		}
		if !static {
			break
		}
		prefix.contents = append(prefix.contents, prompt.Contents[i])
		prefix.partCount += len(parts)
	}
	// Vertex rejects a cache whose contents end on a model turn.
	for n := len(prefix.contents); n > 0 && prefix.contents[n-1].Role == "model"; n = len(prefix.contents) {
		prefix.partCount -= len(prefix.contents[n-1].Parts)
		prefix.contents = prefix.contents[:n-1]
	}
	if prompt.System == nil && len(prefix.contents) == 0 {
		return geminiCachePrefix{}, false
	}
	return prefix, true
}

// stripLeadingGeminiParts drops the first count parts, dropping contents that
// empty out, or returns false when the request holds fewer parts.
func stripLeadingGeminiParts(contents []geminiContent, count int) ([]geminiContent, bool) {
	if count <= 0 {
		return contents, true
	}
	out := []geminiContent{}
	remaining := count
	for _, content := range contents {
		if remaining <= 0 {
			out = append(out, content)
			continue
		}
		if remaining >= len(content.Parts) {
			remaining -= len(content.Parts)
			continue
		}
		content.Parts = content.Parts[remaining:]
		out = append(out, content)
		remaining = 0
	}
	return out, remaining == 0
}

// geminiCacheContentHash identifies a cache by what goes into it; the caller's
// cache key is deliberately not part of it.
func geminiCacheContentHash(model string, prefix geminiCachePrefix, tools any, toolConfig any) string {
	contents := prefix.contents
	if contents == nil {
		contents = []geminiContent{}
	}
	data, _ := json.Marshal(struct {
		Model      string          `json:"model"`
		System     *geminiContent  `json:"system"`
		Contents   []geminiContent `json:"contents"`
		Tools      any             `json:"tools"`
		ToolConfig any             `json:"toolConfig"`
	}{model, prefix.system, contents, tools, toolConfig})
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// geminiCacheDisplayName is the name a cache for this content always gets, so a
// cold driver can find it by listing.
func geminiCacheDisplayName(model string, contentHash string) string {
	short := shortResourceName(model)
	if len(short) > geminiCacheDisplayNameModelLength {
		short = short[:geminiCacheDisplayNameModelLength]
	}
	name := geminiCacheDisplayNamePrefix + ":" + short + ":" + contentHash[:geminiCacheDisplayNameHashLength]
	if len(name) > geminiCacheDisplayNameMaxLength {
		name = name[:geminiCacheDisplayNameMaxLength]
	}
	return name
}

// isGeminiMinimumTokenCountError reports a prefix below the model's minimum size.
func isGeminiMinimumTokenCountError(err error) bool {
	code, _ := errorStatusAndName(err)
	if code != 0 && code != 400 {
		return false
	}
	return geminiMinimumTokenPattern.MatchString(err.Error())
}

// isGeminiExplicitCacheUnsupportedError reports a model without explicit caching.
func isGeminiExplicitCacheUnsupportedError(err error) bool {
	code, _ := errorStatusAndName(err)
	return code == 400 && geminiCacheUnsupportedPattern.MatchString(err.Error())
}

// isGeminiCacheUnusableError reports a named cache that expired, was deleted or
// is no longer accessible.
func isGeminiCacheUnusableError(err error) bool {
	code, _ := errorStatusAndName(err)
	if code == 404 {
		return true
	}
	return (code == 400 || code == 403) && geminiCacheMentionPattern.MatchString(err.Error())
}

// geminiContextCacheRequiredError fails an execution whose prompt_cache_mode is
// required when no cache could be used.
type geminiContextCacheRequiredError struct {
	diagnostic PromptCacheDiagnostic
}

func (e *geminiContextCacheRequiredError) Error() string {
	return fmt.Sprintf("Gemini explicit context cache was required but unavailable (%s)", e.diagnostic.Path)
}

// geminiCachePlan is a payload routed through a cache resource.
type geminiCachePlan struct {
	payload    map[string]any
	managerKey string
	cacheName  string
	diagnostic PromptCacheDiagnostic
}

// cachedContent is the cachedContents resource fields the driver reads.
type cachedContent struct {
	Name        string `json:"name"`
	DisplayName string `json:"displayName"`
	ExpireTime  string `json:"expireTime"`
}

func (c cachedContent) entry(now time.Time, ttlSeconds int) (geminiContextCacheEntry, bool) {
	if c.Name == "" {
		return geminiContextCacheEntry{}, false
	}
	expires, err := time.Parse(time.RFC3339Nano, c.ExpireTime)
	if err != nil {
		expires = now.Add(time.Duration(ttlSeconds) * time.Second)
	}
	return geminiContextCacheEntry{name: c.Name, expiresAt: expires}, true
}

// geminiCacheBase returns the v1beta1 base URL for cachedContents in a location.
func (d *VertexAIDriver) geminiCacheBase(location string) string {
	return d.geminiAPIBase(location)
}

func (d *VertexAIDriver) geminiCacheTTLSeconds(options ExecutionOptions) int {
	if options.PromptCacheTTLSeconds > 0 {
		return options.PromptCacheTTLSeconds
	}
	return d.contextCache.defaultTTLSeconds
}

// planGeminiContextCache builds the cached variant of payload, or returns only a
// diagnostic when the request must go out unchanged. The prefix is derived from
// the payload itself, so the parts it strips are exactly the ones it caches.
func (d *VertexAIDriver) planGeminiContextCache(ctx context.Context, options ExecutionOptions, payload map[string]any, location string, exclude map[string]bool) (*geminiCachePlan, PromptCacheDiagnostic) {
	started := time.Now()
	_, name := splitVertexLocation(options.Model)
	model := shortResourceName(name)
	base := func(path wire.PromptCachePath) PromptCacheDiagnostic {
		return PromptCacheDiagnostic{Path: path, Model: model, PreparationLatencyMs: float64(time.Since(started).Milliseconds())}
	}
	if options.PromptCacheMode == PromptCacheModeOff {
		return nil, base(wire.PromptCachePathDisabled)
	}
	if options.PromptCacheKey == "" {
		return nil, base(wire.PromptCachePathNoKey)
	}
	// Image generation carries its own config shape and no reusable prefix.
	if strings.Contains(strings.ToLower(options.Model), "image") || d.contextCache == nil {
		return nil, base(wire.PromptCachePathDisabled)
	}
	manager := d.contextCache
	if manager.isModelUnsupported(location, model) {
		return nil, base(wire.PromptCachePathUnsupportedModel)
	}
	contents, ok := payload["contents"].([]geminiContent)
	if !ok {
		return nil, base(wire.PromptCachePathFallbackProviderError)
	}
	system, _ := payload["systemInstruction"].(*geminiContent)
	prefix, ok := deriveGeminiCachePrefix(geminiPrompt{Contents: contents, System: system})
	if !ok {
		return nil, base(wire.PromptCachePathFallbackProviderError)
	}
	// Vertex rejects a request that sets systemInstruction, tools or toolConfig
	// alongside cachedContent, so they belong to the cache and to its identity.
	tools, toolConfig := payload["tools"], payload["toolConfig"]
	contentHash := geminiCacheContentHash(model, prefix, tools, toolConfig)
	managerKey := location + ":" + contentHash
	diagnostic := func(path wire.PromptCachePath) PromptCacheDiagnostic {
		out := base(path)
		out.ContentHashPrefix = ptr(contentHash[:12])
		out.CacheablePartCount = ptr(prefix.partCount)
		return out
	}
	if manager.isUncacheable(managerKey) {
		return nil, diagnostic(wire.PromptCachePathMinimumTokenRejection)
	}
	requestContents, ok := stripLeadingGeminiParts(contents, prefix.partCount)
	if !ok || len(requestContents) == 0 {
		return nil, diagnostic(wire.PromptCachePathFallbackProviderError)
	}
	ttlSeconds := d.geminiCacheTTLSeconds(options)
	resolution := manager.resolve(managerKey, func() geminiContextCacheResolution {
		return d.adoptOrCreateGeminiCache(ctx, manager, managerKey, location, model, prefix, tools, toolConfig, ttlSeconds, exclude)
	})
	out := diagnostic(resolution.path)
	out.WaitLatencyMs = resolution.waitLatency
	out.ProviderStatus = resolution.providerStatus
	if resolution.entry == nil {
		return nil, out
	}
	out.ExplicitCacheUsed = true
	cached := make(map[string]any, len(payload))
	for key, value := range payload {
		switch key {
		case "systemInstruction", "tools", "toolConfig":
		default:
			cached[key] = value
		}
	}
	cached["contents"] = requestContents
	cached["cachedContent"] = resolution.entry.name
	return &geminiCachePlan{payload: cached, managerKey: managerKey, cacheName: resolution.entry.name, diagnostic: out}, out
}

// adoptOrCreateGeminiCache extends a memoized resource near expiry, adopts a live
// one found by its display name, or creates one.
func (d *VertexAIDriver) adoptOrCreateGeminiCache(ctx context.Context, manager *geminiContextCacheManager, managerKey, location, model string, prefix geminiCachePrefix, tools, toolConfig any, ttlSeconds int, exclude map[string]bool) geminiContextCacheResolution {
	displayName := geminiCacheDisplayName(model, prefixHashFromKey(managerKey))
	if existing, ok := manager.get(managerKey); ok && existing.expiresAt.After(manager.now()) && !exclude[existing.name] {
		if refreshed, ok := d.extendGeminiCacheTTL(ctx, location, existing, ttlSeconds); ok {
			manager.set(managerKey, refreshed)
			return geminiContextCacheResolution{entry: &refreshed, path: wire.PromptCachePathRefreshed}
		}
		manager.invalidate(managerKey)
	}
	if adopted, ok := d.findGeminiCacheByDisplayName(ctx, location, displayName, exclude); ok {
		if adopted.expiresAt.Sub(manager.now()) < geminiCacheRefreshMargin {
			if refreshed, ok := d.extendGeminiCacheTTL(ctx, location, adopted, ttlSeconds); ok {
				adopted = refreshed
			}
		}
		manager.set(managerKey, adopted)
		return geminiContextCacheResolution{entry: &adopted, path: wire.PromptCachePathProviderListRecovery}
	}
	if until, cooling := manager.localCooldown(); cooling {
		return geminiContextCacheResolution{path: wire.PromptCachePathFallbackQuota, waitLatency: ptr(float64(until.Sub(manager.now()).Milliseconds()))}
	}
	if !manager.acquireCreatePermit(ctx) {
		return geminiContextCacheResolution{path: wire.PromptCachePathFallbackWaitTimeout, waitLatency: ptr(float64(manager.createPermitWindow.Milliseconds()))}
	}
	defer manager.releaseCreatePermit()
	return d.createGeminiCache(ctx, manager, managerKey, location, model, prefix, tools, toolConfig, ttlSeconds, displayName)
}

// prefixHashFromKey returns the content hash of a "location:hash" manager key.
func prefixHashFromKey(key string) string {
	return key[strings.LastIndex(key, ":")+1:]
}

func (d *VertexAIDriver) createGeminiCache(ctx context.Context, manager *geminiContextCacheManager, managerKey, location, model string, prefix geminiCachePrefix, tools, toolConfig any, ttlSeconds int, displayName string) geminiContextCacheResolution {
	body := map[string]any{
		"model":       "projects/" + d.options.Project + "/locations/" + location + "/publishers/google/models/" + model,
		"ttl":         fmt.Sprintf("%ds", ttlSeconds),
		"displayName": displayName,
	}
	if len(prefix.contents) > 0 {
		body["contents"] = prefix.contents
	}
	if prefix.system != nil {
		body["systemInstruction"] = prefix.system
	}
	if tools != nil {
		body["tools"] = tools
	}
	if toolConfig != nil {
		body["toolConfig"] = toolConfig
	}
	var created cachedContent
	err := d.geminiCacheRequest(ctx, http.MethodPost, location, "cachedContents", body, &created)
	if err != nil {
		code, _ := errorStatusAndName(err)
		var status *int
		if code != 0 {
			status = ptr(code)
		}
		switch {
		case isGeminiMinimumTokenCountError(err):
			manager.markUncacheable(managerKey)
			return geminiContextCacheResolution{path: wire.PromptCachePathMinimumTokenRejection, providerStatus: status}
		case isGeminiExplicitCacheUnsupportedError(err):
			manager.markModelUnsupported(location, model)
			return geminiContextCacheResolution{path: wire.PromptCachePathUnsupportedModel, providerStatus: status}
		case code == 429:
			manager.setLocalCooldown(manager.now().Add(geminiCacheQuotaCooldown))
			return geminiContextCacheResolution{path: wire.PromptCachePathFallbackQuota, providerStatus: status}
		}
		return geminiContextCacheResolution{path: wire.PromptCachePathFallbackProviderError, providerStatus: status}
	}
	entry, ok := created.entry(manager.now(), ttlSeconds)
	if !ok {
		return geminiContextCacheResolution{path: wire.PromptCachePathFallbackProviderError}
	}
	manager.set(managerKey, entry)
	return geminiContextCacheResolution{entry: &entry, path: wire.PromptCachePathCreated}
}

// extendGeminiCacheTTL pushes a resource's expiry out, or reports false when
// Vertex would not extend it.
func (d *VertexAIDriver) extendGeminiCacheTTL(ctx context.Context, location string, entry geminiContextCacheEntry, ttlSeconds int) (geminiContextCacheEntry, bool) {
	var updated cachedContent
	if err := d.geminiCacheRequest(ctx, http.MethodPatch, location, entry.name, map[string]any{"ttl": fmt.Sprintf("%ds", ttlSeconds)}, &updated); err != nil {
		return geminiContextCacheEntry{}, false
	}
	if refreshed, ok := updated.entry(d.contextCache.now(), ttlSeconds); ok {
		return refreshed, true
	}
	return geminiContextCacheEntry{name: entry.name, expiresAt: d.contextCache.now().Add(time.Duration(ttlSeconds) * time.Second)}, true
}

// findGeminiCacheByDisplayName scans cachedContents (which has no filter) for a
// live resource with displayName, bounded so a project full of unrelated caches
// cannot stall a completion.
func (d *VertexAIDriver) findGeminiCacheByDisplayName(ctx context.Context, location, displayName string, exclude map[string]bool) (geminiContextCacheEntry, bool) {
	scanned := 0
	pageToken := ""
	for {
		query := url.Values{"pageSize": {fmt.Sprint(geminiCacheListPageSize)}}
		if pageToken != "" {
			query.Set("pageToken", pageToken)
		}
		var page struct {
			CachedContents []cachedContent `json:"cachedContents"`
			NextPageToken  string          `json:"nextPageToken"`
		}
		if err := d.geminiCacheRequest(ctx, http.MethodGet, location, "cachedContents?"+query.Encode(), nil, &page); err != nil {
			return geminiContextCacheEntry{}, false
		}
		for _, cached := range page.CachedContents {
			scanned++
			if scanned > geminiCacheListMaxScanned {
				return geminiContextCacheEntry{}, false
			}
			if cached.DisplayName != displayName || cached.Name == "" || exclude[cached.Name] {
				continue
			}
			expires, err := time.Parse(time.RFC3339Nano, cached.ExpireTime)
			if err != nil || !expires.After(d.contextCache.now()) {
				continue
			}
			return geminiContextCacheEntry{name: cached.Name, expiresAt: expires}, true
		}
		if page.NextPageToken == "" {
			return geminiContextCacheEntry{}, false
		}
		pageToken = page.NextPageToken
	}
}

// geminiCacheRequest calls the cachedContents API. path is "cachedContents..."
// under the project location, or a full resource name.
func (d *VertexAIDriver) geminiCacheRequest(ctx context.Context, method, location, path string, body any, out any) error {
	headers, err := d.authHeaders(ctx)
	if err != nil {
		return err
	}
	base := d.geminiCacheBase(location)
	endpoint := strings.TrimSuffix(base, "/") + "/"
	if strings.HasPrefix(path, "projects/") {
		endpoint += path
	} else {
		endpoint += "projects/" + d.options.Project + "/locations/" + location + "/" + path
	}
	return doJSON(ctx, d.client, method, endpoint, headers, body, out)
}

// sendWithGeminiContextCache sends payload through send, routed through an
// explicit cache when the execution asks for one. An unusable cache is
// rediscovered or recreated once; any other cache trouble sends the full
// uncached request, unless prompt_cache_mode is required. Errors unrelated to the
// cache are returned as they are.
func (d *VertexAIDriver) sendWithGeminiContextCache(ctx context.Context, options ExecutionOptions, payload map[string]any, location string, send func(map[string]any) error) (*PromptCacheDiagnostic, error) {
	plan, diagnostic := d.planGeminiContextCache(ctx, options, payload, location, nil)
	if plan == nil {
		if options.PromptCacheMode == PromptCacheModeRequired {
			return &diagnostic, &geminiContextCacheRequiredError{diagnostic: diagnostic}
		}
		return &diagnostic, send(payload)
	}
	err := send(plan.payload)
	if err == nil || !isGeminiCacheUnusableError(err) {
		return &plan.diagnostic, err
	}
	// The resource is gone. Forget it and plan again: listing may find a peer's
	// live cache before a create, and the dead name is never adopted again.
	d.contextCache.invalidate(plan.managerKey)
	retry, fallback := d.planGeminiContextCache(ctx, options, payload, location, map[string]bool{plan.cacheName: true})
	if retry != nil {
		recreated := retry.diagnostic
		recreated.Path = wire.PromptCachePathUnusableResourceRecreated
		retryErr := send(retry.payload)
		if retryErr == nil || !isGeminiCacheUnusableError(retryErr) {
			return &recreated, retryErr
		}
		d.contextCache.invalidate(retry.managerKey)
	}
	fallback.Path = wire.PromptCachePathFallbackProviderError
	fallback.ExplicitCacheUsed = false
	fallback.ProviderStatus = nil
	if code, _ := errorStatusAndName(err); code != 0 {
		fallback.ProviderStatus = ptr(code)
	}
	if options.PromptCacheMode == PromptCacheModeRequired {
		return &fallback, &geminiContextCacheRequiredError{diagnostic: fallback}
	}
	return &fallback, send(payload)
}
