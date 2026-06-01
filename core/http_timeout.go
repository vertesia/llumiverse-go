package core

import (
	"net"
	"net/http"
)

// hasHTTPTimeout reports whether any llumiverse timeout knob is set.
func hasHTTPTimeout(options HTTPTimeoutOptions) bool {
	return options.HeadersTimeout > 0 ||
		options.BodyTimeout > 0 ||
		options.ConnectTimeout > 0 ||
		options.KeepAliveTimeout > 0
}

// newHTTPClient returns a shallow clone of base with the llumiverse timeout knobs
// applied. BodyTimeout maps to http.Client.Timeout, which covers the full request
// including body reads; HeadersTimeout, ConnectTimeout, and KeepAliveTimeout map
// to the underlying transport when it is an *http.Transport.
func newHTTPClient(base *http.Client, options HTTPTimeoutOptions) *http.Client {
	if base == nil {
		base = http.DefaultClient
	}
	if !hasHTTPTimeout(options) {
		return base
	}
	next := *base
	if options.BodyTimeout > 0 {
		next.Timeout = options.BodyTimeout
	}
	if transport := timeoutTransport(base.Transport, options); transport != nil {
		next.Transport = transport
	}
	return &next
}

func timeoutTransport(rt http.RoundTripper, options HTTPTimeoutOptions) http.RoundTripper {
	if options.HeadersTimeout <= 0 && options.ConnectTimeout <= 0 && options.KeepAliveTimeout <= 0 {
		return nil
	}
	if rt == nil {
		rt = http.DefaultTransport
	}
	transport, ok := rt.(*http.Transport)
	if !ok {
		return rt
	}
	next := transport.Clone()
	if options.HeadersTimeout > 0 {
		next.ResponseHeaderTimeout = options.HeadersTimeout
	}
	if options.KeepAliveTimeout > 0 {
		next.IdleConnTimeout = options.KeepAliveTimeout
	}
	if options.ConnectTimeout > 0 {
		dialer := &net.Dialer{Timeout: options.ConnectTimeout}
		next.DialContext = dialer.DialContext
	}
	return next
}
