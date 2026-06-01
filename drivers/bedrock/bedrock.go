package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"sort"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	bedrock "github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
	bedrockruntime "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// Bedrock defaults to Nova multimodal embeddings, matching the TypeScript client.
const defaultBedrockEmbeddingModel = "amazon.nova-2-multimodal-embeddings-v1:0"

// BedrockOptions configures AWS Bedrock model and runtime clients.
type BedrockOptions struct {
	DriverOptions
	// Region is the AWS region for Bedrock runtime and model APIs.
	Region string
	// Config optionally supplies a preloaded AWS config. If empty, the constructor
	// loads the default AWS config for Region.
	Config aws.Config
}

// bedrockRuntimeAPI abstracts the Bedrock runtime calls used by the driver
// (Converse, ConverseStream, InvokeModel) so tests can inject fake clients.
type bedrockRuntimeAPI interface {
	Converse(ctx context.Context, input *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
	ConverseStream(ctx context.Context, input *bedrockruntime.ConverseStreamInput, optFns ...func(*bedrockruntime.Options)) (bedrockConverseStream, error)
	InvokeModel(ctx context.Context, input *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}

// RuntimeAPI is the injectable subset of the AWS Bedrock Runtime client used by
// NewBedrockDriverWithClient.
type RuntimeAPI = bedrockRuntimeAPI

// bedrockConverseStream is the subset of the ConverseStream event-stream reader
// the driver relies on, abstracted for test injection.
type bedrockConverseStream interface {
	Events() <-chan brtypes.ConverseStreamOutput
	Close() error
	Err() error
}

// ConverseStream is the injectable subset of the AWS ConverseStream reader.
type ConverseStream = bedrockConverseStream

// bedrockRuntimeClient is the production bedrockRuntimeAPI backed by the real
// AWS bedrockruntime client.
type bedrockRuntimeClient struct {
	client *bedrockruntime.Client
}

func (c *bedrockRuntimeClient) Converse(ctx context.Context, input *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	return c.client.Converse(ctx, input, optFns...)
}

// ConverseStream unwraps the AWS output's embedded event stream and returns an
// error when the stream is nil so callers can treat it as a plain reader.
func (c *bedrockRuntimeClient) ConverseStream(ctx context.Context, input *bedrockruntime.ConverseStreamInput, optFns ...func(*bedrockruntime.Options)) (bedrockConverseStream, error) {
	output, err := c.client.ConverseStream(ctx, input, optFns...)
	if err != nil {
		return nil, err
	}
	if output.GetStream() == nil {
		return nil, errors.New("bedrock converse stream is nil")
	}
	return output.GetStream(), nil
}

func (c *bedrockRuntimeClient) InvokeModel(ctx context.Context, input *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	return c.client.InvokeModel(ctx, input, optFns...)
}

// bedrockModelAPI abstracts the Bedrock control-plane model-listing calls used
// by ListModels so tests can inject fake clients.
type bedrockModelAPI interface {
	ListFoundationModels(ctx context.Context, input *bedrock.ListFoundationModelsInput, optFns ...func(*bedrock.Options)) (*bedrock.ListFoundationModelsOutput, error)
	ListCustomModels(ctx context.Context, input *bedrock.ListCustomModelsInput, optFns ...func(*bedrock.Options)) (*bedrock.ListCustomModelsOutput, error)
	ListInferenceProfiles(ctx context.Context, input *bedrock.ListInferenceProfilesInput, optFns ...func(*bedrock.Options)) (*bedrock.ListInferenceProfilesOutput, error)
}

// ModelAPI is the injectable subset of the AWS Bedrock model-listing client
// used by NewBedrockDriverWithClient.
type ModelAPI = bedrockModelAPI

// bedrockModelClient is the production bedrockModelAPI backed by the real AWS
// bedrock control-plane client.
type bedrockModelClient struct {
	client *bedrock.Client
}

func (c *bedrockModelClient) ListFoundationModels(ctx context.Context, input *bedrock.ListFoundationModelsInput, optFns ...func(*bedrock.Options)) (*bedrock.ListFoundationModelsOutput, error) {
	return c.client.ListFoundationModels(ctx, input, optFns...)
}

func (c *bedrockModelClient) ListCustomModels(ctx context.Context, input *bedrock.ListCustomModelsInput, optFns ...func(*bedrock.Options)) (*bedrock.ListCustomModelsOutput, error) {
	return c.client.ListCustomModels(ctx, input, optFns...)
}

func (c *bedrockModelClient) ListInferenceProfiles(ctx context.Context, input *bedrock.ListInferenceProfilesInput, optFns ...func(*bedrock.Options)) (*bedrock.ListInferenceProfilesOutput, error) {
	return c.client.ListInferenceProfiles(ctx, input, optFns...)
}

// BedrockDriver implements Driver using AWS Bedrock Converse, ConverseStream,
// InvokeModel, and Bedrock model-listing APIs.
type BedrockDriver struct {
	options BedrockOptions
	runtime bedrockRuntimeAPI
	models  bedrockModelAPI
}

// NewBedrockDriver creates Bedrock control-plane and runtime clients from AWS config.
func NewBedrockDriver(ctx context.Context, options BedrockOptions) (*BedrockDriver, error) {
	if options.Region == "" {
		return nil, errors.New("region is required")
	}
	cfg := options.Config
	if cfg.Region == "" {
		loadOptions := []func(*awsconfig.LoadOptions) error{awsconfig.WithRegion(options.Region)}
		if hasHTTPTimeout(options.HTTPTimeout) {
			loadOptions = append(loadOptions, awsconfig.WithHTTPClient(newHTTPClient(nil, options.HTTPTimeout)))
		}
		loaded, err := awsconfig.LoadDefaultConfig(ctx, loadOptions...)
		if err != nil {
			return nil, err
		}
		cfg = loaded
	} else if hasHTTPTimeout(options.HTTPTimeout) && cfg.HTTPClient == nil {
		cfg.HTTPClient = newHTTPClient(nil, options.HTTPTimeout)
	}
	return &BedrockDriver{
		options: options,
		runtime: &bedrockRuntimeClient{client: bedrockruntime.NewFromConfig(cfg)},
		models:  &bedrockModelClient{client: bedrock.NewFromConfig(cfg)},
	}, nil
}

// NewBedrockDriverWithClient injects fake or preconfigured clients for tests.
func NewBedrockDriverWithClient(options BedrockOptions, runtime RuntimeAPI, models ModelAPI) *BedrockDriver {
	return &BedrockDriver{options: options, runtime: runtime, models: models}
}

func (d *BedrockDriver) runtimeOptions(timeout HTTPTimeoutOptions) []func(*bedrockruntime.Options) {
	if !hasHTTPTimeout(timeout) {
		return nil
	}
	return []func(*bedrockruntime.Options){
		func(options *bedrockruntime.Options) {
			options.HTTPClient = newHTTPClient(nil, timeout)
		},
	}
}

// Provider returns ProviderBedrock.
func (d *BedrockDriver) Provider() Provider {
	return ProviderBedrock
}

// bedrockPrompt is the prepared Converse payload for text models: the
// conversation messages and system content blocks.
type bedrockPrompt struct {
	Messages []brtypes.Message
	System   []brtypes.SystemContentBlock
	Turn     int
}

// bedrockImagePrompt holds the flattened text/image inputs used to build
// InvokeModel payloads for Bedrock image-generation models.
type bedrockImagePrompt struct {
	Text     string
	System   string
	Negative string
	Images   []string
	Masks    []string
}

// CreatePrompt builds the provider-specific prompt for Bedrock. For image
// models it returns a bedrockImagePrompt; otherwise it assembles Converse
// messages and system blocks from the segments, mapping roles, embedding any
// result schema as a system notice, appending safety messages, and merging
// adjacent same-role messages.
func (d *BedrockDriver) CreatePrompt(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (any, error) {
	if isBedrockImageModel(options.Model) {
		return formatBedrockImagePrompt(ctx, segments)
	}
	prompt := bedrockPrompt{}
	var safety []brtypes.Message
	for _, segment := range segments {
		switch segment.Role {
		case PromptRoleNegative, PromptRoleMask:
			continue
		case PromptRoleSystem:
			if segment.Content != "" {
				prompt.System = append(prompt.System, &brtypes.SystemContentBlockMemberText{Value: segment.Content})
			}
		case PromptRoleTool:
			if segment.ToolUseID == "" {
				return nil, errors.New("tool prompt segment requires ToolUseID")
			}
			content := []brtypes.ToolResultContentBlock{}
			if segment.Content != "" {
				content = append(content, &brtypes.ToolResultContentBlockMemberText{Value: segment.Content})
			}
			for _, file := range segment.Files {
				block, err := bedrockToolResultFileBlock(ctx, file)
				if err != nil {
					return nil, err
				}
				if block != nil {
					content = append(content, block)
				}
			}
			if len(content) == 0 {
				// Bedrock requires at least one content block in a toolResult.
				content = append(content, &brtypes.ToolResultContentBlockMemberText{Value: "[No output]"})
			}
			prompt.Messages = append(prompt.Messages, brtypes.Message{
				Role: brtypes.ConversationRoleUser,
				Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberToolResult{Value: brtypes.ToolResultBlock{
					ToolUseId: aws.String(segment.ToolUseID),
					Content:   content,
				}}},
			})
		default:
			content := []brtypes.ContentBlock{}
			if segment.Content != "" {
				content = append(content, &brtypes.ContentBlockMemberText{Value: segment.Content})
			}
			for _, file := range segment.Files {
				block, err := bedrockFileBlock(ctx, file)
				if err != nil {
					return nil, err
				}
				if block != nil {
					content = append(content, block)
				}
			}
			if len(content) == 0 {
				continue
			}
			message := brtypes.Message{Role: brtypes.ConversationRoleUser, Content: content}
			if segment.Role == PromptRoleAssistant {
				message.Role = brtypes.ConversationRoleAssistant
			}
			if segment.Role == PromptRoleSafety {
				safety = append(safety, message)
			} else {
				prompt.Messages = append(prompt.Messages, message)
			}
		}
	}
	if options.ResultSchema != nil {
		schema, _ := json.MarshalIndent(options.ResultSchema, "", "  ")
		notice := "IMPORTANT: The answer must be a JSON object using the following JSON Schema:\n" + string(schema)
		if len(options.Tools) > 0 {
			notice = "IMPORTANT: When not calling tools, the answer must be a JSON object using the following JSON Schema:\n" + string(schema)
		}
		prompt.System = append(prompt.System, &brtypes.SystemContentBlockMemberText{Value: notice})
	}
	// Safety messages are appended at the end, matching the TS client, then
	// adjacent roles are merged because Bedrock Converse prefers alternating turns.
	prompt.Messages = mergeBedrockMessages(append(prompt.Messages, safety...))
	if len(prompt.Messages) == 0 {
		if len(prompt.System) == 0 {
			return nil, errors.New("prompt must contain at least one message")
		}
		// Converse conversations must start with a user message. When callers
		// only supplied system content, fall back to system-as-user.
		var text string
		for _, block := range prompt.System {
			if value, ok := block.(*brtypes.SystemContentBlockMemberText); ok {
				text += value.Value + "\n"
			}
		}
		prompt.Messages = []brtypes.Message{{Role: brtypes.ConversationRoleUser, Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: strings.TrimSpace(text)}}}}
		prompt.System = nil
	}
	return prompt, nil
}

// bedrockFileBlock maps a DataSource to the appropriate Converse content block
// by MIME type: image, document, video, or text. JSON files become text (the
// Converse JSON block is only valid inside tool results) and unrecognized types
// fall back to text.
func bedrockFileBlock(ctx context.Context, file DataSource) (brtypes.ContentBlock, error) {
	mimeType := file.MIMEType()
	if strings.HasPrefix(mimeType, "image/") {
		rc, err := file.Open(ctx)
		if err != nil {
			return nil, err
		}
		defer func() { _ = rc.Close() }()
		data, err := io.ReadAll(rc)
		if err != nil {
			return nil, err
		}
		return &brtypes.ContentBlockMemberImage{Value: brtypes.ImageBlock{
			Format: bedrockImageFormat(mimeType),
			Source: &brtypes.ImageSourceMemberBytes{Value: data},
		}}, nil
	}
	if bedrockIsJSONFile(file) {
		// Bedrock ContentBlock does not support JSON outside tool results, so JSON
		// files in normal content mode are passed as text.
		text, err := readAllStringFromDataSource(ctx, file)
		if err != nil {
			return nil, err
		}
		return &brtypes.ContentBlockMemberText{Value: text}, nil
	}
	if format, ok := bedrockDocumentFormat(file); ok {
		if strings.HasPrefix(mimeType, "text/") {
			text, err := readAllStringFromDataSource(ctx, file)
			if err != nil {
				return nil, err
			}
			return &brtypes.ContentBlockMemberDocument{Value: brtypes.DocumentBlock{
				Name:   aws.String(bedrockDocumentName(file.Name())),
				Format: format,
				Source: &brtypes.DocumentSourceMemberText{Value: text},
			}}, nil
		}
		rc, err := file.Open(ctx)
		if err != nil {
			return nil, err
		}
		defer func() { _ = rc.Close() }()
		data, err := io.ReadAll(rc)
		if err != nil {
			return nil, err
		}
		return &brtypes.ContentBlockMemberDocument{Value: brtypes.DocumentBlock{
			Name:   aws.String(bedrockDocumentName(file.Name())),
			Format: format,
			Source: &brtypes.DocumentSourceMemberBytes{Value: data},
		}}, nil
	}
	if strings.HasPrefix(mimeType, "video/") {
		block, err := bedrockVideoBlock(ctx, file)
		if err != nil {
			return nil, err
		}
		return &brtypes.ContentBlockMemberVideo{Value: block}, nil
	}
	if strings.HasPrefix(mimeType, "text/") {
		text, err := readAllStringFromDataSource(ctx, file)
		if err != nil {
			return nil, err
		}
		return &brtypes.ContentBlockMemberText{Value: text}, nil
	}
	text, err := readAllStringFromDataSource(ctx, file)
	if err != nil {
		return nil, err
	}
	return &brtypes.ContentBlockMemberText{Value: text}, nil
}

// bedrockToolResultFileBlock maps a DataSource to a tool-result content block.
// Unlike normal content, tool results support a JSON block, so valid JSON files
// are sent as JSON and everything else falls back to image/document/video/text.
func bedrockToolResultFileBlock(ctx context.Context, file DataSource) (brtypes.ToolResultContentBlock, error) {
	mimeType := file.MIMEType()
	if strings.HasPrefix(mimeType, "image/") {
		rc, err := file.Open(ctx)
		if err != nil {
			return nil, err
		}
		defer func() { _ = rc.Close() }()
		data, err := io.ReadAll(rc)
		if err != nil {
			return nil, err
		}
		return &brtypes.ToolResultContentBlockMemberImage{Value: brtypes.ImageBlock{
			Format: bedrockImageFormat(mimeType),
			Source: &brtypes.ImageSourceMemberBytes{Value: data},
		}}, nil
	}
	if bedrockIsJSONFile(file) {
		// Tool-result content can use the Converse JSON block. Fall back to text
		// if the file is not valid JSON, mirroring the TS client.
		text, err := readAllStringFromDataSource(ctx, file)
		if err != nil {
			return nil, err
		}
		var parsed any
		if err := json.Unmarshal([]byte(text), &parsed); err == nil {
			return &brtypes.ToolResultContentBlockMemberJson{Value: brdoc.NewLazyDocument(parsed)}, nil
		}
		return &brtypes.ToolResultContentBlockMemberText{Value: text}, nil
	}
	if format, ok := bedrockDocumentFormat(file); ok {
		block, err := bedrockDocumentBlock(ctx, file, format)
		if err != nil {
			return nil, err
		}
		return &brtypes.ToolResultContentBlockMemberDocument{Value: block}, nil
	}
	if strings.HasPrefix(mimeType, "video/") {
		block, err := bedrockVideoBlock(ctx, file)
		if err != nil {
			return nil, err
		}
		return &brtypes.ToolResultContentBlockMemberVideo{Value: block}, nil
	}
	text, err := readAllStringFromDataSource(ctx, file)
	if err != nil {
		return nil, err
	}
	return &brtypes.ToolResultContentBlockMemberText{Value: text}, nil
}

// bedrockIsJSONFile reports whether a DataSource is JSON by MIME type or .json extension.
func bedrockIsJSONFile(file DataSource) bool {
	mimeType := strings.ToLower(file.MIMEType())
	name := strings.ToLower(file.Name())
	return mimeType == "application/json" || strings.HasSuffix(name, ".json")
}

// bedrockDocumentBlock reads a DataSource fully and wraps it as a byte-source
// Converse document block with a sanitized name.
func bedrockDocumentBlock(ctx context.Context, file DataSource, format brtypes.DocumentFormat) (brtypes.DocumentBlock, error) {
	rc, err := file.Open(ctx)
	if err != nil {
		return brtypes.DocumentBlock{}, err
	}
	defer func() { _ = rc.Close() }()
	data, err := io.ReadAll(rc)
	if err != nil {
		return brtypes.DocumentBlock{}, err
	}
	return brtypes.DocumentBlock{
		Name:   aws.String(bedrockDocumentName(file.Name())),
		Format: format,
		Source: &brtypes.DocumentSourceMemberBytes{Value: data},
	}, nil
}

// bedrockVideoBlock builds a Converse video block, preferring an S3 location
// over inline bytes to avoid loading large media into memory.
func bedrockVideoBlock(ctx context.Context, file DataSource) (brtypes.VideoBlock, error) {
	// Bedrock accepts video by S3 location. Prefer that path because it avoids
	// loading large media into memory and matches the TypeScript behavior. The
	// API takes only the URI here; bucket-owner metadata is intentionally omitted.
	if uri, err := file.URL(ctx); err == nil {
		if s3URI := bedrockS3URI(uri); s3URI != "" {
			return brtypes.VideoBlock{
				Format: bedrockVideoFormat(file.MIMEType()),
				Source: &brtypes.VideoSourceMemberS3Location{Value: brtypes.S3Location{Uri: aws.String(s3URI)}},
			}, nil
		}
	}
	rc, err := file.Open(ctx)
	if err != nil {
		return brtypes.VideoBlock{}, err
	}
	defer func() { _ = rc.Close() }()
	data, err := io.ReadAll(rc)
	if err != nil {
		return brtypes.VideoBlock{}, err
	}
	return brtypes.VideoBlock{
		Format: bedrockVideoFormat(file.MIMEType()),
		Source: &brtypes.VideoSourceMemberBytes{Value: data},
	}, nil
}

// bedrockS3URI normalizes a raw URL into an s3://bucket/key URI, or returns ""
// if it is not a recognized S3 location. It accepts s3:// URLs, path-style and
// virtual-hosted HTTPS S3 URLs, and rejects spoofed hostnames.
func bedrockS3URI(raw string) string {
	// Accept only canonical S3 URLs and anchored AWS hostnames. This prevents
	// treating spoofed hosts like amazonaws.com.evil.example as S3 locations.
	if strings.HasPrefix(raw, "s3://") {
		return raw
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme != "https" {
		return ""
	}
	host := strings.ToLower(parsed.Hostname())
	if strings.HasPrefix(host, "s3.") || host == "s3.amazonaws.com" {
		// Path-style HTTPS URLs are converted from /bucket/key to s3://bucket/key.
		parts := strings.Split(strings.Trim(parsed.Path, "/"), "/")
		if len(parts) < 2 {
			return ""
		}
		return "s3://" + parts[0] + "/" + strings.Join(parts[1:], "/")
	}
	if strings.HasSuffix(host, ".amazonaws.com") && (strings.Contains(host, ".s3.") || strings.HasSuffix(host, ".s3.amazonaws.com")) {
		// Virtual-hosted S3 URLs are converted from bucket.s3.../key to s3://bucket/key.
		bucket := strings.Split(host, ".")[0]
		key := strings.TrimPrefix(parsed.Path, "/")
		if bucket == "" || key == "" {
			return ""
		}
		return "s3://" + bucket + "/" + key
	}
	return ""
}

// bedrockDocumentFormat maps a DataSource to a Bedrock DocumentFormat by MIME
// type or filename extension, returning false when the type is not a supported
// document format.
func bedrockDocumentFormat(file DataSource) (brtypes.DocumentFormat, bool) {
	mimeType := strings.ToLower(file.MIMEType())
	name := strings.ToLower(file.Name())
	switch {
	case mimeType == "application/pdf" || strings.HasSuffix(name, ".pdf"):
		return brtypes.DocumentFormatPdf, true
	case mimeType == "text/csv" || strings.HasSuffix(name, ".csv"):
		return brtypes.DocumentFormatCsv, true
	case mimeType == "text/markdown" || strings.HasSuffix(name, ".md"):
		return brtypes.DocumentFormatMd, true
	case strings.HasPrefix(mimeType, "text/") || strings.HasSuffix(name, ".txt"):
		return brtypes.DocumentFormatTxt, true
	case mimeType == "application/msword" || strings.HasSuffix(name, ".doc"):
		return brtypes.DocumentFormatDoc, true
	case mimeType == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" || strings.HasSuffix(name, ".docx"):
		return brtypes.DocumentFormatDocx, true
	case mimeType == "application/vnd.ms-excel" || strings.HasSuffix(name, ".xls"):
		return brtypes.DocumentFormatXls, true
	case mimeType == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" || strings.HasSuffix(name, ".xlsx"):
		return brtypes.DocumentFormatXlsx, true
	case mimeType == "text/html" || strings.HasSuffix(name, ".html") || strings.HasSuffix(name, ".htm"):
		return brtypes.DocumentFormatHtml, true
	default:
		return "", false
	}
}

func bedrockDocumentName(name string) string {
	// Bedrock document names are restricted to alphanumeric characters,
	// whitespace, hyphens, parentheses, and brackets. Collapse whitespace and
	// drop unsupported characters to avoid request validation failures.
	name = strings.TrimSpace(name)
	if name == "" {
		return "document"
	}
	var b strings.Builder
	lastSpace := false
	for _, r := range name {
		allowed := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '(' || r == ')' || r == '[' || r == ']'
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			if !lastSpace {
				b.WriteByte(' ')
				lastSpace = true
			}
			continue
		}
		if allowed {
			b.WriteRune(r)
			lastSpace = false
		}
	}
	out := strings.TrimSpace(b.String())
	if out == "" {
		return "document"
	}
	return out
}

func readAllStringFromDataSource(ctx context.Context, file DataSource) (string, error) {
	rc, err := file.Open(ctx)
	if err != nil {
		return "", err
	}
	return readAllString(rc)
}

// bedrockImageFormat maps an image MIME type to a Bedrock ImageFormat, defaulting to PNG.
func bedrockImageFormat(mimeType string) brtypes.ImageFormat {
	switch mimeType {
	case "image/jpeg", "image/jpg":
		return brtypes.ImageFormatJpeg
	case "image/gif":
		return brtypes.ImageFormatGif
	case "image/webp":
		return brtypes.ImageFormatWebp
	default:
		return brtypes.ImageFormatPng
	}
}

// bedrockVideoFormat maps a video MIME type to a Bedrock VideoFormat, defaulting to MP4.
func bedrockVideoFormat(mimeType string) brtypes.VideoFormat {
	switch strings.ToLower(mimeType) {
	case "video/quicktime":
		return brtypes.VideoFormatMov
	case "video/x-matroska":
		return brtypes.VideoFormatMkv
	case "video/webm":
		return brtypes.VideoFormatWebm
	case "video/x-flv":
		return brtypes.VideoFormatFlv
	case "video/mpeg":
		return brtypes.VideoFormatMpeg
	case "video/mpg":
		return brtypes.VideoFormatMpg
	case "video/x-ms-wmv":
		return brtypes.VideoFormatWmv
	case "video/3gpp":
		return brtypes.VideoFormatThreeGp
	default:
		return brtypes.VideoFormatMp4
	}
}

// mergeBedrockMessages concatenates the content of consecutive same-role
// messages into a single message, since Bedrock Converse prefers alternating
// user/assistant turns.
func mergeBedrockMessages(messages []brtypes.Message) []brtypes.Message {
	if len(messages) < 2 {
		return messages
	}
	out := []brtypes.Message{messages[0]}
	for _, msg := range messages[1:] {
		last := &out[len(out)-1]
		if last.Role == msg.Role {
			last.Content = append(last.Content, msg.Content...)
		} else {
			out = append(out, msg)
		}
	}
	return out
}

// formatBedrockImagePrompt flattens prompt segments into a bedrockImagePrompt,
// collecting system/user/safety/negative text and base64-encoding image files
// into Images (or Masks for mask-role segments).
func formatBedrockImagePrompt(ctx context.Context, segments []PromptSegment) (bedrockImagePrompt, error) {
	var system []string
	var user []string
	var safety []string
	var negative []string
	prompt := bedrockImagePrompt{}
	for _, segment := range segments {
		switch segment.Role {
		case PromptRoleSystem:
			if segment.Content != "" {
				system = append(system, segment.Content)
			}
		case PromptRoleSafety:
			if segment.Content != "" {
				safety = append(safety, segment.Content)
			}
		case PromptRoleNegative:
			if segment.Content != "" {
				negative = append(negative, segment.Content)
			}
		default:
			if segment.Content != "" {
				user = append(user, segment.Content)
			}
		}
		for _, file := range segment.Files {
			if !strings.HasPrefix(file.MIMEType(), "image/") {
				continue
			}
			encoded, err := dataSourceToBase64(ctx, file)
			if err != nil {
				return bedrockImagePrompt{}, err
			}
			if segment.Role == PromptRoleMask {
				prompt.Masks = append(prompt.Masks, encoded)
			} else {
				prompt.Images = append(prompt.Images, encoded)
			}
		}
	}
	prompt.System = strings.Join(system, "\n\n")
	prompt.Text = strings.TrimSpace(strings.Join([]string{
		strings.Join(user, "\n\n"),
		strings.Join(safety, "\n\n"),
	}, "\n\n"))
	prompt.Negative = strings.Join(negative, ", ")
	return prompt, nil
}

// Execute runs a non-streaming Bedrock completion or image-generation request.
func (d *BedrockDriver) Execute(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (*ExecutionResponse, error) {
	return executeWithPrompt(ctx, d, segments, options, d.requestTextCompletion)
}

// Stream runs a Bedrock ConverseStream request and emits normalized chunks.
func (d *BedrockDriver) Stream(ctx context.Context, segments []PromptSegment, options ExecutionOptions) (CompletionStream, error) {
	return streamWithPrompt(ctx, d, segments, options, d.requestTextCompletionStream)
}

// requestTextCompletion issues a single Converse call (or image generation),
// merging any prior conversation, then assembles the completion and the updated
// conversation state.
func (d *BedrockDriver) requestTextCompletion(ctx context.Context, prompt any, options ExecutionOptions) (*Completion, error) {
	if d.runtime == nil {
		return nil, errors.New("bedrock runtime client is nil")
	}
	if isBedrockImageModel(options.Model) {
		return d.requestBedrockImageGeneration(ctx, prompt.(bedrockImagePrompt), options)
	}
	bp := prompt.(bedrockPrompt)
	bp = bedrockConversationInput(options.Conversation, bp)
	input := d.converseInput(bp, options)
	output, err := d.runtime.Converse(ctx, input, d.runtimeOptions(options.HTTPTimeout)...)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	completion := extractBedrockCompletion(output, options)
	completion.Conversation = finalizeBedrockConversation(appendBedrockResponseToConversation(bp, output), options)
	return completion, nil
}

// requestTextCompletionStream opens a ConverseStream, feeds each event through
// a bedrockStreamState to produce normalized chunks on a channel, and finalizes
// the streamed conversation when the stream ends. Image models cannot stream.
func (d *BedrockDriver) requestTextCompletionStream(ctx context.Context, prompt any, options ExecutionOptions) (CompletionStream, error) {
	if d.runtime == nil {
		return nil, errors.New("bedrock runtime client is nil")
	}
	if isBedrockImageModel(options.Model) {
		return nil, errors.New("bedrock image models do not support streaming")
	}
	bp := prompt.(bedrockPrompt)
	bp = bedrockConversationInput(options.Conversation, bp)
	stream, err := d.runtime.ConverseStream(ctx, d.converseStreamInput(bp, options), d.runtimeOptions(options.HTTPTimeout)...)
	if err != nil {
		return nil, d.formatError(err, options.Model, "stream")
	}
	ch := make(chan streamItem)
	completion := &ExecutionResponse{Prompt: prompt}
	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()
		state := bedrockStreamState{toolBlocks: map[int32]bedrockStreamingToolBlock{}}
		for event := range stream.Events() {
			chunk := state.chunk(event, options)
			if len(chunk.Result) > 0 || len(chunk.ToolUse) > 0 || chunk.FinishReason != "" || chunk.TokenUsage != nil {
				ch <- streamItem{Chunk: chunk}
			}
		}
		if err := stream.Err(); err != nil {
			ch <- streamItem{Err: d.formatError(err, options.Model, "stream")}
		}
	}()
	return newChannelStreamWithFinalizer(ch, completion, stream.Close, func(resp *ExecutionResponse) {
		resp.Conversation = finalizeBedrockConversation(buildBedrockStreamingConversation(bp, &resp.Completion), options)
	}), nil
}

// converseInput builds the Converse request from a prompt and options: messages
// (downgrading tool blocks to text when no tools are configured), inference
// config, tools, thinking/effort additional fields, and Claude cache points.
func (d *BedrockDriver) converseInput(prompt bedrockPrompt, options ExecutionOptions) *bedrockruntime.ConverseInput {
	messages := prompt.Messages
	if len(options.Tools) == 0 && bedrockMessagesContainToolBlocks(messages) {
		messages = convertBedrockToolBlocksToText(messages)
	}
	input := &bedrockruntime.ConverseInput{
		ModelId:         aws.String(options.Model),
		Messages:        messages,
		System:          prompt.System,
		InferenceConfig: bedrockInferenceConfig(options.ModelOptions),
		RequestMetadata: options.Labels,
	}
	if len(options.Tools) > 0 {
		input.ToolConfig = &brtypes.ToolConfiguration{Tools: bedrockTools(options.Tools)}
	}
	additional := bedrockAdditionalFields(options)
	if len(additional) > 0 {
		input.AdditionalModelRequestFields = brdoc.NewLazyDocument(additional)
	}
	if strings.Contains(strings.ToLower(options.Model), "claude") {
		applyBedrockClaudeCache(input, options)
	}
	return input
}

// converseStreamInput builds a ConverseStream request by copying the fields of
// the equivalent Converse request.
func (d *BedrockDriver) converseStreamInput(prompt bedrockPrompt, options ExecutionOptions) *bedrockruntime.ConverseStreamInput {
	converse := d.converseInput(prompt, options)
	return &bedrockruntime.ConverseStreamInput{
		ModelId:                      converse.ModelId,
		Messages:                     converse.Messages,
		System:                       converse.System,
		InferenceConfig:              converse.InferenceConfig,
		ToolConfig:                   converse.ToolConfig,
		AdditionalModelRequestFields: converse.AdditionalModelRequestFields,
		RequestMetadata:              converse.RequestMetadata,
	}
}

// bedrockConversationInput merges the new prompt into any prior conversation
// state, overriding the system blocks, concatenating messages, and repairing
// orphaned tool_use blocks left by an interrupted prior turn.
func bedrockConversationInput(conversation any, prompt bedrockPrompt) bedrockPrompt {
	out := bedrockPrompt{}
	switch value := conversation.(type) {
	case bedrockPrompt:
		out = value
	case *bedrockPrompt:
		if value != nil {
			out = *value
		}
	}
	if len(prompt.System) > 0 {
		out.System = prompt.System
	}
	out.Messages = fixBedrockOrphanedToolUse(mergeBedrockMessages(append(out.Messages, prompt.Messages...)))
	return out
}

// appendBedrockResponseToConversation appends the model's response message to
// the conversation, dropping reasoning blocks (which must not be replayed).
func appendBedrockResponseToConversation(prompt bedrockPrompt, output *bedrockruntime.ConverseOutput) bedrockPrompt {
	if output == nil || output.Output == nil {
		return prompt
	}
	if messageOutput, ok := output.Output.(*brtypes.ConverseOutputMemberMessage); ok {
		msg := messageOutput.Value
		msg.Content = filterBedrockReasoningBlocks(msg.Content)
		prompt.Messages = mergeBedrockMessages(append(prompt.Messages, msg))
	}
	return prompt
}

// buildBedrockStreamingConversation reconstructs the assistant turn from the
// accumulated streamed completion (text plus tool_use blocks) and appends it to
// the conversation, since streaming yields no single response message.
func buildBedrockStreamingConversation(prompt bedrockPrompt, completion *Completion) bedrockPrompt {
	content := []brtypes.ContentBlock{}
	if text := completionResultsToText(completion.Result); text != "" {
		content = append(content, &brtypes.ContentBlockMemberText{Value: text})
	}
	for _, tool := range completion.ToolUse {
		content = append(content, &brtypes.ContentBlockMemberToolUse{Value: brtypes.ToolUseBlock{
			ToolUseId: aws.String(tool.ID),
			Name:      aws.String(tool.ToolName),
			Input:     brdoc.NewLazyDocument(tool.ToolInput),
		}})
	}
	if len(content) > 0 {
		prompt.Messages = mergeBedrockMessages(append(prompt.Messages, brtypes.Message{Role: brtypes.ConversationRoleAssistant, Content: content}))
	}
	return prompt
}

// filterBedrockReasoningBlocks drops reasoning-content blocks from a content slice.
func filterBedrockReasoningBlocks(content []brtypes.ContentBlock) []brtypes.ContentBlock {
	out := make([]brtypes.ContentBlock, 0, len(content))
	for _, block := range content {
		if _, ok := block.(*brtypes.ContentBlockMemberReasoningContent); ok {
			continue
		}
		out = append(out, block)
	}
	return out
}

// bedrockMessagesContainToolBlocks reports whether any message carries a
// tool_use or tool_result block.
func bedrockMessagesContainToolBlocks(messages []brtypes.Message) bool {
	for _, msg := range messages {
		for _, block := range msg.Content {
			switch block.(type) {
			case *brtypes.ContentBlockMemberToolUse, *brtypes.ContentBlockMemberToolResult:
				return true
			}
		}
	}
	return false
}

// convertBedrockToolBlocksToText rewrites tool_use and tool_result blocks as
// human-readable text, used to replay tool history when the current request has
// no tool configuration (Converse rejects tool blocks without a ToolConfig).
func convertBedrockToolBlocksToText(messages []brtypes.Message) []brtypes.Message {
	out := make([]brtypes.Message, 0, len(messages))
	for _, msg := range messages {
		next := msg
		next.Content = make([]brtypes.ContentBlock, 0, len(msg.Content))
		for _, block := range msg.Content {
			switch value := block.(type) {
			case *brtypes.ContentBlockMemberToolUse:
				next.Content = append(next.Content, &brtypes.ContentBlockMemberText{Value: fmt.Sprintf("[Tool call: %s(%s)]", aws.ToString(value.Value.Name), truncateForConversation(toolInputString(documentToAny(value.Value.Input)), 500))})
			case *brtypes.ContentBlockMemberToolResult:
				next.Content = append(next.Content, &brtypes.ContentBlockMemberText{Value: "[Tool result: " + truncateForConversation(bedrockToolResultText(value.Value.Content), 500) + "]"})
			default:
				next.Content = append(next.Content, block)
			}
		}
		out = append(out, next)
	}
	return out
}

// bedrockToolResultText joins the text blocks of a tool result, returning
// "No content" when there are none.
func bedrockToolResultText(content []brtypes.ToolResultContentBlock) string {
	var parts []string
	for _, block := range content {
		if text, ok := block.(*brtypes.ToolResultContentBlockMemberText); ok {
			parts = append(parts, text.Value)
		}
	}
	if len(parts) == 0 {
		return "No content"
	}
	return strings.Join(parts, "\n")
}

// fixBedrockOrphanedToolUse repairs conversations where an assistant tool_use
// has no matching tool_result in the following user turn (e.g. the user
// interrupted before the tool ran), by synthesizing an "interrupted" result.
// Converse requires every tool_use to be answered by a tool_result.
func fixBedrockOrphanedToolUse(messages []brtypes.Message) []brtypes.Message {
	if len(messages) < 2 {
		return messages
	}
	out := make([]brtypes.Message, len(messages))
	copy(out, messages)
	for i := 0; i < len(out)-1; i++ {
		current := out[i]
		if current.Role != brtypes.ConversationRoleAssistant {
			continue
		}
		var tools []brtypes.ToolUseBlock
		for _, block := range current.Content {
			if tool, ok := block.(*brtypes.ContentBlockMemberToolUse); ok {
				tools = append(tools, tool.Value)
			}
		}
		if len(tools) == 0 || out[i+1].Role != brtypes.ConversationRoleUser {
			continue
		}
		results := map[string]bool{}
		for _, block := range out[i+1].Content {
			if result, ok := block.(*brtypes.ContentBlockMemberToolResult); ok {
				results[aws.ToString(result.Value.ToolUseId)] = true
			}
		}
		var synthetic []brtypes.ContentBlock
		for _, tool := range tools {
			id := aws.ToString(tool.ToolUseId)
			if id == "" || results[id] {
				continue
			}
			name := aws.ToString(tool.Name)
			if name == "" {
				name = "unknown"
			}
			synthetic = append(synthetic, &brtypes.ContentBlockMemberToolResult{Value: brtypes.ToolResultBlock{
				ToolUseId: aws.String(id),
				Content: []brtypes.ToolResultContentBlock{&brtypes.ToolResultContentBlockMemberText{
					Value: fmt.Sprintf(`[Tool interrupted: The user stopped the operation before "%s" could execute.]`, name),
				}},
			}})
		}
		if len(synthetic) > 0 {
			out[i+1].Content = append(synthetic, out[i+1].Content...)
		}
	}
	return out
}

// applyBedrockClaudeCache strips any pre-existing cache points and, when caching
// is enabled, inserts fresh cache-point blocks for Claude-on-Bedrock: after the
// system content, after the tools, and on the second-to-last message (so the
// stable prefix of a multi-turn conversation is cached).
func applyBedrockClaudeCache(input *bedrockruntime.ConverseInput, options ExecutionOptions) {
	input.Messages = stripBedrockMessageCachePoints(input.Messages)
	input.System = stripBedrockSystemCachePoints(input.System)
	if input.ToolConfig != nil {
		input.ToolConfig.Tools = stripBedrockToolCachePoints(input.ToolConfig.Tools)
	}
	if !optionBool(options.ModelOptions, "cache_enabled") {
		return
	}
	cache := bedrockCachePoint(options)
	if len(input.System) > 0 {
		input.System = append(input.System, &brtypes.SystemContentBlockMemberCachePoint{Value: cache})
	}
	if input.ToolConfig != nil && len(input.ToolConfig.Tools) > 0 {
		input.ToolConfig.Tools = append(input.ToolConfig.Tools, &brtypes.ToolMemberCachePoint{Value: cache})
	}
	if len(input.Messages) >= 4 {
		pivot := &input.Messages[len(input.Messages)-2]
		if len(pivot.Content) > 0 {
			pivot.Content = append(pivot.Content, &brtypes.ContentBlockMemberCachePoint{Value: cache})
		}
	}
}

// bedrockCachePoint builds a default cache-point block, applying a custom TTL
// from cache_ttl when provided.
func bedrockCachePoint(options ExecutionOptions) brtypes.CachePointBlock {
	cache := brtypes.CachePointBlock{Type: brtypes.CachePointTypeDefault}
	if ttl := optionString(options.ModelOptions, "cache_ttl"); ttl != "" {
		cache.Ttl = brtypes.CacheTTL(ttl)
	}
	return cache
}

// stripBedrockMessageCachePoints removes cache-point blocks from message content
// so they can be reinserted cleanly per request.
func stripBedrockMessageCachePoints(messages []brtypes.Message) []brtypes.Message {
	out := make([]brtypes.Message, 0, len(messages))
	for _, msg := range messages {
		next := msg
		next.Content = make([]brtypes.ContentBlock, 0, len(msg.Content))
		for _, block := range msg.Content {
			if _, ok := block.(*brtypes.ContentBlockMemberCachePoint); ok {
				continue
			}
			next.Content = append(next.Content, block)
		}
		out = append(out, next)
	}
	return out
}

// stripBedrockSystemCachePoints removes cache-point blocks from system content.
func stripBedrockSystemCachePoints(system []brtypes.SystemContentBlock) []brtypes.SystemContentBlock {
	out := make([]brtypes.SystemContentBlock, 0, len(system))
	for _, block := range system {
		if _, ok := block.(*brtypes.SystemContentBlockMemberCachePoint); ok {
			continue
		}
		out = append(out, block)
	}
	return out
}

// stripBedrockToolCachePoints removes cache-point entries from the tool list.
func stripBedrockToolCachePoints(tools []brtypes.Tool) []brtypes.Tool {
	out := make([]brtypes.Tool, 0, len(tools))
	for _, tool := range tools {
		if _, ok := tool.(*brtypes.ToolMemberCachePoint); ok {
			continue
		}
		out = append(out, tool)
	}
	return out
}

// bedrockInferenceConfig maps the generic model options (max_tokens,
// temperature, top_p, stop_sequence) to a Converse InferenceConfiguration,
// returning nil when none are set.
func bedrockInferenceConfig(options map[string]any) *brtypes.InferenceConfiguration {
	cfg := &brtypes.InferenceConfiguration{}
	if v := optionInt(options, "max_tokens"); v != nil {
		cfg.MaxTokens = aws.Int32(int32(*v))
	}
	if v := optionFloat(options, "temperature"); v != nil {
		cfg.Temperature = aws.Float32(float32(*v))
	}
	if v := optionFloat(options, "top_p"); v != nil {
		cfg.TopP = aws.Float32(float32(*v))
	}
	if stops := optionStringSlice(options, "stop_sequence"); len(stops) > 0 {
		cfg.StopSequences = stops
	}
	if cfg.MaxTokens == nil && cfg.Temperature == nil && cfg.TopP == nil && len(cfg.StopSequences) == 0 {
		return nil
	}
	return cfg
}

// bedrockAdditionalFields assembles the AdditionalModelRequestFields document.
// These carry model-specific knobs that Converse does not model directly: Claude
// thinking/reasoning config (explicit budget vs. adaptive effort vs. disabled,
// including the 128k output beta for claude-3-7-sonnet), top_k (when thinking is
// off and the model permits it), and reasoning_effort for non-Claude models.
func bedrockAdditionalFields(options ExecutionOptions) map[string]any {
	out := map[string]any{}
	claude := strings.Contains(strings.ToLower(options.Model), "claude")
	thinkingActive := false
	if claude && isClaudeVersionGTE(options.Model, 3, 7) {
		// Claude thinking fields are passed as additional model request fields in
		// Bedrock Converse, not as normal inference config fields. Explicit budgets
		// take priority over adaptive effort-based thinking.
		if v := optionInt(options.ModelOptions, "thinking_budget_tokens"); v != nil {
			out["reasoning_config"] = map[string]any{"type": "enabled", "budget_tokens": *v}
			thinkingActive = true
			if strings.Contains(options.Model, "claude-3-7-sonnet") {
				if maxTokens := optionInt(options.ModelOptions, "max_tokens"); (maxTokens != nil && *maxTokens > 64000) || *v > 64000 {
					out["anthropic_beta"] = []string{"output-128k-2025-02-19"}
				}
			}
		} else if effort := optionString(options.ModelOptions, "effort"); effort != "" && claudeSupportsAdaptiveThinking(options.Model) {
			// include_thoughts controls display only; adaptive thinking is enabled by effort.
			display := "omitted"
			if optionBool(options.ModelOptions, "include_thoughts") {
				display = "summarized"
			}
			out["reasoning_config"] = map[string]any{"type": "adaptive", "display": display}
			out["output_config"] = map[string]any{"effort": effort}
			thinkingActive = true
		} else {
			// Older thinking-capable Claude models are disabled unless the caller
			// provides an explicit budget.
			out["reasoning_config"] = map[string]any{"type": "disabled"}
		}
		if effort := optionString(options.ModelOptions, "effort"); effort != "" {
			out["output_config"] = map[string]any{"effort": effort}
		}
	}
	if v := optionInt(options.ModelOptions, "top_k"); v != nil && !thinkingActive && !claudeHasSamplingParameterRestriction(options.Model) {
		out["top_k"] = *v
	}
	if v, ok := options.ModelOptions["reasoning_effort"].(string); ok && !claude {
		out["reasoning_effort"] = v
	}
	return out
}

// bedrockTools converts generic tool definitions into Converse tool specifications.
func bedrockTools(tools []ToolDefinition) []brtypes.Tool {
	out := make([]brtypes.Tool, 0, len(tools))
	for _, tool := range tools {
		out = append(out, &brtypes.ToolMemberToolSpec{Value: brtypes.ToolSpecification{
			Name:        aws.String(tool.Name),
			Description: aws.String(tool.Description),
			InputSchema: &brtypes.ToolInputSchemaMemberJson{Value: brdoc.NewLazyDocument(tool.InputSchema)},
		}})
	}
	return out
}

// extractBedrockCompletion converts a Converse response into a Completion,
// collecting text and tool_use blocks (and reasoning text when thoughts are
// requested), and reporting a tool_use finish reason when tool calls are present.
func extractBedrockCompletion(output *bedrockruntime.ConverseOutput, options ExecutionOptions) *Completion {
	completion := &Completion{
		TokenUsage:   bedrockUsage(output.Usage),
		FinishReason: bedrockFinishReason(output.StopReason),
	}
	if options.IncludeOriginalResponse {
		completion.OriginalResponse = output
	}
	if output.Output == nil {
		completion.Result = []CompletionResult{{Type: ResultTypeText, Value: ""}}
		return completion
	}
	if messageOutput, ok := output.Output.(*brtypes.ConverseOutputMemberMessage); ok {
		for _, block := range messageOutput.Value.Content {
			switch value := block.(type) {
			case *brtypes.ContentBlockMemberText:
				completion.Result = append(completion.Result, CompletionResult{Type: ResultTypeText, Value: value.Value})
			case *brtypes.ContentBlockMemberReasoningContent:
				if bedrockIncludeThoughts(options) {
					if text := bedrockReasoningContentText(value.Value); text != "" {
						completion.Result = append(completion.Result, CompletionResult{Type: ResultTypeText, Value: text})
					}
				}
			case *brtypes.ContentBlockMemberToolUse:
				input := documentToAny(value.Value.Input)
				completion.ToolUse = append(completion.ToolUse, ToolUse{
					ID:        aws.ToString(value.Value.ToolUseId),
					ToolName:  aws.ToString(value.Value.Name),
					ToolInput: input,
				})
			}
		}
	}
	if len(completion.Result) == 0 {
		completion.Result = []CompletionResult{{Type: ResultTypeText, Value: ""}}
	}
	if len(completion.ToolUse) > 0 {
		completion.FinishReason = "tool_use"
	}
	return completion
}

// documentToAny unmarshals a Smithy lazy document into a plain Go value,
// returning nil on failure.
func documentToAny(doc brdoc.Interface) any {
	if doc == nil {
		return nil
	}
	var out any
	if err := doc.UnmarshalSmithyDocument(&out); err != nil {
		return nil
	}
	return out
}

// bedrockUsage maps Converse token usage into ExecutionTokenUsage, folding cache
// read/write tokens into the total prompt count while also exposing them
// separately.
func bedrockUsage(usage *brtypes.TokenUsage) *ExecutionTokenUsage {
	if usage == nil {
		return nil
	}
	prompt := int32Value(usage.InputTokens)
	result := int32Value(usage.OutputTokens)
	total := int32Value(usage.TotalTokens)
	cacheRead := int32Value(usage.CacheReadInputTokens)
	cacheWrite := int32Value(usage.CacheWriteInputTokens)
	return &ExecutionTokenUsage{
		Prompt:           prompt + cacheRead + cacheWrite,
		PromptNew:        prompt,
		Result:           result,
		Total:            total,
		PromptCached:     cacheRead,
		PromptCacheWrite: cacheWrite,
	}
}

func int32Value(v *int32) int {
	if v == nil {
		return 0
	}
	return int(*v)
}

// bedrockFinishReason normalizes a Converse stop reason to the common finish
// reason vocabulary (stop/length/tool_use), passing through other values.
func bedrockFinishReason(reason brtypes.StopReason) string {
	switch reason {
	case brtypes.StopReasonEndTurn:
		return "stop"
	case brtypes.StopReasonMaxTokens:
		return "length"
	case brtypes.StopReasonToolUse:
		return "tool_use"
	default:
		return string(reason)
	}
}

// bedrockStreamingToolBlock tracks the id and name of an in-progress tool_use
// block whose input arrives incrementally across delta events.
type bedrockStreamingToolBlock struct {
	id   string
	name string
}

// bedrockStreamState is the streaming state machine for ConverseStream. It keeps
// per-index tool blocks so partial tool_use input deltas can be attributed to
// the correct call as events arrive.
type bedrockStreamState struct {
	toolBlocks map[int32]bedrockStreamingToolBlock
}

// chunk advances the state machine for one ConverseStream event and returns the
// resulting normalized CompletionChunk (content start/delta/stop, message stop,
// or metadata).
func (s *bedrockStreamState) chunk(event brtypes.ConverseStreamOutput, options ExecutionOptions) CompletionChunk {
	switch value := event.(type) {
	case *brtypes.ConverseStreamOutputMemberContentBlockStart:
		return s.contentBlockStart(value.Value)
	case *brtypes.ConverseStreamOutputMemberContentBlockDelta:
		return s.contentBlockDelta(value.Value, options)
	case *brtypes.ConverseStreamOutputMemberContentBlockStop:
		delete(s.toolBlocks, bedrockContentBlockIndex(value.Value.ContentBlockIndex))
	case *brtypes.ConverseStreamOutputMemberMessageStop:
		return CompletionChunk{FinishReason: bedrockFinishReason(value.Value.StopReason)}
	case *brtypes.ConverseStreamOutputMemberMetadata:
		return CompletionChunk{TokenUsage: bedrockUsage(value.Value.Usage)}
	}
	return CompletionChunk{}
}

// contentBlockStart records a starting tool_use block (its id and name) and
// emits an initial empty-input tool-use chunk.
func (s *bedrockStreamState) contentBlockStart(event brtypes.ContentBlockStartEvent) CompletionChunk {
	toolStart, ok := event.Start.(*brtypes.ContentBlockStartMemberToolUse)
	if !ok {
		return CompletionChunk{}
	}
	index := bedrockContentBlockIndex(event.ContentBlockIndex)
	block := bedrockStreamingToolBlock{
		id:   aws.ToString(toolStart.Value.ToolUseId),
		name: aws.ToString(toolStart.Value.Name),
	}
	s.toolBlocks[index] = block
	return CompletionChunk{ToolUse: []ToolUse{{ID: block.id, ToolName: block.name, ToolInput: ""}}}
}

// contentBlockDelta turns a delta event into a chunk: text deltas, incremental
// tool_use input (attributed to the tracked block by index), or reasoning text
// when thoughts are requested.
func (s *bedrockStreamState) contentBlockDelta(event brtypes.ContentBlockDeltaEvent, options ExecutionOptions) CompletionChunk {
	switch delta := event.Delta.(type) {
	case *brtypes.ContentBlockDeltaMemberText:
		return CompletionChunk{Result: []CompletionResult{{Type: ResultTypeText, Value: delta.Value}}}
	case *brtypes.ContentBlockDeltaMemberToolUse:
		index := bedrockContentBlockIndex(event.ContentBlockIndex)
		block := s.toolBlocks[index]
		return CompletionChunk{ToolUse: []ToolUse{{
			ID:        block.id,
			ToolName:  "",
			ToolInput: aws.ToString(delta.Value.Input),
		}}}
	case *brtypes.ContentBlockDeltaMemberReasoningContent:
		if !bedrockIncludeThoughts(options) {
			return CompletionChunk{}
		}
		if text := bedrockReasoningDeltaText(delta.Value); text != "" {
			return CompletionChunk{Result: []CompletionResult{{Type: ResultTypeText, Value: text}}}
		}
	}
	return CompletionChunk{}
}

// bedrockContentBlockIndex dereferences a content-block index pointer, using -1
// as the sentinel for a missing index.
func bedrockContentBlockIndex(v *int32) int32 {
	if v == nil {
		return -1
	}
	return *v
}

// bedrockIncludeThoughts reports whether reasoning content should be surfaced:
// always for DeepSeek-R1 (which has no separate display toggle) and otherwise
// when include_thoughts is set.
func bedrockIncludeThoughts(options ExecutionOptions) bool {
	model := strings.ToLower(options.Model)
	return (strings.Contains(model, "deepseek") && strings.Contains(model, "r1")) || optionBool(options.ModelOptions, "include_thoughts")
}

// bedrockReasoningDeltaText extracts displayable text from a streamed reasoning
// delta, rendering redacted content as a placeholder and signatures as a
// paragraph break.
func bedrockReasoningDeltaText(delta brtypes.ReasoningContentBlockDelta) string {
	switch value := delta.(type) {
	case *brtypes.ReasoningContentBlockDeltaMemberText:
		return value.Value
	case *brtypes.ReasoningContentBlockDeltaMemberRedactedContent:
		return "[Redacted thinking: " + string(value.Value) + "]"
	case *brtypes.ReasoningContentBlockDeltaMemberSignature:
		return "\n\n"
	default:
		return ""
	}
}

// bedrockReasoningContentText extracts displayable text from a complete
// reasoning content block, appending a paragraph break when the block carries a
// signature and rendering redacted content as a placeholder.
func bedrockReasoningContentText(block brtypes.ReasoningContentBlock) string {
	switch value := block.(type) {
	case *brtypes.ReasoningContentBlockMemberReasoningText:
		text := aws.ToString(value.Value.Text)
		if value.Value.Signature != nil {
			return text + "\n\n"
		}
		return text
	case *brtypes.ReasoningContentBlockMemberRedactedContent:
		return "[Redacted thinking: " + string(value.Value) + "]"
	default:
		return ""
	}
}

// ListModels returns the supported Bedrock models: filtered foundation models
// plus any custom models and inference profiles, sorted by ID.
func (d *BedrockDriver) ListModels(ctx context.Context, _ *ModelSearchPayload) ([]AIModel, error) {
	if d.models == nil {
		return nil, errors.New("bedrock model client is nil")
	}
	output, err := d.models.ListFoundationModels(ctx, &bedrock.ListFoundationModelsInput{})
	if err != nil {
		return nil, d.formatError(err, "", "listModels")
	}
	models := make([]AIModel, 0, len(output.ModelSummaries))
	for _, summary := range output.ModelSummaries {
		if !bedrockModelSupported(summary) {
			continue
		}
		models = append(models, bedrockFoundationAIModel(summary))
	}
	if custom, err := d.models.ListCustomModels(ctx, &bedrock.ListCustomModelsInput{}); err == nil {
		for _, summary := range custom.ModelSummaries {
			models = append(models, bedrockCustomAIModel(summary))
		}
	}
	if profiles, err := d.models.ListInferenceProfiles(ctx, &bedrock.ListInferenceProfilesInput{}); err == nil {
		for _, summary := range profiles.InferenceProfileSummaries {
			if model, ok := bedrockInferenceProfileAIModel(summary); ok {
				models = append(models, model)
			}
		}
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	return models, nil
}

// bedrockModelSupported filters foundation models to those this driver can
// serve: a supported publisher, on-demand inference, non-embedding output, and
// not on the per-publisher unsupported list or a rerank/embed/marengo model.
func bedrockModelSupported(summary bedrocktypes.FoundationModelSummary) bool {
	id := strings.ToLower(aws.ToString(summary.ModelId))
	if !bedrockPublisherSupported(aws.ToString(summary.ProviderName)) {
		return false
	}
	if len(summary.InferenceTypesSupported) > 0 {
		supportsOnDemand := false
		for _, inferenceType := range summary.InferenceTypesSupported {
			if inferenceType == bedrocktypes.InferenceTypeOnDemand {
				supportsOnDemand = true
				break
			}
		}
		if !supportsOnDemand {
			return false
		}
	}
	for _, modality := range summary.OutputModalities {
		if modality == bedrocktypes.ModelModalityEmbedding {
			return false
		}
	}
	for _, unsupported := range bedrockUnsupportedModels(aws.ToString(summary.ProviderName)) {
		if strings.Contains(id, unsupported) {
			return false
		}
	}
	if strings.Contains(id, "rerank") || strings.Contains(id, "embed") || strings.Contains(id, "marengo") {
		return false
	}
	return true
}

// bedrockFoundationAIModel maps a foundation model summary to an AIModel,
// preferring the ARN as the ID.
func bedrockFoundationAIModel(summary bedrocktypes.FoundationModelSummary) AIModel {
	id := aws.ToString(summary.ModelArn)
	if id == "" {
		id = aws.ToString(summary.ModelId)
	}
	return AIModel{
		ID:               id,
		Name:             strings.TrimSpace(aws.ToString(summary.ProviderName) + " " + aws.ToString(summary.ModelName)),
		Provider:         ProviderBedrock,
		Owner:            aws.ToString(summary.ProviderName),
		Type:             "text",
		CanStream:        aws.ToBool(summary.ResponseStreamingSupported),
		InputModalities:  bedrockModalities(summary.InputModalities),
		OutputModalities: bedrockModalities(summary.OutputModalities),
		ToolSupport:      bedrockLikelySupportsTools(id),
	}
}

// bedrockCustomAIModel maps a custom model summary to an AIModel, inferring
// modalities and tool support from the model ID.
func bedrockCustomAIModel(summary bedrocktypes.CustomModelSummary) AIModel {
	id := aws.ToString(summary.ModelArn)
	name := aws.ToString(summary.ModelName)
	if name == "" {
		name = id
	}
	return AIModel{
		ID:               id,
		Name:             name,
		Provider:         ProviderBedrock,
		Owner:            "custom",
		Description:      "Custom model from " + aws.ToString(summary.BaseModelName),
		Type:             "text",
		IsCustom:         true,
		InputModalities:  bedrockCapabilitiesInput(id),
		OutputModalities: bedrockCapabilitiesOutput(id),
		ToolSupport:      bedrockLikelySupportsTools(id),
	}
}

// bedrockInferenceProfileAIModel maps an inference profile to an AIModel,
// returning false when it lacks an ID or its inferred publisher is unsupported.
func bedrockInferenceProfileAIModel(summary bedrocktypes.InferenceProfileSummary) (AIModel, bool) {
	id := aws.ToString(summary.InferenceProfileArn)
	if id == "" {
		id = aws.ToString(summary.InferenceProfileId)
	}
	if id == "" {
		return AIModel{}, false
	}
	provider := bedrockProviderFromProfile(summary)
	if provider == "" || !bedrockPublisherSupported(provider) {
		return AIModel{}, false
	}
	name := aws.ToString(summary.InferenceProfileName)
	if name == "" {
		name = id
	}
	return AIModel{
		ID:               id,
		Name:             name,
		Provider:         ProviderBedrock,
		Owner:            provider,
		Type:             "text",
		InputModalities:  bedrockCapabilitiesInput(id),
		OutputModalities: bedrockCapabilitiesOutput(id),
		ToolSupport:      bedrockLikelySupportsTools(id),
	}, true
}

// bedrockProviderFromProfile infers the publisher of an inference profile by
// scanning its name, id, and ARN for a known supported publisher token.
func bedrockProviderFromProfile(summary bedrocktypes.InferenceProfileSummary) string {
	candidates := []string{aws.ToString(summary.InferenceProfileName), aws.ToString(summary.InferenceProfileId), aws.ToString(summary.InferenceProfileArn)}
	for _, candidate := range candidates {
		normalized := strings.ToLower(candidate)
		for _, provider := range bedrockSupportedPublishers() {
			if strings.Contains(normalized, provider) {
				return provider
			}
		}
	}
	return ""
}

// bedrockPublisherSupported reports whether a provider name contains a supported
// publisher token.
func bedrockPublisherSupported(provider string) bool {
	normalized := strings.ToLower(provider)
	for _, supported := range bedrockSupportedPublishers() {
		if strings.Contains(normalized, supported) {
			return true
		}
	}
	return false
}

// bedrockSupportedPublishers lists the model publishers this driver exposes.
func bedrockSupportedPublishers() []string {
	return []string{"amazon", "anthropic", "cohere", "ai21", "mistral", "meta", "deepseek", "writer", "openai", "twelvelabs", "qwen"}
}

// bedrockUnsupportedModels returns model-id substrings to exclude for a given
// publisher (e.g. image/video/audio/rerank/embed variants not served here).
func bedrockUnsupportedModels(provider string) []string {
	normalized := strings.ToLower(provider)
	switch {
	case strings.Contains(normalized, "amazon"):
		return []string{"titan-image-generator", "nova-reel", "nova-sonic", "rerank"}
	case strings.Contains(normalized, "cohere"):
		return []string{"rerank", "embed"}
	case strings.Contains(normalized, "twelvelabs"):
		return []string{"marengo"}
	default:
		return nil
	}
}

// bedrockModalities maps Bedrock model modalities to the common modality strings.
func bedrockModalities(modalities []bedrocktypes.ModelModality) []string {
	out := make([]string, 0, len(modalities))
	for _, modality := range modalities {
		switch modality {
		case bedrocktypes.ModelModalityText:
			out = append(out, "text")
		case bedrocktypes.ModelModalityImage:
			out = append(out, "image")
		case bedrocktypes.ModelModalityEmbedding:
			out = append(out, "embedding")
		default:
			out = append(out, strings.ToLower(string(modality)))
		}
	}
	return out
}

// bedrockCapabilitiesInput infers input modalities from a model ID when the API
// does not report them (custom models and inference profiles), distinguishing
// image-generation, multimodal embedding, and text/image embedding families.
func bedrockCapabilitiesInput(model string) []string {
	model = strings.ToLower(model)
	if isBedrockImageModel(model) {
		return []string{"text", "image"}
	}
	if strings.Contains(model, "embed") {
		if strings.Contains(model, "nova") || strings.Contains(model, "twelvelabs.marengo") {
			return []string{"text", "image", "video", "audio"}
		}
		if strings.Contains(model, "titan-embed-image") || strings.HasPrefix(model, "cohere.embed") {
			return []string{"text", "image"}
		}
		return []string{"text"}
	}
	return []string{"text"}
}

// bedrockCapabilitiesOutput infers output modalities from a model ID
// (image, embedding, or text).
func bedrockCapabilitiesOutput(model string) []string {
	model = strings.ToLower(model)
	if isBedrockImageModel(model) {
		return []string{"image"}
	}
	if strings.Contains(model, "embed") {
		return []string{"embedding"}
	}
	return []string{"text"}
}

// bedrockLikelySupportsTools heuristically reports tool support from the model
// ID (Claude 3/4 and Nova families).
func bedrockLikelySupportsTools(model string) bool {
	model = strings.ToLower(model)
	return strings.Contains(model, "claude-3") || strings.Contains(model, "claude-4") || strings.Contains(model, "nova")
}

// ValidateConnection verifies AWS Bedrock access by listing models.
func (d *BedrockDriver) ValidateConnection(ctx context.Context) error {
	_, err := d.ListModels(ctx, nil)
	return err
}

// GenerateEmbeddings routes Bedrock embedding requests to the supported InvokeModel schema.
func (d *BedrockDriver) GenerateEmbeddings(ctx context.Context, options EmbeddingsOptions) (*EmbeddingsResult, error) {
	if d.runtime == nil {
		return nil, errors.New("bedrock runtime client is nil")
	}
	normalized, err := normalizeEmbeddingsOptions(options)
	if err != nil {
		return nil, err
	}
	model := options.Model
	if model == "" {
		model = defaultBedrockEmbeddingModel
	}
	normalized.Model = model
	// Bedrock embedding models use incompatible InvokeModel schemas, so route
	// by family rather than trying to build a single generic request shape.
	switch {
	case strings.Contains(model, "twelvelabs.marengo"):
		return d.generateBedrockMarengoEmbeddings(ctx, normalized, model)
	case strings.HasPrefix(model, "cohere.embed"):
		return d.generateBedrockCohereEmbeddings(ctx, normalized, model)
	case strings.Contains(model, "titan-embed"):
		return d.generateBedrockTitanEmbeddings(ctx, normalized, model)
	default:
		return d.generateBedrockNovaEmbeddings(ctx, normalized, model)
	}
}

// invokeBedrockEmbeddingJSON marshals a family-specific payload, calls
// InvokeModel, and unmarshals the JSON response into out.
func (d *BedrockDriver) invokeBedrockEmbeddingJSON(ctx context.Context, model string, payload any, out any, timeout HTTPTimeoutOptions) error {
	// All supported Bedrock embedding models are synchronous InvokeModel JSON
	// calls. The payload and response schema vary by model family.
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	output, err := d.runtime.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(model),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		Body:        body,
	}, d.runtimeOptions(timeout)...)
	if err != nil {
		return d.formatError(err, model, "embeddings")
	}
	if output == nil || len(output.Body) == 0 {
		return fmt.Errorf("invokeModel returned empty response body for model: %s", model)
	}
	return json.Unmarshal(output.Body, out)
}

// generateBedrockNovaEmbeddings issues one SINGLE_EMBEDDING InvokeModel call per
// input for the Nova v2 multimodal model (text/image/audio/video).
func (d *BedrockDriver) generateBedrockNovaEmbeddings(ctx context.Context, options EmbeddingsOptions, model string) (*EmbeddingsResult, error) {
	// Nova v2 multimodal embeddings accept exactly one input per synchronous
	// SINGLE_EMBEDDING request and support text, image, audio, and video. The
	// synchronous InvokeModel path is limited by Nova's documented media caps
	// (for example short audio/video and 25 MB inline media); longer media would
	// require StartAsyncInvoke, which does not fit this await-style interface.
	items := make([]EmbeddingResultItem, 0, len(options.Inputs))
	for _, input := range options.Inputs {
		params := map[string]any{
			"embeddingPurpose": bedrockNovaEmbeddingPurpose(input.TaskType),
		}
		if options.Dimensions > 0 {
			params["embeddingDimension"] = options.Dimensions
		}
		modal, err := bedrockNovaEmbeddingModalParams(ctx, input)
		if err != nil {
			return nil, err
		}
		for key, value := range modal {
			params[key] = value
		}
		payload := map[string]any{
			"taskType":              "SINGLE_EMBEDDING",
			"singleEmbeddingParams": params,
		}
		var response struct {
			Embeddings []struct {
				Embedding []float64 `json:"embedding"`
			} `json:"embeddings"`
		}
		if err := d.invokeBedrockEmbeddingJSON(ctx, model, payload, &response, options.HTTPTimeout); err != nil {
			return nil, err
		}
		if len(response.Embeddings) == 0 || len(response.Embeddings[0].Embedding) == 0 {
			return nil, fmt.Errorf("nova embeddings response missing 'embeddings[0].embedding' for input type '%s'", input.Type)
		}
		items = append(items, EmbeddingResultItem{
			Outputs: []EmbeddingOutput{{Values: response.Embeddings[0].Embedding, Modality: input.Type}},
		})
	}
	return &EmbeddingsResult{Model: model, Results: items}, nil
}

// bedrockNovaEmbeddingPurpose maps the task type to Nova's embeddingPurpose.
func bedrockNovaEmbeddingPurpose(taskType EmbeddingTaskType) string {
	// TS maps query text to retrieval and all other purposes to generic index.
	if taskType == EmbeddingTaskQuery {
		return "GENERIC_RETRIEVAL"
	}
	return "GENERIC_INDEX"
}

// bedrockNovaEmbeddingModalParams builds the per-modality parameter object for a
// Nova SINGLE_EMBEDDING request, validating the media format and resolving the
// source. Video uses AUDIO_VIDEO_COMBINED embedding mode.
func bedrockNovaEmbeddingModalParams(ctx context.Context, input EmbeddingInput) (map[string]any, error) {
	switch input.Type {
	case embeddingInputText:
		return map[string]any{"text": map[string]any{"truncationMode": "END", "value": input.Text}}, nil
	case embeddingInputImage:
		format, err := bedrockEmbeddingImageFormat(input.Source.MIMEType())
		if err != nil {
			return nil, err
		}
		source, err := bedrockNovaSource(ctx, input.Source)
		if err != nil {
			return nil, err
		}
		return map[string]any{"image": map[string]any{"format": format, "source": source}}, nil
	case embeddingInputAudio:
		format, err := bedrockEmbeddingAudioFormat(input.Source.MIMEType())
		if err != nil {
			return nil, err
		}
		source, err := bedrockNovaSource(ctx, input.Source)
		if err != nil {
			return nil, err
		}
		return map[string]any{"audio": map[string]any{"format": format, "source": source}}, nil
	case embeddingInputVideo:
		format, err := bedrockEmbeddingVideoFormat(input.Source.MIMEType())
		if err != nil {
			return nil, err
		}
		source, err := bedrockNovaSource(ctx, input.Source)
		if err != nil {
			return nil, err
		}
		return map[string]any{"video": map[string]any{"format": format, "source": source, "embeddingMode": "AUDIO_VIDEO_COMBINED"}}, nil
	default:
		return nil, fmt.Errorf("nova embeddings do not support '%s' input", input.Type)
	}
}

// bedrockNovaSource builds a Nova media source, preferring an S3 location and
// falling back to inline base64 bytes.
func bedrockNovaSource(ctx context.Context, source DataSource) (map[string]any, error) {
	// Nova can consume S3 locations directly; inline base64 is the fallback.
	if uri, err := source.URL(ctx); err == nil {
		if s3URI := bedrockS3URI(uri); s3URI != "" {
			return map[string]any{"s3Location": map[string]any{"uri": s3URI}}, nil
		}
	}
	encoded, err := dataSourceToBase64(ctx, source)
	if err != nil {
		return nil, err
	}
	return map[string]any{"bytes": encoded}, nil
}

// generateBedrockTitanEmbeddings issues one InvokeModel call per input for Titan
// embeddings, selecting the text-only or text/image schema by model ID and
// accumulating Titan's reported input-token usage.
func (d *BedrockDriver) generateBedrockTitanEmbeddings(ctx context.Context, options EmbeddingsOptions, model string) (*EmbeddingsResult, error) {
	// Titan has separate schemas for text-only and text/image embedding models.
	imageModel := strings.Contains(model, "titan-embed-image")
	items := make([]EmbeddingResultItem, 0, len(options.Inputs))
	totalTokens := 0
	for _, input := range options.Inputs {
		if input.Type == embeddingInputAudio || input.Type == embeddingInputVideo {
			return nil, fmt.Errorf("titan embeddings do not support '%s' input", input.Type)
		}
		payload := map[string]any{}
		if imageModel {
			switch input.Type {
			case embeddingInputText:
				payload["inputText"] = input.Text
			case embeddingInputImage:
				encoded, err := dataSourceToBase64(ctx, input.Source)
				if err != nil {
					return nil, err
				}
				payload["inputImage"] = encoded
			default:
				return nil, fmt.Errorf("titan embeddings do not support '%s' input", input.Type)
			}
			if options.Dimensions > 0 {
				payload["embeddingConfig"] = map[string]any{"outputEmbeddingLength": options.Dimensions}
			}
		} else {
			if input.Type != embeddingInputText {
				return nil, fmt.Errorf("titan text embeddings model '%s' only supports text input", model)
			}
			payload["inputText"] = input.Text
			if options.Dimensions > 0 {
				payload["dimensions"] = options.Dimensions
			}
		}
		var response struct {
			Embedding           []float64 `json:"embedding"`
			InputTextTokenCount int       `json:"inputTextTokenCount"`
			Message             string    `json:"message"`
		}
		if err := d.invokeBedrockEmbeddingJSON(ctx, model, payload, &response, options.HTTPTimeout); err != nil {
			return nil, err
		}
		if response.Message != "" {
			return nil, fmt.Errorf("titan image embedding error: %s", response.Message)
		}
		if len(response.Embedding) == 0 {
			if imageModel {
				return nil, errors.New("titan image embedding response missing 'embedding'")
			}
			return nil, fmt.Errorf("titan text embedding response missing 'embedding' (model %s)", model)
		}
		tokens := response.InputTextTokenCount
		totalTokens += tokens
		items = append(items, EmbeddingResultItem{
			Outputs:     []EmbeddingOutput{{Values: response.Embedding, Modality: input.Type}},
			InputTokens: tokens,
		})
	}
	var usage *EmbeddingsTokenUsage
	if totalTokens > 0 {
		usage = &EmbeddingsTokenUsage{InputTokens: totalTokens, InputTextTokens: totalTokens}
	}
	return &EmbeddingsResult{Model: model, Results: items, Usage: usage}, nil
}

// generateBedrockCohereEmbeddings batches Cohere text inputs by input_type into
// shared requests and sends image inputs individually, then reassembles results
// in the original input order.
func (d *BedrockDriver) generateBedrockCohereEmbeddings(ctx context.Context, options EmbeddingsOptions, model string) (*EmbeddingsResult, error) {
	// Cohere batches text inputs by effective input_type so per-input task_type
	// overrides are preserved. Image inputs must be sent one at a time as data URIs.
	type inputEntry struct {
		index int
		input EmbeddingInput
	}
	textGroups := map[string][]inputEntry{}
	var imageInputs []inputEntry
	for i, input := range options.Inputs {
		switch input.Type {
		case embeddingInputText:
			key := bedrockCohereInputType(input.TaskType)
			textGroups[key] = append(textGroups[key], inputEntry{index: i, input: input})
		case embeddingInputImage:
			imageInputs = append(imageInputs, inputEntry{index: i, input: input})
		default:
			return nil, fmt.Errorf("cohere embeddings do not support '%s' input", input.Type)
		}
	}
	items := make([]EmbeddingResultItem, len(options.Inputs))
	for inputType, group := range textGroups {
		texts := make([]string, 0, len(group))
		for _, entry := range group {
			texts = append(texts, entry.input.Text)
		}
		payload := map[string]any{"texts": texts}
		if inputType != "" {
			payload["input_type"] = inputType
		}
		var response struct {
			Embeddings [][]float64 `json:"embeddings"`
		}
		if err := d.invokeBedrockEmbeddingJSON(ctx, model, payload, &response, options.HTTPTimeout); err != nil {
			return nil, err
		}
		if len(response.Embeddings) != len(group) {
			return nil, fmt.Errorf("cohere returned %d embeddings for %d texts (model %s)", len(response.Embeddings), len(group), model)
		}
		for i, entry := range group {
			items[entry.index] = EmbeddingResultItem{Outputs: []EmbeddingOutput{{Values: response.Embeddings[i], Modality: embeddingInputText}}}
		}
	}
	for _, entry := range imageInputs {
		encoded, err := dataSourceToBase64(ctx, entry.input.Source)
		if err != nil {
			return nil, err
		}
		payload := map[string]any{
			"images":     []string{"data:" + entry.input.Source.MIMEType() + ";base64," + encoded},
			"input_type": "image",
		}
		var response struct {
			Embeddings [][]float64 `json:"embeddings"`
		}
		if err := d.invokeBedrockEmbeddingJSON(ctx, model, payload, &response, options.HTTPTimeout); err != nil {
			return nil, err
		}
		if len(response.Embeddings) == 0 {
			return nil, fmt.Errorf("cohere returned no embedding for image input (model %s)", model)
		}
		items[entry.index] = EmbeddingResultItem{Outputs: []EmbeddingOutput{{Values: response.Embeddings[0], Modality: embeddingInputImage}}}
	}
	return &EmbeddingsResult{Model: model, Results: items}, nil
}

// bedrockCohereInputType maps a task type to Cohere's input_type
// (search_query/search_document), or "" when unspecified.
func bedrockCohereInputType(taskType EmbeddingTaskType) string {
	switch taskType {
	case EmbeddingTaskQuery:
		return "search_query"
	case EmbeddingTaskDocument:
		return "search_document"
	default:
		return ""
	}
}

// bedrockMarengoSegment is one embedding segment from a TwelveLabs Marengo
// response; audio/video responses carry start/end times and an embedding option.
type bedrockMarengoSegment struct {
	Embedding       []float64 `json:"embedding"`
	StartSec        *float64  `json:"startSec"`
	EndSec          *float64  `json:"endSec"`
	EmbeddingOption string    `json:"embeddingOption"`
}

// generateBedrockMarengoEmbeddings issues one InvokeModel call per input for
// TwelveLabs Marengo, flattening each response's segment embeddings into outputs.
func (d *BedrockDriver) generateBedrockMarengoEmbeddings(ctx context.Context, options EmbeddingsOptions, model string) (*EmbeddingsResult, error) {
	// Marengo may return either one embedding object or an envelope of segment
	// embeddings; normalize both into EmbeddingOutput values.
	items := make([]EmbeddingResultItem, 0, len(options.Inputs))
	for _, input := range options.Inputs {
		payload, err := bedrockMarengoRequest(ctx, input)
		if err != nil {
			return nil, err
		}
		var raw json.RawMessage
		if err := d.invokeBedrockEmbeddingJSON(ctx, model, payload, &raw, options.HTTPTimeout); err != nil {
			return nil, err
		}
		segments, err := bedrockParseMarengoSegments(raw)
		if err != nil {
			return nil, err
		}
		outputs := make([]EmbeddingOutput, 0, len(segments))
		for _, segment := range segments {
			if len(segment.Embedding) == 0 {
				continue
			}
			outputs = append(outputs, EmbeddingOutput{
				Values:          segment.Embedding,
				Modality:        input.Type,
				StartSec:        segment.StartSec,
				EndSec:          segment.EndSec,
				EmbeddingOption: segment.EmbeddingOption,
			})
		}
		if len(outputs) == 0 {
			return nil, fmt.Errorf("marengo response did not contain embedding values for input type '%s'", input.Type)
		}
		items = append(items, EmbeddingResultItem{Outputs: outputs})
	}
	return &EmbeddingsResult{Model: model, Results: items}, nil
}

// bedrockMarengoRequest builds a Marengo request body for one input, attaching
// segment controls (startSec/lengthSec and video clip options) for audio/video.
func bedrockMarengoRequest(ctx context.Context, input EmbeddingInput) (map[string]any, error) {
	// Marengo uses mediaSource for non-text inputs and reserves the segment
	// controls for audio/video requests. It accepts only one embeddingOption per
	// request; callers submit separate inputs when they need multiple views.
	switch input.Type {
	case embeddingInputText:
		return map[string]any{"inputType": "text", "inputText": input.Text}, nil
	case embeddingInputImage:
		media, err := bedrockMediaSource(ctx, input.Source)
		if err != nil {
			return nil, err
		}
		return map[string]any{"inputType": "image", "mediaSource": media}, nil
	case embeddingInputVideo, embeddingInputAudio:
		media, err := bedrockMediaSource(ctx, input.Source)
		if err != nil {
			return nil, err
		}
		payload := map[string]any{"inputType": input.Type, "mediaSource": media}
		if input.StartSec != nil {
			payload["startSec"] = *input.StartSec
		}
		if input.LengthSec != nil {
			payload["lengthSec"] = *input.LengthSec
		}
		if input.Type == embeddingInputVideo {
			if input.UseFixedLength != nil {
				payload["useFixedLengthSec"] = *input.UseFixedLength
			}
			if input.MinClipSec != nil {
				payload["minClipSec"] = *input.MinClipSec
			}
			if len(input.EmbeddingOption) > 1 {
				return nil, fmt.Errorf("marengo accepts only one embedding_option per request; received [%s]. submit separate inputs, one per option", strings.Join(input.EmbeddingOption, ", "))
			}
			if len(input.EmbeddingOption) == 1 {
				payload["embeddingOption"] = input.EmbeddingOption[0]
			}
		}
		return payload, nil
	default:
		return nil, fmt.Errorf("marengo embeddings do not support '%s' input", input.Type)
	}
}

// bedrockMediaSource builds a Marengo media source, preferring an S3 location
// and falling back to an inline base64 string.
func bedrockMediaSource(ctx context.Context, source DataSource) (map[string]any, error) {
	// TwelveLabs-on-Bedrock accepts S3 locations or inline base64 strings.
	if uri, err := source.URL(ctx); err == nil {
		if s3URI := bedrockS3URI(uri); s3URI != "" {
			return map[string]any{"s3Location": map[string]any{"uri": s3URI}}, nil
		}
	}
	encoded, err := dataSourceToBase64(ctx, source)
	if err != nil {
		return nil, err
	}
	return map[string]any{"base64String": encoded}, nil
}

// bedrockParseMarengoSegments parses a Marengo response, accepting both an
// {embeddings:[...]} envelope and a single bare segment object.
func bedrockParseMarengoSegments(raw json.RawMessage) ([]bedrockMarengoSegment, error) {
	// The Bedrock wrapper has returned both bare segment objects and
	// {embeddings:[...]} envelopes; accept both for compatibility.
	var envelope struct {
		Embeddings []bedrockMarengoSegment `json:"embeddings"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, err
	}
	if len(envelope.Embeddings) > 0 {
		return envelope.Embeddings, nil
	}
	var segment bedrockMarengoSegment
	if err := json.Unmarshal(raw, &segment); err != nil {
		return nil, err
	}
	return []bedrockMarengoSegment{segment}, nil
}

// bedrockEmbeddingImageFormat maps an image MIME type to the format string the
// embedding APIs expect, erroring on unsupported types.
func bedrockEmbeddingImageFormat(mimeType string) (string, error) {
	switch strings.ToLower(mimeType) {
	case "image/jpeg", "image/jpg":
		return "jpeg", nil
	case "image/png":
		return "png", nil
	case "image/gif":
		return "gif", nil
	case "image/webp":
		return "webp", nil
	default:
		return "", fmt.Errorf("unsupported image MIME type for Bedrock: '%s'", mimeType)
	}
}

// bedrockEmbeddingAudioFormat maps an audio MIME type to the embedding format
// string, erroring on unsupported types.
func bedrockEmbeddingAudioFormat(mimeType string) (string, error) {
	switch strings.ToLower(mimeType) {
	case "audio/mpeg", "audio/mp3":
		return "mp3", nil
	case "audio/wav", "audio/wave", "audio/x-wav":
		return "wav", nil
	case "audio/ogg":
		return "ogg", nil
	default:
		return "", fmt.Errorf("unsupported audio MIME type for Bedrock: '%s'", mimeType)
	}
}

// bedrockEmbeddingVideoFormat maps a video MIME type to the embedding format
// string, erroring on unsupported types.
func bedrockEmbeddingVideoFormat(mimeType string) (string, error) {
	switch strings.ToLower(mimeType) {
	case "video/mp4":
		return "mp4", nil
	case "video/quicktime":
		return "mov", nil
	case "video/x-matroska":
		return "mkv", nil
	case "video/webm":
		return "webm", nil
	case "video/x-flv":
		return "flv", nil
	case "video/mpeg":
		return "mpeg", nil
	case "video/mpg":
		return "mpg", nil
	case "video/x-ms-wmv":
		return "wmv", nil
	case "video/3gpp":
		return "3gp", nil
	default:
		return "", fmt.Errorf("unsupported video MIME type for Bedrock: '%s'", mimeType)
	}
}

// requestBedrockImageGeneration invokes an image model, decodes the returned
// images into data-URI results, and surfaces any model-reported error.
func (d *BedrockDriver) requestBedrockImageGeneration(ctx context.Context, prompt bedrockImagePrompt, options ExecutionOptions) (*Completion, error) {
	payload := bedrockImagePayload(prompt, options)
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	output, err := d.runtime.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(options.Model),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		Body:        body,
	}, d.runtimeOptions(options.HTTPTimeout)...)
	if err != nil {
		return nil, d.formatError(err, options.Model, "execute")
	}
	var response struct {
		Images []string `json:"images"`
		Error  any      `json:"error"`
	}
	if err := json.Unmarshal(output.Body, &response); err != nil {
		return nil, err
	}
	results := make([]CompletionResult, 0, len(response.Images))
	for _, image := range response.Images {
		value := image
		if !strings.HasPrefix(value, "data:") {
			value = "data:image/png;base64," + value
		}
		results = append(results, CompletionResult{Type: ResultTypeImage, Value: value})
	}
	completion := &Completion{Result: results}
	if response.Error != nil {
		completion.Error = &ResultValidationError{Code: "bedrock_image_error", Message: toolInputString(response.Error), Data: results}
	}
	if options.IncludeOriginalResponse {
		completion.OriginalResponse = response
	}
	return completion, nil
}

// bedrockImagePayload builds the InvokeModel body for an image model, choosing
// the Stable Diffusion or Nova/Titan schema by model ID.
func bedrockImagePayload(prompt bedrockImagePrompt, options ExecutionOptions) map[string]any {
	model := strings.ToLower(options.Model)
	if strings.Contains(model, "stable-diffusion") || strings.Contains(model, "stability.") {
		return bedrockStableDiffusionPayload(prompt, options)
	}
	return bedrockNovaTitanImagePayload(prompt, options)
}

// bedrockNovaTitanImagePayload builds the Nova/Titan image payload, selecting
// the task-specific params block (text-to-image, variation, color-guided,
// background removal, inpainting, or outpainting) from the requested taskType.
func bedrockNovaTitanImagePayload(prompt bedrockImagePrompt, options ExecutionOptions) map[string]any {
	taskType := optionString(options.ModelOptions, "taskType", "task_type")
	if taskType == "" {
		taskType = "TEXT_IMAGE"
	}
	config := removeNilMapValues(map[string]any{
		"quality":        optionString(options.ModelOptions, "quality"),
		"width":          optionIntValue(options.ModelOptions, "width"),
		"height":         optionIntValue(options.ModelOptions, "height"),
		"numberOfImages": optionIntValue(options.ModelOptions, "numberOfImages"),
		"seed":           optionIntValue(options.ModelOptions, "seed"),
		"cfgScale":       optionFloatValue(options.ModelOptions, "cfgScale"),
	})
	text := bedrockImageText(prompt)
	payload := map[string]any{
		"taskType":              taskType,
		"imageGenerationConfig": config,
	}
	switch taskType {
	case "IMAGE_VARIATION":
		payload["imageVariationParams"] = removeNilMapValues(map[string]any{
			"images":             prompt.Images,
			"text":               text,
			"negativeText":       prompt.Negative,
			"similarityStrength": optionFloatValue(options.ModelOptions, "similarityStrength"),
		})
	case "COLOR_GUIDED_GENERATION":
		payload["colorGuidedGenerationParams"] = removeNilMapValues(map[string]any{
			"colors":         optionStringSlice(options.ModelOptions, "colors"),
			"text":           text,
			"referenceImage": firstString(prompt.Images),
			"negativeText":   prompt.Negative,
		})
	case "BACKGROUND_REMOVAL":
		payload["backgroundRemovalParams"] = map[string]any{"image": firstString(prompt.Images)}
	case "INPAINTING":
		payload["inPaintingParams"] = removeNilMapValues(map[string]any{
			"image":        firstString(prompt.Images),
			"maskImage":    firstString(prompt.Masks),
			"text":         text,
			"negativeText": prompt.Negative,
		})
	case "OUTPAINTING":
		payload["outPaintingParams"] = removeNilMapValues(map[string]any{
			"image":           firstString(prompt.Images),
			"maskImage":       firstString(prompt.Masks),
			"text":            text,
			"negativeText":    prompt.Negative,
			"outPaintingMode": optionStringDefault(options.ModelOptions, "outPaintingMode", "DEFAULT"),
		})
	default:
		payload["textToImageParams"] = removeNilMapValues(map[string]any{
			"text":            text,
			"conditionImage":  firstString(prompt.Images),
			"controlMode":     optionString(options.ModelOptions, "controlMode"),
			"controlStrength": optionFloatValue(options.ModelOptions, "controlStrength"),
			"negativeText":    prompt.Negative,
		})
	}
	return payload
}

// bedrockStableDiffusionPayload builds the Stable Diffusion image payload,
// expressing the negative prompt as a weighted negative text prompt.
func bedrockStableDiffusionPayload(prompt bedrockImagePrompt, options ExecutionOptions) map[string]any {
	text := bedrockImageText(prompt)
	prompts := []map[string]any{{"text": text, "weight": 1}}
	if prompt.Negative != "" {
		prompts = append(prompts, map[string]any{"text": prompt.Negative, "weight": -1})
	}
	return removeNilMapValues(map[string]any{
		"text_prompts": prompts,
		"cfg_scale":    optionFloatValue(options.ModelOptions, "cfgScale"),
		"seed":         optionIntValue(options.ModelOptions, "seed"),
		"steps":        optionIntValue(options.ModelOptions, "steps"),
		"width":        optionIntValue(options.ModelOptions, "width"),
		"height":       optionIntValue(options.ModelOptions, "height"),
	})
}

func firstString(values []string) any {
	if len(values) == 0 {
		return nil
	}
	return values[0]
}

// bedrockImageText combines the user prompt text with any system text, appending
// the system text as an "IMPORTANT:" instruction since image models take a
// single prompt string.
func bedrockImageText(prompt bedrockImagePrompt) string {
	text := strings.TrimSpace(prompt.Text)
	if prompt.System != "" {
		if text != "" {
			text += "\n\n\nIMPORTANT: "
		}
		text += prompt.System
	}
	return strings.TrimSpace(text)
}

func optionStringDefault(options map[string]any, key string, fallback string) string {
	if value := optionString(options, key); value != "" {
		return value
	}
	return fallback
}

func optionIntValue(options map[string]any, key string) any {
	if v := optionInt(options, key); v != nil {
		return *v
	}
	return nil
}

func optionFloatValue(options map[string]any, key string) any {
	if v := optionFloat(options, key); v != nil {
		return *v
	}
	return nil
}

// isBedrockImageModel reports whether a model ID is a Bedrock image-generation
// model (Titan Image, Stable Diffusion, or Nova Canvas).
func isBedrockImageModel(model string) bool {
	model = strings.ToLower(model)
	return strings.Contains(model, "titan-image") || strings.Contains(model, "stable-diffusion") || strings.Contains(model, "nova-canvas")
}

// formatError wraps an AWS API error as a LlumiverseError with provider/model/
// operation context and a computed retryable flag; non-API errors pass through.
func (d *BedrockDriver) formatError(err error, model string, operation string) error {
	var apiErr interface {
		ErrorCode() string
		ErrorMessage() string
		ErrorFault() string
	}
	if errors.As(err, &apiErr) {
		return newLlumiverseError(apiErr.ErrorMessage(), bedrockRetryable(apiErr.ErrorCode(), 0, apiErr.ErrorFault()), LlumiverseErrorContext{
			Provider:  ProviderBedrock,
			Model:     model,
			Operation: operation,
		}, err, 0, apiErr.ErrorCode())
	}
	return err
}

// bedrockRetryable decides whether a Bedrock error is retryable from its
// exception name, falling back to the fault type (server/client) and then the
// HTTP status and message.
func bedrockRetryable(name string, status int, fault string) *bool {
	switch name {
	case "ThrottlingException", "ServiceUnavailableException", "InternalServerException", "ServiceQuotaExceededException":
		return boolPtr(true)
	case "ValidationException", "AccessDeniedException", "ResourceNotFoundException", "ConflictException", "ResourceInUseException":
		return boolPtr(false)
	}
	if fault == "server" {
		return boolPtr(true)
	}
	if fault == "client" {
		return boolPtr(false)
	}
	return retryableFromStatusAndMessage(status, name)
}
