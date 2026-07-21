# LM-Kit.NET Amazon Bedrock integration

Use Amazon Bedrock embedding models (Amazon Titan Text Embeddings and Cohere Embed) anywhere
LM-Kit.NET consumes embeddings, including `RagEngine`, via the provider-agnostic
`LMKit.Abstractions.IEmbedder` contract.

## Install

This project references the `AWSSDK.BedrockRuntime` SDK and LM-Kit.NET. Credentials and region are
resolved by the AWS SDK (environment variables, a shared profile, or an IAM role); this library
never handles secrets directly.

## Usage

```csharp
using Amazon;
using LMKit.Integrations.Aws.Embeddings;
using LMKit.Retrieval;

// Region-based constructor uses the default AWS credential provider chain.
var embedder = new BedrockEmbedder(
    modelId: "amazon.titan-embed-text-v2:0",
    region: RegionEndpoint.USEast1,
    dimensions: 1024);

// Plug it straight into RAG. A remote embedder needs a local tokenizer model for chunking
// during import; querying an already-populated store needs no local model.
var rag = new RagEngine(embedder, tokenizerModel);
await rag.ImportTextAsync("...", "docs", "s1");
var matches = await rag.FindMatchingPartitionsAsync("...", topK: 5);
```

Cohere models are batched and query-aware automatically:

```csharp
var cohere = new BedrockEmbedder("cohere.embed-multilingual-v3", RegionEndpoint.EUWest1);
float[][] passages = await cohere.GetEmbeddingsAsync(new[] { "doc a", "doc b" });
float[] query = await cohere.GetQueryEmbeddingsAsync("search text"); // input_type = search_query
```

Inject a pre-configured `IAmazonBedrockRuntime` when you need full control over the client:

```csharp
var embedder = new BedrockEmbedder(myBedrockRuntimeClient, "amazon.titan-embed-text-v2:0");
```

## Notes

- The model family (Titan vs Cohere) is inferred from the model id, or set explicitly via the
  `BedrockEmbedder.ModelFamily` parameter.
- `EmbeddingSize` is known up front when `dimensions` is supplied, otherwise it is learned from the
  first response.
- Strong-name signing is disabled here; wire in the organization key before publishing.
