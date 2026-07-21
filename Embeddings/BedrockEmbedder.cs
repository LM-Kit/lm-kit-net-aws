using System.Text;
using System.Text.Json;
using Amazon;
using Amazon.BedrockRuntime;
using Amazon.BedrockRuntime.Model;
using LMKit.Abstractions;

namespace LMKit.Integrations.Aws.Embeddings
{
    /// <summary>
    /// An <see cref="IEmbedder"/> backed by Amazon Bedrock, so Bedrock-hosted embedding models
    /// (Amazon Titan Text Embeddings and Cohere Embed) can be used anywhere LMKit consumes
    /// embeddings, including <c>RagEngine</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The request and response shape differs per model family; this class detects the family from
    /// the model id (or an explicit override) and formats each call accordingly:
    /// </para>
    /// <list type="bullet">
    /// <item><description>
    /// <b>Amazon Titan</b> (for example <c>amazon.titan-embed-text-v2:0</c>) embeds a single text per
    /// request, so a batch is issued as sequential calls. The optional output dimension and
    /// normalization flag are forwarded when supported.
    /// </description></item>
    /// <item><description>
    /// <b>Cohere</b> (for example <c>cohere.embed-english-v3</c>) embeds a batch in one request and
    /// distinguishes queries from passages via <c>input_type</c>, which this class sets when
    /// <see cref="IQueryEmbedder"/> is used.
    /// </description></item>
    /// </list>
    /// <para>
    /// Credentials and region are resolved by the AWS SDK. Pass an <see cref="IAmazonBedrockRuntime"/>
    /// you have configured, or use the region-based constructor, which relies on the default AWS
    /// credential provider chain (environment variables, shared profile, or an IAM role).
    /// </para>
    /// </remarks>
    public sealed class BedrockEmbedder : EmbedderBase
    {
        /// <summary>
        /// Identifies the Bedrock embedding model family, which determines the request and response
        /// payload shape.
        /// </summary>
        public enum ModelFamily
        {
            /// <summary>Amazon Titan Text Embeddings (single input per request).</summary>
            Titan,

            /// <summary>Cohere Embed (batched input, query/passage aware).</summary>
            Cohere
        }

        private readonly IAmazonBedrockRuntime _client;
        private readonly string _modelId;
        private readonly ModelFamily _family;
        private readonly int? _dimensions;
        private readonly bool _normalize;
        private int _embeddingSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="BedrockEmbedder"/> class with an
        /// already-configured Bedrock runtime client.
        /// </summary>
        /// <param name="client">The Bedrock runtime client to use. Cannot be null.</param>
        /// <param name="modelId">The Bedrock model id, for example <c>amazon.titan-embed-text-v2:0</c>. Cannot be null or empty.</param>
        /// <param name="family">The model family; inferred from <paramref name="modelId"/> when null.</param>
        /// <param name="dimensions">
        /// The output dimension to request (Titan v2 supports 256, 512, or 1024). When null, the
        /// model default is used.
        /// </param>
        /// <param name="normalize">Whether to request normalized vectors (Titan). Cohere always normalizes.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="client"/> or <paramref name="modelId"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the family cannot be inferred from <paramref name="modelId"/>.</exception>
        public BedrockEmbedder(
            IAmazonBedrockRuntime client,
            string modelId,
            ModelFamily? family = null,
            int? dimensions = null,
            bool normalize = true)
        {
            _client = client ?? throw new ArgumentNullException(nameof(client));

            if (string.IsNullOrWhiteSpace(modelId))
            {
                throw new ArgumentNullException(nameof(modelId));
            }

            _modelId = modelId;
            _family = family ?? InferFamily(modelId);
            _dimensions = dimensions;
            _normalize = normalize;
            _embeddingSize = dimensions ?? 0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BedrockEmbedder"/> class, creating a Bedrock
        /// runtime client for the given region using the default AWS credential provider chain.
        /// </summary>
        /// <param name="modelId">The Bedrock model id, for example <c>amazon.titan-embed-text-v2:0</c>. Cannot be null or empty.</param>
        /// <param name="region">The AWS region hosting the model. Cannot be null.</param>
        /// <param name="family">The model family; inferred from <paramref name="modelId"/> when null.</param>
        /// <param name="dimensions">The output dimension to request (Titan v2). When null, the model default is used.</param>
        /// <param name="normalize">Whether to request normalized vectors (Titan).</param>
        public BedrockEmbedder(
            string modelId,
            RegionEndpoint region,
            ModelFamily? family = null,
            int? dimensions = null,
            bool normalize = true)
            : this(
                  new AmazonBedrockRuntimeClient(region ?? throw new ArgumentNullException(nameof(region))),
                  modelId,
                  family,
                  dimensions,
                  normalize)
        {
        }

        /// <inheritdoc/>
        public override string ModelId => _modelId;

        /// <inheritdoc/>
        public override int EmbeddingSize => _embeddingSize;

        /// <inheritdoc/>
        public override async Task<float[][]> GetEmbeddingsAsync(
            IEnumerable<string> texts,
            CancellationToken cancellationToken = default)
        {
            if (texts is null)
            {
                throw new ArgumentNullException(nameof(texts));
            }

            return await EmbedAsync(texts, isQuery: false, cancellationToken).ConfigureAwait(false);
        }

        /// <inheritdoc/>
        public override async Task<float[]> GetQueryEmbeddingsAsync(
            string query,
            CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(query))
            {
                throw new ArgumentNullException(nameof(query));
            }

            float[][] result = await EmbedAsync(new[] { query }, isQuery: true, cancellationToken).ConfigureAwait(false);
            return result[0];
        }

        private async Task<float[][]> EmbedAsync(
            IEnumerable<string> texts,
            bool isQuery,
            CancellationToken cancellationToken)
        {
            var inputs = texts as IList<string> ?? texts.ToList();

            float[][] vectors = _family == ModelFamily.Cohere
                ? await EmbedCohereAsync(inputs, isQuery, cancellationToken).ConfigureAwait(false)
                : await EmbedTitanAsync(inputs, cancellationToken).ConfigureAwait(false);

            if (_embeddingSize == 0 && vectors.Length > 0)
            {
                _embeddingSize = vectors[0].Length;
            }

            return vectors;
        }

        private async Task<float[][]> EmbedTitanAsync(IList<string> inputs, CancellationToken cancellationToken)
        {
            var result = new float[inputs.Count][];

            for (int i = 0; i < inputs.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var payload = new Dictionary<string, object> { ["inputText"] = inputs[i] ?? string.Empty };
                if (_dimensions.HasValue)
                {
                    payload["dimensions"] = _dimensions.Value;
                }
                payload["normalize"] = _normalize;

                using JsonDocument doc = await InvokeAsync(payload, cancellationToken).ConfigureAwait(false);
                result[i] = ReadVector(doc.RootElement.GetProperty("embedding"));
            }

            return result;
        }

        private async Task<float[][]> EmbedCohereAsync(IList<string> inputs, bool isQuery, CancellationToken cancellationToken)
        {
            var payload = new Dictionary<string, object>
            {
                ["texts"] = inputs,
                ["input_type"] = isQuery ? "search_query" : "search_document",
                ["truncate"] = "END"
            };

            using JsonDocument doc = await InvokeAsync(payload, cancellationToken).ConfigureAwait(false);
            JsonElement embeddings = doc.RootElement.GetProperty("embeddings");

            var result = new float[embeddings.GetArrayLength()][];
            int index = 0;
            foreach (JsonElement vector in embeddings.EnumerateArray())
            {
                result[index++] = ReadVector(vector);
            }

            return result;
        }

        private async Task<JsonDocument> InvokeAsync(Dictionary<string, object> payload, CancellationToken cancellationToken)
        {
            byte[] body = JsonSerializer.SerializeToUtf8Bytes(payload);

            var request = new InvokeModelRequest
            {
                ModelId = _modelId,
                ContentType = "application/json",
                Accept = "application/json",
                Body = new MemoryStream(body)
            };

            InvokeModelResponse response = await _client.InvokeModelAsync(request, cancellationToken).ConfigureAwait(false);

            using var reader = new StreamReader(response.Body, Encoding.UTF8);
            string json = await reader.ReadToEndAsync().ConfigureAwait(false);
            return JsonDocument.Parse(json);
        }

        private static float[] ReadVector(JsonElement array)
        {
            var vector = new float[array.GetArrayLength()];
            int i = 0;
            foreach (JsonElement value in array.EnumerateArray())
            {
                vector[i++] = value.GetSingle();
            }
            return vector;
        }

        private static ModelFamily InferFamily(string modelId)
        {
            string id = modelId.ToLowerInvariant();

            if (id.Contains("cohere"))
            {
                return ModelFamily.Cohere;
            }

            if (id.Contains("titan-embed") || id.Contains("titan-e1t"))
            {
                return ModelFamily.Titan;
            }

            throw new ArgumentException(
                $"Unable to infer the Bedrock embedding model family from model id '{modelId}'. " +
                "Pass an explicit BedrockEmbedder.ModelFamily.",
                nameof(modelId));
        }
    }
}
