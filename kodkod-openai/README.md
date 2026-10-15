# kodkod-openai

OpenAI-compatible model adapters for Kodkod, including stateless Responses replay.

### Native context compaction

`OpenAiResponsesProvider::with_native_compaction(true)` enables the documented
standalone `/responses/compact` operation used by the core's persistent context
loop. It is opt-in for arbitrary compatible gateways. The fixed public ChatGPT
provider opts in; deployment/model access remains subject to the actual service.

The adapter preserves the **entire** returned input window, including retained
messages and encrypted checkpoints, and replays it before subsequent messages.
The portable active conversation starts empty, retaining its instructions. Native
state is validated against its model, request profile and endpoint; hosts must
also bind checkpoints to the relevant account. Do not transfer opaque checkpoints
between unrelated models or gateways. Debug output never prints their contents.

Native input and provider continuation are saved together by the core's checkpoint
API. Merely storing the visible `Conversation` or resuming with a fresh continuation
would lose the compacted history. This implementation uses explicit standalone
compaction; inline and speculative/background compaction are separate future
optimizations, not assumed capabilities.
