<?php

namespace Laravel\Ai\Gateway\Laravel;

use Generator;
use Illuminate\Http\Client\PendingRequest;
use Illuminate\Support\Collection;
use Illuminate\Support\Str;
use InvalidArgumentException;
use Laravel\Ai\Contracts\Providers\TextProvider;
use Laravel\Ai\Enums\Lab;
use Laravel\Ai\Files\ProviderDocument;
use Laravel\Ai\Files\ProviderImage;
use Laravel\Ai\Gateway\Anthropic\AnthropicGateway;
use Laravel\Ai\Gateway\StepContext;
use Laravel\Ai\Gateway\StepResponse;
use Laravel\Ai\Gateway\TextGenerationOptions;
use Laravel\Ai\Messages\AssistantMessage;
use Laravel\Ai\Messages\Message;
use Laravel\Ai\Providers\Provider;
use Laravel\Ai\Streaming\Events\StreamStart;

class LaravelAnthropicGateway extends AnthropicGateway
{
    use Concerns\HandlesLaravelErrors;

    /**
     * {@inheritdoc}
     */
    public function generateTextStep(
        TextProvider $provider,
        string $model,
        ?string $instructions,
        array $messages,
        array $tools,
        ?array $schema,
        ?TextGenerationOptions $options,
        ?int $timeout,
        StepContext $stepContext,
    ): StepResponse {
        $response = parent::generateTextStep($provider, $this->catalogModel($model), $instructions, $messages, $tools, $schema, $options, $timeout, $stepContext);

        // The gateway answers with the upstream model ID, such as a Bedrock ARN, rather than the catalog ID that was requested...
        $response->meta->model = $model;

        return $response;
    }

    /**
     * {@inheritdoc}
     */
    public function generateStreamStep(
        string $invocationId,
        TextProvider $provider,
        string $model,
        ?string $instructions,
        array $messages,
        array $tools,
        ?array $schema,
        ?TextGenerationOptions $options,
        ?int $timeout,
        StepContext $stepContext,
    ): Generator {
        $stream = parent::generateStreamStep($invocationId, $provider, $this->catalogModel($model), $instructions, $messages, $tools, $schema, $options, $timeout, $stepContext);

        foreach ($stream as $event) {
            if ($event instanceof StreamStart) {
                $event->model = $model;
            }

            yield $event;
        }

        $response = $stream->getReturn();

        if ($response instanceof StepResponse) {
            $response->meta->model = $model;
        }

        return $response;
    }

    /**
     * Get the Anthropic-format model ID the Messages endpoint expects, such as "claude-opus-5-5" for "anthropic/claude-opus-5-5".
     */
    protected function catalogModel(string $model): string
    {
        return Str::after($model, 'anthropic/');
    }

    /**
     * Get an HTTP client for the gateway's Anthropic Messages endpoint.
     */
    protected function client(Provider $provider, ?int $timeout = null): PendingRequest
    {
        $config = $provider->additionalConfiguration();
        $key = $provider->providerCredentials()['key'] ?? null;

        return $this->createClient(
            $this->baseUrl($provider),
            array_filter([
                'Authorization' => filled($key) ? 'Bearer '.$key : null,
                'anthropic-version' => $config['version'] ?? '2023-06-01',
                'anthropic-beta' => $config['anthropic_beta'] ?? null,
            ]),
            $config['headers'] ?? [],
            $timeout ?? 60,
        );
    }

    /**
     * Get the base URL for the gateway's Anthropic Messages endpoint.
     */
    protected function baseUrl(Provider $provider): string
    {
        return LaravelGateway::rootUrl($provider).'/v1/anthropic/v1';
    }

    /**
     * Resolve the Anthropic options, overridden by the options set for the Laravel provider itself.
     *
     * @return array<string, mixed>|null
     */
    protected function resolveProviderOptions(?TextGenerationOptions $options, Provider $provider): ?array
    {
        if (! $options instanceof TextGenerationOptions) {
            return null;
        }

        return [
            ...($options->providerOptions(Lab::Anthropic) ?? []),
            ...($options->providerOptions(Lab::Laravel) ?? []),
        ];
    }

    /**
     * Map the given attachments, rejecting files stored with another provider.
     */
    protected function mapAttachments(Collection $attachments): array
    {
        $attachments->each(function ($attachment): void {
            if ($attachment instanceof ProviderImage || $attachment instanceof ProviderDocument) {
                throw new InvalidArgumentException('The Laravel Cloud AI gateway does not support provider file IDs. Attach the file contents instead.');
            }
        });

        return parent::mapAttachments($attachments);
    }

    /**
     * Map an assistant message, replaying only the state an Anthropic model produced.
     */
    protected function mapAssistantMessage(AssistantMessage|Message $message, array &$mapped): void
    {
        if ($message instanceof AssistantMessage && LaravelGateway::hasOpenResponsesItems($message->replayBlocks)) {
            $message = new AssistantMessage($message->content, $message->toolCalls);
        }

        parent::mapAssistantMessage($message, $mapped);
    }

    /**
     * Get the content of the paused steps, ignoring steps an Open Responses model produced.
     *
     * @param  array<int, Message>  $messages
     * @return array<int, array<string, mixed>>
     */
    protected function pausedContent(array $messages): array
    {
        $content = parent::pausedContent($messages);

        return LaravelGateway::hasOpenResponsesItems($content) ? [] : $content;
    }
}
