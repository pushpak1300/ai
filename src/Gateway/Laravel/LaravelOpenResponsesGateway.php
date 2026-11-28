<?php

namespace Laravel\Ai\Gateway\Laravel;

use Illuminate\Support\Collection;
use InvalidArgumentException;
use Laravel\Ai\Contracts\HasProviderOptions;
use Laravel\Ai\Enums\Lab;
use Laravel\Ai\Files\ProviderDocument;
use Laravel\Ai\Files\ProviderImage;
use Laravel\Ai\Files\RemoteDocument;
use Laravel\Ai\Gateway\OpenResponses\OpenResponsesGateway;
use Laravel\Ai\Gateway\TextGenerationOptions;
use Laravel\Ai\Messages\AssistantMessage;
use Laravel\Ai\Messages\Message;
use Laravel\Ai\Providers\Provider;

class LaravelOpenResponsesGateway extends OpenResponsesGateway
{
    use Concerns\HandlesLaravelErrors;

    /**
     * Get the base URL for the gateway's Open Responses endpoint.
     */
    protected function baseUrl(Provider $provider): string
    {
        return LaravelGateway::rootUrl($provider).'/v1/openai/v1';
    }

    /**
     * Determine if the gateway should receive full stateless conversation history.
     *
     * The gateway does not store responses, so it always receives the full history and never a previous response ID.
     */
    protected function isStateless(Provider $provider): bool
    {
        return true;
    }

    /**
     * Resolve the Open Responses options, overridden by the options set for the Laravel provider itself.
     *
     * @return array<string, mixed>|null
     */
    protected function resolveProviderOptions(?TextGenerationOptions $options, Provider $provider): ?array
    {
        if (! $options instanceof TextGenerationOptions) {
            return null;
        }

        return [
            ...($options->providerOptions(Lab::OpenResponses) ?? []),
            ...($options->providerOptions(Lab::Laravel) ?? []),
        ];
    }

    /**
     * Map the given attachments, rejecting provider files and inlining remote documents.
     */
    protected function mapAttachments(Collection $attachments, Provider $provider): array
    {
        $providerKey = Lab::tryFrom($provider->driver()) ?? $provider->driver();

        return $attachments->flatMap(function ($attachment) use ($provider, $providerKey): array {
            if ($attachment instanceof ProviderImage || $attachment instanceof ProviderDocument) {
                throw new InvalidArgumentException('The Laravel Cloud AI gateway does not support provider file IDs. Attach the file contents instead.');
            }

            if (! $attachment instanceof RemoteDocument) {
                return parent::mapAttachments(new Collection([$attachment]), $provider);
            }

            // The gateway only accepts inline file data for non-Claude models, so remote documents are fetched here...
            $part = [
                'type' => 'input_file',
                'file_data' => 'data:'.($attachment->mimeType() ?? 'application/octet-stream').';base64,'.base64_encode($attachment->content()),
                'filename' => $attachment->name() ?: $this->fallbackFilename($attachment->mimeType()),
            ];

            return [$attachment instanceof HasProviderOptions
                ? array_merge($attachment->providerOptions($providerKey), $part)
                : $part];
        })->all();
    }

    /**
     * Map an assistant message, replaying only the state an Open Responses model produced.
     */
    protected function mapAssistantMessage(AssistantMessage|Message $message, array &$input, Provider $provider): void
    {
        if ($message instanceof AssistantMessage
            && filled($message->replayBlocks)
            && ! LaravelGateway::hasOnlyOpenResponsesItems($message->replayBlocks)) {
            $message = new AssistantMessage($message->content, $message->toolCalls);
        }

        parent::mapAssistantMessage($message, $input, $provider);
    }
}
