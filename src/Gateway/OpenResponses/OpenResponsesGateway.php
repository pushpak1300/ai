<?php

namespace Laravel\Ai\Gateway\OpenResponses;

use Illuminate\Contracts\Events\Dispatcher;
use Illuminate\Support\Collection;
use InvalidArgumentException;
use Laravel\Ai\Contracts\Gateway\StepTextGateway;
use Laravel\Ai\Enums\Lab;
use Laravel\Ai\Exceptions\AiException;
use Laravel\Ai\Files\ProviderDocument;
use Laravel\Ai\Files\ProviderImage;
use Laravel\Ai\Gateway\Concerns\HandlesFailoverErrors;
use Laravel\Ai\Gateway\Concerns\ParsesServerSentEvents;
use Laravel\Ai\Gateway\OpenAi\Concerns as OpenAi;
use Laravel\Ai\Gateway\TextGenerationOptions;
use Laravel\Ai\Providers\Provider;

class OpenResponsesGateway implements StepTextGateway
{
    use Concerns\CreatesOpenResponsesClient;
    use HandlesFailoverErrors;
    use OpenAi\BuildsTextRequests;
    use OpenAi\HandlesTextGeneration;
    use OpenAi\HandlesTextSteps;
    use OpenAi\MapsAttachments {
        mapAttachments as protected mapOpenAiAttachments;
    }
    use OpenAi\MapsMessages {
        mapMessagesToInput as protected mapOpenAiMessagesToInput;
    }
    use OpenAi\MapsTools;
    use OpenAi\ParsesTextResponses;
    use ParsesServerSentEvents;

    public function __construct(protected Dispatcher $events)
    {
        //
    }

    /**
     * Determine if the server should receive full stateless conversation history.
     *
     * Open Responses servers are not required to store responses, so full history is sent unless "store" is enabled.
     */
    protected function isStateless(Provider $provider): bool
    {
        return filter_var(
            $provider->additionalConfiguration()['store'] ?? false,
            FILTER_VALIDATE_BOOL,
            FILTER_NULL_ON_FAILURE,
        ) !== true;
    }

    /**
     * Resolve the explicit store flag to include on the request body.
     */
    protected function storeRequestValue(Provider $provider): ?bool
    {
        return ! $this->isStateless($provider);
    }

    /**
     * Determine if the model supports encrypted reasoning content.
     */
    protected function isReasoningModel(string $model): bool
    {
        return false;
    }

    /**
     * Resolve the provider options the agent and prompt set for this provider.
     *
     * @return array<string, mixed>|null
     */
    protected function resolveProviderOptions(?TextGenerationOptions $options, Provider $provider): ?array
    {
        return $options?->providerOptions($provider->name());
    }

    /**
     * Map the given Laravel messages to Open Responses input items with required message item types.
     */
    protected function mapMessagesToInput(array $messages, ?string $instructions, Provider $provider): array
    {
        return array_map(
            fn (array $item): array => ! isset($item['type']) && isset($item['role'])
                ? ['type' => 'message', ...$item]
                : $item,
            $this->mapOpenAiMessagesToInput($messages, $instructions, $provider),
        );
    }

    /**
     * Map the given Laravel attachments to Open Responses content parts.
     */
    protected function mapAttachments(Collection $attachments, Provider $provider): array
    {
        foreach ($attachments as $attachment) {
            if ($attachment instanceof ProviderImage || $attachment instanceof ProviderDocument) {
                throw new InvalidArgumentException(
                    'Open Responses does not support provider-hosted file IDs ['.$attachment::class.'].'
                );
            }
        }

        return $this->mapOpenAiAttachments($attachments, $provider);
    }

    /**
     * Resolve the provider key passed to attachment provider options callbacks.
     */
    protected function attachmentProviderKey(Provider $provider): Lab|string
    {
        return Lab::tryFrom($provider->name()) ?? $provider->name();
    }

    /**
     * Validate the Open Responses response data.
     *
     * @throws AiException
     */
    protected function validateTextResponse(array $data): void
    {
        if (! $data || isset($data['error'])) {
            $error = $data['error'] ?? null;

            $code = is_array($error)
                ? (($error['code'] ?? null) ?: ($error['type'] ?? null) ?: 'unknown')
                : 'unknown';

            $message = is_array($error)
                ? ($error['message'] ?? 'Unknown Open Responses error.')
                : (is_string($error) && $error !== '' ? $error : 'Unknown Open Responses error.');

            throw new AiException(sprintf('Open Responses Error: [%s] %s', $code, $message));
        }

        if (($data['status'] ?? '') === 'failed') {
            throw new AiException('Open Responses Error: [unknown] The response failed without an error message.');
        }
    }
}
