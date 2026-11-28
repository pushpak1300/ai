<?php

namespace Laravel\Ai\Gateway\Laravel;

use Generator;
use Illuminate\Contracts\Events\Dispatcher;
use Laravel\Ai\Contracts\Gateway\StepTextGateway;
use Laravel\Ai\Contracts\Providers\TextProvider;
use Laravel\Ai\Gateway\StepContext;
use Laravel\Ai\Gateway\StepResponse;
use Laravel\Ai\Gateway\TextGenerationOptions;
use Laravel\Ai\Providers\Provider;

class LaravelGateway implements StepTextGateway
{
    protected ?LaravelAnthropicGateway $anthropicGateway = null;

    protected ?LaravelOpenResponsesGateway $openResponsesGateway = null;

    public function __construct(protected Dispatcher $events)
    {
        //
    }

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
        return $this->gatewayFor($model)->generateTextStep(
            $provider, $model, $instructions, $messages, $tools, $schema, $options, $timeout, $stepContext,
        );
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
        return yield from $this->gatewayFor($model)->generateStreamStep(
            $invocationId, $provider, $model, $instructions, $messages, $tools, $schema, $options, $timeout, $stepContext,
        );
    }

    /**
     * Get the gateway that serves the given model.
     *
     * Claude models are served by the gateway's Anthropic Messages endpoint, while vendor-prefixed models such as "openai/gpt-5-4" are served by its Open Responses endpoint.
     */
    public function gatewayFor(string $model): StepTextGateway
    {
        if (str_contains($model, '/')) {
            return $this->openResponsesGateway ??= new LaravelOpenResponsesGateway($this->events);
        }

        return $this->anthropicGateway ??= new LaravelAnthropicGateway($this->events);
    }

    /**
     * Get the root URL of the gateway, without any API path the configured URL may include.
     */
    public static function rootUrl(Provider $provider): string
    {
        $url = rtrim((string) ($provider->additionalConfiguration()['url'] ?? ''), '/');

        if ($url === '') {
            return 'https://gateway.laravel.cloud';
        }

        foreach (['/v1/anthropic/v1', '/v1/anthropic', '/v1/openai/v1', '/v1/openai'] as $suffix) {
            if (str_ends_with($url, $suffix)) {
                return substr($url, 0, -strlen($suffix));
            }
        }

        return $url;
    }

    /**
     * Determine if any of the given replay blocks is a Responses output item rather than an Anthropic content block.
     *
     * @param  array<array-key, mixed>  $blocks
     */
    public static function hasOpenResponsesItems(array $blocks): bool
    {
        return array_any($blocks, fn ($block): bool => is_array($block) && static::isOpenResponsesItem($block));
    }

    /**
     * Determine if every one of the given replay blocks is a Responses output item.
     *
     * @param  array<array-key, mixed>  $blocks
     */
    public static function hasOnlyOpenResponsesItems(array $blocks): bool
    {
        return array_all($blocks, fn ($block): bool => is_array($block) && static::isOpenResponsesItem($block));
    }

    /**
     * Determine if the given replay block is a Responses output item.
     *
     * @param  array<string, mixed>  $block
     */
    protected static function isOpenResponsesItem(array $block): bool
    {
        $type = (string) ($block['type'] ?? '');

        return isset($block['role'])
            || in_array($type, ['message', 'reasoning', 'function_call', 'function_call_output'], true)
            || str_ends_with($type, '_call')
            || str_contains($type, ':');
    }
}
