<?php

use Illuminate\Http\Client\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Sleep;
use Laravel\Ai\AiManager;
use Laravel\Ai\Enums\Lab;
use Laravel\Ai\Exceptions\InvalidApiKeyException;
use Laravel\Ai\Exceptions\ProviderOverloadedException;
use Laravel\Ai\Exceptions\RateLimitedException;
use Laravel\Ai\Files\Document;
use Laravel\Ai\Files\Image;
use Laravel\Ai\Messages\AssistantMessage;
use Laravel\Ai\Messages\UserMessage;
use Laravel\Ai\Providers\LaravelProvider;
use Laravel\Ai\Responses\AgentResponse;
use Laravel\Ai\Streaming\Events\StreamEnd;
use Laravel\Ai\Streaming\Events\StreamStart;
use Tests\Fixtures\Agents\LabProviderOptionsAgent;
use Tests\Fixtures\Agents\MultiStepToolAgent;

use function Laravel\Ai\agent;

beforeEach(function (): void {
    config(['ai.providers.laravel' => [
        'driver' => 'laravel',
        'key' => 'cloud-key',
        'url' => 'https://gateway.laravel.cloud',
    ]]);
});

function laravelAnthropicResponse(string $text = 'Hello from Claude', string $model = 'arn:aws:bedrock:us-east-1::inference-profile/claude'): array
{
    return [
        'id' => 'msg_1',
        'type' => 'message',
        'role' => 'assistant',
        'model' => $model,
        'content' => [['type' => 'text', 'text' => $text]],
        'stop_reason' => 'end_turn',
        'usage' => ['input_tokens' => 4, 'output_tokens' => 2],
    ];
}

function laravelResponsesResponse(string $text = 'Hello from GPT', string $model = 'openai/gpt-5-4'): array
{
    return [
        'id' => 'resp_1',
        'status' => 'completed',
        'model' => $model,
        'output' => [[
            'type' => 'message',
            'id' => 'msg_1',
            'role' => 'assistant',
            'status' => 'completed',
            'content' => [['type' => 'output_text', 'text' => $text, 'annotations' => []]],
        ]],
        'usage' => ['input_tokens' => 4, 'output_tokens' => 2],
    ];
}

function laravelSse(array $events): string
{
    return implode('', array_map(fn (array $event): string => 'data: '.json_encode($event)."\n\n", $events));
}

function laravelBody(Request $request): array
{
    return json_decode($request->body(), true);
}

test('the laravel provider is the default text provider with Claude defaults', function (): void {
    $config = require __DIR__.'/../../../../config/ai.php';

    expect($config['default'])->toBe('laravel')
        ->and($config['providers']['laravel'])->toMatchArray(['driver' => 'laravel'])
        ->and(Lab::Laravel->value)->toBe('laravel');

    $provider = app(AiManager::class)->textProvider('laravel');

    expect($provider)->toBeInstanceOf(LaravelProvider::class)
        ->and($provider->defaultTextModel())->toBe('anthropic/claude-sonnet-5')
        ->and($provider->cheapestTextModel())->toBe('anthropic/claude-haiku-4-5')
        ->and($provider->smartestTextModel())->toBe('anthropic/claude-opus-5-5');
});

test('the default models can be overridden', function (): void {
    config(['ai.providers.laravel.models.text' => ['default' => 'openai/gpt-5-4', 'cheapest' => 'openai/gpt-oss-120b']]);

    $provider = app(AiManager::class)->textProvider('laravel');

    expect($provider->defaultTextModel())->toBe('openai/gpt-5-4')
        ->and($provider->cheapestTextModel())->toBe('openai/gpt-oss-120b')
        ->and($provider->smartestTextModel())->toBe('anthropic/claude-opus-5-5');
});

test('anthropic prefixed models are sent unprefixed to the Anthropic endpoint with bearer auth', function (): void {
    Http::fake(['*' => Http::response(laravelAnthropicResponse())]);

    $response = agent()->prompt('Hello', provider: 'laravel', model: 'anthropic/claude-opus-5-5');

    expect($response->text)->toBe('Hello from Claude')
        ->and($response->meta->provider)->toBe('laravel')
        ->and($response->meta->model)->toBe('anthropic/claude-opus-5-5');

    Http::assertSent(fn (Request $request): bool => $request->url() === 'https://gateway.laravel.cloud/v1/anthropic/v1/messages'
        && $request->hasHeader('Authorization', 'Bearer cloud-key')
        && $request->hasHeader('anthropic-version', '2023-06-01')
        && ! $request->hasHeader('anthropic-beta')
        && ! $request->hasHeader('x-api-key')
        && laravelBody($request)['model'] === 'claude-opus-5-5');
});

test('the anthropic version and beta headers can be configured', function (): void {
    config(['ai.providers.laravel.version' => '2024-01-01', 'ai.providers.laravel.anthropic_beta' => 'web-fetch-2025-09-10']);

    Http::fake(['*' => Http::response(laravelAnthropicResponse())]);

    agent()->prompt('Hello', provider: 'laravel', model: 'anthropic/claude-haiku-4-5');

    Http::assertSent(fn (Request $request): bool => $request->hasHeader('anthropic-version', '2024-01-01')
        && $request->hasHeader('anthropic-beta', 'web-fetch-2025-09-10'));
});

test('every other model is sent statelessly to the Open Responses endpoint', function (string $model): void {
    Http::fake(['*' => Http::response(laravelResponsesResponse(model: $model))]);

    $response = agent()->prompt('Hello', provider: 'laravel', model: $model);

    expect($response->text)->toBe('Hello from GPT')
        ->and($response->meta->model)->toBe($model);

    Http::assertSent(fn (Request $request): bool => $request->url() === 'https://gateway.laravel.cloud/v1/openai/v1/responses'
        && $request->hasHeader('Authorization', 'Bearer cloud-key')
        && laravelBody($request)['model'] === $model
        && laravelBody($request)['store'] === false
        && ! isset(laravelBody($request)['include']));
})->with(['openai/gpt-5-4', 'openai/gpt-oss-120b', 'deepseek/deepseek-v3-2', 'claude-opus-5-5']);

test('the store option cannot make the Open Responses route stateful', function (): void {
    config(['ai.providers.laravel.store' => true]);

    Http::fake(['*' => Http::sequence()
        ->push([...laravelResponsesResponse(), 'id' => 'resp_tool', 'output' => [[
            'type' => 'function_call', 'id' => 'fc_1', 'call_id' => 'call_1', 'name' => 'FixedNumberGenerator', 'arguments' => '{}', 'status' => 'completed',
        ]]])
        ->push(laravelResponsesResponse('The number is 42'))]);

    $response = (new MultiStepToolAgent)->prompt('Number?', provider: 'laravel', model: 'openai/gpt-5-4');

    expect($response->text)->toBe('The number is 42');

    $requests = Http::recorded()->map(fn (array $pair): array => laravelBody($pair[0]));

    expect($requests)->toHaveCount(2)
        ->and($requests[1])->not->toHaveKey('previous_response_id')
        ->and($requests[1]['store'])->toBeFalse()
        ->and(collect($requests[1]['input'])->pluck('type')->filter()->values()->all())
        ->toBe(['message', 'message', 'function_call', 'function_call_output']);
});

test('the configured url is normalized to the gateway root', function (string $url): void {
    config(['ai.providers.laravel.url' => $url]);

    Http::fake([
        '*/v1/anthropic/v1/messages' => Http::response(laravelAnthropicResponse()),
        '*/v1/openai/v1/responses' => Http::response(laravelResponsesResponse()),
    ]);

    agent()->prompt('Hello', provider: 'laravel', model: 'anthropic/claude-sonnet-5');
    agent()->prompt('Hello', provider: 'laravel', model: 'openai/gpt-5-4');

    $urls = Http::recorded()->map(fn (array $pair): string => $pair[0]->url())->all();

    expect($urls)->toBe([
        'https://gw.example.test/v1/anthropic/v1/messages',
        'https://gw.example.test/v1/openai/v1/responses',
    ]);
})->with([
    'root' => ['https://gw.example.test'],
    'trailing slash' => ['https://gw.example.test/'],
    'anthropic v1' => ['https://gw.example.test/v1/anthropic/v1'],
    'anthropic' => ['https://gw.example.test/v1/anthropic/'],
    'openai v1' => ['https://gw.example.test/v1/openai/v1/'],
    'openai' => ['https://gw.example.test/v1/openai'],
]);

test('each route merges its own lab options with the laravel options', function (): void {
    Http::fake([
        '*/v1/anthropic/v1/messages' => Http::response(laravelAnthropicResponse()),
        '*/v1/openai/v1/responses' => Http::response(laravelResponsesResponse()),
    ]);

    $agent = new LabProviderOptionsAgent([
        'anthropic' => ['top_k' => 5, 'metadata' => ['user_id' => 'anthropic']],
        'open-responses' => ['top_logprobs' => 2, 'metadata' => ['user_id' => 'open-responses']],
        'laravel' => ['metadata' => ['user_id' => 'laravel']],
    ]);

    $agent->prompt('Hello', provider: 'laravel', model: 'anthropic/claude-sonnet-5');
    $agent->prompt('Hello', provider: 'laravel', model: 'openai/gpt-5-4');

    [$anthropic, $responses] = Http::recorded()->map(fn (array $pair): array => laravelBody($pair[0]))->all();

    expect($anthropic['top_k'])->toBe(5)
        ->and($anthropic)->not->toHaveKey('top_logprobs')
        ->and($anthropic['metadata'])->toBe(['user_id' => 'laravel'])
        ->and($responses['top_logprobs'])->toBe(2)
        ->and($responses)->not->toHaveKey('top_k')
        ->and($responses['metadata'])->toBe(['user_id' => 'laravel']);
});

test('provider file IDs are rejected on both routes', function (string $model, object $attachment): void {
    Http::fake();

    expect(fn (): AgentResponse => agent()->prompt('Read this', [$attachment], provider: 'laravel', model: $model))
        ->toThrow(InvalidArgumentException::class, 'does not support provider file IDs');

    Http::assertNothingSent();
})->with([
    'anthropic image' => fn () => ['anthropic/claude-sonnet-5', Image::fromId('file_1')],
    'anthropic document' => fn () => ['anthropic/claude-sonnet-5', Document::fromId('file_1')],
    'responses image' => fn () => ['openai/gpt-5-4', Image::fromId('file_1')],
    'responses document' => fn () => ['openai/gpt-5-4', Document::fromId('file_1')],
]);

test('remote documents are inlined as file data on the Open Responses route', function (): void {
    Http::fake([
        'files.example.com/*' => Http::response('%PDF-1.4 report', 200, ['Content-Type' => 'application/pdf']),
        '*/v1/openai/v1/responses' => Http::response(laravelResponsesResponse()),
    ]);

    agent()->prompt('Summarize', [Document::fromUrl('https://files.example.com/report.pdf')], provider: 'laravel', model: 'openai/gpt-5-4');

    Http::assertSent(function (Request $request): bool {
        if (! str_ends_with($request->url(), '/responses')) {
            return false;
        }

        $part = laravelBody($request)['input'][0]['content'][1];

        return $part === [
            'type' => 'input_file',
            'file_data' => 'data:application/pdf;base64,'.base64_encode('%PDF-1.4 report'),
            'filename' => 'report.pdf',
        ];
    });
});

test('Anthropic history is replayed generically on the Open Responses route', function (): void {
    Http::fake(['*' => Http::response(laravelResponsesResponse())]);

    agent(messages: [
        new UserMessage('Hi'),
        new AssistantMessage('Earlier answer', replayBlocks: [
            ['type' => 'thinking', 'thinking' => 'Hmm', 'signature' => 'sig'],
            ['type' => 'text', 'text' => 'Earlier answer'],
        ]),
    ])->prompt('Again', provider: 'laravel', model: 'openai/gpt-5-4');

    Http::assertSent(function (Request $request): bool {
        $input = laravelBody($request)['input'];

        return $input[1] === ['type' => 'message', 'role' => 'assistant', 'content' => [['type' => 'output_text', 'text' => 'Earlier answer']]]
            && ! collect($input)->contains(fn (array $item): bool => ($item['type'] ?? null) === 'thinking');
    });
});

test('Open Responses history is replayed generically on the Anthropic route', function (): void {
    Http::fake(['*' => Http::response(laravelAnthropicResponse())]);

    agent(messages: [
        new UserMessage('Hi'),
        new AssistantMessage('Earlier answer', replayBlocks: [
            ['type' => 'reasoning', 'id' => 'rs_1', 'summary' => []],
            ['type' => 'message', 'role' => 'assistant', 'content' => [['type' => 'output_text', 'text' => 'Earlier answer']]],
        ]),
    ])->prompt('Again', provider: 'laravel', model: 'anthropic/claude-sonnet-5');

    Http::assertSent(fn (Request $request): bool => laravelBody($request)['messages'][1] === [
        'role' => 'assistant',
        'content' => [['type' => 'text', 'text' => 'Earlier answer']],
    ]);
});

test('matching history is replayed verbatim on each route', function (): void {
    Http::fake([
        '*/v1/anthropic/v1/messages' => Http::response(laravelAnthropicResponse()),
        '*/v1/openai/v1/responses' => Http::response(laravelResponsesResponse()),
    ]);

    $thinking = ['type' => 'thinking', 'thinking' => 'Hmm', 'signature' => 'sig'];
    $reasoning = ['type' => 'reasoning', 'id' => 'rs_1', 'summary' => [], 'encrypted_content' => 'enc'];

    agent(messages: [new UserMessage('Hi'), new AssistantMessage('A', replayBlocks: [$thinking, ['type' => 'text', 'text' => 'A']])])
        ->prompt('Again', provider: 'laravel', model: 'anthropic/claude-sonnet-5');
    agent(messages: [new UserMessage('Hi'), new AssistantMessage('B', replayBlocks: [$reasoning, ['type' => 'message', 'role' => 'assistant', 'content' => [['type' => 'output_text', 'text' => 'B']]]])])
        ->prompt('Again', provider: 'laravel', model: 'openai/gpt-5-4');

    [$anthropic, $responses] = Http::recorded()->map(fn (array $pair): array => laravelBody($pair[0]))->all();

    expect($anthropic['messages'][1]['content'][0])->toBe($thinking)
        ->and($responses['input'][1])->toBe($reasoning);
});

test('streamed Claude responses report the requested catalog model', function (): void {
    Http::fake(['*' => Http::response(laravelSse([
        ['type' => 'message_start', 'message' => ['id' => 'msg_1', 'model' => 'us.anthropic.claude-opus-5-5-v1:0', 'role' => 'assistant', 'content' => [], 'usage' => ['input_tokens' => 3, 'output_tokens' => 0]]],
        ['type' => 'content_block_start', 'index' => 0, 'content_block' => ['type' => 'text', 'text' => '']],
        ['type' => 'content_block_delta', 'index' => 0, 'delta' => ['type' => 'text_delta', 'text' => 'Streamed']],
        ['type' => 'content_block_stop', 'index' => 0],
        ['type' => 'message_delta', 'delta' => ['stop_reason' => 'end_turn'], 'usage' => ['output_tokens' => 1]],
        ['type' => 'message_stop'],
    ]), 200, ['Content-Type' => 'text/event-stream'])]);

    $stream = agent()->stream('Hello', provider: 'laravel', model: 'anthropic/claude-opus-5-5');

    foreach ($stream as $event) {
        //
    }

    expect($stream->text)->toBe('Streamed')
        ->and($stream->events->whereInstanceOf(StreamStart::class)->first()->model)->toBe('anthropic/claude-opus-5-5')
        ->and($stream->events->whereInstanceOf(StreamEnd::class)->last()->steps->last()->meta->model)->toBe('anthropic/claude-opus-5-5');

    Http::assertSent(fn (Request $request): bool => str_ends_with($request->url(), '/v1/anthropic/v1/messages'));
});

test('gateway errors map to failover exceptions without retrying', function (int $status, array $body, string $exception): void {
    Sleep::fake();

    Http::fake(['*' => Http::response($body, $status, ['Retry-After' => '1'])]);

    expect(fn (): AgentResponse => agent()->prompt('Hello', provider: 'laravel', model: 'anthropic/claude-sonnet-5'))
        ->toThrow($exception);

    Http::assertSentCount($exception === InvalidApiKeyException::class ? 1 : 2);
})->with([
    'unauthorized' => [401, ['error' => ['code' => 'unauthorized']], InvalidApiKeyException::class],
    'key rate limited' => [429, ['error' => ['code' => 'rate_limited', 'message' => 'Key limit reached.']], InvalidApiKeyException::class],
    'upstream throttled' => [429, ['error' => ['code' => 'throttled']], RateLimitedException::class],
    'internal error' => [500, ['error' => ['code' => 'upstream_error']], ProviderOverloadedException::class],
    'overloaded' => [529, ['error' => ['code' => 'overloaded']], ProviderOverloadedException::class],
]);

test('the Open Responses route uses the same error handling', function (): void {
    Sleep::fake();

    Http::fake(['*' => Http::response(['error' => ['code' => 'rate_limited']], 429, ['Retry-After' => '1'])]);

    expect(fn (): AgentResponse => agent()->prompt('Hello', provider: 'laravel', model: 'openai/gpt-5-4'))
        ->toThrow(InvalidApiKeyException::class);

    Http::assertSentCount(1);
});
