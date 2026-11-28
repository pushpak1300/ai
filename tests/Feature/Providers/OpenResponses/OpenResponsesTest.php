<?php

use Illuminate\Http\Client\Request;
use Illuminate\Http\Client\RequestException;
use Illuminate\Support\Facades\Http;
use Laravel\Ai\Ai;
use Laravel\Ai\Enums\Lab;
use Laravel\Ai\Exceptions\AiException;
use Laravel\Ai\Exceptions\InvalidApiKeyException;
use Laravel\Ai\Exceptions\ProviderOverloadedException;
use Laravel\Ai\Exceptions\RateLimitedException;
use Laravel\Ai\Exceptions\StreamErrorException;
use Laravel\Ai\Files\Base64Document;
use Laravel\Ai\Files\Base64Image;
use Laravel\Ai\Files\ProviderDocument;
use Laravel\Ai\Files\ProviderImage;
use Laravel\Ai\Files\RemoteDocument;
use Laravel\Ai\Files\RemoteImage;
use Laravel\Ai\Messages\AssistantMessage;
use Laravel\Ai\Messages\UserMessage;
use Laravel\Ai\PendingStep;
use Laravel\Ai\Providers\OpenResponsesProvider;
use Laravel\Ai\Providers\Tools\ToolSearch;
use Laravel\Ai\Providers\Tools\WebSearch;
use Laravel\Ai\Responses\AgentResponse;
use Laravel\Ai\Streaming\Events\Error;
use Laravel\Ai\Streaming\Events\ReasoningDelta;
use Laravel\Ai\Streaming\Events\ReasoningEnd;
use Laravel\Ai\Streaming\Events\ReasoningStart;
use Laravel\Ai\Streaming\Events\StreamEnd;
use Laravel\Ai\Streaming\Events\ToolCall as ToolCallEvent;
use Laravel\Ai\Streaming\Events\ToolInputDelta;
use Laravel\Ai\Streaming\Events\ToolInputStart;
use Laravel\Ai\Streaming\Events\ToolResult as ToolResultEvent;
use Laravel\Ai\ToolChoice;
use Tests\Fixtures\Agents\AssistantAgent;
use Tests\Fixtures\Agents\LabProviderOptionsAgent;
use Tests\Fixtures\Agents\NestedStructuredAgent;
use Tests\Fixtures\Agents\StructuredAgent;
use Tests\Fixtures\Agents\ToolUsingAgent;
use Tests\Fixtures\Tools\FixedNumberGenerator;

use function Laravel\Ai\agent;

beforeEach(function (): void {
    config(['ai.providers.open-responses' => [
        'driver' => 'open-responses',
        'url' => 'http://localhost:8080/v1/',
        'key' => 'test-key',
        'models' => ['text' => ['default' => 'local-model']],
    ]]);
});

function openResponsesBody(array $output, array $overrides = []): array
{
    return [
        'id' => 'resp_1',
        'object' => 'response',
        'status' => 'completed',
        'model' => 'local-model',
        'output' => $output,
        'usage' => ['input_tokens' => 3, 'output_tokens' => 2],
        ...$overrides,
    ];
}

function openResponsesText(string $text): array
{
    return openResponsesBody([[
        'type' => 'message',
        'id' => 'msg_1',
        'status' => 'completed',
        'role' => 'assistant',
        'content' => [['type' => 'output_text', 'text' => $text, 'annotations' => []]],
    ]]);
}

function openResponsesStream(array $events): string
{
    return implode('', array_map(fn (array $event): string => 'data: '.json_encode($event)."\n\n", $events));
}

test('the provider resolves from the open-responses driver', function (): void {
    expect(Lab::OpenResponses->value)->toBe('open-responses')
        ->and(app('Laravel\Ai\AiManager')->textProvider('open-responses'))->toBeInstanceOf(OpenResponsesProvider::class);
});

test('the shipped config registers the provider', function (): void {
    $config = require __DIR__.'/../../../../config/ai.php';

    expect($config['providers']['open-responses'])->toMatchArray(['driver' => 'open-responses'])
        ->toHaveKeys(['url', 'key']);
});

test('text requests post to the configured responses endpoint with bearer auth', function (): void {
    Http::fake(['*' => Http::response(openResponsesText('Hello there'))]);

    $response = agent()->prompt('Hello', provider: 'open-responses');

    expect($response->text)->toBe('Hello there')
        ->and($response->meta->provider)->toBe('open-responses')
        ->and($response->meta->model)->toBe('local-model');

    Http::assertSent(fn (Request $request): bool => $request->url() === 'http://localhost:8080/v1/responses'
        && $request->hasHeader('Authorization', 'Bearer test-key')
        && $request['model'] === 'local-model'
        && $request['store'] === false
        && ! isset($request['include']));
});

test('the authorization header is omitted without a key', function (?string $key): void {
    config(['ai.providers.open-responses.key' => $key]);

    Http::fake(['*' => Http::response(openResponsesText('Hi'))]);

    agent()->prompt('Hello', provider: 'open-responses');

    Http::assertSent(fn (Request $request): bool => ! $request->hasHeader('Authorization'));
})->with(['null' => [null], 'blank' => ['']]);

test('a url is required', function (?string $url): void {
    config(['ai.providers.open-responses.url' => $url]);

    expect(fn (): AgentResponse => agent()->prompt('Hello', provider: 'open-responses'))
        ->toThrow(InvalidArgumentException::class, "requires a 'url'");
})->with(['null' => [null], 'blank' => ['  ']]);

test('a default model is required unless one is passed', function (): void {
    config(['ai.providers.open-responses.models' => []]);

    expect(fn (): AgentResponse => agent()->prompt('Hello', provider: 'open-responses'))
        ->toThrow(InvalidArgumentException::class, 'requires a default text model');

    Http::fake(['*' => Http::response(openResponsesText('Hi'))]);

    agent()->prompt('Hello', provider: 'open-responses', model: 'explicit-model');

    Http::assertSent(fn (Request $request): bool => $request['model'] === 'explicit-model');
});

test('the cheapest and smartest models fall back to the default', function (): void {
    $provider = app('Laravel\Ai\AiManager')->textProvider('open-responses');

    expect($provider->cheapestTextModel())->toBe('local-model')
        ->and($provider->smartestTextModel())->toBe('local-model');
});

test('storage is opt in through the store option', function (): void {
    config(['ai.providers.open-responses.store' => true]);

    Http::fake(['*' => Http::response(openResponsesText('Hi'))]);

    agent()->prompt('Hello', provider: 'open-responses');

    Http::assertSent(fn (Request $request): bool => $request['store'] === true);
});

test('provider options resolve by the configured provider name', function (): void {
    config(['ai.providers.my-server' => config('ai.providers.open-responses')]);

    Http::fake(['*' => Http::response(openResponsesText('Hi'))]);

    (new LabProviderOptionsAgent([
        'my-server' => ['top_logprobs' => 2],
        'open-responses' => ['ignored' => true],
    ]))->prompt('Hello', provider: 'my-server');

    Http::assertSent(fn (Request $request): bool => $request['top_logprobs'] === 2 && ! isset($request['ignored']));
});

test('a failed response throws an Open Responses error', function (): void {
    Http::fake(['*' => Http::response(openResponsesBody([], [
        'status' => 'failed',
        'error' => ['type' => 'server_error', 'message' => 'Model crashed.'],
    ]))]);

    expect(fn (): AgentResponse => agent()->prompt('Hello', provider: 'open-responses'))
        ->toThrow(AiException::class, 'Open Responses Error: [server_error] Model crashed.');
});

test('text across messages and trailing extension items is parsed', function (): void {
    Http::fake(['*' => Http::response(openResponsesBody([
        ['type' => 'message', 'role' => 'assistant', 'status' => 'completed', 'content' => [['type' => 'output_text', 'text' => 'One']]],
        ['type' => 'message', 'role' => 'assistant', 'status' => 'completed', 'content' => [['type' => 'output_text', 'text' => ' two']]],
        ['type' => 'acme:trace', 'id' => 'tr_1'],
    ]))]);

    $response = agent()->prompt('Hello', provider: 'open-responses');

    expect($response->text)->toBe('One two')
        ->and($response->steps->last()->finishReason->value)->toBe('stop');
});

test('a stream error without a code reports its type', function (): void {
    Http::fake(['*' => Http::response(openResponsesStream([
        ['type' => 'response.created', 'response' => ['id' => 'resp_1', 'model' => 'local-model']],
        ['type' => 'error', 'error' => ['type' => 'invalid_request_error', 'message' => 'Bad input.']],
    ]), 200, ['Content-Type' => 'text/event-stream'])]);

    $events = [];

    try {
        foreach (agent()->stream('Hello', provider: 'open-responses') as $event) {
            $events[] = $event;
        }
    } catch (StreamErrorException) {
        //
    }

    $error = collect($events)->first(fn ($event): bool => $event instanceof Error);

    expect($error)->not->toBeNull()
        ->and($error->type)->toBe('invalid_request_error')
        ->and($error->message)->toBe('Bad input.');
});

test('streamed reasoning without deltas is kept from the completed response', function (): void {
    Http::fake(['*' => Http::response(openResponsesStream([
        ['type' => 'response.created', 'response' => ['id' => 'resp_1', 'model' => 'local-model']],
        ['type' => 'response.output_text.delta', 'delta' => 'Answer'],
        ['type' => 'response.output_text.done', 'text' => 'Answer'],
        ['type' => 'response.completed', 'response' => openResponsesBody([
            ['type' => 'reasoning', 'id' => 'rs_1', 'summary' => [], 'content' => [['type' => 'reasoning_text', 'text' => 'Buffered thought']]],
            ['type' => 'message', 'role' => 'assistant', 'status' => 'completed', 'content' => [['type' => 'output_text', 'text' => 'Answer']]],
        ])],
    ]), 200, ['Content-Type' => 'text/event-stream'])]);

    $stream = agent()->stream('Hello', provider: 'open-responses');

    foreach ($stream as $event) {
        //
    }

    $end = $stream->events->whereInstanceOf(StreamEnd::class)->last();

    expect($stream->text)->toBe('Answer')
        ->and($end->steps->last()->reasoning)->toBe('Buffered thought');
});

test('multi-step tool loops are stateless by default and replay encrypted reasoning', function (): void {
    Http::fake([
        '*' => Http::sequence([
            Http::response(openResponsesBody([
                ['type' => 'reasoning', 'id' => 'rs_1', 'summary' => [], 'encrypted_content' => 'enc_blob_1'],
                ['type' => 'function_call', 'id' => 'fc_1', 'call_id' => 'call_1', 'name' => 'FixedNumberGenerator', 'arguments' => '{}', 'status' => 'completed'],
            ])),
            Http::response(openResponsesText('The number is 72019')),
        ]),
    ]);

    $response = (new ToolUsingAgent(fixed: true))->prompt('Generate a number', provider: 'open-responses');

    expect($response->text)->toBe('The number is 72019');

    $recorded = Http::recorded();
    expect($recorded)->toHaveCount(2);

    $followUp = json_decode((string) $recorded[1][0]->body(), true);

    expect($followUp)->not->toHaveKey('previous_response_id')
        ->and($followUp['store'] ?? null)->toBeFalse();

    $input = collect($followUp['input']);

    expect($input->contains(fn ($item): bool => ($item['type'] ?? null) === 'reasoning'
        && ($item['encrypted_content'] ?? null) === 'enc_blob_1'))->toBeTrue()
        ->and($input->contains(fn ($item): bool => ($item['type'] ?? null) === 'function_call_output'
            && ($item['call_id'] ?? null) === 'call_1'))->toBeTrue();
});

test('opt-in store true uses previous_response_id in multi-step tool loops', function (): void {
    config(['ai.providers.open-responses.store' => true]);

    Http::fake([
        '*' => Http::sequence([
            Http::response(openResponsesBody([
                ['type' => 'function_call', 'id' => 'fc_1', 'call_id' => 'call_1', 'name' => 'FixedNumberGenerator', 'arguments' => '{}', 'status' => 'completed'],
            ])),
            Http::response(openResponsesText('Done')),
        ]),
    ]);

    (new ToolUsingAgent(fixed: true))->prompt('Generate a number', provider: 'open-responses');

    $first = json_decode((string) Http::recorded()[0][0]->body(), true);
    $followUp = json_decode((string) Http::recorded()[1][0]->body(), true);

    expect($first['store'] ?? null)->toBeTrue()
        ->and($followUp['previous_response_id'] ?? null)->toBe('resp_1')
        ->and($followUp['store'] ?? null)->toBeTrue();
});

test('structured output with Strict attribute sends strict true', function (): void {
    Http::fake(['*' => Http::response(openResponsesText('{"symbol":"Au"}'))]);

    (new StructuredAgent)->prompt('Gold?', provider: 'open-responses');

    Http::assertSent(function (Request $request): bool {
        $format = data_get(json_decode($request->body(), true), 'text.format');

        return ($format['type'] ?? null) === 'json_schema'
            && ($format['strict'] ?? null) === true;
    });
});

test('structured output without Strict attribute sends strict false', function (): void {
    Http::fake(['*' => Http::response(openResponsesText('{"elements":[]}'))]);

    (new NestedStructuredAgent)->prompt('Elements?', provider: 'open-responses');

    Http::assertSent(function (Request $request): bool {
        $format = data_get(json_decode($request->body(), true), 'text.format');

        return ($format['type'] ?? null) === 'json_schema'
            && ($format['strict'] ?? null) === false;
    });
});

test('unsupported provider tools are stripped and tool search is unwrapped to plain functions', function (): void {
    Http::fake(['*' => Http::response(openResponsesText('Done'))]);

    agent(tools: [
        new WebSearch,
        new ToolSearch([new FixedNumberGenerator]),
    ])->prompt('Hello', provider: 'open-responses');

    $body = json_decode((string) Http::recorded()[0][0]->body(), true);

    expect($body['tools'])->toHaveCount(1)
        ->and($body['tools'][0]['type'])->toBe('function')
        ->and($body['tools'][0]['name'])->toBe('FixedNumberGenerator');
});

test('input messages include required message item type', function (): void {
    Http::fake(['*' => Http::response(openResponsesText('Hi'))]);

    agent(
        instructions: 'Be brief.',
        messages: [
            new UserMessage('First'),
            new AssistantMessage('Second'),
        ],
    )->prompt('Third', provider: Lab::OpenResponses);

    $body = json_decode((string) Http::recorded()[0][0]->body(), true);

    expect(array_column($body['input'], 'type'))->toBe(['message', 'message', 'message', 'message'])
        ->and(array_column($body['input'], 'role'))->toBe(['system', 'user', 'assistant', 'user']);
});

test('streamed open responses reasoning deltas emit reasoning events and populate stream reasoning', function (string $deltaType): void {
    Http::fake(['*' => Http::response(openResponsesStream([
        ['type' => 'response.created', 'response' => ['id' => 'resp_1', 'model' => 'local-model']],
        ['type' => $deltaType, 'delta' => 'Step 1. '],
        ['type' => $deltaType, 'delta' => 'Step 2.'],
        ['type' => 'response.output_item.done', 'item' => ['type' => 'reasoning', 'id' => 'rs_1', 'summary' => []]],
        ['type' => 'response.output_text.delta', 'delta' => 'Done'],
        ['type' => 'response.output_text.done', 'text' => 'Done'],
        ['type' => 'response.completed', 'response' => openResponsesBody([
            ['type' => 'reasoning', 'id' => 'rs_1', 'summary' => [], 'content' => [['type' => 'reasoning_text', 'text' => 'Step 1. Step 2.']]],
            ['type' => 'message', 'role' => 'assistant', 'status' => 'completed', 'content' => [['type' => 'output_text', 'text' => 'Done']]],
        ])],
    ]), 200, ['Content-Type' => 'text/event-stream'])]);

    $stream = agent()->stream('Think', provider: Lab::OpenResponses);
    $events = [];

    foreach ($stream as $event) {
        $events[] = $event;
    }

    $collection = collect($events);

    expect($collection->whereInstanceOf(ReasoningStart::class))->toHaveCount(1)
        ->and($collection->whereInstanceOf(ReasoningDelta::class)->pluck('delta')->all())->toBe(['Step 1. ', 'Step 2.'])
        ->and($collection->whereInstanceOf(ReasoningEnd::class))->toHaveCount(1)
        ->and($stream->reasoning)->toBe('Step 1. Step 2.')
        ->and($stream->text)->toBe('Done');
})->with([
    'raw reasoning delta' => ['response.reasoning.delta'],
    'summary reasoning delta' => ['response.reasoning_summary_text.delta'],
]);

test('flat stream error and failed response without nested error fall back to unknown_error', function (array $errorEvent, string $expectedMessage): void {
    Http::fake(['*' => Http::response(openResponsesStream([
        ['type' => 'response.created', 'response' => ['id' => 'resp_1', 'model' => 'local-model']],
        $errorEvent,
    ]), 200, ['Content-Type' => 'text/event-stream'])]);

    $events = [];

    try {
        foreach (agent()->stream('Hello', provider: Lab::OpenResponses) as $event) {
            $events[] = $event;
        }
    } catch (StreamErrorException) {
        //
    }

    $error = collect($events)->first(fn ($event): bool => $event instanceof Error);

    expect($error)->not->toBeNull()
        ->and($error->type)->toBe('unknown_error')
        ->and($error->message)->toBe($expectedMessage);
})->with([
    'flat error event without code' => [['type' => 'error', 'message' => 'Flat failure'], 'Flat failure'],
    'response.failed with null error' => [['type' => 'response.failed', 'response' => ['error' => null]], 'Unknown error'],
]);

test('malformed 200 responses and failed status without error object throw AiException', function ($body, string $expectedMessage): void {
    Http::fake(['*' => Http::response($body, 200)]);

    expect(fn (): AgentResponse => agent()->prompt('Hello', provider: Lab::OpenResponses))
        ->toThrow(AiException::class, $expectedMessage);
})->with([
    'non-json body' => ['<html>Bad Gateway</html>', 'Open Responses Error: [unknown] Unknown Open Responses error.'],
    'string error field' => [['error' => 'upstream timeout'], 'Open Responses Error: [unknown] upstream timeout'],
    'failed status without error' => [openResponsesBody([], ['status' => 'failed', 'error' => null]), 'The response failed without an error message.'],
]);

test('http status errors map to failover exceptions', function (int $status, array $body, string $exceptionClass): void {
    Http::fake(['*' => Http::response($body, $status)]);

    expect(fn (): AgentResponse => agent()->prompt('Hello', provider: Lab::OpenResponses))
        ->toThrow($exceptionClass);
})->with([
    '401 invalid api key' => [401, ['error' => ['message' => 'Unauthorized']], InvalidApiKeyException::class],
    '429 rate limited' => [429, ['error' => ['message' => 'Too many requests']], RateLimitedException::class],
    '503 overloaded' => [503, ['error' => ['message' => 'Service unavailable']], ProviderOverloadedException::class],
    '400 bad request' => [400, ['error' => ['message' => 'Bad request']], RequestException::class],
]);

test('attachments map images and documents and resolve provider options by provider name', function (): void {
    config(['ai.providers.my-server' => config('ai.providers.open-responses')]);

    Http::fake(['*' => Http::response(openResponsesText('Seen'))]);

    $seenProvider = null;

    $image = (new Base64Image(base64_encode('png-bytes'), 'image/png'))
        ->withProviderOptions(function (Lab|string $provider) use (&$seenProvider): array {
            $seenProvider = $provider;

            return ['detail' => 'high'];
        });

    agent()->prompt(
        'Inspect these.',
        attachments: [
            $image,
            new RemoteImage('https://example.com/photo.png'),
            (new Base64Document(base64_encode('%PDF-1.1'), 'application/pdf'))->as('doc.pdf'),
            (new RemoteDocument('https://example.com/report.pdf'))->as('report.pdf'),
        ],
        provider: 'my-server',
    );

    $content = json_decode((string) Http::recorded()[0][0]->body(), true)['input'][0]['content'];

    expect($seenProvider)->toBe('my-server')
        ->and($content[1])->toMatchArray(['type' => 'input_image', 'detail' => 'high'])
        ->and($content[2])->toMatchArray(['type' => 'input_image', 'image_url' => 'https://example.com/photo.png'])
        ->and($content[3])->toMatchArray(['type' => 'input_file', 'filename' => 'doc.pdf'])
        ->and($content[4])->toMatchArray(['type' => 'input_file', 'file_url' => 'https://example.com/report.pdf', 'filename' => 'report.pdf']);
});

test('provider-hosted file ID attachments are rejected', function (object $attachment): void {
    Http::fake(['*' => Http::response(openResponsesText('Never'))]);

    expect(fn (): AgentResponse => agent()->prompt('Inspect', attachments: [$attachment], provider: Lab::OpenResponses))
        ->toThrow(InvalidArgumentException::class, 'provider-hosted file IDs');
})->with([
    'ProviderImage' => [new ProviderImage('file_img_1')],
    'ProviderDocument' => [new ProviderDocument('file_doc_1')],
]);

test('streaming multi-step tool calls emit tool input, call, and result events', function (): void {
    Http::fake([
        '*' => Http::sequence([
            Http::response(openResponsesStream([
                ['type' => 'response.created', 'response' => ['id' => 'resp_1', 'model' => 'local-model']],
                ['type' => 'response.output_item.added', 'output_index' => 0, 'item' => ['type' => 'function_call', 'id' => 'fc_1', 'call_id' => 'call_1', 'name' => 'FixedNumberGenerator']],
                ['type' => 'response.function_call_arguments.delta', 'output_index' => 0, 'item_id' => 'fc_1', 'delta' => '{}'],
                ['type' => 'response.function_call_arguments.done', 'output_index' => 0, 'item_id' => 'fc_1', 'arguments' => '{}'],
                ['type' => 'response.completed', 'response' => openResponsesBody([
                    ['type' => 'function_call', 'id' => 'fc_1', 'call_id' => 'call_1', 'name' => 'FixedNumberGenerator', 'arguments' => '{}', 'status' => 'completed'],
                ])],
            ]), 200, ['Content-Type' => 'text/event-stream']),
            Http::response(openResponsesStream([
                ['type' => 'response.created', 'response' => ['id' => 'resp_2', 'model' => 'local-model']],
                ['type' => 'response.output_text.delta', 'delta' => 'Result is 72019'],
                ['type' => 'response.output_text.done', 'text' => 'Result is 72019'],
                ['type' => 'response.completed', 'response' => openResponsesText('Result is 72019')],
            ]), 200, ['Content-Type' => 'text/event-stream']),
        ]),
    ]);

    $stream = agent(tools: [new FixedNumberGenerator])->stream('Generate a number', provider: Lab::OpenResponses);
    $events = [];

    foreach ($stream as $event) {
        $events[] = $event;
    }

    $collection = collect($events);

    expect($collection->whereInstanceOf(ToolInputStart::class))->toHaveCount(1)
        ->and($collection->whereInstanceOf(ToolInputDelta::class))->toHaveCount(1)
        ->and($collection->whereInstanceOf(ToolCallEvent::class))->toHaveCount(1)
        ->and($collection->whereInstanceOf(ToolResultEvent::class))->toHaveCount(1)
        ->and($stream->text)->toBe('Result is 72019');
});

test('custom headers, tool choice, and on-demand Ai::build work for open-responses', function (): void {
    Http::fake([
        '*' => Http::sequence([
            Http::response(openResponsesBody([
                ['type' => 'function_call', 'id' => 'fc_1', 'call_id' => 'call_1', 'name' => 'FixedNumberGenerator', 'arguments' => '{}', 'status' => 'completed'],
            ])),
            Http::response(openResponsesText('72019')),
        ]),
    ]);

    $provider = Ai::build([
        'driver' => 'open-responses',
        'url' => 'http://localhost:9090/v1',
        'key' => 'secret-key',
        'headers' => ['X-Custom-Header' => 'abc123'],
        'models' => ['text' => ['default' => 'on-demand-model']],
    ]);

    $response = (new AssistantAgent)
        ->withTools([new FixedNumberGenerator])
        ->withMiddleware([fn (PendingStep $step, Closure $next) => $next($step->withToolChoice($step->isFirstStep() ? ToolChoice::tool('FixedNumberGenerator') : ToolChoice::none))])
        ->prompt('Fetch a number.', provider: $provider);

    $recorded = Http::recorded();
    $first = json_decode((string) $recorded[0][0]->body(), true);
    $second = json_decode((string) $recorded[1][0]->body(), true);

    expect($response->text)->toBe('72019')
        ->and($recorded[0][0]->url())->toBe('http://localhost:9090/v1/responses')
        ->and($recorded[0][0]->hasHeader('X-Custom-Header', 'abc123'))->toBeTrue()
        ->and($first['model'])->toBe('on-demand-model')
        ->and($first['tool_choice'])->toBe(['type' => 'function', 'name' => 'FixedNumberGenerator'])
        ->and($second['tool_choice'])->toBe('none');
});
