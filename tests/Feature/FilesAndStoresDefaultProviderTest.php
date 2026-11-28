<?php

use Illuminate\Http\Client\Request;
use Illuminate\Support\Facades\Http;
use Laravel\Ai\Files;
use Laravel\Ai\Stores;

beforeEach(function (): void {
    config([
        'ai.default' => 'laravel',
        'ai.providers.openai.key' => 'openai-key',
    ]);
});

test('files use the default file provider rather than the text default', function (): void {
    Http::fake(['*' => Http::response(['id' => 'file_1', 'object' => 'file', 'filename' => 'a.txt', 'bytes' => 1])]);

    Files::get('file_1');

    Http::assertSent(fn (Request $request): bool => $request->url() === 'https://api.openai.com/v1/files/file_1');
});

test('stores use the default store provider rather than the text default', function (): void {
    Http::fake(['*' => Http::response(['id' => 'vs_1', 'object' => 'vector_store', 'name' => 'Docs', 'file_counts' => ['completed' => 0, 'in_progress' => 0, 'failed' => 0, 'cancelled' => 0, 'total' => 0], 'status' => 'completed'])]);

    Stores::get('vs_1');

    Http::assertSent(fn (Request $request): bool => $request->url() === 'https://api.openai.com/v1/vector_stores/vs_1');
});

test('files and stores fall back to the default provider when no specific default is set', function (): void {
    config(['ai.default' => 'openai', 'ai.default_for_files' => null, 'ai.default_for_stores' => null]);

    Http::fake(['*/files/*' => Http::response(['id' => 'file_1']), '*/vector_stores/*' => Http::response(['id' => 'vs_1', 'name' => 'Docs', 'file_counts' => ['completed' => 0, 'in_progress' => 0, 'failed' => 0, 'cancelled' => 0, 'total' => 0], 'status' => 'completed'])]);

    Files::get('file_1');
    Stores::get('vs_1');

    Http::assertSentCount(2);
});

test('an explicit provider still wins over the defaults', function (): void {
    config(['ai.default_for_files' => 'laravel']);

    Http::fake(['*' => Http::response(['id' => 'file_1'])]);

    Files::get('file_1', provider: 'openai');

    Http::assertSent(fn (Request $request): bool => str_starts_with($request->url(), 'https://api.openai.com/'));
});

test('the shipped config defaults files and stores to openai', function (): void {
    $config = require __DIR__.'/../../config/ai.php';

    expect($config['default_for_files'])->toBe('openai')
        ->and($config['default_for_stores'])->toBe('openai');
});
