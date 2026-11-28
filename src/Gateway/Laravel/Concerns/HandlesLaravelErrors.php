<?php

namespace Laravel\Ai\Gateway\Laravel\Concerns;

use Illuminate\Http\Client\RequestException;

trait HandlesLaravelErrors
{
    /**
     * Determine if the gateway rejected the request for a missing or invalid API key.
     *
     * The gateway answers a key that has hit its own request limit with a 429 "rate_limited" code, which a retry cannot fix.
     */
    protected function isInvalidApiKey(RequestException $exception): bool
    {
        $status = $exception->response?->status();

        return $status === 401
            || ($status === 429 && $exception->response->json('error.code') === 'rate_limited');
    }

    /**
     * The status codes that indicate the gateway or its upstream is transiently unavailable.
     *
     * @return list<int>
     */
    protected function overloadedStatusCodes(): array
    {
        return [500, 502, 503, 504, 520, 522, 524, 529];
    }
}
