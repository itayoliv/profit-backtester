# Employee agent guide

**Use the `bt` CLI and nothing else.** Never read, write or clean up `workspace/` by
hand: the worker owns it, and editing it corrupts in-flight jobs. Everything you need
is on stdout as a single JSON object. Human-readable logs go to stderr.

## Setup (once)

```bash
pip install -e .
cp .env.example .env     # fill in ALPACA_API_KEY and ALPACA_SECRET_KEY
bt worker                # leave running in its own terminal
```

Jobs make no progress unless a worker is running. Confirm one is alive by checking
that `worker_heartbeat_at` in `bt status --json` is less than 120 seconds old.

## The loop

1. Write a request JSON file.
2. `bt submit request.json` returns `{"job_id": "..."}`.
3. Poll `bt status --job-id <id> --json` every 2 to 5 seconds until `phase` is `done`
   or `failed`.
4. `bt results <id> --json` for the metrics, then read the `report_path` it gives you.

## Exit codes

| Command | 0 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| `bt submit <file>` | queued, prints `job_id` | validation error (bad schema, unknown strategy, duplicate `job_id`) | - | - |
| `bt status [--job-id ID]` | always | - | - | - |
| `bt results <id>` | done, prints the summary | - | still queued or running | failed, or unknown job |

Exit 1 means an unexpected internal error on any command; report it rather than
retrying in a loop.

## Request schema

Unknown fields are rejected. Only `tickers`, `timeframe`, `start`, `end` and
`strategy` are required; everything else below shows its default.

```json
{
  "job_id": "optional-uuid",
  "source": "agent",
  "tickers": ["AAPL"],
  "timeframe": "1Day",
  "start": "2023-01-01",
  "end": "2024-01-01",
  "strategy": "sma_cross",
  "params": {"fast": 10, "slow": 50},
  "engine": "vectorbt",
  "capital": {
    "initial_cash": 100000,
    "sizing": "percent_equity",
    "size_value": 1.0,
    "commission": 0.0,
    "slippage_bps": 5,
    "allow_short": false
  },
  "rules": {"max_trades_per_day": null, "no_overnight": false},
  "filters": {"weekdays": null, "regime": null}
}
```

- `timeframe`: `1Min`, `5Min`, `15Min`, `1Hour`, `1Day`.
- `engine`: `vectorbt` or `backtesting`. Both honour the same rules and report the
  same metric definitions, so results are comparable.
- `sizing`: `percent_equity` (a fraction of equity, so `size_value` must be at most 1),
  `fixed_cash`, or `fixed_shares`.
- `strategy` must be registered. Run `bt strategies --json` for the list and each
  one's parameter schema. `params` are validated against that schema.
- `filters.weekdays` is a list with Monday as 0. `filters.regime` looks like
  `{"symbol": "SPY", "timeframe": "1Day", "indicator": "sma", "period": 200, "condition": "above"}`.
- Multi-ticker requests become one child job per ticker, named `<job_id>-<TICKER>`.
  Poll the parent: its status carries a `children` list, and its results aggregate the
  per-ticker metrics and list `failed_children`. A batch is `done` once at least one
  ticker succeeds, and `failed` only if all of them fail.

## Phases

`queued` -> `fetching_data` -> `filtering` -> `running_strategy` -> `applying_rules`
-> `reporting` -> `done` | `failed`

Only `done` and `failed` are terminal.

## Results

`bt results <id> --json` prints the summary plus absolute paths to `report.md`,
`trades.csv` and `equity.parquet`. The summary always carries `total_return`,
`sharpe`, `max_drawdown`, `win_rate`, `trade_count`, `rule_violations`, `params`,
`rules`, `data_coverage`, `engine` and `strategy`.

Returns are fractions, not percentages. `max_drawdown` is negative. Sharpe is
annualised with no risk-free rate. `rule_violations` should always be 0; anything
else is an engine bug worth reporting, and the metrics should not be trusted.

`report.md` has five sections: Setup, Metrics, Rules, Data coverage and Notes. Read
Notes: it is where per-run caveats appear.

## Error codes

| Code | Meaning | What to do |
| --- | --- | --- |
| `validation_error` | the request did not pass the schema | fix the request from `details` and resubmit |
| `no_data` | the ticker returned no bars for that window | drop that ticker or widen the window; other tickers in a batch keep going |
| `feed_not_permitted` | `ALPACA_FEED=sip` without a subscription | tell a human; do not retry |
| `stale_worker` | the worker died mid-job twice | resubmitting once is fine |
| `all_children_failed` | every ticker in a batch failed | read the children's own errors first |

## When a job looks stuck

Jobs heartbeat at least every 5 seconds. A job with no heartbeat for 120 seconds is
requeued once automatically, and fails with `stale_worker` the second time, so you do
not need to intervene.

If a job stays non-terminal **and** `worker_heartbeat_at` is more than 120 seconds
old, the worker itself is down. Report that to a human and stop. Do not resubmit in a
loop: the queue is already holding your job and it will run when a worker returns.

## Data notes

- Bars come from Alpaca, split and dividend adjusted, stored in UTC, cached locally.
  Re-running the same window costs nothing: only missing date ranges are downloaded.
- Session logic uses `America/New_York`. For intraday timeframes only regular-session
  bars (09:30 to 16:00 ET, by bar open time) are tradable.
- Hourly bars are labelled by open time, so the 09:00 bar straddles the open and is
  excluded. The first tradable hourly bar is 10:00 and the last is 15:00; a forced
  end-of-day exit fills at the 15:00 bar's close.
- `no_overnight` on the `1Day` timeframe closes every position on its entry bar, which
  is rarely what you want.

## Out of scope

No live or paper trading, no multi-asset portfolios (each ticker is backtested
independently), and no survivorship-bias-free universes.
