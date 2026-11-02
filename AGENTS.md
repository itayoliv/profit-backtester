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
that `worker_heartbeat_at` in `bt status` is less than 120 seconds old.

## The loop

1. Write a request JSON file.
2. `bt submit request.json` returns `{"job_id": "..."}`.
3. Poll `bt status --job-id <id>` every 2 to 5 seconds until `phase` is `done`
   or `failed`.
4. `bt results <id>` for the metrics, then read the `report_path` it gives you.

Every command prints JSON already; there is no `--json` flag, and passing one is an
error.

## Exit codes

| Command | 0 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| `bt submit <file>` | queued, prints `job_id` | validation error (bad schema, unknown strategy, duplicate `job_id`) | - | - |
| `bt status [--job-id ID]` | always | - | - | - |
| `bt results <id>` | done, prints the summary | - | still queued or running | failed, or unknown job |
| `bt compare <id> [<id> ...]` | ranked table printed | bad `--sort-by` | at least one still queued or running | none of them finished |
| `bt add-strategy <file> --author NAME` | installed, prints the record | a check failed, see `details` | - | - |
| `bt remove-strategy <name>` | removed | built in or never added | - | - |

Exit 1 means an unexpected internal error on any command; report it rather than
retrying in a loop.

## Request schema

Unknown fields are rejected. Only `tickers`, `timeframe`, `start`, `end` and
`strategy` are required; everything else below shows its default.

```json
{
  "job_id": "optional, defaults to TICKER_START_END_STRATEGY_<8 hex>",
  "source": "agent",
  "tickers": ["AAPL"],
  "timeframe": "1Day",
  "start": "2023-01-01",
  "end": "2024-01-01",
  "strategy": "sma_cross",
  "params": {"fast": 10, "slow": 50},
  "params_grid": null,
  "engine": "vectorbt",
  "fill": "close",
  "stop": null,
  "capital": {
    "initial_cash": 500,
    "sizing": "percent_equity",
    "size_value": 1.0,
    "commission": 0.0,
    "slippage_bps": 1,
    "allow_short": false,
    "allow_fractional": false
  },
  "rules": {"max_trades_per_day": null, "no_overnight": false, "indicators_rth_only": false},
  "filters": {"weekdays": null, "regime": null}
}
```

`examples/` holds ready requests: `request.json` (sma_cross), `macd_cross.json`,
`intraday_daytrading.json` (rules and filters) and `sweep_macd.json` (a sweep).

- `tickers`: symbols, each backtested on its own with the same settings.
- `timeframe`: `1Min`, `5Min`, `15Min`, `1Hour`, `1Day`. Indicator lengths in
  `params` are counted in bars of this timeframe, so `slow: 26` means 26 days on
  `1Day` and 26 hours on `1Hour`.
- `start`, `end`: the backtest window, `YYYY-MM-DD`. `start` must be before `end`.
  Indicators need warm-up bars inside the window, so the first signal comes
  `slow`-ish bars after `start`.
- `source`: a free-text tag for who launched the job. It does not affect results.
- `capital.initial_cash`: starting account value in dollars. Defaults to 500.
- `capital.allow_fractional`: when false (the default), orders round down to whole
  shares, so a $500 account buying a $176 stock gets 2 shares. When true, vectorbt
  buys fractions of a share. The backtesting engine always buys whole shares, so a
  request that sets this with `engine: backtesting` is rejected.
- `capital.commission`: per-side fee as a fraction of trade value (`0.001` is 0.1%
  on entry and again on exit).
- `capital.slippage_bps`: every fill is this many basis points worse than the bar
  close. Defaults to 1, which is 0.01% per fill. Keep it the same across every run
  you compare.
- `capital.allow_short`: lets the engine open shorts, but only for strategies that
  emit short signals. `sma_cross` and `macd_cross` are long only.
- `rules.max_trades_per_day`: keeps only the first N entry signals per session;
  `null` means no cap. Exits are never capped.
- `rules.no_overnight`: forces an exit on the last bar of each session and blocks
  entries on that bar.
- `rules.indicators_rth_only`: default false, and that default does not change
  results. Extended-hours bars are removed before indicators either way (they
  must not move an average or fill a trade). False then runs the strategy once
  on the continuous regular-session series, so a moving average can use the
  previous session. True rebuilds indicators on each regular session alone
  (09:30–16:00 ET by bar open), which changes entry times. `no_overnight` still
  flattens on the last bar of the session.
- `fill`: `close` (default) fills on the signal bar's close, which is what every
  existing job does. `next_open` fills at the next bar's open. With
  `no_overnight`, an exit that would land in the next session is filled at the
  last regular-session bar's close instead, so the position is flat before the
  session ends and is not held overnight.
- `stop`: `null` (default) leaves exits to the strategy. `{"type": "fixed",
  "dollars": 5}` or `{"type": "atr", "atr_mult": 2, "atr_period": 14}` attaches
  a protective stop on both engines, measured from the fill (below a long, above
  a short). It is live after the entry, including when `no_overnight` flattens
  at the close. A hit is `exit_reason` `stop`, ahead of `eod_close`. ATR is
  Wilder's ATR and uses only bars up to the decision bar.
- `engine`: `vectorbt` or `backtesting`. Both honour the same rules and report the
  same metric definitions, so results are comparable.
- `sizing`: `percent_equity` (a fraction of equity, so `size_value` must be at most 1),
  `fixed_cash`, or `fixed_shares`.
- `strategy` must be registered. Run `bt strategies` before choosing one: it
  gives each strategy's `description` (when it enters and exits) and its
  `params_schema`, where every parameter has a `description`, `default`, `minimum`
  and `maximum`, plus `origin` (`builtin` or `custom`) and, for custom ones, the
  `author` who added it. Read the descriptions; do not guess what a parameter does
  from its name. `params` are validated against that schema, and omitted ones take the
  default.
- `params_grid` runs a sweep: `{"fast": [8, 12], "slow": [21, 26]}` makes one child
  job per combination, each with that combination merged over `params`. A
  parameter goes in `params` or `params_grid`, not both. Combinations the strategy
  rejects (such as `fast >= slow`) are skipped and listed in `bt submit`'s
  `skipped_combinations`. Tickers x combinations may not exceed 200.
- `filters.weekdays` is a list with Monday as 0. `filters.regime` looks like
  `{"symbol": "SPY", "timeframe": "1Day", "indicator": "sma", "period": 200, "condition": "above"}`.
- When `job_id` is omitted it becomes `TICKER_START_END_STRATEGY_<8 hex>`, for
  example `AAPL_2023-01-01_2024-01-01_sma_cross_a1b2c3d4`. Several tickers are joined
  with `-`. Pass your own `job_id` to override. The 8 hex characters keep a repeat
  of the same ticker, window and strategy unique.
- Multi-ticker requests become one child job per ticker, named `<job_id>-<TICKER>`.
  A sweep names its children `<job_id>-<combo>` (for example
  `sweep1-fast8_slow21`), or `<job_id>-<TICKER>-<combo>` with several tickers.
  Poll the parent: its status carries a `children` list, and its results aggregate the
  per-run metrics, list `failed_children`, and carry a ranked `comparison`. A batch is
  `done` once at least one run succeeds, and `failed` only if all of them fail.

## Writing a new strategy

You may write your own strategy when the registered ones cannot express an idea.

1. Copy `examples/strategy_template.py` to a new file outside `workspace/` and
   `custom_strategies/` (for example `drafts/my_idea.py`) and edit it. One file
   defines one strategy and calls `register()` once. Import only from
   `backtester.strategies`, `pandas`, `numpy` and `pydantic`.
2. `bt add-strategy drafts/my_idea.py --author <your name>`. Always pass your own
   name (Grok passes `--author grok`): it is recorded against the strategy and
   shown in `bt strategies`, every summary (`strategy_origin`), every `report.md`
   and `bt compare` (`strategy_author`). Exit 0 prints the installed record, exit 2
   prints `details` saying what to fix.
3. It is then usable like any other: `bt strategies`, then `bt submit`.

`bt add-strategy` checks that:

- `name` is lowercase snake_case and not a built-in; `description` says when it
  enters and exits; every parameter has a docstring.
- The default `params` validate.
- `generate_signals` runs on synthetic daily and 15-minute bars, returns `Signals`
  on the input frame's index, and does not modify the bars.
- It does not look ahead: signals on the early bars must not change when later bars
  are added. Use only the current and earlier rows (`rolling`, `ewm`, `shift(1)`),
  never `shift(-1)`, centred windows or whole-series statistics.

To change a strategy you added, edit your draft and rerun with `--replace`. Never
edit files in `custom_strategies/` directly: a strategy whose file no longer
matches what was checked stops loading and shows up in `bt strategies` under
`load_errors`. `bt remove-strategy <name>` uninstalls one; built-ins cannot be
removed.

## Phases

`queued` -> `fetching_data` -> `filtering` -> `running_strategy` -> `applying_rules`
-> `reporting` -> `done` | `failed`

Only `done` and `failed` are terminal.

## Results

`bt results <id>` prints the summary plus absolute paths to `report.md`,
`trades.csv` and `equity.parquet`. The summary always carries `total_return`,
`sharpe`, `max_drawdown`, `win_rate`, `trade_count`, `rule_violations`, `params`,
`rules`, `data_coverage`, `engine` and `strategy`.

Returns are fractions, not percentages. `max_drawdown` is negative. Sharpe is
annualised with no risk-free rate. `rule_violations` should always be 0; anything
else is an engine bug worth reporting, and the metrics should not be trusted.

`report.md` has five sections: Setup, Metrics, Rules, Data coverage and Notes. Read
Notes: it is where per-run caveats appear.

## Comparing runs

To answer "which settings work best", run several backtests and compare them.

1. **Change one thing at a time.** Keep tickers, timeframe, window, engine, capital,
   rules and filters identical, and vary only what you are testing. `bt compare`
   lists what differs between runs under `varied`, and warns when it is more than
   parameters.
2. **Prefer one sweep over many requests.** Put the values to try in `params_grid`.
   For anything a grid cannot express (two strategies, two engines), submit separate
   requests with your own descriptive `job_id`s, such as `AAPL_macd_f8_s21`.
3. **Submit everything first, then poll.** The worker runs jobs in parallel.
4. **Compare with `bt compare <id> [<id> ...] --sort-by sharpe`.** A sweep or batch
   parent expands into its runs. Each row carries `rank`, `ticker`, `params`,
   `total_return`, `sharpe`, `max_drawdown`, `return_over_drawdown`, `win_rate`,
   `trade_count` and `rule_violations`. Runs with rule violations always rank last.
5. **Rank on risk-adjusted numbers.** Use `sharpe` or `return_over_drawdown`, not
   `total_return` alone. Treat runs under 10 trades as noise; `warnings` flags them.
   Ignore a parent's headline metrics for a sweep: they average different settings.
6. **Check the winner out of sample.** Choose settings on one window, then rerun only
   the winner on a later window it has not seen, and report both. A best-of-many
   result on a single window is usually overfit.
7. **Report** a table with one row per run (params and the metrics above), the
   winner, its out-of-sample result, and any warnings.

## Error codes

| Code | Meaning | What to do |
| --- | --- | --- |
| `validation_error` | the request did not pass the schema | fix the request from `details` and resubmit |
| `no_data` | the ticker returned no bars for that window | drop that ticker or widen the window; other tickers in a batch keep going |
| `feed_not_permitted` | `ALPACA_FEED=sip` without a subscription | tell a human; do not retry |
| `stale_worker` | the worker died mid-job twice | resubmitting once is fine |
| `all_children_failed` | every run in a batch or sweep failed | read the children's own errors first |

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
- Hourly bars are built from 15-minute bars aligned to the 09:30 open, like
  TradingView's exchange chart: they open at 09:30, 10:30, ... 15:30, and the 15:30 bar
  covers only the last half hour. A forced end-of-day exit fills at the 16:00 close.
- `no_overnight` on the `1Day` timeframe closes every position on its entry bar, which
  is rarely what you want.

## Out of scope

No live or paper trading, no multi-asset portfolios (each ticker is backtested
independently), and no survivorship-bias-free universes.
