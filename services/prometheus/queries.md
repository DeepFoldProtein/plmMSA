# plmMSA PromQL cookbook

Copy-paste reference for the Prometheus instance that ships under the
`observability` compose profile. Open the Graph UI at
`http://localhost:${PROMETHEUS_HOST_PORT:-9091}/graph` and paste any of
these into the query box.

> Prometheus has no server-side "saved queries" — bookmarks are URLs.
> Once a query is on the page, the address bar is a shareable link.

Sections:

- [Health at a glance](#health-at-a-glance)
- [API performance](#api-performance)
- [Worker / pipeline](#worker--pipeline)
- [GPU telemetry](#gpu-telemetry)
- [Capacity / saturation](#capacity--saturation)
- [Cross-cutting troubleshooting](#cross-cutting-troubleshooting)
- [PromQL working tips](#promql-working-tips)

## Health at a glance

```promql
# Are all six scrape targets up? (1 = up, 0 = down)
up
```

```promql
# Per-service request rate, last minute.
sum by (service) (rate(plmmsa_http_requests_total[1m]))
```

```promql
# 5xx rate per service — first place to look when something feels off.
sum by (service) (rate(plmmsa_http_requests_total{status=~"5.."}[5m]))
```

```promql
# Currently-firing alerts (alongside /alerts).
ALERTS{alertstate="firing"}
```

## API performance

```promql
# api p95 latency by route, 5-minute window.
histogram_quantile(0.95,
  sum by (le, route) (
    rate(plmmsa_http_request_duration_seconds_bucket{service="api"}[5m])))
```

```promql
# api p99 latency overall (a single number for SLO tracking).
histogram_quantile(0.99,
  sum by (le) (
    rate(plmmsa_http_request_duration_seconds_bucket{service="api"}[5m])))
```

```promql
# In-flight requests right now, per route — quick saturation glance.
plmmsa_http_in_flight_requests
```

```promql
# Rate of /v2/msa POST submissions (real workload entry-point).
rate(plmmsa_http_requests_total{service="api",route="/v2/msa",method="POST"}[5m])
```

## Worker / pipeline

```promql
# Job throughput by terminal status, last 5 min.
rate(plmmsa_worker_jobs_processed_total[5m])
```

```promql
# Cumulative outcomes since worker start.
plmmsa_worker_jobs_processed_total
```

```promql
# Pipeline duration p50 / p95 / p99.
histogram_quantile(0.50, rate(plmmsa_worker_pipeline_duration_seconds_bucket[10m]))
histogram_quantile(0.95, rate(plmmsa_worker_pipeline_duration_seconds_bucket[10m]))
histogram_quantile(0.99, rate(plmmsa_worker_pipeline_duration_seconds_bucket[10m]))
```

```promql
# Live queue depth (sampled every 5 s by the worker loop).
plmmsa_worker_queue_depth
```

```promql
# Failed-job ratio over the window — a clean stack should be ~0.
sum(rate(plmmsa_worker_jobs_processed_total{status="failed"}[10m]))
  /
sum(rate(plmmsa_worker_jobs_processed_total[10m]))
```

## GPU telemetry

```promql
# Utilisation per card (%).
DCGM_FI_DEV_GPU_UTIL
```

```promql
# Memory used / free per card (MiB) — sanity check vs settings.toml device pins.
DCGM_FI_DEV_FB_USED
DCGM_FI_DEV_FB_FREE
```

```promql
# Memory utilisation as a fraction (0 – 1).
DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)
```

```promql
# Temperature (°C), power (W), per card.
DCGM_FI_DEV_GPU_TEMP
DCGM_FI_DEV_POWER_USAGE
```

```promql
# ECC errors — should always be zero. Non-zero = a card is going bad.
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL
```

```promql
# Hottest card observed in the last 5 minutes.
max_over_time(DCGM_FI_DEV_GPU_TEMP[5m])
```

## Capacity / saturation

```promql
# Free memory headroom on each GPU (MiB) — leading indicator of OOM.
min by (gpu) (DCGM_FI_DEV_FB_FREE)
```

```promql
# 5xx error budget burn — fraction of requests returning 5xx.
sum(rate(plmmsa_http_requests_total{status=~"5.."}[5m]))
  /
sum(rate(plmmsa_http_requests_total[5m]))
```

```promql
# Time spent in pipeline vs available worker concurrency.
# (Useful when you scale workers horizontally — each adds 1.0 to the
# capacity.)
sum(rate(plmmsa_worker_pipeline_duration_seconds_sum[10m]))
```

## Cross-cutting troubleshooting

Two complementary queries to graph side-by-side — paste each into its
own row in the Graph UI.

```promql
# Embedding latency p95.
histogram_quantile(0.95,
  sum by (le) (
    rate(plmmsa_http_request_duration_seconds_bucket{service="embedding"}[1m])))
```

```promql
# GPU 0 utilisation.
DCGM_FI_DEV_GPU_UTIL{gpu="0"}
```

When latency and utilisation rise together → GPU pressure. When
latency rises but utilisation stays low → upstream stall (Redis,
shard store, network).

```promql
# Result-cache hits leave a fingerprint as very-fast worker pipelines.
# A spike of the p10 dropping to ~0.01s usually means hits are landing.
histogram_quantile(0.10, rate(plmmsa_worker_pipeline_duration_seconds_bucket[10m]))
```

```promql
# Total HTTP traffic across the whole stack.
sum(rate(plmmsa_http_requests_total[1m]))
```

## PromQL working tips

- **Range vector windows.** `[1m]` is twitchy and noisy; `[5m]` is the
  steady-state workhorse for `rate()`; `[10m]+` for histograms with
  sparse traffic. Smaller window = fewer samples = noisier.
- **`rate()` is for counters, `irate()` is also for counters but uses
  the last 2 samples in the window** (sharper, but less stable).
- **Don't `rate()` a gauge.** Gauges (`plmmsa_worker_queue_depth`,
  `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_GPU_TEMP`) — use them directly,
  or wrap with `avg_over_time` / `max_over_time` for smoothing.
- **`histogram_quantile` needs `_bucket` series and `rate()` underneath.**
  Pattern: `histogram_quantile(0.95, rate(<metric>_bucket[5m]))`.
- **`sum by (label)`** before `rate()` works; **`rate()` then `sum`**
  also works. Same arithmetic. Pick whichever reads better.
- **Filter early.** `{service="embedding",route="/embed/bin"}` cuts
  the cardinality before aggregation — faster queries and clearer
  labels.
- **Bookmarkable.** Once a query is on screen, copy the URL.
  Prometheus encodes the full query+range into the address bar.
