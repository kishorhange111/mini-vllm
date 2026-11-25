| workload | policy | max seqs | KV blocks | prefix cache | tok/s | latency mean / p90 (s) | TTFT (s) | steps | mean batch | preempt | prefix-hit tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | static | 32 | 4096 | on | 356 | 49.3 / 88.5 | 47.7 | 8480 | 4.1 | 0 | 0 |
| random | continuous | 32 | 4096 | on | 1468 | 12.7 / 21.9 | 9.9 | 1218 | 28.9 | 0 | 0 |
| random | continuous | 128 | 4096 | on | 1838 | 12.1 / 18.0 | 4.4 | 436 | 80.7 | 0 | 0 |
| random | continuous | 128 | 1024 | on | 1568 | 12.2 / 20.8 | 8.5 | 887 | 39.7 | 84 | 48 |
| shared-prefix | continuous | 128 | 4096 | off | 1488 | 15.3 / 23.2 | 6.1 | 450 | 79.8 | 21 | 0 |
| shared-prefix | continuous | 128 | 4096 | on | 1677 | 13.4 / 20.4 | 4.6 | 425 | 84.5 | 0 | 64,000 |
