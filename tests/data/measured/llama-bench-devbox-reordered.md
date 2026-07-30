Same run as llama-bench-devbox.md, columns in a different order.
llama-bench only prints the columns that vary between runs, so position is not
stable across two invocations -- the parser has to match on header names.

|          test |                  t/s | threads |     params |       size | backend    | model                          |
| ------------: | -------------------: | ------: | ---------: | ---------: | ---------- | ------------------------------ |
|         pp512 |         25.48 ± 0.31 |      10 |     8.03 B |   4.58 GiB | CPU        | llama 8B Q4_K - Medium         |
|         tg128 |          6.07 ± 0.03 |      10 |     8.03 B |   4.58 GiB | CPU        | llama 8B Q4_K - Medium         |
