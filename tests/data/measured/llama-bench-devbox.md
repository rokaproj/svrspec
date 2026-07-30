$ llama-bench -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -t 10 -p 512 -n 128
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      10 |         pp512 |         25.48 ± 0.31 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      10 |         tg128 |          6.07 ± 0.03 |

build: 8f3a1c2d (4785)
