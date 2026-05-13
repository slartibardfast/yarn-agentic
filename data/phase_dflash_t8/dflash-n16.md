=============================== NCCL main communicator initialized
| model                          |       size |     params | backend    | ngl | type_k | type_v |    sm | ts           |         spec |  nd |     acc% |       ma | prompt                   |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | ----: | ------------ | -----------: | --: | -------: | -------: | ------------------------ | ------------: | ---------------: |
    Device 0:  110.812 MiB
    Device 1:  110.812 MiB
| qwen35 27B F16                 |  30.51 GiB |    51.25 B | CUDA       | 999 |   q4_0 |   q4_0 | graph | 1.00/1.00    |       dflash |   4 |    0.000 |     0.00 | prompt-quicksort.txt     |         pp512 |   1027.51 ± 0.00 |
    Device 0:  110.812 MiB
    Device 1:  110.812 MiB
| qwen35 27B F16                 |  30.51 GiB |    51.25 B | CUDA       | 999 |   q4_0 |   q4_0 | graph | 1.00/1.00    |       dflash |   4 |    0.429 |     1.71 | prompt-quicksort.txt     |          tg16 |      0.83 ± 0.00 |

build: 8d05ea69 (4592)
