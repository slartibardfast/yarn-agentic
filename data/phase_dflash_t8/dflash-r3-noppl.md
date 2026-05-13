=============================== NCCL main communicator initialized
| model                          |       size |     params | backend    | ngl | type_k | type_v |    sm | ts           |         spec |  nd |     acc% |       ma | prompt                   |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | ----: | ------------ | -----------: | --: | -------: | -------: | ------------------------ | ------------: | ---------------: |
    Device 0:  110.812 MiB
    Device 1:  110.812 MiB
| qwen35 27B F16                 |  30.51 GiB |    51.25 B | CUDA       | 999 |   q4_0 |   q4_0 | graph | 1.00/1.00    |       dflash |   4 |    0.000 |     0.00 | prompt-quicksort.txt     |         pp512 | 1246.83 ± 171.54 |
    Device 0:  110.812 MiB
    Device 1:  110.812 MiB
[New LWP 398139]
[New LWP 398136]
[New LWP 398135]
[New LWP 398134]
[New LWP 398133]
[New LWP 398132]
[New LWP 398106]
[New LWP 398105]
[New LWP 398104]
[New LWP 398103]
[New LWP 398102]
[New LWP 398094]

This GDB supports auto-downloading debuginfo from the following URLs:
  <https://debuginfod.archlinux.org>
Enable debuginfod for this session? (y or [n]) [answered N; input not from terminal]
Debuginfod has been disabled.
To make this setting permanent, add 'set debuginfod enabled off' to .gdbinit.
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
0x00007fcc4b49ef32 in ?? () from /usr/lib/libc.so.6
#0  0x00007fcc4b49ef32 in ?? () from /usr/lib/libc.so.6
#1  0x00007fcc4b49339c in ?? () from /usr/lib/libc.so.6
#2  0x00007fcc4b4933e4 in ?? () from /usr/lib/libc.so.6
#3  0x00007fcc4b50367f in wait4 () from /usr/lib/libc.so.6
#4  0x00007fcc5153ec79 in ggml_abort () from /opt/llm/build-dflash/ggml/src/libggml.so
#5  0x00007fcc5155108b in ggml_new_tensor_impl.constprop () from /opt/llm/build-dflash/ggml/src/libggml.so
#6  0x00007fcc51581e6e in ggml_view_2d () from /opt/llm/build-dflash/ggml/src/libggml.so
#7  0x00007fcc613d96d6 in delta_net::build_layer_attn_linear_core(ggml_context*, ggml_cgraph*, ggml_tensor*, ggml_tensor*, ggml_tensor*, unsigned int, bool, int, std::function<void (ggml_tensor*, char const*, int)> const&, bool) const () from /opt/llm/build-dflash/src/libllama.so
#8  0x00007fcc613dad98 in delta_net::build_layer_attn_linear(ggml_context*, ggml_cgraph*, ggml_tensor*, ggml_tensor*, int, std::function<void (ggml_tensor*, char const*, int)> const&) const () from /opt/llm/build-dflash/src/libllama.so
#9  0x00007fcc614163c0 in llm_build_context::build_qwen35() () from /opt/llm/build-dflash/src/libllama.so
#10 0x00007fcc613bac3c in llm_build_context::llama_build_graph(llama_context&, llama_batch const&, bool, int) () from /opt/llm/build-dflash/src/libllama.so
#11 0x00007fcc612c6ccc in llama_decode_internal(llama_context&, llama_batch) () from /opt/llm/build-dflash/src/libllama.so
#12 0x00007fcc612c947a in llama_decode () from /opt/llm/build-dflash/src/libllama.so
#13 0x0000564788e5cabe in main ()
[Inferior 1 (process 398093) detached]
