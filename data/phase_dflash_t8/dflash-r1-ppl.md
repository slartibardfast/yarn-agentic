=============================== NCCL main communicator initialized
| model                          |       size |     params | backend    | ngl | type_k | type_v |    sm | ts           |         spec |  nd |     acc% |       ma | prompt                   |    ppl_out |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | ----: | ------------ | -----------: | --: | -------: | -------: | ------------------------ | ---------: | ------------: | ---------------: |
    Device 0:  110.812 MiB
    Device 1:  110.812 MiB
| qwen35 27B F16                 |  30.51 GiB |    51.25 B | CUDA       | 999 |   q4_0 |   q4_0 | graph | 1.00/1.00    |       dflash |   4 |    0.000 |     0.00 | prompt-quicksort.txt     |          - |         pp512 |   1041.51 ± 0.00 |
    Device 0:  110.812 MiB
    Device 1:  110.812 MiB
[New LWP 398258]
[New LWP 398255]
[New LWP 398254]
[New LWP 398253]
[New LWP 398252]
[New LWP 398251]
[New LWP 398248]
[New LWP 398247]
[New LWP 398246]
[New LWP 398245]
[New LWP 398244]
[New LWP 398236]

This GDB supports auto-downloading debuginfo from the following URLs:
  <https://debuginfod.archlinux.org>
Enable debuginfod for this session? (y or [n]) [answered N; input not from terminal]
Debuginfod has been disabled.
To make this setting permanent, add 'set debuginfod enabled off' to .gdbinit.
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
0x00007fd0f9c9ef32 in ?? () from /usr/lib/libc.so.6
#0  0x00007fd0f9c9ef32 in ?? () from /usr/lib/libc.so.6
#1  0x00007fd0f9c9339c in ?? () from /usr/lib/libc.so.6
#2  0x00007fd0f9c933e4 in ?? () from /usr/lib/libc.so.6
#3  0x00007fd0f9d0367f in wait4 () from /usr/lib/libc.so.6
#4  0x00007fd0ffd3ec79 in ggml_abort () from /opt/llm/build-dflash/ggml/src/libggml.so
#5  0x00007fd0ffd5108b in ggml_new_tensor_impl.constprop () from /opt/llm/build-dflash/ggml/src/libggml.so
#6  0x00007fd0ffd81e6e in ggml_view_2d () from /opt/llm/build-dflash/ggml/src/libggml.so
#7  0x00007fd10fbd96d6 in delta_net::build_layer_attn_linear_core(ggml_context*, ggml_cgraph*, ggml_tensor*, ggml_tensor*, ggml_tensor*, unsigned int, bool, int, std::function<void (ggml_tensor*, char const*, int)> const&, bool) const () from /opt/llm/build-dflash/src/libllama.so
#8  0x00007fd10fbdad98 in delta_net::build_layer_attn_linear(ggml_context*, ggml_cgraph*, ggml_tensor*, ggml_tensor*, int, std::function<void (ggml_tensor*, char const*, int)> const&) const () from /opt/llm/build-dflash/src/libllama.so
#9  0x00007fd10fc163c0 in llm_build_context::build_qwen35() () from /opt/llm/build-dflash/src/libllama.so
#10 0x00007fd10fbbac3c in llm_build_context::llama_build_graph(llama_context&, llama_batch const&, bool, int) () from /opt/llm/build-dflash/src/libllama.so
#11 0x00007fd10fac6ccc in llama_decode_internal(llama_context&, llama_batch) () from /opt/llm/build-dflash/src/libllama.so
#12 0x00007fd10fac947a in llama_decode () from /opt/llm/build-dflash/src/libllama.so
#13 0x000055cba124d1e6 in compute_ppl_of_output(llama_context*, std::vector<int, std::allocator<int> > const&, std::vector<int, std::allocator<int> > const&) ()
#14 0x000055cba124492f in main ()
[Inferior 1 (process 398235) detached]
