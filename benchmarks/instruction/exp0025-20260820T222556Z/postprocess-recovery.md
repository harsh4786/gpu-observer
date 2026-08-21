# Post-processing harness recovery

The original experiment process exited after capture because it passed cuobjdump's `SASS text section 1722` to `--function-index`. That option expects a symbol-table function index, so cuobjdump reported `Invalid ELF` even though `file` and `readelf` validated the copied module and `--list-text` had already parsed it successfully.

The raw runtime evidence was complete before this error: all nine responses reached SSE `[DONE]`, the semantic trace had zero gaps/loss, and the device snapshot contained 760 launch records and 14,363 zero-drop samples. The run was recovered without rerunning inference by selecting the active mangled function name from `device-probe.summary.tsv`, asking cuobjdump for that function, filtering the result to the executed `sm_120` image, and running the offline join. The join passed every identity, PC, geometry, ownership, loss, and accounting check.

The reproducible runner now uses this corrected selection method. The failed interpretation is retained in `vllm_C.text-sections.txt`; raw capture files were not modified.
