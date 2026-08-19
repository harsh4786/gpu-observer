# EXP-0002 analysis

This run prevented a false positive. The patch and bridge were loaded, but the semantic ring stayed empty because the instrumented method was not the active dispatch path.

The important architectural correction is that process topology and scheduling topology are separate questions. Scheduler decisions and CUDA calls still occur in one EngineCore process, but that process can execute them through a queued asynchronous method. Future support must instrument both paths and define end-of-step as future completion, not queue insertion.

EXP-0003 changes exactly one server variable, disables async scheduling, and tests the synchronous boundary.
