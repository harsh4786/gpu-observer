# Observer core

A `no_std + alloc` crate containing the fixed event layout, bounded SPSC
transport, nonblocking emitter, and dense correlation index.

Heap allocation is limited to ring construction and cold correlation output.
The counting-allocator integration test enforces zero allocation for
steady-state push/pop.
