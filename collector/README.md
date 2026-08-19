# Collector

The collector is the cold `std` boundary. It validates JSONL, optionally makes
an exact raw copy before parsing, translates names to numeric IDs, invokes the
compact core, and writes a report atomically.

Production ingestion will consume one ring per source and preserve compact raw
records before aggregation. JSONL is a debugging/compatibility format, not the
probe transport.
