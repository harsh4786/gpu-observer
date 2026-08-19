# Aborted overhead ladder

This ladder was stopped before accepting the first SASS measurement.

The requested device-event capacity was 1,048,576 records, but the subscriber's
compiled maximum was 262,144. Its parser silently fell back to 65,536 records.
That would have overflowed during the 64-request workload and invalidated the
SASS comparison.

Completed clean, scheduler, and packed points remain raw diagnostic artifacts
only. They must not be combined with the corrected ladder. The bounded maximum
was raised to 1,048,576 64-byte records (64 MiB), and the corrected runner now
fails closed unless the live server log reports the exact requested capacity.
