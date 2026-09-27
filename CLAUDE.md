# Contributor guidance

Read docs/MAINTENANCE.md and the relevant section of docs/DESIGN.md before
changing pipeline behavior. Use uv for Python and run tests appropriate to the
change. Keep public API documentation focused on evidence, coordinate and seat
contracts, failure behavior and effects.

Mahjong rules are hard constraints: conserve tile counts and distinguish red
fives; never turn missing observations into proof of absence. Ground truth comes
from human review. Do not label video tiles by eye with a language model.

Ship code, package assets, small intentional regression fixtures and docs.
Keep videos, models, full annotation corpora, caches and outputs local. Preserve
reviewer answers in labels/ when invalidating work/ caches. Runtime assets live
in src/video2tenhou/assets; writable paths use VIDEO2TENHOU_HOME.

A legal reconstruction is not necessarily well evidenced. Keep ambiguous draws
and unseen indicators visible in review; replay every hand before export. New
performance paths must preserve evidence and be measured on the same workload.
Do not claim full-video speedups from cache hits or isolated microbenchmarks.
