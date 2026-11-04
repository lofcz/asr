# Checkpoint comparison viewer

Routes /evaluations, /evaluations/100h and /evaluations/150h serve the new
viewer. Completed future milestones are discovered from periodic/history.json.
Uncompleted milestones return 404 instead of silently showing a different run.
The old /evaluations/100h/comparison.json remains available for compatibility.

/api/comparisons reads the approved reference and saved hypotheses on CPU only.
No inference, paid API calls, label edits, or training changes are involved.
The vanilla output is reused. Reference speaker turns define stable scroll
anchors; hypothesis text is aligned globally and assigned to those turns.
Raw output segments are available separately, grouped by midpoint for review.
Speaker IDs remain local to each model output.

Features: arbitrary checkpoint pair selection, timeline and clickable metric
trends, number-normalized WER/DER deltas, lexical/diacritic/number-format highlights,
fixed/new difference groups, filters, search, recurring-error table, timed audio,
segment inspection, mobile layout, and keyboard navigation ([ / ], j / k).
Switching snapshots preserves the visible reference turn and pixel offset,
expanded timing panels, filters, search and audio playback. Language switching
remembers each sample's scroll position. URL parameters preserve selected models,
language and filter. Refresh discovers newly completed evaluations.

Review spans use Python SequenceMatcher with autojunk disabled and NFC/casefold
word keys. They are not Levenshtein WER edit counts; scored metrics remain exactly
those of the existing evaluator. Punctuation and whitespace do not create word
changes. Number equivalence uses the same language-specific normalizer as scoring.
An old error changing into another overlapping error is not counted as fixed.
The recurring table groups differences by reference location and shows the first
observed wording as an example; cells navigate to that snapshot and passage.

Validation: four alignment tests; Playwright desktop/mobile checks for routes,
selection, highlights, zero-pixel scroll drift, retained timing expansions,
filters/search, seeking/playback, language switching, no horizontal overflow,
byte ranges and no JavaScript errors. Additional check verifies repeated j/k
navigation advances between passages. Screenshots saved alongside this file.

Revision review correction: default highlighting now compares the two selected
hypotheses directly, so shared gold errors remain unhighlighted. Gold comparison
is an explicit alternate mode. Pairwise highlighting uses removed/changed and
added/changed colors, not correctness colors. Difference navigation follows the
active mode. Reference-error category filters select gold mode explicitly.
Charts default to the selected snapshot interval with actual percentage axes;
the dashed reference is the selected left checkpoint. Full-history charts remain
available. Both controls persist in the URL. Regression browser checks verified
shared-error turns have no highlights, identical snapshots have no highlights,
chart bounds change with the comparator, and scroll anchors stay fixed.
