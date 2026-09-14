# Highlights notes -- Paper F

Source of record: `paper_f/PAPER_F.tex`.
Upload artifact: `paper_f/Highlights.txt` (bullets only).

Requirement (Neurocomputing Guide for Authors): 3-5 bullet points, each at
most 85 characters including spaces; upload as a separate editable file with
"Highlights" in the file name.

Alignment with the manuscript: bullet 1-2 = objective vs range (`s50`,
C1/C2); bullet 3 = `s51` Gate-C-topk (C4b); bullet 4 = `s52` soft routing
(C5); bullet 5 = `s66` external selective-SSM, scoped (not a SoTA claim).

Corrected 2026-09-14: the earlier bullet 5 read "Gate beats a selective-SSM
baseline (~1.34x params) 10/10 seeds". The 1.34x figure counted the frozen
external arm's 24,864 *untrained* host parameters as trainable; in trained
parameters F's gate arm holds 3.0x MORE than that arm, so the parameter
ratio must not be used as a Highlights selling point. See the Limitations
paragraph of `PAPER_F.tex`.
