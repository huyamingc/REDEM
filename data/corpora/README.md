# Real-text corpus for the Paper D real-text benchmark (S23)

Two public-domain English texts, downloaded from Project Gutenberg
(https://www.gutenberg.org), header/footer license blocks stripped, body
only. Both are public domain in the United States (Project Gutenberg
License applies to the electronic versions).

| File | Source (Gutenberg ebook) | Size (chars) |
|---|---|---|
| `alice.txt` | Lewis Carroll, *Alice's Adventures in Wonderland* (ebook #11) | ~148k |
| `dickens.txt` | Charles Dickens, *A Tale of Two Cities* (ebook #98) | ~774k |

Usage: `scripts/s23_ssm_p4_realtext.py` builds a two-source character-level
streaming domain-drift task (alternating windows, known switch instants,
s18 protocol) with a 32-symbol vocabulary (31 most frequent chars + UNK).

License: Project Gutenberg License; see
https://www.gutenberg.org/policy/license.html .
