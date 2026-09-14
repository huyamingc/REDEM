#!/usr/bin/env python3
"""
Regenerate a cover letter / highlights .docx from a plain-text source
=====================================================================
Type:           EXPLORE
Experiment:     gen_cover_letter_docx
Purpose:        Emit a minimal, valid WordprocessingML .docx from a UTF-8
                plain-text source (cover letter or highlights list). Uses
                only the standard library (zipfile + xml.sax.saxutils)
                because no docx toolchain is available on this host:
                pandoc / LibreOffice are absent, Word COM automation is
                unavailable, and pip installs are blocked by the sandbox.
                No third-party dependency is added to the project
                environment.

Usage:
    python gen_cover_letter_docx.py                    # legacy: paper_d/COVER_LETTER_D_NC.txt -> paper_d/COVER_LETTER.docx
    python gen_cover_letter_docx.py paper_e COVER_LETTER   # paper_e/COVER_LETTER.txt -> paper_e/COVER_LETTER.docx
    python gen_cover_letter_docx.py paper_e Highlights     # paper_e/Highlights.txt -> paper_e/Highlights.docx
    python gen_cover_letter_docx.py paper_f COVER_LETTER   # paper_f/COVER_LETTER.txt -> paper_f/COVER_LETTER.docx

Note:           The .txt remains the source of truth; re-run this script
                after every edit to the .txt.
=====================================================================
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent.parent

CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '</Types>'
)

RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
    '</Relationships>'
)

DOC_OPEN = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:body>'
)
DOC_CLOSE = (
    '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
    '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>'
    '</w:sectPr></w:body></w:document>'
)


def para(text):
    """One body paragraph; empty text yields an empty paragraph."""
    if not text.strip():
        return '<w:p/>'
    return (
        '<w:p><w:r><w:t xml:space="preserve">'
        + escape(text)
        + '</w:t></w:r></w:p>'
    )


def paragraphs_from_lines(lines):
    """Join consecutive non-empty source lines into one Word paragraph.

    The .txt is hard-wrapped; without this, each visual line becomes its
    own paragraph and mid-sentence breaks read as line breaks in Word.
    """
    out = []
    buf = []
    for ln in lines:
        if not ln.strip():
            if buf:
                out.append(' '.join(buf))
                buf = []
            out.append('')
        else:
            buf.append(ln.strip())
    if buf:
        out.append(' '.join(buf))
    return out


def resolve_paths(paper_dir, base):
    """Map (paper_dir, base) to (txt_path, docx_path).

    With no arguments the legacy Paper D mapping is kept:
    paper_d/COVER_LETTER_D_NC.txt -> paper_d/COVER_LETTER.docx.
    """
    if paper_dir is None:
        return (HERE / 'paper_d' / 'COVER_LETTER_D_NC.txt',
                HERE / 'paper_d' / 'COVER_LETTER.docx')
    return (HERE / paper_dir / f'{base}.txt',
            HERE / paper_dir / f'{base}.docx')


def main():
    ap = argparse.ArgumentParser(
        description='Regenerate a .docx from a UTF-8 plain-text source.')
    ap.add_argument('paper_dir', nargs='?', default=None,
                    help='paper directory under the repo root (e.g. paper_e); '
                         'omit for the legacy Paper D cover letter')
    ap.add_argument('base', nargs='?', default=None,
                    help='source base name without extension '
                         '(e.g. COVER_LETTER, Highlights)')
    args = ap.parse_args()
    if args.paper_dir is not None and args.base is None:
        ap.error('base name required when paper_dir is given')

    txt_path, out_path = resolve_paths(args.paper_dir, args.base)
    if not txt_path.exists():
        raise SystemExit(f'ERROR: source not found: {txt_path}')
    lines = txt_path.read_text(encoding='utf-8').splitlines()
    paras = paragraphs_from_lines(lines)
    body = ''.join(para(p) for p in paras)
    document = DOC_OPEN + body + DOC_CLOSE

    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', CONTENT_TYPES)
        z.writestr('_rels/.rels', RELS)
        z.writestr('word/document.xml', document)

    print(f'[OK] wrote {out_path}  ({out_path.stat().st_size} bytes, '
          f'{len(paras)} paragraphs from {len(lines)} source lines)')


if __name__ == '__main__':
    main()
