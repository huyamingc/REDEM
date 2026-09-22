#!/usr/bin/env python3
"""
Update a cover letter / highlights .docx in place from a plain-text draft
=========================================================================
Type:           FIG
Experiment:     gen_cover_letter_docx
Purpose:        Write or replace the body of a minimal, valid
                WordprocessingML .docx from plain text (cover letter or
                highlights list). Uses only the standard library
                (zipfile + xml.sax.saxutils) because no docx toolchain is
                available on this host: pandoc / LibreOffice are absent,
                Word COM automation is unavailable, and pip installs are
                blocked by the sandbox. No third-party dependency is added
                to the project environment.

Workflow (docx is the single source of truth since 2026-09-17; the
sibling .txt files were removed from the repository):

    1. Read the current .docx body (paragraph text).
    2. Edit that text (this script prints it; or keep the edited text in
       a scratch file outside the repo while drafting).
    3. Pipe the final text back through this script to rewrite the docx:

    python gen_cover_letter_docx.py --text-file draft.txt paper_d COVER_LETTER
    python gen_cover_letter_docx.py --print paper_d Highlights   # dump current text

    With no --text-file/--print, the script prints the current body and
    exits (read-only).

The document keeps its original zip structure (styles, sectPr) and only
the word/document.xml body paragraphs are replaced.
=====================================================================
"""
import argparse
import os
import re
import sys
import zipfile
from xml.sax.saxutils import escape

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

DOC_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def read_docx_text(path):
    """Extract paragraph texts from word/document.xml."""
    with zipfile.ZipFile(path) as z:
        xml = z.read('word/document.xml').decode('utf-8')
    paras = []
    for m in re.finditer(r'<w:p[ >].*?</w:p>|<w:p/>', xml, re.S):
        block = m.group(0)
        runs = re.findall(r'<w:t(?:\s[^>]*)?>(.*?)</w:t>', block, re.S)
        text = ''.join(runs)
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        paras.append(text)
    return paras


def write_docx_text(path, paragraphs):
    """Replace body paragraphs, preserving everything else in the zip."""
    with zipfile.ZipFile(path) as z:
        xml = z.read('word/document.xml').decode('utf-8')
        names = z.namelist()

    def _repl(m):
        return '<w:p><w:r><w:t xml:space="preserve">%s</w:t></w:r></w:p>' % escape(m.group(1))

    # Substitute every paragraph block (or empty <w:p/>) with the new sequence.
    new_blocks = ''.join(
        '<w:p><w:r><w:t xml:space="preserve">%s</w:t></w:r></w:p>' % escape(p) if p.strip()
        else '<w:p/>'
        for p in paragraphs
    )
    body_start = xml.find('<w:body>') + len('<w:body>')
    sect_pos = xml.rfind('<w:sectPr')
    close_pos = xml.rfind('</w:body>')
    if sect_pos != -1 and sect_pos > body_start:
        body_end = sect_pos
    elif close_pos != -1 and close_pos > body_start:
        body_end = close_pos
    else:
        raise SystemExit('ERROR: unexpected document.xml structure in %s' % path)
    new_xml = xml[:body_start] + new_blocks + xml[body_end:]

    tmp = str(path) + '.tmp'
    with zipfile.ZipFile(path) as zin, \
            zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == 'word/document.xml':
                data = new_xml.encode('utf-8')
            zout.writestr(item, data)
    import os as _os
    _os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(
        description='Print or update the text body of a submission .docx '
                    '(docx is the single source of truth).')
    ap.add_argument('paper_dir', help='paper directory under the repo root (e.g. paper_d)')
    ap.add_argument('base', help='document base name without extension '
                                 '(e.g. COVER_LETTER, Highlights)')
    ap.add_argument('--print', dest='show', action='store_true',
                    help='print the current body text and exit (default action)')
    ap.add_argument('--text-file', metavar='FILE',
                    help='rewrite the docx body from this UTF-8 text file '
                         '(blank line = paragraph break)')
    args = ap.parse_args()

    docx = os.path.join(args.paper_dir, args.base + '.docx')
    if not os.path.exists(docx):
        raise SystemExit('ERROR: not found: %s' % docx)

    if args.text_file is None:
        paras = read_docx_text(docx)
        for p in paras:
            print(p)
        return

    with open(args.text_file, encoding='utf-8') as f:
        raw = f.read()
    blocks = re.split(r'\n\s*\n', raw.strip())
    paragraphs = [' '.join(ln.strip() for ln in b.split('\n')) for b in blocks]
    paragraphs = [p for p in paragraphs if p or True]
    write_docx_text(docx, paragraphs)
    print('[OK] updated %s (%d paragraphs)' % (docx, len(paragraphs)))


if __name__ == '__main__':
    main()
