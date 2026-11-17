"""Turns each supported file type into Markdown-like text.

Headings are kept as "#" lines because the chunker uses them to split the text
into sections and to label each chunk with where it came from.
"""

import re
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".html", ".htm"}


class LoaderError(Exception):
    """Raised when a file can't be read or contains no text."""


def load_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".md", ".txt"):
        # errors="replace" keeps going if the file has a few bad bytes instead of crashing.
        text = path.read_text(encoding="utf-8", errors="replace")
    elif ext == ".pdf":
        text = _load_pdf(path)
    elif ext == ".docx":
        text = _load_docx(path)
    elif ext in (".html", ".htm"):
        text = _load_html(path)
    else:
        raise LoaderError(f"unsupported file type {ext}")

    text = clean_text(text)
    if not text.strip():
        # Scanned PDFs are images, so they have no extractable text. OCR is out of scope.
        raise LoaderError("no extractable text (scanned PDF or empty file?)")
    return text


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported here so the other formats work even without pypdf

    try:
        reader = PdfReader(str(path))
        # PDFs have no real headings, so pages are joined with blank lines (= paragraph breaks).
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise LoaderError(f"could not read PDF: {exc}") from exc


def _load_docx(path: Path) -> str:
    from docx import Document

    try:
        doc = Document(str(path))
    except Exception as exc:
        raise LoaderError(f"could not read Word file: {exc}") from exc

    lines = []
    for para in doc.paragraphs:
        style = (para.style.name or "").lower() if para.style else ""
        # Word styles "Heading 1", "Heading 2"... become "#", "##"... so section structure survives.
        match = re.match(r"heading (\d)", style)
        if match and para.text.strip():
            lines.append("#" * int(match.group(1)) + " " + para.text.strip())
        else:
            lines.append(para.text)
        lines.append("")  # blank line between paragraphs
    # Tables are flattened to "cell | cell" rows and kept together as one paragraph.
    for table in doc.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        lines.append("\n".join(rows))
        lines.append("")
    return "\n".join(lines)


def _load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    # Remove page furniture (menus, scripts, footers) that would pollute the chunks.
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    blocks = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "tr"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name.startswith("h"):
            blocks.append("#" * int(el.name[1]) + " " + text)  # <h2> -> "## "
        elif el.name == "li":
            blocks.append("- " + text)
        else:
            blocks.append(text)
    return "\n\n".join(blocks)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # normalise Windows/Mac line endings
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        # Drop lines that are only a page number, like "12" or "Page 3 of 10".
        if re.fullmatch(r"(page\s+)?\d+(\s+of\s+\d+)?", stripped, flags=re.IGNORECASE):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text)  # collapse runs of blank lines to one paragraph break
