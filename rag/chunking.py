"""Splits a document into retrieval chunks that each make sense on their own.

Good chunks are the biggest lever on answer accuracy: a chunk that cuts a procedure
in half, or a 15-token fragment with no context, retrieves badly and misleads the LLM.
Strategy (DESIGN.md section 6):

1. Sections: split on Markdown headings and remember each section's heading path,
   e.g. ["Leave Policy", "Parental Leave", "Eligibility"]. "#" lines inside code
   blocks are not treated as headings.
2. Merge tiny sections: neighbouring sections under the same topic that are shorter
   than `min_tokens` are combined (e.g. Parental Leave's Eligibility, Duration and
   How to Apply become one chunk), so a chunk holds a complete answer.
3. Blocks: a section body is split into structural blocks (paragraph, list, table,
   code). Each block is kept whole when it fits, so steps and tables are never cut.
   A lead-in line ending in ":" stays attached to the list or table it introduces.
4. Oversized blocks are split at natural boundaries: sentences for prose, items for
   lists, rows for tables (repeating the header row), lines for code. Words are only
   split as a last resort, and tokens never.
5. Overlap: when a section needs several chunks, the next chunk starts with the last
   sentences (up to `overlap` tokens) of the previous one, so a fact on the boundary
   is retrievable from both sides.
6. Every chunk starts with its heading path, so it is understandable in isolation and
   the embedding captures the topic as well as the text.
"""

import re
from dataclasses import dataclass, field

import tiktoken

# cl100k_base is the tokenizer used by OpenAI's text-embedding-3 models, so our
# token counts match what the embedding model actually sees.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Bump this whenever the chunking logic changes. It is stored with the index, so an
# index built by older logic is detected and the user is told to rebuild.
CHUNKER_VERSION = 2

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")  # "## Title" (optional closing #'s)
_FENCE_RE = re.compile(r"^\s*(```|~~~)")                 # start/end of a code block
_LIST_RE = re.compile(r"^(\s*)(?:[-*+•]|\d+[.)])\s+")     # "- item", "* item", "1. item", "2) item"
_TABLE_RE = re.compile(r"^\s*\|")                         # Markdown table row
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}")          # the |---|---| line under a table header
# A sentence ends at . ! or ? followed by whitespace and a capital letter, digit,
# quote or bracket. Requiring the capital avoids splitting "e.g. the" or "approx. 5".
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class TextChunk:
    text: str     # what gets embedded and sent to the LLM (includes the heading prefix)
    title: str    # top-level heading, usually the document title
    section: str  # heading path below the title; merged sections are joined with " | "


@dataclass
class _Section:
    path: list[str]  # heading path, title first
    body: str


@dataclass
class _Block:
    kind: str       # "text", "list", "table" or "code"
    text: str
    lead: str = ""  # lead-in sentence such as "To reset your password:" (kept with lists/tables)

    @property
    def full(self) -> str:
        return f"{self.lead}\n{self.text}" if self.lead else self.text


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


# ------------------------------------------------------------------ 1. sections


def _split_sections(text: str, fallback_title: str) -> list[_Section]:
    sections: list[_Section] = []
    path: list[str] = []
    body: list[str] = []
    in_fence = False

    for line in text.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence  # "#" lines inside ``` blocks are code comments, not headings
        match = None if in_fence else _HEADING_RE.match(line)
        if match and match.group(2):
            if "\n".join(body).strip():
                sections.append(_Section(list(path), "\n".join(body)))
            body = []
            level = len(match.group(1))  # number of "#" characters = heading depth
            # Keep only the parent headings above this level, then add this heading.
            path = path[: level - 1] + [match.group(2)]
        else:
            body.append(line)

    if "\n".join(body).strip():
        sections.append(_Section(list(path), "\n".join(body)))

    # Plain text or PDFs with no headings: use the file name as the title.
    return [_Section(s.path or [fallback_title], s.body) for s in sections]


# ------------------------------------------------------------------ 2. merge tiny sections


def _topic(section: _Section) -> tuple[str, ...]:
    # Sections are only merged within the same topic: same title + same top-level heading.
    return tuple(section.path[:2])


def _section_label(path: list[str]) -> str:
    # Text directly under the title (before any sub-heading) is labelled "Introduction".
    return " > ".join(path[1:]) or "Introduction"


def _merge_small_sections(sections: list[_Section], min_tokens: int, budget: int) -> list[list[_Section]]:
    groups: list[list[_Section]] = []
    group_tokens = 0
    for s in sections:
        tokens = count_tokens(s.body) + count_tokens(s.path[-1])  # body + its heading line
        if groups:
            last = groups[-1]
            # A group holding only the title-level preamble (intro text before the first
            # sub-heading) may join the first topic; otherwise topics must match.
            is_preamble = len(last) == 1 and len(last[0].path) == 1
            same_topic = is_preamble or _topic(last[-1]) == _topic(s)
            too_small = group_tokens < min_tokens or tokens < min_tokens
            if same_topic and too_small and group_tokens + tokens <= budget:
                last.append(s)
                group_tokens += tokens
                continue
        groups.append([s])
        group_tokens = tokens
    return groups


def _common_path(paths: list[list[str]]) -> list[str]:
    common = paths[0]
    for p in paths[1:]:
        n = 0
        while n < min(len(common), len(p)) and common[n] == p[n]:
            n += 1
        common = common[:n]
    return common or paths[0][:1]


def _group_body(group: list[_Section], common: list[str]) -> str:
    if len(group) == 1:
        return group[0].body
    parts = []
    for s in group:
        rest = s.path[len(common):]
        # Keep each merged section's own heading inside the text so the LLM still sees
        # where one topic ends and the next begins.
        heading = f"{' > '.join(rest)}:" if rest else ""
        parts.append(f"{heading}\n{s.body.strip()}".strip())
    return "\n\n".join(parts)


# ------------------------------------------------------------------ 3. blocks


def _is_continuation(line: str) -> bool:
    return bool(line.strip()) and line[:1].isspace()  # indented line = part of the item above


def _split_blocks(body: str) -> list[_Block]:
    lines = body.split("\n")
    blocks: list[_Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if _FENCE_RE.match(line):  # code block: everything up to the closing fence
            j = i + 1
            while j < len(lines) and not _FENCE_RE.match(lines[j]):
                j += 1
            blocks.append(_Block("code", "\n".join(lines[i : j + 1])))
            i = j + 1
            continue

        if _TABLE_RE.match(line):  # table: consecutive "|" rows
            j = i
            while j < len(lines) and _TABLE_RE.match(lines[j]):
                j += 1
            blocks.append(_Block("table", "\n".join(lines[i:j])))
            i = j
            continue

        if _LIST_RE.match(line):  # list: items, indented continuations, and blank lines between items
            j = i + 1
            while j < len(lines):
                if _LIST_RE.match(lines[j]) or _is_continuation(lines[j]):
                    j += 1
                    continue
                if not lines[j].strip():
                    k = j
                    while k < len(lines) and not lines[k].strip():
                        k += 1
                    # A blank line only ends the list if no further item follows it.
                    if k < len(lines) and (_LIST_RE.match(lines[k]) or _is_continuation(lines[k])):
                        j = k
                        continue
                break
            blocks.append(_Block("list", "\n".join(l for l in lines[i:j] if l.strip())))
            i = j
            continue

        # Paragraph: runs until a blank line or the start of another block type.
        j = i
        while (
            j < len(lines)
            and lines[j].strip()
            and not (j > i and (_LIST_RE.match(lines[j]) or _TABLE_RE.match(lines[j]) or _FENCE_RE.match(lines[j])))
        ):
            j += 1
        # Soft-wrapped lines (common in PDFs and Markdown) are joined into one line of prose.
        blocks.append(_Block("text", " ".join(l.strip() for l in lines[i:j])))
        i = j

    # Attach a lead-in like "Follow these steps:" to the list/table that follows it.
    merged: list[_Block] = []
    for b in blocks:
        prev = merged[-1] if merged else None
        if prev and prev.kind == "text" and prev.text.endswith(":") and b.kind in ("list", "table"):
            merged[-1] = _Block(b.kind, b.text, lead=prev.text)
        else:
            merged.append(b)
    return merged


# ------------------------------------------------------------------ 4. splitting oversized blocks


def _units(block: _Block) -> list[str]:
    """The natural pieces a block may be split into."""
    if block.kind == "text":
        return [s for s in _SENTENCE_RE.split(block.text) if s.strip()]
    if block.kind == "list":
        items: list[str] = []
        for line in block.text.split("\n"):
            match = _LIST_RE.match(line)
            if (match and not match.group(1)) or not items:  # a new top-level (unindented) item
                items.append(line)
            else:
                items[-1] += "\n" + line  # nested item or continuation stays with its parent
        return items
    return block.text.split("\n")  # table rows or code lines


def _pack(units: list[str], budget: int, sep: str) -> list[str]:
    """Greedily groups units into pieces of at most `budget` tokens."""
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        tokens = count_tokens(unit)
        if tokens > budget:  # a single unit is too big: split it into words (last resort)
            if current:
                pieces.append(sep.join(current))
                current, current_tokens = [], 0
            pieces.extend(_split_words(unit, budget))
            continue
        # +1 per unit is a safe allowance for the separator token.
        if current and current_tokens + tokens + 1 > budget:
            pieces.append(sep.join(current))
            current, current_tokens = [], 0
        current.append(unit)
        current_tokens += tokens + 1
    if current:
        pieces.append(sep.join(current))
    return pieces


def _split_words(text: str, budget: int) -> list[str]:
    words = text.split(" ")
    if any(count_tokens(w) > budget for w in words):
        # Pathological input (e.g. a huge URL with no spaces): only now cut by tokens.
        tokens = _ENCODING.encode(text)
        return [_ENCODING.decode(tokens[i : i + budget]) for i in range(0, len(tokens), budget)]
    return _pack(words, budget, " ")


def _split_block(block: _Block, budget: int) -> list[_Block]:
    units = _units(block)
    header = ""
    if block.kind == "table" and len(units) > 1:
        # Repeat the header row (and its |---| line) in every piece so each piece's
        # columns stay labelled.
        n = 2 if _TABLE_SEP_RE.match(units[1]) else 1
        header, units = "\n".join(units[:n]), units[n:]
    # The lead-in and table header are repeated in every piece, so reserve room for them.
    repeated = "\n".join(p for p in (block.lead, header) if p)
    piece_budget = max(20, budget - count_tokens(repeated) - 1)
    sep = " " if block.kind == "text" else "\n"
    pieces = _pack(units, piece_budget, sep)
    return [_Block(block.kind, "\n".join(p for p in (header, piece) if p), lead=block.lead) for piece in pieces]


# ------------------------------------------------------------------ 5. packing with overlap


def _overlap_tail(block: _Block, overlap: int) -> str:
    """The last whole sentences of a prose block that fit within `overlap` tokens."""
    if block.kind != "text" or overlap <= 0:
        return ""  # repeating half a list or table would confuse more than help
    tail: list[str] = []
    tokens = 0
    for sentence in reversed(_units(block)):
        t = count_tokens(sentence)
        if tokens + t > overlap:
            break
        tail.insert(0, sentence)
        tokens += t
    return " ".join(tail)


def _chunk_body(body: str, budget: int, overlap: int) -> list[str]:
    blocks: list[_Block] = []
    for b in _split_blocks(body):
        if count_tokens(b.full) > budget:
            # Leave room for the overlap that may be prepended to each piece.
            blocks.extend(_split_block(b, budget - overlap))
        else:
            blocks.append(b)

    chunks: list[list[_Block]] = []
    current: list[_Block] = []
    current_tokens = 0
    for b in blocks:
        tokens = count_tokens(b.full) + 2  # +2 for the blank line between blocks
        if current and current_tokens + tokens > budget:
            chunks.append(current)
            tail = _overlap_tail(current[-1], overlap)
            tail_tokens = count_tokens(tail) + 2 if tail else 0
            # Start the next chunk with the overlap, unless it would push the chunk over budget.
            if tail and tail_tokens + tokens <= budget:
                current, current_tokens = [_Block("text", tail)], tail_tokens
            else:
                current, current_tokens = [], 0
        current.append(b)
        current_tokens += tokens
    if current:
        chunks.append(current)

    return ["\n\n".join(b.full for b in chunk) for chunk in chunks]


# ------------------------------------------------------------------ public entry point


def chunk_document(
    text: str, fallback_title: str, size: int, overlap: int, min_tokens: int = 0
) -> list[TextChunk]:
    """Splits one document into chunks of at most about `size` tokens.

    size:       maximum tokens per chunk, including the heading prefix
    overlap:    tokens of trailing sentences repeated at the start of the next chunk
    min_tokens: sections smaller than this are merged with neighbours on the same topic
    """
    chunks: list[TextChunk] = []
    sections = _split_sections(text, fallback_title)
    # Merging uses a conservative budget so a merged group still fits with its prefix.
    merge_budget = size - 40

    for group in _merge_small_sections(sections, min_tokens, merge_budget):
        common = _common_path([s.path for s in group])
        prefix = " > ".join(common)  # breadcrumb shown at the top of the chunk
        section = " | ".join(_section_label(s.path) for s in group)
        # The prefix uses part of the token budget, so subtract it from the body limit.
        budget = max(50, size - count_tokens(prefix) - 2)

        for body in _chunk_body(_group_body(group, common), budget, overlap):
            chunks.append(TextChunk(f"{prefix}\n\n{body}", common[0], section))

    return chunks
