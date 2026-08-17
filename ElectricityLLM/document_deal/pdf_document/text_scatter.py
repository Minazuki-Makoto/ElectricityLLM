"""Extract a PDF into font-aware hierarchical JSON chunks."""

from __future__ import annotations
from AiChat.GLM_Chat import chat, get_client
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from ElasticSearch import embed

import fitz

from pdf_deal_functions import (
    find_pictures_and_save,
    find_table,
    get_page_image_rects,
    is_caption_text,
    is_page_number_text,
    remove_all_images,
    vector2pic,
)


class PDFExtract:
    """Pre-process a PDF and split its text according to font hierarchy."""

    CHAPTER_RE = re.compile(
        r"^\s*(?:第\s*[一二三四五六七八九十百千万0-9]+\s*[章节篇部]\s*\S+|"
        r"[一二三四五六七八九十百千万]+、\s*\S+)"
    )
    NUMBERED_HEADING_RE = re.compile(
        r"^\s*(\d+(?:\.\d+){0,3})(?!\d)(?:\s*[、．.]\s*|\s+|"
        r"(?=[A-Za-z\u4e00-\u9fff]))(\S.*)$"
    )
    PAREN_HEADING_RE = re.compile(
        r"^\s*[（(]([一二三四五六七八九十百千万]+|\d+)[）)]\s*\S+"
    )
    APPENDIX_RE = re.compile(
        r"^\s*(?:附\s*录(?:\s*[A-Za-zＡ-Ｚａ-ｚ0-9一二三四五六七八九十]*)|"
        r"Appendix(?:\s+[A-Za-z0-9]+)?)\s*\S*",
        re.IGNORECASE,
    )
    # These sections and everything after them are auxiliary material rather
    # than handbook正文, so exclude_appendix=True treats them as a hard stop.
    TERMINAL_SECTION_TITLES = frozenset({
        "缩略词列表",
        "图示清单",
        "参考资料",
        "微信公众号技术文章集锦",
    })
    EXCLUDED_FRONT_SECTION_TITLES = frozenset({"编写说明"})
    FRONT_MATTER_NOTE_RE = re.compile(
        r"^\s*附[：:].*(?:微信公众(?:号|平台)|技术文章集锦).*$"
    )
    FRONT_MATTER_DATE_LABEL_RE = re.compile(
        r"^\s*(?:发布|出版)日期\s*[：:]?\s*$"
    )
    FRONT_MATTER_DATE_VALUE_RE = re.compile(
        r"^\s*\d{4}\s*(?:[/.-]\s*\d{1,2}\s*[/.-]\s*\d{1,2}|"
        r"年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?)\s*$"
    )
    PUBLICATION_INFO_START_RE = re.compile(
        r"^\s*(?:图书在版编目(?:[（(]?\s*CIP\s*[）)]?)?数据|"
        r"CIP\s*数据|中国版本图书馆\s*CIP\s*数据).*$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        pdf_file_address,
        cache_address,
        bucket_name,
        title,
        minio_endpoint,
        minio_access_key,
        minio_secret_key,
        secure=True,
    ):
        self.pdf_file_address = Path(pdf_file_address)
        self.cache_address = Path(cache_address)
        self.bucket_name = str(bucket_name).strip()
        if not self.bucket_name:
            raise ValueError("bucket_name 不能为空")
        # The MinIO bucket is shared by all documents. JSON title and
        # chunk_id keep the actual document title instead.
        self.title = str(title or "").strip() or self.pdf_file_address.stem
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.secure = secure
        self._lines_cache: Optional[List[Dict[str, Any]]] = None
        self._body_size_cache: Optional[float] = None
        self._repeated_margin_texts_cache: Optional[set] = None
        self._toc_pages_cache: Optional[set] = None
        self._document_title_line_cache: Optional[Dict[str, Any]] = None
        self._detected_title_cache: Optional[str] = None
        self._source_watermark_phrases: set = set()
        self._toc_heading_map = self._load_toc_heading_map(
            self.pdf_file_address
        )

    @staticmethod
    def _load_toc_heading_map(pdf_path: Path) -> Dict[str, str]:
        """Return canonical numbered headings from the PDF outline.

        Arabic headings use their number directly (for example ``12.5``).
        Chinese chapter headings use a namespaced key (for example
        ``chapter:十二``) so a visually fragmented heading can be restored
        from the PDF's canonical outline entry.
        """
        heading_map: Dict[str, str] = {}
        ambiguous_numbers = set()
        try:
            with fitz.open(pdf_path) as document:
                toc = document.get_toc()
        except Exception:
            return heading_map

        for _level, title, _page in toc:
            title = str(title or "").strip()
            match = re.match(
                r"^\s*(\d+(?:\.\d+){0,3})(?:\s+|[、．.]\s*)(\S.*)$",
                title,
            )
            if match:
                number = match.group(1)
                heading = match.group(2).strip()
                if (
                    number in heading_map
                    and heading_map[number] != heading
                ):
                    # Broken/generated outlines sometimes reuse one number for
                    # several different sections (for example five distinct
                    # ``46.1`` entries).  Such a key cannot safely reconstruct
                    # body headings, so trust the visible heading text instead.
                    heading_map.pop(number, None)
                    ambiguous_numbers.add(number)
                elif number not in ambiguous_numbers:
                    heading_map[number] = heading
                continue

            chinese_chapter_match = re.match(
                r"^\s*([一二三四五六七八九十百千万]+)、\s*(\S.*)$",
                title,
            )
            if chinese_chapter_match:
                heading_map[
                    f"chapter:{chinese_chapter_match.group(1)}"
                ] = title
        return heading_map

    def primary_deal(self, pdf_file_address=None, cache_address=None):
        """Rasterize tables/vector graphics and use the result downstream."""
        source_path = Path(pdf_file_address or self.pdf_file_address)
        cache_dir = Path(cache_address or self.cache_address)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Capture complete repeated overlays before vector/table processing can
        # clip them into different text fragments on each page.
        original_lines = self._extract_lines()
        self._source_watermark_phrases = {
            re.sub(r"\s+", "", phrase)
            for phrase in self._repeated_margin_texts(original_lines)
            if re.sub(r"[#\s]+", "", phrase)
        }

        table_pdf = cache_dir / f"{source_path.stem}_table2pic.pdf"
        processed_pdf = cache_dir / f"{source_path.stem}_processed.pdf"

        # Convert tables first.  Their vector borders are removed here, so the
        # following generic vector pass will not process the same area twice.
        find_table(
            filename=source_path,
            output_filename=table_pdf,
        )
        vector2pic(
            filename=table_pdf,
            output_filename=processed_pdf,
        )

        # Picture upload and text extraction must read the processed PDF;
        # otherwise primary_deal() produces files but has no downstream effect.
        self.pdf_file_address = processed_pdf
        self._lines_cache = None
        self._body_size_cache = None
        self._repeated_margin_texts_cache = None
        self._toc_pages_cache = None
        self._document_title_line_cache = None
        self._detected_title_cache = None
        return processed_pdf

    def pdf_picture_clear(self, pdf_file_address=None):
        """Upload pictures using the existing project helper."""
        document_title = self.detect_document_title()
        return find_pictures_and_save(
            pdf_file_address or self.pdf_file_address,
            document_title,
            self.minio_endpoint,
            self.minio_access_key,
            self.minio_secret_key,
            self.bucket_name,
            self.secure,
            save_address=self.cache_address,
        )

    def remove_visual_content(self, pdf_file_address=None):
        """Remove uploaded images/tables and use the text-only PDF downstream."""
        source_path = Path(pdf_file_address or self.pdf_file_address)
        output_path = self.cache_address / f"{source_path.stem}_text_only.pdf"
        remove_all_images(source_path, output_path)
        self.pdf_file_address = output_path
        self._lines_cache = None
        self._body_size_cache = None
        self._repeated_margin_texts_cache = None
        self._toc_pages_cache = None
        self._document_title_line_cache = None
        self._detected_title_cache = None
        return output_path

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\u00a0", " ").replace("\u3000", " ")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def _is_formula_font(font_name: str) -> bool:
        """Return whether a PDF span uses a dedicated math/formula font."""
        normalized = re.sub(r"[^a-z0-9]+", "", str(font_name).lower())
        return any(
            marker in normalized
            for marker in (
                "cambr math",
                "cambriamath",
                "math",
                "symbol",
                "mtextra",
                "mathtype",
                "euclid",
                "stix",
                "asana",
                "xits",
                "libertinusmath",
                "latinmodernmath",
                "cmsy",
                "cmmi",
                "cmex",
                "msam",
                "msbm",
                "timesnewromanpsitalicmt",
            )
        )

    @classmethod
    def _embedding_line(cls, spans: List[Dict[str, Any]]) -> str:
        """Build embedding text without replacing equal text elsewhere."""
        span_items = []
        for index, span in enumerate(spans):
            text = str(span.get("text", ""))
            font = str(span.get("font", ""))
            normalized_font = re.sub(r"[^a-z0-9]+", "", font.lower())
            is_formula = cls._is_formula_font(font)
            previous = str(spans[index - 1].get("text", "")) if index else ""
            following = (
                str(spans[index + 1].get("text", ""))
                if index + 1 < len(spans)
                else ""
            )

            # Preserve a multiplication dot when it is glued between unit
            # components, as in kV⋅A. An isolated multiplication dot in an
            # equation remains a formula span.
            if (
                text.strip() in {"⋅", "·"}
                and previous
                and following
                and not previous[-1].isspace()
                and not following[0].isspace()
                and previous[-1].isalnum()
                and following[0].isalnum()
            ):
                is_formula = False

            # A book may use italic Times for both variables and the final
            # letter of a unit/acronym (for example, the A in 100kVA). Do not
            # replace a single italic letter glued to another alphanumeric
            # span; standalone variables still remain formula spans.
            if (
                normalized_font == "timesnewromanpsitalicmt"
                and re.fullmatch(r"[A-Za-z]", text.strip())
            ):
                neighbor_window = "".join(
                    str(item.get("text", ""))
                    for item in spans[
                        max(0, index - 2) : min(len(spans), index + 3)
                    ]
                )
                attached_to_word = bool(
                    (previous and not previous[-1].isspace() and previous[-1].isalnum())
                    or (
                        following
                        and not following[0].isspace()
                        and following[0].isalnum()
                    )
                    or bool(re.search(r"[A-Za-z]-$", previous))
                    or bool(re.match(r"^-[A-Za-z]", following))
                    or bool(re.search(r"[A-Za-z]-[A-Za-z]", neighbor_window))
                )
                if attached_to_word:
                    is_formula = False
            span_items.append((text, is_formula))

        original = cls._clean_text("".join(text for text, _ in span_items))
        has_formula_span = any(is_formula for _, is_formula in span_items)
        has_math_syntax = bool(re.search(
            r"[=+−∑∏√∫∞≤≥πωφθ]",
            original,
        ))

        # A line containing Chinese prose is normally an explanation or a
        # variable definition (for example ``Rsh 为并联电阻`` or ``错开φ角``),
        # not a standalone display equation.  Preserve the complete sentence;
        # display equations are handled on their own Chinese-free lines and
        # subsequently collapsed into one marker by _collapse_formula_blocks.
        if re.search(r"[\u4e00-\u9fff]", original):
            return original

        # Display equations are commonly fragmented across regular Times,
        # italic Times, Symbol and Euclid spans. Once a dedicated formula font
        # identifies a Chinese-free line as an equation, replace the whole
        # line so that no variable/operator fragments leak into embeddings.
        if (
            not re.search(r"[\u4e00-\u9fff]", original)
            and (has_formula_span or has_math_syntax)
        ):
            return "[公式]"

        pieces = [
            "[公式]" if is_formula and text.strip() else text
            for text, is_formula in span_items
        ]
        result = cls._clean_text("".join(pieces))
        return re.sub(r"(?:\[公式\]\s*){2,}", "[公式]", result).strip()

    @staticmethod
    def _collapse_formula_blocks(
        text_lines: List[str],
        embedding_lines: List[str],
    ) -> List[str]:
        """Collapse one PDF equation block into a single ``[公式]`` marker."""
        collapsed = list(embedding_lines)

        def is_formula_fragment(text: str, embedding_text: str) -> bool:
            compact = re.sub(r"\s+", "", text)
            if "[公式]" in embedding_text:
                return True
            if re.fullmatch(r"[（(]?\d+(?:[.-]\d+)+[）)]?", compact):
                return True
            if re.search(r"[\u4e00-\u9fff]", compact) or len(compact) > 24:
                return False
            # PDF equations frequently expose subscripts such as s, sh and br
            # as separate lines in an otherwise formula-font block.
            return bool(re.fullmatch(
                r"[0-9A-Za-z_{}()[\].,+\-*/=<>\u2212\u2211\u220f\u221a"
                r"\u222b\u221e\u2264\u2265\u03b1-\u03c9⎡-⎦]+",
                compact,
            ))

        index = 0
        while index < len(text_lines):
            if not is_formula_fragment(text_lines[index], collapsed[index]):
                index += 1
                continue
            end = index
            has_formula_anchor = False
            while end < len(text_lines) and is_formula_fragment(
                text_lines[end], collapsed[end]
            ):
                has_formula_anchor = (
                    has_formula_anchor or "[公式]" in collapsed[end]
                )
                end += 1
            if has_formula_anchor:
                collapsed[index] = "[公式]"
                for fragment_index in range(index + 1, end):
                    collapsed[fragment_index] = ""
            index = end
        return collapsed

    def _extract_lines(self) -> List[Dict[str, Any]]:
        lines: List[Dict[str, Any]] = []
        with fitz.open(self.pdf_file_address) as document:
            for page_number, page in enumerate(document, start=1):
                image_rects = get_page_image_rects(page)
                page_dict = page.get_text("dict", sort=True)
                page_fragments: List[Dict[str, Any]] = []
                for block in page_dict.get("blocks", []):
                    for line in block.get("lines", []):
                        # Preserve whitespace-only spans until after joining.
                        # A PDF may store the separator in "10.6.2.1 300MW"
                        # as its own span; dropping it makes the numbering regex
                        # consume 300 as part of the clause number.
                        spans = line.get("spans", [])
                        if not spans:
                            continue
                        if not self._clean_text(
                            "".join(str(s.get("text", "")) for s in spans)
                        ):
                            continue
                        page_fragments.append({
                            "spans": spans,
                            "bbox": tuple(line["bbox"]),
                        })

                # Some PDFs split one visual line into alternating font runs
                # represented as independent ``line`` objects.  PyMuPDF's
                # reading order can then become ``依 ... 赖 ...`` instead of
                # the left-to-right sentence.  Rebuild each visual baseline
                # before any heading detection or chunk packing.
                visual_lines = self._merge_visual_line_fragments(
                    page_fragments
                )
                for visual_line in visual_lines:
                    spans = visual_line["spans"]
                    text = self._clean_text(
                        "".join(str(s.get("text", "")) for s in spans)
                    )
                    if not text:
                        continue
                    embedding_text = self._embedding_line(spans)
                    char_count = sum(
                        max(len(s["text"].strip()), 1) for s in spans
                    )
                    size = sum(
                        float(s["size"]) * max(len(s["text"].strip()), 1)
                        for s in spans
                    ) / char_count
                    bold = any(
                        "bold" in s.get("font", "").lower()
                        or int(s.get("flags", 0)) & 16
                        for s in spans
                    )
                    lines.append(
                        {
                            "text": text,
                            "embedding_text": embedding_text,
                            "size": round(size, 2),
                            "bold": bold,
                            "page": page_number,
                            "bbox": visual_line["bbox"],
                            "page_width": float(page.rect.width),
                            "page_height": float(page.rect.height),
                            "image_rects": [
                                tuple(rect) for rect in image_rects
                            ],
                        }
                    )
        return lines

    @staticmethod
    def _merge_visual_line_fragments(
        fragments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge same-baseline PDF fragments in left-to-right order.

        The horizontal-gap split prevents unrelated columns or diagram labels
        on the same y-coordinate from being joined into one text line.
        """
        if not fragments:
            return []

        def same_baseline(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
            left_box = left["bbox"]
            right_box = right["bbox"]
            overlap = min(left_box[3], right_box[3]) - max(
                left_box[1], right_box[1]
            )
            min_height = min(
                left_box[3] - left_box[1],
                right_box[3] - right_box[1],
            )
            return (
                overlap >= min_height * 0.55
                or abs(left_box[3] - right_box[3]) <= 2.5
            )

        baseline_groups: List[List[Dict[str, Any]]] = []
        for fragment in sorted(
            fragments,
            key=lambda item: (item["bbox"][1], item["bbox"][0]),
        ):
            matching_group = next(
                (
                    group
                    for group in baseline_groups
                    if any(same_baseline(fragment, member) for member in group)
                ),
                None,
            )
            if matching_group is None:
                baseline_groups.append([fragment])
            else:
                matching_group.append(fragment)

        visual_lines: List[Dict[str, Any]] = []
        for group in baseline_groups:
            ordered = sorted(group, key=lambda item: item["bbox"][0])
            runs: List[List[Dict[str, Any]]] = []
            current_run: List[Dict[str, Any]] = []
            for fragment in ordered:
                if current_run:
                    previous = current_run[-1]
                    sizes = [
                        float(span.get("size", 0))
                        for item in (previous, fragment)
                        for span in item["spans"]
                    ]
                    typical_size = max(sizes, default=10.0)
                    horizontal_gap = (
                        fragment["bbox"][0] - previous["bbox"][2]
                    )
                    if horizontal_gap > max(18.0, typical_size * 1.8):
                        runs.append(current_run)
                        current_run = []
                current_run.append(fragment)
            if current_run:
                runs.append(current_run)

            for run in runs:
                run = sorted(run, key=lambda item: item["bbox"][0])
                spans = [span for item in run for span in item["spans"]]
                visual_lines.append({
                    "spans": spans,
                    "bbox": (
                        min(item["bbox"][0] for item in run),
                        min(item["bbox"][1] for item in run),
                        max(item["bbox"][2] for item in run),
                        max(item["bbox"][3] for item in run),
                    ),
                })

        return sorted(
            visual_lines,
            key=lambda item: (item["bbox"][1], item["bbox"][0]),
        )

    @staticmethod
    def _body_font_size(lines: Iterable[Dict[str, Any]]) -> float:
        """Estimate a conservative body size from prose, not formulas/captions."""
        all_sizes: Counter = Counter()
        prose_sizes: Counter = Counter()
        for line in lines:
            text = str(line.get("text", ""))
            effective_length = len(re.sub(r"\s+", "", text))
            size = round(float(line["size"]), 1)
            all_sizes[size] += max(effective_length, 1)
            chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
            line_width = line["bbox"][2] - line["bbox"][0]
            is_prose = (
                chinese_count >= 8
                and effective_length >= 15
                and line_width >= line["page_width"] * 0.25
                and not is_page_number_text(text)
                and not is_caption_text(text)
                and not re.search(r"[.．…·]{4,}", text)
            )
            if is_prose:
                prose_sizes[size] += min(effective_length, 120)

        if prose_sizes:
            total_weight = sum(prose_sizes.values())
            target_weight = total_weight * 0.75
            cumulative_weight = 0
            for size, weight in sorted(prose_sizes.items()):
                cumulative_weight += weight
                if cumulative_weight >= target_weight:
                    return float(size)

        if not all_sizes:
            raise ValueError("PDF 中没有可提取的文字层，请先执行 OCR。")
        return float(all_sizes.most_common(1)[0][0])

    @staticmethod
    def _near_image(line: Dict[str, Any], max_gap: float = 90.0) -> bool:
        x0, y0, x1, y1 = line["bbox"]
        line_center = (x0 + x1) / 2
        for image_rect in line["image_rects"]:
            ix0, iy0, ix1, iy1 = image_rect
            vertical_gap = min(abs(y0 - iy1), abs(iy0 - y1))
            horizontally_related = x1 >= ix0 - 30 and x0 <= ix1 + 30
            center_related = ix0 - 60 <= line_center <= ix1 + 60
            if vertical_gap <= max_gap and (horizontally_related or center_related):
                return True
        return False

    @staticmethod
    def _repeated_margin_texts(lines: List[Dict[str, Any]]) -> set:
        """Detect repeated headers, footers, and full-page watermark text."""
        page_count = max((line["page"] for line in lines), default=1)
        occurrences: Dict[str, set] = {}
        for line in lines:
            key = re.sub(r"\d+", "#", line["text"]).strip()
            if key:
                occurrences.setdefault(key, set()).add(line["page"])
        # Watermarks may alternate between odd/even pages or several variants.
        minimum = max(2, (page_count + 4) // 5)
        return {
            key for key, pages in occurrences.items()
            if len(pages) >= minimum
        }

    @staticmethod
    def _detect_toc_pages(lines: List[Dict[str, Any]]) -> set:
        """Detect contents pages even when entries have no dot leaders."""
        pages: Dict[int, List[str]] = {}
        for line in lines:
            pages.setdefault(line["page"], []).append(line["text"])

        toc_pages = set()
        toc_started = False
        toc_item_re = re.compile(
            r"^\s*\d+(?:\.\d+){0,3}(?:\s+|(?=[\u4e00-\u9fff]))\S+"
        )
        for page_number in sorted(pages):
            page_lines = pages[page_number]
            compact_lines = [
                re.sub(r"\s+", "", text)
                for text in page_lines
                if not is_page_number_text(text)
            ]
            has_toc_title = any(
                text.lower() in {"\u76ee\u5f55", "\u76ee\u6b21", "contents"}
                for text in compact_lines
            )
            item_count = sum(
                bool(toc_item_re.match(text)) and len(text) <= 100
                for text in page_lines
            )
            prose_count = sum(
                len(re.sub(r"\s+", "", text)) >= 45
                for text in page_lines
            )
            # A正文起始页 can contain four or more numbered headings as well
            # as many wrapped prose lines.  Those headings alone must not make
            # it a continuation of the contents pages.
            sentence_line_count = sum(
                len(re.sub(r"\s+", "", text)) >= 20
                and bool(re.search(r"[，。；：,;:]", text))
                for text in page_lines
            )
            dot_leader_count = sum(
                bool(re.search(r"(?:\.{4,}|·{4,})\s*\d+\s*$", text))
                for text in page_lines
            )
            continuation = (
                dot_leader_count >= 4
                or (
                    item_count >= 4
                    and prose_count <= 2
                    and sentence_line_count <= 3
                )
            )

            if has_toc_title:
                toc_started = True
                toc_pages.add(page_number)
            elif toc_started and continuation:
                toc_pages.add(page_number)
            elif toc_started:
                toc_started = False
        return toc_pages

    @staticmethod
    def _detect_leading_cover_pages(
        lines: List[Dict[str, Any]],
        body_size: float,
    ) -> set:
        """Remove a sparse first-page cover without dropping normal page one."""
        first_page_lines = [line for line in lines if line["page"] == 1]
        if not first_page_lines:
            return set()
        effective_chars = sum(
            len(re.sub(r"\s+", "", line["text"]))
            for line in first_page_lines
            if not is_page_number_text(line["text"])
        )
        has_cover_title = any(
            line["size"] >= body_size * 1.80
            and line["bbox"][1] <= line["page_height"] * 0.55
            for line in first_page_lines
        )
        return {1} if has_cover_title and effective_chars <= 180 else set()
    def _detect_trailing_noncontent_pages(
        self,
        lines: List[Dict[str, Any]],
        body_size: float,
        repeated_margin_texts: set,
        min_body_chars: int = 40,
    ) -> set:
        """Detect a consecutive tail made only of covers/promotional titles."""
        body_chars_by_page: Dict[int, int] = {}
        for line in lines:
            page = line["page"]
            body_chars_by_page.setdefault(page, 0)
            if self._is_noise_line(line, body_size, repeated_margin_texts):
                continue
            if line["size"] <= body_size * 1.12:
                body_chars_by_page[page] += len(re.sub(r"\s+", "", line["text"]))

        trailing_pages = set()
        for page in sorted(body_chars_by_page, reverse=True):
            if body_chars_by_page[page] >= min_body_chars:
                break
            trailing_pages.add(page)
        return trailing_pages

    def _is_noise_line(
        self,
        line: Dict[str, Any],
        body_size: float,
        repeated_margin_texts: set,
    ) -> bool:
        text = line["text"]
        _, y0, _, y1 = line["bbox"]
        height = line["page_height"]
        margin_key = re.sub(r"\d+", "#", text).strip()
        in_outer_margin = y0 <= height * 0.05 or y1 >= height * 0.95
        compact_text = re.sub(r"\s+", "", text)
        clipped_source_watermark = (
            line["size"] >= body_size * 1.50
            and any(
                compact_text in phrase or phrase in compact_text
                for phrase in self._source_watermark_phrases
            )
        )
        if clipped_source_watermark:
            return True
        if is_page_number_text(text):
            return True
        if margin_key in repeated_margin_texts:
            return True
        if in_outer_margin and line["size"] <= body_size * 1.05:
            return True
        if is_caption_text(text) and (
            self._near_image(line) or line["size"] <= body_size * 1.05
        ):
            return True
        return False

    def _detect_document_title(
        self,
        lines: List[Dict[str, Any]],
        body_size: float,
        repeated_margin_texts: set,
    ) -> Optional[Dict[str, Any]]:
        candidates = []
        for index, line in enumerate(lines):
            text = line["text"]
            x0, y0, x1, _ = line["bbox"]
            if line["page"] > 3 or len(text) > 80:
                continue
            if self._is_noise_line(line, body_size, repeated_margin_texts):
                continue
            if (
                line["page"] in self._toc_pages_cache
                or self.CHAPTER_RE.match(text)
                or self.NUMBERED_HEADING_RE.match(text)
            ):
                continue
            if line["size"] < body_size * 1.25:
                continue
            centered = 1.0 - min(
                abs(((x0 + x1) / 2) - line["page_width"] / 2)
                / (line["page_width"] / 2),
                1.0,
            )
            upper_score = max(0.0, 1.0 - y0 / (line["page_height"] * 0.5))
            score = line["size"] / body_size + centered * 0.8 + upper_score * 0.4
            candidates.append((score, index, line))
        return max(candidates, default=(None, None, None), key=lambda item: item[0])[2]

    def _document_analysis(self):
        """Analyze layout once so title detection always precedes other work."""
        if self._lines_cache is None:
            self._lines_cache = self._extract_lines()
            self._body_size_cache = self._body_font_size(self._lines_cache)
            self._repeated_margin_texts_cache = self._repeated_margin_texts(
                self._lines_cache
            )
            self._toc_pages_cache = self._detect_toc_pages(self._lines_cache)
            self._document_title_line_cache = self._detect_document_title(
                self._lines_cache,
                self._body_size_cache,
                self._repeated_margin_texts_cache,
            )
            detected_title = (
                self._document_title_line_cache["text"]
                if self._document_title_line_cache
                else ""
            )
            configured_title = str(self.title or "").strip()
            detected_key = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", detected_title)
            configured_key = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", configured_title)
            title_matches = (
                bool(detected_key and configured_key)
                and (
                    detected_key in configured_key
                    or configured_key in detected_key
                )
            )
            if configured_title and not title_matches:
                # The layout candidate was not the document title. Keep it in
                # the content stream so a real chapter heading is not lost.
                self._document_title_line_cache = None
                self._detected_title_cache = configured_title
            else:
                self._detected_title_cache = detected_title or configured_title
            print(f"Detected document title: {self._detected_title_cache}")
        return (
            self._lines_cache,
            self._body_size_cache,
            self._repeated_margin_texts_cache,
            self._document_title_line_cache,
            self._detected_title_cache,
            self._toc_pages_cache,
        )

    def detect_document_title(self) -> str:
        """Return the real title detected from the PDF layout."""
        return self._document_analysis()[4]

    def _heading_level(self, line: Dict[str, Any], body_size: float) -> int:
        text, size = line["text"], line["size"]
        if self.CHAPTER_RE.match(text):
            # A wrapped standard/reference title can start with text such as
            # ``第1部分：通用要求》、GB/T ...``.  It matches CHAPTER_RE's
            # ``第X部`` branch, but it is ordinary body text rather than a
            # chapter heading.  Real chapter headings are normally prominent;
            # reference continuations contain a standard identifier and a
            # closing Chinese book-title mark.
            is_standard_reference_continuation = bool(
                re.match(
                    r"^\s*第\s*[一二三四五六七八九十百千万0-9]+\s*部(?:分)?\s*[：:]",
                    text,
                )
                and "》" in text
                and re.search(
                    r"(?:GB(?:/T)?|DL(?:/T)?|NB(?:/T)?|IEC|IEEE)\s*[/.-]?\s*\d",
                    text,
                    re.IGNORECASE,
                )
                and not line["bold"]
                and size < body_size * 1.12
            )
            if is_standard_reference_continuation:
                return 0
            return 1
        numbered_match = self.NUMBERED_HEADING_RE.match(text)
        if numbered_match:
            # Do not treat chart legends or equations such as
            # "2 = hard magnetic material" as numbered headings.
            heading_text = numbered_match.group(2).strip()
            # Numbered headings must contain semantic text after the number.
            # Reject measurements/results such as "71.3%。" and
            # "47.1%，is ...", where a decimal percentage resembles a
            # hierarchical number.
            if (
                not re.search(r"[0-9A-Za-z\u4e00-\u9fff]", heading_text)
                or re.match(r"^[%％‰]", heading_text)
                or re.match(r"^[([{（]", heading_text)
            ):
                return 0
            if re.search(r"[。！？!?；;]\s*$", heading_text):
                return 0
            if (
                len(re.sub(r"\s+", "", heading_text)) < 2
                or re.match(r"^[=≈~<>≤≥+\-×÷*/]", heading_text)
            ):
                return 0
            level = min(numbered_match.group(1).count(".") + 1, 4)
            has_single_number_prefix = level == 1
            if has_single_number_prefix:
                has_dot_delimiter = bool(
                    re.match(r"^\s*\d+\s*[.．、]", text)
                )
                if (
                    not has_dot_delimiter
                    and line["bold"]
                    and size >= body_size * 1.35
                ):
                    # "1 Chapter title" without a dot is a common book style.
                    return 1
                # A prominent "1. Title" is a section; a body-size bold
                # item is a third-level heading; ordinary enumerations are text.
                if size >= body_size * 1.12:
                    level = 2
                elif line["bold"]:
                    level = 3
                else:
                    return 0
            elif (
                level >= 3
                and line["bold"]
                and size >= body_size * 1.12
            ):
                # Some source books carry stale/mistyped numeric prefixes.
                # Strong typography wins when no matching parent is present.
                level = 2
            return level
        if self.PAREN_HEADING_RE.match(text):
            # Parenthesized items inherit depth from typography: prominent
            # ones are sections, body-size bold ones are fourth-level labels.
            if size >= body_size * 1.12:
                return 2
            if line["bold"]:
                return 4
            return 0
        compact_text = re.sub(r"\s+", "", text)
        if re.fullmatch(r"第\s*\d+\s*版", text) or (len(compact_text) < 3 and compact_text not in {"前言", "序言", "绪论", "概述"}):
            return 0
        if not re.search(r"[0-9A-Za-z\u4e00-\u9fff]", compact_text):
            return 0
        if len(text) > 100 or re.search(r"[。！？!?；;]$", text):
            return 0
        if size >= body_size * 1.40:
            # Font size alone is insufficient for a chapter: diagrams and
            # equations often contain isolated large labels. Explicit chapter
            # syntax is handled above; generic large headings are sections.
            if compact_text in {"前言", "序言"}:
                return 1
            if (
                size >= body_size * 2.0
                and len(re.findall(r"[\u4e00-\u9fff]", text)) >= 2
                and (
                    line["bold"]
                    or line["bbox"][1] <= line["page_height"] * 0.25
                )
            ):
                return 1
            if not line["bold"] and len(re.findall(r"[\u4e00-\u9fff]", text)) < 3:
                return 0
            return 2
        if size >= body_size * 1.12 and (line["bold"] or len(text) <= 60):
            return 2
        return 0

    @staticmethod
    def _split_long_text(text: str, max_chars: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        sentences = [part.strip() for part in re.split(r"(?<=[。！？!?；;])\s*|\n+", text) if part.strip()]
        result, current = [], ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                if current:
                    result.append(current)
                    current = ""
                result.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
            elif current and len(current) + len(sentence) + 1 > max_chars:
                result.append(current)
                current = sentence
            else:
                current = f"{current}\n{sentence}".strip()
        if current:
            result.append(current)
        return result

    @staticmethod
    def _pack_paired_lines(
        text_lines: List[str],
        embedding_lines: List[str],
        max_chars: int,
        min_target_chars: int = 500,
        overlap_sentences: int = 1,
    ) -> List[tuple]:
        """Pack paired PDF lines without cutting ordinary sentences."""
        if len(text_lines) != len(embedding_lines):
            raise ValueError("text 与 embedding_text 行数不一致")

        sentence_end_re = re.compile(r"[。！？!?；;][”’\"）)]*\s*$")
        weak_end_re = re.compile(r"[；;：:，,][”’\"）)]*\s*$")

        # A semantic unit contains all visual PDF lines up to a real sentence
        # boundary. This reconnects sentences wrapped by the page layout.
        units: List[List[tuple]] = []
        current_unit: List[tuple] = []
        for line_text, line_embedding in zip(text_lines, embedding_lines):
            current_unit.append((line_text, line_embedding))
            if sentence_end_re.search(line_text):
                units.append(current_unit)
                current_unit = []
        if current_unit:
            units.append(current_unit)

        def pair_text(pairs: List[tuple]) -> str:
            return "\n".join(text for text, _ in pairs).strip()

        def split_oversized_unit(unit: List[tuple]) -> List[List[tuple]]:
            fragments: List[List[tuple]] = []
            remaining = list(unit)
            while remaining:
                fit_count = 0
                for count in range(1, len(remaining) + 1):
                    if len(pair_text(remaining[:count])) > max_chars:
                        break
                    fit_count = count
                if fit_count == 0:
                    fit_count = 1
                if fit_count < len(remaining):
                    # Prefer a clause boundary near the size limit before
                    # falling back to the last visual line that fits.
                    preferred = next(
                        (
                            count
                            for count in range(fit_count, 0, -1)
                            if weak_end_re.search(remaining[count - 1][0])
                        ),
                        None,
                    )
                    if preferred is not None:
                        fit_count = preferred
                fragments.append(remaining[:fit_count])
                remaining = remaining[fit_count:]
            return fragments

        normalized_units: List[List[tuple]] = []
        for unit in units:
            if len(pair_text(unit)) <= max_chars:
                normalized_units.append(unit)
            else:
                normalized_units.extend(split_oversized_unit(unit))

        packed_units: List[List[List[tuple]]] = []
        current_units: List[List[tuple]] = []
        for unit in normalized_units:
            candidate = [pair for item in current_units + [unit] for pair in item]
            if current_units and len(pair_text(candidate)) > max_chars:
                packed_units.append(current_units)
                current_units = []
            current_units.append(unit)
        if current_units:
            packed_units.append(current_units)

        # Repeat at most the final complete sentence of the previous chunk.
        # A size guard prevents an unusually long sentence from consuming most
        # of the following chunk. Heading changes call flush(), so overlap never
        # crosses structural boundaries.
        if overlap_sentences > 0:
            for index in range(1, len(packed_units)):
                previous_units = packed_units[index - 1]
                overlap_units = previous_units[-overlap_sentences:]
                overlap_pairs = [pair for unit in overlap_units for pair in unit]
                current_pairs = [
                    pair for unit in packed_units[index] for pair in unit
                ]
                overlap_text = pair_text(overlap_pairs)
                if (
                    overlap_text
                    and sentence_end_re.search(overlap_text)
                    and len(overlap_text) <= max_chars * 0.20
                    and len(pair_text(overlap_pairs + current_pairs)) <= max_chars
                ):
                    packed_units[index] = overlap_units + packed_units[index]

        # ``min_target_chars`` is intentionally a target rather than a hard
        # constraint: heading changes may legitimately produce shorter chunks,
        # while preserving complete sentences has priority over filling space.
        _ = min_target_chars
        return [
            (
                pair_text([pair for unit in chunk_units for pair in unit]),
                "\n".join(
                    embedding
                    for unit in chunk_units
                    for _, embedding in unit
                    if embedding
                ).strip(),
            )
            for chunk_units in packed_units
        ]

    @staticmethod
    def _is_redundant_heading_text(
        text: str,
        headings: Dict[int, str],
        max_effective_chars: int = 30,
    ) -> bool:
        """Return whether a short body is only a duplicate of its heading."""
        compact_text = re.sub(r"\s+", "", text or "")
        if not compact_text or len(compact_text) >= max_effective_chars:
            return False

        def normalize_heading(value: str) -> str:
            compact = re.sub(r"\s+", "", value or "")
            # Remove common numeric heading prefixes such as ``4.`` and
            # ``2.2.4`` before comparing them with the preserved heading body.
            return re.sub(
                r"^\d+(?:\.\d+)*(?:[.．、:：]|(?=[\u4e00-\u9fffA-Za-z]))",
                "",
                compact,
            )

        return any(
            compact_text == normalize_heading(heading)
            for heading in headings.values()
            if heading
        )

    @staticmethod
    def _strip_heading_number(text: str) -> str:
        """Remove structural numbering while preserving technical numbers."""
        value = str(text or "").strip()
        patterns = (
            # 第1章、第四章、第 2 节、第一篇
            r"^第\s*[一二三四五六七八九十百千万0-9]+\s*[章节篇部卷]\s*",
            # 2.2.1、3.1（多级编号必须作为整体删除）
            r"^\d+(?:\.\d+)+(?:[.．、:：]\s*|\s+|$)",
            # 3.、4．、1、（单级编号）
            r"^\d+(?:[.．、:：]\s*|\s+)",
            # （1）、(2)、（一）
            r"^[（(](?:\d+|[一二三四五六七八九十]+)[）)]\s*",
        )
        while value:
            for pattern in patterns:
                updated = re.sub(pattern, "", value, count=1).strip()
                if updated != value:
                    value = updated
                    break
            else:
                break
        return value

    @staticmethod
    def _normalize_heading_spacing(text: str) -> str:
        """Remove PDF layout spaces only when they separate Chinese chars."""
        value = str(text or "").replace("\u3000", " ").strip()
        value = re.sub(r"[ \t]+", " ", value)
        previous = None
        while value != previous:
            previous = value
            value = re.sub(
                r"(?<=[\u3400-\u4dbf\u4e00-\u9fff])\s+"
                r"(?=[\u3400-\u4dbf\u4e00-\u9fff])",
                "",
                value,
            )
        return value.strip()

    @classmethod
    def _clean_heading(cls, text: str) -> str:
        return cls._normalize_heading_spacing(cls._strip_heading_number(text))

    def split_text(
        self,
        max_chars: int = 1500,
        exclude_appendix: bool = True,
        overlap_sentences: int = 1,
    ) -> List[Dict[str, str]]:
        """Return chunks with separate title fields for heading levels 1-4."""
        (
            lines,
            body_size,
            repeated_margin_texts,
            document_title_line,
            document_title,
            toc_pages,
        ) = self._document_analysis()
        ignored_pages = self._detect_leading_cover_pages(
            lines, body_size
        ) | self._detect_trailing_noncontent_pages(
            lines,
            body_size,
            repeated_margin_texts,
        )
        headings = {1: "", 2: "", 3: "", 4: ""}
        text_lines: List[str] = []
        embedding_text_lines: List[str] = []
        chunks: List[Dict[str, str]] = []
        pending_heading_level: Optional[int] = None
        last_heading_line: Optional[Dict[str, Any]] = None
        pending_front_matter_date = False
        skipping_publication_info = False
        skipping_excluded_front_section = False
        reconstructed_heading_line: Optional[Dict[str, Any]] = None
        skipping_exercises = False
        skipping_toc = False
        toc_start_page: Optional[int] = None

        def heading_is_complete(text: str) -> bool:
            return bool(re.search(r"[。！？!?；;.]\s*$", text))

        def join_wrapped_heading(left: str, right: str) -> str:
            if not left:
                return right
            if (
                left[-1].isascii()
                and left[-1].isalnum()
                and right
                and right[0].isascii()
                and right[0].isalnum()
            ):
                return f"{left} {right}"
            return f"{left}{right}"

        def flush():
            nonlocal text_lines, embedding_text_lines
            embedding_text_lines = self._collapse_formula_blocks(
                text_lines, embedding_text_lines
            )
            paired_parts = self._pack_paired_lines(
                text_lines,
                embedding_text_lines,
                max_chars=max_chars,
                min_target_chars=min(500, max_chars),
                overlap_sentences=overlap_sentences,
            )

            for part, embedding_part in paired_parts:
                if len(re.sub(r"\s+", "", part)) < 8:
                    continue
                if self._is_redundant_heading_text(part, headings):
                    continue
                embedding_context = [document_title]
                embedding_context.extend(
                    heading
                    for heading in (
                        headings[1], headings[2], headings[3], headings[4]
                    )
                    if heading
                )
                contextual_embedding_text = "\n".join(
                    embedding_context + ([embedding_part] if embedding_part else [])
                ).strip()
                part_chunk = {
                        "title": document_title,
                        "chapter_title": headings[1],
                        "section_title": headings[2],
                        "third_title": headings[3],
                        "fourth_title": headings[4],
                        "text": part,
                        "embedding_text": contextual_embedding_text,
                        "description": "",
                    }
                chunks.append(part_chunk)
            text_lines = []
            embedding_text_lines = []

        for line in lines:
            if line["page"] in ignored_pages:
                continue
            if line["page"] in toc_pages:
                if not skipping_toc:
                    flush()
                    last_heading_line = None
                    pending_heading_level = None
                skipping_toc = True
                continue
            if skipping_toc:
                skipping_toc = False
            if reconstructed_heading_line is not None:
                same_heading_baseline = (
                    line["page"] == reconstructed_heading_line["page"]
                    and abs(line["size"] - reconstructed_heading_line["size"])
                    <= max(line["size"], reconstructed_heading_line["size"])
                    * 0.05
                    and line["bbox"][1]
                    <= reconstructed_heading_line["bbox"][3] + 2
                    and line["bbox"][3]
                    >= reconstructed_heading_line["bbox"][1] - 2
                )
                if same_heading_baseline:
                    # A mixed-font PDF may expose alternating glyphs from one
                    # visual heading as several lines on the same baseline.
                    continue
                reconstructed_heading_line = None
            compact_line = re.sub(r"\s+", "", line["text"]).lower()
            if line["page"] <= 3:
                if self.FRONT_MATTER_NOTE_RE.match(line["text"]):
                    pending_front_matter_date = False
                    continue
                if self.FRONT_MATTER_DATE_LABEL_RE.match(line["text"]):
                    pending_front_matter_date = True
                    continue
                if (
                    pending_front_matter_date
                    and self.FRONT_MATTER_DATE_VALUE_RE.match(line["text"])
                ):
                    pending_front_matter_date = False
                    continue
                pending_front_matter_date = False
            else:
                pending_front_matter_date = False
            if (
                exclude_appendix
                and self.APPENDIX_RE.match(line["text"])
                and line["size"] >= body_size * 1.15
                and not re.search(r"[.．…·]{4,}", line["text"])
            ):
                flush()
                print(
                    f"Detected appendix at PDF page {line['page']}; "
                    "remaining content was skipped."
                )
                break
            if line is document_title_line or line["text"] == document_title:
                continue
            level = self._heading_level(line, body_size)
            heading_source_text = line["text"]
            chinese_chapter_match = re.match(
                r"^\s*([一二三四五六七八九十百千万]+)、",
                heading_source_text,
            )
            chinese_chapter_key = (
                f"chapter:{chinese_chapter_match.group(1)}"
                if chinese_chapter_match
                else ""
            )
            if (
                chinese_chapter_key in self._toc_heading_map
                and (
                    line["bold"]
                    or line["size"] >= body_size * 1.15
                )
            ):
                heading_source_text = self._toc_heading_map[
                    chinese_chapter_key
                ]
                level = 1
                reconstructed_heading_line = line

            outline_number_match = re.match(
                r"^\s*(\d+(?:\.\d+){0,3})(?:\s+|[、．.]\s*|$)",
                heading_source_text,
            )
            if (
                outline_number_match
                and outline_number_match.group(1) in self._toc_heading_map
                and (
                    line["bold"]
                    or line["size"] >= body_size * 1.15
                )
            ):
                outline_number = outline_number_match.group(1)
                heading_source_text = (
                    f"{outline_number} "
                    f"{self._toc_heading_map[outline_number]}"
                )
                level = min(outline_number.count(".") + 1, 4)
                reconstructed_heading_line = line

            # A chapter title can be split into a standalone bold number plus
            # several same-baseline glyph runs.  Recognize the canonical TOC
            # heading before the generic page-number filter; ordinary footer
            # page numbers do not satisfy the bold/large-font guard above.
            reconstructed_from_outline = (
                heading_source_text != line["text"]
            )
            if (
                not reconstructed_from_outline
                and self._is_noise_line(
                    line, body_size, repeated_margin_texts
                )
            ):
                continue

            clean_line_heading = self._clean_heading(heading_source_text)
            if skipping_excluded_front_section:
                if level != 1:
                    continue
                skipping_excluded_front_section = False
                last_heading_line = None
                pending_heading_level = None
            elif (
                level
                and clean_line_heading in self.EXCLUDED_FRONT_SECTION_TITLES
            ):
                flush()
                skipping_excluded_front_section = True
                last_heading_line = None
                pending_heading_level = None
                continue
            if skipping_publication_info:
                is_content_restart = bool(
                    level == 1
                    and (
                        re.fullmatch(r"(?:前言|序言|绪论|引言)", clean_line_heading)
                        or self.CHAPTER_RE.match(line["text"])
                    )
                )
                if not is_content_restart:
                    continue
                skipping_publication_info = False
                last_heading_line = None
                pending_heading_level = None
            elif self.PUBLICATION_INFO_START_RE.match(line["text"]):
                flush()
                skipping_publication_info = True
                last_heading_line = None
                pending_heading_level = None
                continue
            terminal_title = re.sub(r"[\s:：]+", "", line["text"])
            if (
                exclude_appendix
                and level
                and terminal_title in self.TERMINAL_SECTION_TITLES
            ):
                flush()
                print(
                    f"Detected excluded terminal section "
                    f"'{terminal_title}' at PDF page {line['page']}; "
                    "this section and all remaining content were skipped."
                )
                break
            normalized_heading = re.sub(r"\s+", "", line["text"])
            is_exercise_heading = bool(re.fullmatch(
                r"(?:\d+(?:\.\d+)*[、.]?)?(?:习题|练习题|复习题|思考题|自测题)\d*",
                normalized_heading,
            ))
            if is_exercise_heading:
                flush()
                skipping_exercises = True
                last_heading_line = None
                pending_heading_level = None
                continue
            if skipping_exercises:
                if level != 1:
                    continue
                skipping_exercises = False
            if level:
                pending_heading_level = None
                is_wrapped_heading = (
                    last_heading_line is not None
                    and not text_lines
                    and last_heading_line["level"] == level
                    and last_heading_line["page"] == line["page"]
                    and abs(last_heading_line["size"] - line["size"])
                    <= max(last_heading_line["size"], line["size"]) * 0.05
                    and abs(last_heading_line["bbox"][0] - line["bbox"][0]) <= 20
                    and 0 <= line["bbox"][1] - last_heading_line["bbox"][3]
                    <= body_size * 1.5
                )
                if is_wrapped_heading:
                    headings[level] = self._clean_heading(
                        join_wrapped_heading(headings[level], heading_source_text)
                    )
                    last_heading_line = {**line, "level": level}
                    continue
                flush()
                headings[level] = self._clean_heading(heading_source_text)
                for deeper_level in range(level + 1, 5):
                    headings[deeper_level] = ""
                # In technical standards, level-3/4 numbered entries are
                # frequently both a structural heading and the first line of
                # the requirement itself.  Preserve the text after the number
                # so short one-line requirements are not lost.
                numbered_match = self.NUMBERED_HEADING_RE.match(
                    heading_source_text
                )
                if level >= 3 and numbered_match:
                    heading_body = numbered_match.group(2).strip()
                    text_lines.append(heading_body)
                    embedding_text_lines.append(heading_body)
                    if (
                        not heading_is_complete(line["text"])
                        and (
                            line["bold"]
                            or line["size"] >= body_size * 1.08
                        )
                    ):
                        pending_heading_level = level
                last_heading_line = {**line, "level": level}
            else:
                can_extend_pending_heading = (
                    pending_heading_level is not None
                    and last_heading_line is not None
                    and last_heading_line["page"] == line["page"]
                    and abs(last_heading_line["size"] - line["size"])
                    <= max(last_heading_line["size"], line["size"]) * 0.05
                    and abs(last_heading_line["bbox"][0] - line["bbox"][0]) <= 20
                    and 0 <= line["bbox"][1] - last_heading_line["bbox"][3]
                    <= body_size * 1.5
                )
                text_lines.append(line["text"])
                embedding_text_lines.append(line.get("embedding_text", line["text"]))
                if can_extend_pending_heading:
                    headings[pending_heading_level] = self._clean_heading(
                        join_wrapped_heading(
                            headings[pending_heading_level], line["text"]
                        )
                    )
                    if heading_is_complete(line["text"]):
                        pending_heading_level = None
                    else:
                        last_heading_line = {**line, "level": pending_heading_level}
                else:
                    pending_heading_level = None
                    last_heading_line = None
        flush()

        # Remove only consecutive leading cover/copyright chunks. Stop as
        # soon as normal front matter or chapter content begins.
        removed_cover_chunks = 0
        while chunks:
            first_chunk = chunks[0]
            heading_text = " ".join(
                first_chunk.get(field, "")
                for field in (
                    "chapter_title", "section_title",
                    "third_title", "fourth_title",
                )
            )
            body_text = first_chunk.get("text", "")
            is_cover_or_copyright = bool(
                re.search(r"出版社|出版集团|出版发行|第\s*\d+\s*版|ISBN", heading_text, re.IGNORECASE)
                or re.search(r"版权所有|盗版必究|翻印必究|ISBN", body_text, re.IGNORECASE)
            )
            if not is_cover_or_copyright:
                break
            chunks.pop(0)
            removed_cover_chunks += 1
        if removed_cover_chunks:
            print(f"Removed {removed_cover_chunks} leading cover/copyright chunks.")
        return chunks

    @staticmethod
    def save_json(chunks: List[Dict[str, str]], output_file) -> Path:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
        return output_path

    @staticmethod
    def _chunk_identity(chunk: Dict[str, str]) -> tuple:
        return tuple(
            chunk.get(field, "")
            for field in (
                "title",
                "chapter_title",
                "section_title",
                "third_title",
                "fourth_title",
                "text",
            )
        )

    def _load_existing_descriptions(self, output_file) -> Dict[tuple, str]:
        output_path = Path(output_file)
        if not output_path.exists():
            return {}
        try:
            existing_chunks = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"无法读取已有摘要，将重新生成：{error}")
            return {}
        return {
            self._chunk_identity(chunk): chunk.get("description", "").strip()
            for chunk in existing_chunks
            if isinstance(chunk, dict) and chunk.get("description", "").strip()
        }

    def add_descriptions(
        self,
        chunks: List[Dict[str, str]],
        output_file,
        min_chars: int = 50,
        checkpoint_interval: int = 1,
        max_consecutive_failures: int = 3,
    ) -> List[Dict[str, str]]:
        """Generate missing summaries with thresholding and checkpoint saves."""
        short_text_reused = 0
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            effective_length = len(re.sub(r"\s+", "", text))
            if (
                text
                and effective_length < min_chars
                and not chunk.get("description", "").strip()
            ):
                chunk["description"] = text
                short_text_reused += 1

        pending = [
            chunk
            for chunk in chunks
            if not chunk.get("description", "").strip()
            and len(re.sub(r"\s+", "", chunk.get("text", ""))) >= min_chars
        ]
        if not pending:
            self.save_json(chunks, output_file)
            print(
                f"无需生成新摘要；短文本沿用正文 {short_text_reused} 条。"
            )
            return chunks

        client = get_client()
        print(
            f"需要生成摘要 {len(pending)} 条，"
            f"短文本沿用正文 {short_text_reused} 条，"
            f"每成功 {checkpoint_interval} 条保存一次。"
        )
        successful_since_checkpoint = 0
        consecutive_failures = 0
        summary_index = 0
        total_pending = len(pending)
        for chunk_index, chunk in enumerate(chunks, start=1):
            effective_length = len(re.sub(r"\s+", "", chunk.get("text", "")))
            if effective_length < min_chars or chunk.get("description", "").strip():
                continue
            summary_index += 1
            try:
                summarized_chunk = chat(chunk, client=client)
                chunk["description"] = summarized_chunk["description"]
                consecutive_failures = 0
                successful_since_checkpoint += 1
                print(
                    f"摘要生成成功：{summary_index}/{total_pending}，"
                    f"原 chunk 序号={chunk_index}"
                )
            except Exception as error:
                chunk["description"] = ""
                consecutive_failures += 1
                print(
                    f"摘要生成失败：{summary_index}/{total_pending}，"
                    f"原 chunk 序号={chunk_index}，错误：{error}"
                )
                if consecutive_failures >= max_consecutive_failures:
                    self.save_json(chunks, output_file)
                    raise RuntimeError(
                        f"连续 {max_consecutive_failures} 个摘要失败，"
                        "已保存当前进度并停止，请检查 API Key、余额和网络。"
                    ) from error
            if successful_since_checkpoint >= checkpoint_interval:
                self.save_json(chunks, output_file)
                print(
                    f"摘要已写入 JSON：{summary_index}/{total_pending}，"
                    f"文件={output_file}"
                )
                successful_since_checkpoint = 0

        self.save_json(chunks, output_file)
        return chunks

    def run_text_pipeline(
        self,
        output_file,
        max_chars: int = 1500,
        exclude_appendix: bool = True,
        overlap_sentences: int = 1,
        generate_descriptions: bool = True,
        generate_embedding:bool = True,
        description_min_chars: int = 50,
        checkpoint_interval: int = 1,
        max_consecutive_failures: int = 3,
    ) -> List[Dict[str, str]]:
        existing_descriptions = self._load_existing_descriptions(output_file)
        chunks = self.split_text(
            max_chars=max_chars,
            exclude_appendix=exclude_appendix,
            overlap_sentences=overlap_sentences,
        )
        for chunk_index, chunk in enumerate(chunks, start=1):
            document_title = chunk.get("title") or Path(output_file).stem
            chunk["chunk_id"] = f"{document_title}_{chunk_index:06d}"
            chunk["chunk_index"] = chunk_index

        for chunk in chunks:
            old_description = existing_descriptions.get(self._chunk_identity(chunk))
            if old_description:
                chunk["description"] = old_description

        if generate_descriptions:
            self.add_descriptions(
                chunks,
                output_file=output_file,
                min_chars=description_min_chars,
                checkpoint_interval=checkpoint_interval,
                max_consecutive_failures=max_consecutive_failures,
            )
        if generate_embedding:
            start = 1
            for chunk in chunks:
                if chunk.get("text_embedding", "") and chunk.get("description_embedding", ""):
                    start += 1
                    continue

                chunk["text_embedding"] = embed(
                    chunk.get("embedding_text") or chunk.get("text", "")
                )
                chunk["description_embedding"] = embed(chunk.get("description", ""))

                print(f"chunk{start}已编码完成")
                start += 1
        # Always persist the split result, even when descriptions and embeddings
        # are both disabled.
        self.save_json(chunks, output_file)
        return chunks

# Backward compatibility for existing imports in the project.
pdf_extract = PDFExtract
