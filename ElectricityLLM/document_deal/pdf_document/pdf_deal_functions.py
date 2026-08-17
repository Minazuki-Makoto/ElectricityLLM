import hashlib
import re
import unicodedata
from pathlib import Path

import fitz

def vector2pic(
        filename,
        dpi = 200.0,
        min_length=100,
        min_height=80,
        output_filename=None
):

    if output_filename is None:
        source_path = Path(filename)
        output_filename = source_path.with_name(
            f"{source_path.stem}_vector2pic.pdf"
        )

    picture_count = 0

    with fitz.open(filename) as doc:
        for index, page in enumerate(doc, start=1):
            graphic_rect = []

            for rect in page.cluster_drawings():
                rect = fitz.Rect(rect)
                if _is_margin_image(rect, page.rect):
                    continue
                if rect.width < min_length or rect.height < min_height:
                    continue

                graphic_area = fitz.Rect(
                    max(page.rect.x0, rect.x0 - 1),
                    max(page.rect.y0, rect.y0 - 1),
                    min(page.rect.x1, rect.x1 + 1),
                    min(page.rect.y1, rect.y1 + 1)
                )
                graphic_rect.extend(
                    _split_graphic_rect_by_caption(
                        page=page,
                        graphic_rect=graphic_area,
                        min_length=min_length,
                        min_height=min_height
                    )
                )

            rendered_graphics = []
            matrix = fitz.Matrix(dpi / 72, dpi / 72)

            for rect in graphic_rect:
                pixmap = page.get_pixmap(
                    matrix=matrix,
                    clip=rect,
                    alpha=False
                )
                rendered_graphics.append((rect, pixmap))

            for rect in graphic_rect:
                page.add_redact_annot(rect, fill=(1, 1, 1))

            if graphic_rect:
                page.apply_redactions()

            for rect, pixmap in rendered_graphics:
                page.insert_image(
                    rect,
                    pixmap=pixmap,
                    keep_proportion=False,
                    overlay=True
                )
                picture_count += 1

            print(
                f"Page {index}: converted "
                f"{len(rendered_graphics)} vector regions"
            )

        doc.save(output_filename, garbage=4, deflate=True)

    print(f"Converted {picture_count} vector regions in total")
    return Path(output_filename)


def find_table(
        filename,
        dpi=200,
        min_height=40,
        min_length=20,
        output_filename=None
):
    if output_filename is None:
        source_path = Path(filename)
        output_filename = source_path.with_name(
            f"{source_path.stem}_table2pic.pdf"
        )

    table_count = 0

    with fitz.open(filename) as doc:
        for index, page in enumerate(doc, start=1):
            tables = page.find_tables()
            table_rect = []

            for table in tables.tables:
                rect = fitz.Rect(table.bbox)
                if rect.width < min_length or rect.height < min_height:
                    continue
                table_rect.append(rect)

            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            table_area = []

            for rect in table_rect:
                pixmap = page.get_pixmap(
                    matrix=matrix,
                    clip=rect,
                    alpha=False
                )
                table_area.append((rect, pixmap))

            for rect in table_rect:
                page.add_redact_annot(rect, fill=(1, 1, 1))

            if table_rect:
                page.apply_redactions()

            for rect, pixmap in table_area:
                page.insert_image(
                    rect,
                    pixmap=pixmap,
                    keep_proportion=False,
                    overlay=True
                )
                table_count += 1

            print(
                f"Page {index}: converted "
                f"{len(table_area)} table regions"
            )

        doc.save(output_filename, garbage=4, deflate=True)

    print(f"Converted {table_count} table regions in total")
    return Path(output_filename)


FIGURE_REFERENCE_PATTERN = re.compile(
    r"(?:\u56fe\s*\d+(?:\s*[-\u2014\u2013.]\s*\d+)*|"
    r"(?:Figure|Fig\.?)\s*\d+(?:\s*[-\u2014\u2013.]\s*\d+)*)",
    re.IGNORECASE
)

TABLE_REFERENCE_PATTERN = re.compile(
    r"(?:\u8868\s*\d+(?:\s*[-\u2014\u2013.]\s*\d+)*|"
    r"(?:Table|Tab\.?)\s*\d+(?:\s*[-\u2014\u2013.]\s*\d+)*)",
    re.IGNORECASE
)

# A caption must start with a figure/table number and have a separator or
# whitespace before its description.  This avoids treating sentences such as
# "图1显示了……" as captions merely because they mention a figure.
CAPTION_LINE_PATTERN = re.compile(
    r"^\s*(?:(?:图|表)\s*[A-Za-z]?\d+(?:\s*[-—–.]\s*\d+)*|"
    r"(?:Figure|Fig\.?|Table|Tab\.?)\s*[A-Za-z]?\d+"
    r"(?:\s*[-—–.]\s*\d+)*)"
    r"(?:\s+|[:：.．、-])\S.*$",
    re.IGNORECASE,
)

REFERENCE_SENTENCE_START_PATTERN = re.compile(
    r"^\s*(?:(?:图|表)\s*[A-Za-z]?\d+(?:\s*[-—–.]\s*\d+)*|"
    r"(?:Figure|Fig\.?|Table|Tab\.?)\s*[A-Za-z]?\d+"
    r"(?:\s*[-—–.]\s*\d+)*)\s*(?:所示|显示|为)",
    re.IGNORECASE,
)

PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(?:第\s*)?[-—–]?\s*\d+\s*[-—–]?(?:\s*页)?\s*$",
    re.IGNORECASE,
)


def is_caption_text(text):
    """Return True only for a standalone figure/table caption line."""
    normalized = re.sub(r"[ \t]+", " ", str(text)).strip()
    return bool(CAPTION_LINE_PATTERN.match(normalized))


def is_page_number_text(text):
    """Recognize common standalone page-number forms."""
    return bool(PAGE_NUMBER_PATTERN.match(str(text)))


def get_page_image_rects(page):
    """Return every displayed occurrence rectangle of raster images."""
    rects = []
    seen = set()
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        for rect in page.get_image_rects(xref):
            if _is_margin_image(rect, page.rect):
                continue
            key = tuple(round(value, 2) for value in rect)
            if key not in seen:
                seen.add(key)
                rects.append(fitz.Rect(rect))
    return rects


def remove_all_images(filename, output_filename):
    """Save a copy of the PDF with every raster image object removed."""
    source_path = Path(filename)
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    removed_xrefs = set()
    with fitz.open(source_path) as document:
        for page in document:
            for image_info in page.get_images(full=True):
                xref = image_info[0]
                if xref in removed_xrefs:
                    continue
                page.delete_image(xref)
                removed_xrefs.add(xref)
        document.save(output_path, garbage=4, deflate=True)

    print(f"Removed {len(removed_xrefs)} image objects: {output_path}")
    return output_path


def _split_graphic_rect_by_caption(
        page,
        graphic_rect,
        min_length=100,
        min_height=80,
        padding=2,
        row_tolerance=8,
):
    """Split a merged graphic into rows and side-by-side captioned panels."""
    graphic_rect = fitz.Rect(graphic_rect)
    nearby_blocks = page.get_text("blocks", clip=graphic_rect, sort=True)
    caption_rects = []
    for block in nearby_blocks:
        block_text = re.sub(r"\s+", " ", block[4]).strip()
        if not is_caption_text(block_text):
            continue
        block_rect = fitz.Rect(block[:4])
        if block_rect.y0 - graphic_rect.y0 >= min_height:
            caption_rects.append(block_rect)

    if not caption_rects:
        return [graphic_rect]

    caption_rows = []
    for caption_rect in sorted(caption_rects, key=lambda item: (item.y0, item.x0)):
        for row in caption_rows:
            row_y0 = min(item.y0 for item in row)
            row_y1 = max(item.y1 for item in row)
            if caption_rect.y0 <= row_y1 + row_tolerance and caption_rect.y1 >= row_y0 - row_tolerance:
                row.append(caption_rect)
                break
        else:
            caption_rows.append([caption_rect])

    result_rects = []
    current_y = graphic_rect.y0
    for row in caption_rows:
        row = sorted(row, key=lambda item: item.x0)
        image_bottom = min(item.y0 for item in row) - padding
        if image_bottom - current_y < min_height:
            current_y = max(current_y, max(item.y1 for item in row) + padding)
            continue

        boundaries = [graphic_rect.x0]
        for left_caption, right_caption in zip(row, row[1:]):
            boundaries.append((left_caption.x1 + right_caption.x0) / 2)
        boundaries.append(graphic_rect.x1)
        for index in range(len(row)):
            panel_rect = fitz.Rect(
                boundaries[index], current_y,
                boundaries[index + 1], image_bottom,
            )
            if panel_rect.width >= min_length and panel_rect.height >= min_height:
                result_rects.append(panel_rect)
        current_y = max(item.y1 for item in row) + padding

    lower_rect = fitz.Rect(graphic_rect.x0, current_y, graphic_rect.x1, graphic_rect.y1)
    if lower_rect.width >= min_length and lower_rect.height >= min_height:
        result_rects.append(lower_rect)
    return result_rects if len(result_rects) >= 2 else [graphic_rect]
def _extract_table_caption(page, image_rect, caption_height=80):
    caption_rect = fitz.Rect(
        image_rect.x0,
        max(image_rect.y0-caption_height,page.rect.y0),
        image_rect.x1,
        image_rect.y0
    )

    nearby_blocks = page.get_text(
        "blocks",
        clip=caption_rect,
        sort=True
    )

    # Prefer the closest matching block above the table.
    for block in reversed(nearby_blocks):
        block_text = re.sub(r"\s+", " ", block[4]).strip()
        if REFERENCE_SENTENCE_START_PATTERN.match(block_text):
            continue
        reference_match = TABLE_REFERENCE_PATTERN.search(block_text)
        if reference_match:
            return block_text[reference_match.start():].strip()

    return ""


def _normalize_object_component(value):
    """Normalize one side of the title:caption object key."""
    value = re.sub(r"\s+", " ", str(value)).strip()
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Cc"
    )
    value = value.replace(":", "\uff1a")
    value = value.replace("/", "\uff0f").replace("\\", "\uff3c")
    return value.strip(" .")


def _safe_object_name(title, caption, max_bytes=240):
    """Build a MinIO-safe, deterministic object name."""
    object_name = f"{title}:{caption}"
    encoded_name = object_name.encode("utf-8")
    if len(encoded_name) <= max_bytes:
        return object_name

    digest = hashlib.sha256(encoded_name).hexdigest()[:12]
    suffix = f"-{digest}"
    byte_budget = max_bytes - len(suffix.encode("ascii"))
    shortened = encoded_name[:byte_budget]
    while shortened:
        try:
            prefix = shortened.decode("utf-8")
            break
        except UnicodeDecodeError:
            shortened = shortened[:-1]
    else:
        prefix = "image"
    return f"{prefix.rstrip()}{suffix}"


def _extract_figure_caption(page, image_rect, caption_height=80):
    """Read a standard figure caption immediately below an image."""
    caption_rect = fitz.Rect(
        image_rect.x0,
        image_rect.y1,
        image_rect.x1,
        min(page.rect.y1, image_rect.y1 + caption_height)
    )
    nearby_blocks = page.get_text(
        "blocks",
        clip=caption_rect,
        sort=True
    )

    for block in nearby_blocks:
        block_text = re.sub(r"\s+", " ", block[4]).strip()
        if REFERENCE_SENTENCE_START_PATTERN.match(block_text):
            continue
        reference_match = FIGURE_REFERENCE_PATTERN.search(block_text)
        if reference_match:
            return block_text[reference_match.start():].strip()

    return ""


def _image_content_type(extension):
    extension = extension.lower()
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "jpx": "image/jp2",
        "jp2": "image/jp2",
        "tif": "image/tiff",
        "tiff": "image/tiff",
        "bmp": "image/bmp"
    }.get(extension, "application/octet-stream")


def _is_margin_image(
        image_rect,
        page_rect,
        margin_ratio=0.10,
        minimum_body_overlap=0.80,
):
    """Reject images outside the main body, including partial margin images."""
    image_rect = fitz.Rect(image_rect)
    page_rect = fitz.Rect(page_rect)
    if image_rect.is_empty or image_rect.get_area() <= 0:
        return True
    body_rect = fitz.Rect(
        page_rect.x0,
        page_rect.y0 + page_rect.height * margin_ratio,
        page_rect.x1,
        page_rect.y1 - page_rect.height * margin_ratio,
    )
    overlap_rect = image_rect & body_rect
    overlap_ratio = overlap_rect.get_area() / image_rect.get_area() if not overlap_rect.is_empty else 0.0
    center = (image_rect.tl + image_rect.br) / 2
    return center not in body_rect or overlap_ratio < minimum_body_overlap


def remove_picture_reference_sentences(text, picture_records):
    """Remove sentences that reference images listed in picture_records."""
    references = {
        reference
        for record in picture_records
        for reference in record.get("references", [])
        if reference
    }

    cleaned_text = text
    for reference in sorted(references, key=len, reverse=True):
        escaped_reference = re.escape(reference)
        sentence_pattern = re.compile(
            rf"[^\u3002\uff01\uff1f.!?\n]*{escaped_reference}"
            rf"[^\u3002\uff01\uff1f.!?\n]*[\u3002\uff01\uff1f.!?]?",
            re.IGNORECASE
        )
        cleaned_text = sentence_pattern.sub("", cleaned_text)

    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def _group_touching_page_images(page, edge_tolerance=1.5, overlap_ratio=0.80):
    """Group image strips that form one contiguous visual region."""
    items = []
    for image_index, image_info in enumerate(page.get_images(full=True), start=1):
        xref = image_info[0]
        for occurrence_index, rect in enumerate(page.get_image_rects(xref), start=1):
            rect = fitz.Rect(rect)
            if not _is_margin_image(rect, page.rect):
                items.append({"xref": xref, "image_index": image_index, "occurrence_index": occurrence_index, "rect": rect})
    parent = list(range(len(items)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    for left in range(len(items)):
        a = items[left]["rect"]
        for right in range(left + 1, len(items)):
            b = items[right]["rect"]
            x_overlap = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
            y_overlap = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
            vertical_gap = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
            horizontal_gap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
            vertically_joined = vertical_gap <= edge_tolerance and x_overlap >= min(a.width, b.width) * overlap_ratio
            horizontally_joined = horizontal_gap <= edge_tolerance and y_overlap >= min(a.height, b.height) * overlap_ratio
            if vertically_joined or horizontally_joined:
                union(left, right)
    grouped = {}
    for index, item in enumerate(items):
        grouped.setdefault(find(index), []).append(item)
    groups, lookup = [], {}
    for members in grouped.values():
        rect = fitz.Rect(members[0]["rect"])
        for member in members[1:]:
            rect |= member["rect"]
        group_index = len(groups)
        groups.append({"rect": rect, "members": members})
        for member in members:
            key = (member["xref"], member["occurrence_index"])
            lookup[key] = group_index
    return groups, lookup


def _picture_object_name(caption, extension):
    """Use a figure/table reference as the object name when available."""
    reference = None
    patterns = (
        ()
        if re.match(r"^图表\d", str(caption))
        else (FIGURE_REFERENCE_PATTERN, TABLE_REFERENCE_PATTERN)
    )
    for pattern in patterns:
        match = pattern.search(str(caption))
        if match:
            reference = re.sub(r"\s+", "", match.group(0))
            break
    base_name = reference or _normalize_object_component(caption) or "image"
    normalized_extension = str(extension).lower().lstrip(".") or "png"
    return f"{base_name}.{normalized_extension}"


def find_pictures_and_save(
        filename,
        title,
        minio_endpoint,
        minio_access_key,
        minio_secret_key,
        minio_bucket_name,
        secure=False,
        caption_height=80,
        save_address=None
):
    # Keep MinIO optional for callers that only use PDF text/layout helpers.
    from minio_service.minio_service import MinioService

    minio_server = MinioService(
        minio_endpoint,
        minio_access_key,
        minio_secret_key,
        minio_bucket_name,
        secure
    )

    normalized_title = _normalize_object_component(title)
    local_save_dir = Path(save_address) if save_address else None
    if local_save_dir is not None:
        local_save_dir.mkdir(parents=True, exist_ok=True)

    image_cache = {}
    uploaded_object_names = set()
    picture_records = []

    with fitz.open(filename) as doc:
        for page_index, page in enumerate(doc, start=1):
            image_groups, image_group_lookup = _group_touching_page_images(page)
            handled_image_groups = set()
            for image_index, image_info in enumerate(
                    page.get_images(full=True),
                    start=1
            ):
                xref = image_info[0]
                if xref not in image_cache:
                    image_cache[xref] = doc.extract_image(xref)

                image = image_cache[xref]
                image_data = image["image"]
                extension = image["ext"]
                content_type = _image_content_type(extension)
                image_rects = page.get_image_rects(xref)

                for occurrence_index, image_rect in enumerate(
                        image_rects,
                        start=1
                ):
                    group_index = image_group_lookup.get((xref, occurrence_index))
                    if group_index is not None:
                        if group_index in handled_image_groups:
                            continue
                        handled_image_groups.add(group_index)
                        image_group = image_groups[group_index]
                        image_rect = image_group["rect"]
                        if len(image_group["members"]) > 1:
                            pixmap = page.get_pixmap(
                                matrix=fitz.Matrix(2, 2),
                                clip=image_rect,
                                alpha=False,
                            )
                            image_data = pixmap.tobytes("png")
                            extension = "png"
                            content_type = "image/png"

                    if _is_margin_image(image_rect, page.rect):
                        print(
                            "Skipped header/footer image: "
                            f"page={page_index}, xref={xref}, "
                            f"rect={tuple(image_rect)}"
                        )
                        continue

                    local_path = None
                    if local_save_dir is not None:
                        local_filename = (
                            f"page_{page_index:04d}_image_{image_index:03d}_"
                            f"occurrence_{occurrence_index:02d}.{extension}"
                        )
                        local_path = local_save_dir / local_filename
                        local_path.write_bytes(image_data)
                        print(f"Saved local image: {local_path}")

                    caption = _extract_figure_caption(
                        page,
                        fitz.Rect(image_rect),
                        caption_height=caption_height
                    )
                    caption_type = "figure"

                    if not caption:
                        caption = _extract_table_caption(
                            page,
                            fitz.Rect(image_rect),
                            caption_height=caption_height
                        )
                        caption_type = "table"

                    if not caption:
                        caption = (
                            f"\u56fe\u8868{page_index}-"
                            f"{image_index}-{occurrence_index}"
                        )
                        caption_type = "generated"

                    caption = _normalize_object_component(caption)
                    caption_object_name = _picture_object_name(
                        caption, extension
                    )
                    object_name = _safe_object_name(
                        normalized_title,
                        caption_object_name,
                    )

                    if object_name not in uploaded_object_names:
                        minio_server.upload_file(
                            object_name=object_name,
                            data=image_data,
                            content_type=content_type,
                            bucket_name=minio_bucket_name
                        )
                        uploaded_object_names.add(object_name)
                        print(
                            "MinIO upload succeeded: "
                            f"bucket={minio_bucket_name}, "
                            f"object={object_name}"
                        )

                    references = [
                        match.group(0)
                        for pattern in (
                            FIGURE_REFERENCE_PATTERN,
                            TABLE_REFERENCE_PATTERN
                        )
                        for match in pattern.finditer(caption)
                    ]
                    picture_records.append({
                        "page": page_index,
                        "xref": xref,
                        "caption": caption,
                        "caption_type": caption_type,
                        "references": references,
                        "object_name": object_name,
                        "bucket_name": minio_bucket_name,
                        "document_title": normalized_title,
                        "local_path": str(local_path) if local_path else "",
                        "rect": tuple(image_rect)
                    })

    return picture_records

