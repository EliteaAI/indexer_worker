"""Image processing helpers for the indexer worker.

All image-heavy work (S3 download, base64 encoding, PIL thumbnailing)
"""

import mimetypes
from base64 import b64encode
from io import BytesIO

from pylon.core.tools import log


# Thumbnail settings: 512 px longest edge, JPEG at 75 % quality.
THUMBNAIL_MAX_PX = 512
THUMBNAIL_JPEG_QUALITY = 75


def create_thumbnail_base64(
    image_bytes: bytes,
    max_px: int = THUMBNAIL_MAX_PX,
    quality: int = THUMBNAIL_JPEG_QUALITY,
) -> str | None:
    """Create a JPEG thumbnail and return it as a ``data:image/jpeg;base64,…`` URL.

    Returns ``None`` on any failure (missing Pillow, corrupt image, etc.).
    The caller stores ``None`` in the thumbnails dict; elitea_core will
    exclude such entries when building the final response.
    """
    img = buf = None
    try:
        from PIL import Image

        img = Image.open(BytesIO(image_bytes))

        # Resize FIRST — PIL.thumbnail() works on any mode, so we shrink the
        # pixel data immediately.  All subsequent mode conversions then operate
        # on the small image instead of the original full-resolution frame.
        if max(img.size) > max_px:
            img.thumbnail((max_px, max_px), Image.LANCZOS)

        # JPEG does not support transparency — flatten to RGB *after* resize
        # so the background allocation and alpha compositing are cheap.
        if img.mode in ('RGBA', 'LA'):
            alpha = img.split()[-1]
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=alpha)
            img.close()
            img = background
        elif img.mode != 'RGB':
            converted = img.convert('RGB')
            img.close()
            img = converted

        buf = BytesIO()
        img.save(buf, format='JPEG', quality=quality, optimize=True)
        return f"data:image/jpeg;base64,{b64encode(buf.getvalue()).decode('ascii')}"

    except ImportError:
        log.error("Pillow not available — cannot create thumbnail")
    except Exception as exc:
        log.error("Thumbnailing failed: %s", exc)
    finally:
        if img is not None:
            img.close()
        if buf is not None:
            buf.close()

    return None


# ------------------------------------------------------------------
# Filepath-scheme resolution
# ------------------------------------------------------------------

_FILEPATH_PREFIX = 'filepath:'


def _parse_filepath(filepath: str) -> tuple[str, str]:
    """Parse ``/{bucket}/{key}`` into (bucket, key)."""
    path = filepath.lstrip('/')
    parts = path.split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid filepath format: {filepath}")
    return parts[0], parts[1]


def resolve_filepath_images(content, client):
    """Resolve ``filepath:`` image URLs in *content* by downloading from S3.

    For each ``image_url`` chunk whose URL starts with ``filepath:``:

    * The **original unmodified bytes** are base64-encoded and injected
      into the chunk so the LLM receives the full-quality image.
    * A **JPEG thumbnail** (512 px, 75 % quality) is created from the
      same downloaded bytes — **no second S3 read** — and collected for
      the caller to include in the final response emitted to the UI.

    Args:
        content: Either a plain ``str`` (text-only message — returned
                 unchanged) or a ``list`` of multimodal chunk dicts.
        client:  ``EliteAClient`` instance (post-fork) for S3 downloads.

    Returns:
        tuple[content, dict]:
            *content* — the (possibly mutated) input.
            *thumbnails* — ``{filepath: thumbnail_data_url}`` for every
            image that was successfully resolved.
    """
    thumbnails: dict[str, str] = {}

    if not isinstance(content, list):
        return content, thumbnails

    for chunk in content:
        if not isinstance(chunk, dict):
            continue
        if chunk.get('type') != 'image_url':
            continue

        url = (chunk.get('image_url') or {}).get('url', '')
        if not url.startswith(_FILEPATH_PREFIX):
            continue

        filepath = url[len(_FILEPATH_PREFIX):]
        try:
            bucket, key = _parse_filepath(filepath)
            raw = client.download_artifact_s3(bucket, key)
            if isinstance(raw, dict):
                log.error("S3 download error for %s: %s", filepath, raw)
                continue

            # Create thumbnail FIRST while raw is still in scope.
            # None on failure — elitea_core will exclude the entry.
            thumbnails[filepath] = create_thumbnail_base64(raw)

            # Detect MIME from file extension for the original image.
            mime, _ = mimetypes.guess_type(key)
            mime = mime or 'image/png'

            # Inject ORIGINAL unmodified bytes into the chunk for the LLM.
            # Inline b64 encoding avoids a named intermediate string.
            chunk['image_url']['url'] = (
                f"data:{mime};base64,{b64encode(raw).decode('ascii')}"
            )
            del raw
        except Exception as exc:
            log.error("Failed to resolve filepath image %s: %s", filepath, exc)

    return content, thumbnails


def resolve_filepaths_to_thumbnails(
    filepaths: list[str],
    client,
) -> dict[str, str]:
    """Download images by filepath and return a mapping of filepath to thumbnail data URL.

    Each filepath is ``/<bucket>/<key>``.  Downloads the image from S3,
    generates a JPEG thumbnail via ``create_thumbnail_base64``, and maps
    the filepath (without leading ``/``) to the resulting data URL.

    Failures are logged and skipped — callers fall back to ``filepath:`` scheme.
    """
    thumbnails: dict[str, str] = {}
    for fp in filepaths:
        stripped = fp.lstrip('/')
        try:
            bucket, key = _parse_filepath(fp)
            raw = client.download_artifact_s3(bucket, key)
            if isinstance(raw, dict):
                log.error("S3 download error for generated image %s: %s", fp, raw)
                continue
            thumb = create_thumbnail_base64(raw)
            del raw
            if thumb:
                thumbnails[stripped] = thumb
        except Exception as exc:
            log.error("Failed to thumbnail generated image %s: %s", fp, exc)
    return thumbnails


def _reuse_thumbnails_for_copied_images(
    modified_files: list[dict],
    existing_thumbnails: dict[str, str],
) -> dict[str, str]:
    """Map copied-image destination filepaths to already-resolved source thumbnails."""
    reused: dict[str, str] = {}
    for file_info in modified_files:
        if file_info.get('media_type') != 'image':
            continue
        if file_info.get('operation_type') != 'copy':
            continue
        source = (file_info.get('meta') or {}).get('source_filepath', '')
        dest = file_info.get('filepath', '')
        if not source or not dest:
            continue
        source_key = source.lstrip('/')
        if source_key in existing_thumbnails and existing_thumbnails[source_key]:
            reused[dest.lstrip('/')] = existing_thumbnails[source_key]
    return reused


def resolve_generated_image_thumbnails(
    elitea_custom_callback,
    image_thumbnails: dict[str, str],
    client,
) -> None:
    """Resolve thumbnails for tool-generated images into *image_thumbnails*.

    First reuses thumbnails from copied-image sources, then downloads and
    thumbnails the remaining filepaths via S3.  Mutates *image_thumbnails* in place.
    """
    if not elitea_custom_callback.generated_image_filepaths:
        return
    reused = _reuse_thumbnails_for_copied_images(
        elitea_custom_callback.modified_files, image_thumbnails
    )
    image_thumbnails.update(reused)
    remaining = [
        fp for fp in elitea_custom_callback.generated_image_filepaths
        if fp.lstrip('/') not in image_thumbnails
    ]
    if remaining:
        image_thumbnails.update(resolve_filepaths_to_thumbnails(remaining, client))


def is_anthropic_model(model_name: str) -> bool:
    """Return True if model_name identifies an Anthropic model.

    Matches the same heuristic used by ``EliteAClient.get_llm()`` so that
    provider detection stays consistent across the platform.
    """
    if not model_name:
        return False
    lower = model_name.lower()
    return 'claude' in lower or 'anthropic' in lower


def strip_image_chunks_from_assistant_messages(messages: list) -> list:
    """Remove ``image_url`` chunks from assistant-role messages.

    Some providers (e.g. Anthropic) do not permit ``image`` content blocks
    inside assistant turns.  Tool-generated images are stored as assistant
    attachments, so this filter must run before the history reaches those
    providers.

    Only ``image_url`` chunks are removed — sibling ``text`` chunks
    (``Image file: …``) are preserved so the LLM still knows about the
    attachment.

    Operates **in-place** and returns the same list.
    """
    for message in messages:
        if isinstance(message, dict):
            role = message.get('role', '')
            content = message.get('content')
        else:
            role = getattr(message, 'type', '') or getattr(message, 'role', '')
            content = getattr(message, 'content', None)

        # Only filter assistant messages — user/tool messages are fine.
        if role not in ('assistant', 'ai'):
            continue

        if not isinstance(content, list):
            continue

        filtered = [
            chunk for chunk in content
            if not (
                isinstance(chunk, dict)
                and chunk.get('type') == 'image_url'
            )
        ]

        if len(filtered) != len(content):
            if isinstance(message, dict):
                message['content'] = filtered
            else:
                message.content = filtered

    return messages


def strip_stale_filepath_image_chunks(messages: list) -> list:
    """Remove ``image_url`` chunks that still carry ``filepath:`` URLs.

    Only the ``image_url`` chunks are removed — sibling ``text`` chunks
    (``Image file: …``) are preserved so the LLM still knows about the
    attachment.

    Operates **in-place** and returns the same list.
    """
    for message in messages:
        content = (
            message.get('content')
            if isinstance(message, dict)
            else getattr(message, 'content', None)
        )

        if not isinstance(content, list):
            continue

        filtered = [
            chunk for chunk in content
            if not (
                isinstance(chunk, dict)
                and chunk.get('type') == 'image_url'
                and (chunk.get('image_url') or {}).get('url', '').startswith(_FILEPATH_PREFIX)
            )
        ]

        if len(filtered) != len(content):
            if isinstance(message, dict):
                message['content'] = filtered
            else:
                message.content = filtered

    return messages
