"""Fetching a member's photo on the server: one bounded way for every fetch, the Daily Fates thumbnails
(fates_api.blur_thumbs), the biodata PDF photo (biodata_pdf.fetch_photo) and the face checks
(main._download_image_bytes, face_match.decode).

- No redirects: a Storage or CDN link answers itself, and a redirect could lead anywhere.
- A total byte cap (MAX_BYTES, 12 MB; a caller may ask for less) and a total time cap (TOTAL_SEC), checked after
  every read, and each read returns what has arrived (urllib3's read1): neither a huge body nor a slow drip holds a
  worker past the caps.
- The image's own header is read before anything decodes it (image_size: Pillow reads only the header). More than
  MAX_PIXELS (40 megapixels) is never decoded, so a small file that decodes to a huge bitmap (a decompression bomb)
  can't take the memory. A file Pillow can't name is not decoded by OpenCV either (decode_cv2): OpenCV reads formats
  Pillow doesn't, and those have no header check here.
"""
from __future__ import annotations

import io
import time
import warnings
from typing import Any, Iterator, Mapping, Optional, Tuple

import requests

MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 40_000_000
TOTAL_SEC = 15.0
TIMEOUT = (3.0, 5.0)          # (connect, each read)
CHUNK = 64 * 1024

_clock = time.monotonic       # the tests move time on


class FetchError(Exception):
    """Why a fetch gave no bytes: 'status' (not 200; a redirect too), 'too_large', 'too_slow' or 'network'."""

    def __init__(self, reason: str, status: int = 0):
        super().__init__(reason)
        self.reason, self.status = reason, status


def fetch(url: str, *, max_bytes: int = MAX_BYTES, total_sec: float = TOTAL_SEC,
          timeout: Tuple[float, float] = TIMEOUT, session: Any = None,
          headers: Optional[Mapping[str, str]] = None) -> bytes:
    """The body of url (200 only), at most max_bytes (never above MAX_BYTES), all of it within total_sec. Redirects
    are not followed. Raises FetchError. session: a requests.Session to use (else requests itself)."""
    max_bytes = min(int(max_bytes), MAX_BYTES)
    deadline = _clock() + float(total_sec)
    kw: dict = {"timeout": timeout, "allow_redirects": False, "stream": True}
    if headers:
        kw["headers"] = dict(headers)
    try:
        resp = (session or requests).get(url, **kw)
    except Exception as e:
        raise FetchError("network") from e
    try:
        if resp.status_code != 200:
            raise FetchError("status", int(resp.status_code or 0))
        try:
            declared = int((resp.headers or {}).get("Content-Length") or 0)
        except (TypeError, ValueError):
            declared = 0
        if declared > max_bytes:
            raise FetchError("too_large")
        buf = bytearray()
        for chunk in _chunks(resp):
            buf += chunk
            if len(buf) > max_bytes:
                raise FetchError("too_large")
            if _clock() > deadline:
                raise FetchError("too_slow")
        return bytes(buf)
    except FetchError:
        raise
    except Exception as e:
        raise FetchError("network") from e
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _chunks(resp: Any) -> Iterator[bytes]:
    """The body as it arrives. urllib3's read1 returns after one read of the socket (at most CHUNK bytes), so the
    caps are checked between reads; a response without read1 (an old urllib3) falls back to iter_content."""
    read1 = getattr(getattr(resp, "raw", None), "read1", None)
    if callable(read1):
        while True:
            chunk = read1(CHUNK, decode_content=True)
            if not chunk:
                return
            yield chunk
    else:
        for chunk in resp.iter_content(CHUNK):
            if chunk:
                yield chunk


def image_size(data: bytes) -> Optional[Tuple[int, int]]:
    """(width, height) from the image's header, nothing decoded; None when Pillow can't tell what the file is.
    A header past Pillow's own bomb limit reads as larger than MAX_PIXELS."""
    from PIL import Image
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as img:
                w, h = img.size
    except Image.DecompressionBombError:
        return (MAX_PIXELS + 1, 1)
    except Exception:
        return None
    return int(w), int(h)


def pixels_ok(data: bytes, max_pixels: int = MAX_PIXELS) -> bool:
    """A picture Pillow can name, at most max_pixels (never above MAX_PIXELS) by its header."""
    size = image_size(data)
    return size is not None and size[0] > 0 and size[1] > 0 and size[0] * size[1] <= min(max_pixels, MAX_PIXELS)


def too_many_pixels(data: bytes) -> bool:
    """The header says more than MAX_PIXELS (a file Pillow can't name is not judged here)."""
    size = image_size(data)
    return size is not None and size[0] * size[1] > MAX_PIXELS


def decode_cv2(data: bytes, min_width: int = 0) -> Any:
    """An OpenCV BGR image, or None: decoded only after pixels_ok. min_width > 0 lets a JPEG decode at 1/2, 1/4 or
    1/8 of its size (IMREAD_REDUCED_*) while it stays at least that wide: a thumbnail never needs the full bitmap."""
    size = image_size(data)
    if size is None or size[0] <= 0 or size[1] <= 0 or size[0] * size[1] > MAX_PIXELS:
        return None
    import cv2
    import numpy as np
    flags = cv2.IMREAD_COLOR
    if min_width > 0:
        for factor, flag in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
                             (2, cv2.IMREAD_REDUCED_COLOR_2)):
            if size[0] // factor >= min_width:
                flags = flag
                break
    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)
