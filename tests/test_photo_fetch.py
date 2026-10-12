"""Every server fetch of a member's photo is bounded (photo_fetch): Daily Fates thumbnails (fates_api.blur_thumbs),
the biodata PDF photo (biodata_pdf.fetch_photo) and the face checks (main._download_image_bytes, face_match.decode).

- streamed with a total byte cap (12 MB, less where a caller asks) and a total time cap, no redirects;
- the image header is read before anything decodes it: above 40 megapixels nothing is decoded, so a small
  decompression-bomb PNG (a few KB on the wire, ~48 megapixels once decoded) never reaches OpenCV or Pillow's decoder.

The network is a fake response; the bomb PNG is generated here.
"""
import io
import struct
import zlib
from urllib.parse import quote

import pytest
from PIL import Image

import biodata_pdf as bp
import face_match
import fates_api
import main
import photo_fetch

PROD = "shaadiparrot.firebasestorage.app"
OWN = f"https://firebasestorage.googleapis.com/v0/b/{PROD}/o/{quote('users/u1/photos/a.jpg', safe='')}?alt=media&token=t"
CDN = "https://photos.shaadiparrot.com/users/me1/variants/k_960.jpg"


def png_bomb(w: int = 8000, h: int = 6000) -> bytes:
    """A 1-bit greyscale PNG of w x h black pixels: a few KB, 48 megapixels (144 MB as an OpenCV colour bitmap)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = (b"\x00" + b"\x00" * ((w + 7) // 8)) * h       # filter byte 0, then the packed row
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 1, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


BOMB = png_bomb()


def jpeg(w=900, h=1200) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 120, 90)).save(buf, "JPEG")
    return buf.getvalue()


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Raw:
    """urllib3's raw response: read1 gives what one read brings (here: the next chunk)."""

    def __init__(self, resp):
        self._resp = resp

    def read1(self, amt=None, decode_content=None):
        return self._resp.next_chunk()


class Resp:
    """A streamed response. chunks: the body as it arrives; every read moves the clock on by `step` seconds.
    .content and iter_content are there for code that reads the body another way (it then gets all of it)."""

    def __init__(self, chunks, status=200, headers=None, clock=None, step=0.0):
        self.status_code, self.headers, self.closed = status, headers or {}, False
        self._chunks, self._clock, self._step = iter(chunks), clock, step
        self.served = 0
        self.raw = Raw(self)

    def next_chunk(self) -> bytes:
        if self._clock is not None:
            self._clock.t += self._step
        c = next(self._chunks, b"")
        self.served += len(c)
        return c

    def iter_content(self, n=None):
        while True:
            c = self.next_chunk()
            if not c:
                return
            yield c

    @property
    def content(self) -> bytes:
        return b"".join(self.iter_content())

    def close(self):
        self.closed = True


def pieces(data: bytes, n: int):
    return [data[i:i + n] for i in range(0, len(data), n)]


@pytest.fixture()
def net(monkeypatch):
    """requests.get answers with net.resp; every call is recorded. The clock is photo_fetch's."""
    from types import SimpleNamespace as NS
    state = NS(resp=None, calls=[], clock=Clock())

    def get(url, **kw):
        state.calls.append((url, kw))
        return state.resp

    import requests
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(photo_fetch, "_clock", state.clock, raising=False)
    for k in ("FIREBASE_STORAGE_BUCKET", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        monkeypatch.delenv(k, raising=False)
    return state


@pytest.fixture()
def no_decode(monkeypatch):
    """Records every OpenCV decode (none may happen for a bomb)."""
    import cv2
    seen = []
    real = cv2.imdecode

    def spy(buf, flags):
        seen.append(len(buf))
        return real(buf, flags)

    monkeypatch.setattr(cv2, "imdecode", spy)
    return seen


# ------------------------------------------------------------------ photo_fetch itself
def test_the_bomb_is_small_on_the_wire_and_huge_when_decoded():
    assert len(BOMB) < 64 * 1024
    assert Image.open(io.BytesIO(BOMB)).size == (8000, 6000)
    assert photo_fetch.image_size(BOMB) == (8000, 6000)
    assert not photo_fetch.pixels_ok(BOMB)
    assert photo_fetch.pixels_ok(jpeg())
    assert not photo_fetch.pixels_ok(b"not an image")               # a file Pillow can't name is never decoded
    assert photo_fetch.decode_cv2(BOMB) is None


def test_fetch_streams_with_a_total_byte_cap(net):
    net.resp = Resp([b"\0" * 65536] * 640)                           # 40 MB
    with pytest.raises(photo_fetch.FetchError) as e:
        photo_fetch.fetch(OWN)
    assert e.value.reason == "too_large" and net.resp.closed
    assert net.resp.served <= photo_fetch.MAX_BYTES + 65536          # stopped at the cap, not after the whole body
    net.resp = Resp([b"x"], headers={"Content-Length": str(photo_fetch.MAX_BYTES + 1)})
    with pytest.raises(photo_fetch.FetchError):
        photo_fetch.fetch(OWN)
    assert net.resp.served == 0                                      # the declared size alone refuses it


def test_fetch_has_a_total_time_cap(net):
    net.resp = Resp(pieces(jpeg(), 64), clock=net.clock, step=1.0)  # a slow drip: 64 bytes a second
    with pytest.raises(photo_fetch.FetchError) as e:
        photo_fetch.fetch(OWN, total_sec=8)
    assert e.value.reason == "too_slow" and net.resp.closed and net.resp.served <= 9 * 64


def test_fetch_never_follows_a_redirect(net):
    net.resp = Resp([], status=302, headers={"Location": "http://169.254.169.254/"})
    with pytest.raises(photo_fetch.FetchError) as e:
        photo_fetch.fetch(OWN)
    assert e.value.reason == "status" and e.value.status == 302
    assert net.calls[0][1]["allow_redirects"] is False and net.calls[0][1]["stream"] is True


# ------------------------------------------------------------------ Daily Fates thumbnails
def test_a_thumbnail_of_a_bomb_is_never_decoded(net, no_decode):
    net.resp = Resp(pieces(BOMB, 4096))
    assert fates_api.blur_thumbs(OWN, "u1") == (None, None)
    assert no_decode == []


def test_a_thumbnail_is_read_with_the_byte_and_time_caps(net):
    net.resp = Resp([b"\0" * 65536] * 640)
    assert fates_api.blur_thumbs(OWN, "u1") == (None, None)
    assert net.resp.served <= photo_fetch.MAX_BYTES + 65536
    net.resp = Resp(pieces(jpeg(), 64), clock=net.clock, step=1.0)
    assert fates_api.blur_thumbs(OWN, "u1") == (None, None)
    net.resp = Resp(pieces(jpeg(), 4096))                              # the same photo, on time
    heavy, peek = fates_api.blur_thumbs(OWN, "u1")
    assert heavy.startswith("data:image/jpeg;base64,") and peek.startswith("data:image/jpeg;base64,")
    assert net.calls[-1][1]["allow_redirects"] is False and net.calls[-1][1]["stream"] is True


# ------------------------------------------------------------------ the biodata PDF photo
def test_the_biodata_photo_has_a_total_time_cap(net):
    net.resp = Resp(pieces(jpeg(1200, 900), 256), clock=net.clock, step=1.0)
    assert bp.fetch_photo(CDN) is None and net.resp.closed
    net.resp = Resp(pieces(jpeg(1200, 900), 8192))
    assert bp.fetch_photo(CDN) is not None


def test_the_biodata_photo_of_a_bomb_is_never_decoded(net, monkeypatch):
    decoded = []
    monkeypatch.setattr(Image.Image, "load", lambda self: decoded.append(self.size))
    net.resp = Resp(pieces(BOMB, 4096))
    assert bp.fetch_photo(CDN) is None and decoded == []


# ------------------------------------------------------------------ the face checks
class Session:
    def __init__(self, state):
        self._state = state

    def get(self, url, **kw):
        self._state.calls.append((url, kw))
        return self._state.resp


def test_a_face_check_download_refuses_a_bomb_and_never_follows_a_redirect(net, monkeypatch):
    monkeypatch.setattr(main, "_HTTP", Session(net))
    net.resp = Resp(pieces(BOMB, 4096))
    with pytest.raises(main.HTTPException) as e:
        main._download_image_bytes(OWN, max_mb=12)
    assert e.value.status_code == 413
    assert net.calls[-1][1]["allow_redirects"] is False and net.calls[-1][1]["stream"] is True
    net.resp = Resp([], status=302, headers={"Location": "https://evil.example/x.jpg"})
    with pytest.raises(main.HTTPException) as e:
        main._download_image_bytes(OWN)
    assert e.value.status_code == 400


def test_a_face_check_download_has_a_total_time_cap(net, monkeypatch):
    monkeypatch.setattr(main, "_HTTP", Session(net))
    net.resp = Resp(pieces(jpeg(), 64), clock=net.clock, step=1.0)
    with pytest.raises(main.HTTPException) as e:
        main._download_image_bytes(OWN)
    assert e.value.status_code == 400 and "too_slow" in e.value.detail and net.resp.closed
    net.resp = Resp(pieces(jpeg(), 8192))
    assert main._download_image_bytes(OWN) == jpeg()


def test_face_match_never_decodes_a_bomb(no_decode):
    assert face_match.decode(BOMB) is None and no_decode == []
    assert face_match.decode(jpeg()) is not None and len(no_decode) == 1
