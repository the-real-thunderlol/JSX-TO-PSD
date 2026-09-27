#!/usr/bin/env python3
"""Convert the data in file.jsx to a layered PSD.

No Photoshop or JavaScript runtime is used. Words become separate RGBA paint
layers, not editable type layers. This reads the supplied JSX's layout format;
it is not a general interpreter for arbitrary Photoshop scripts.

Install: python -m pip install -r requirements.txt
Run:     python jsx_to_psd.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

try:
    from PIL import Image, ImageCms, ImageDraw, ImageFont
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, PixelLayer
    from psd_tools.constants import BlendMode, Compression, Resource
    from psd_tools.psd.image_resources import ImageResource, ResoulutionInfo
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}\n"
        "Run: python -m pip install -r requirements.txt"
    ) from exc


@dataclass(frozen=True)
class Word:
    region: int
    text: str
    x: int
    y: int
    width: int
    height: int
    font_size: float
    angle: float
    color: tuple[int, int, int]
    quote: int
    colors: tuple[int, ...]


@dataclass
class Layout:
    width: int
    height: int
    names: dict[int, str]
    order: list[int]
    words: list[Word]


def read_json_variable(source: str, name: str):
    """Decode a literal array without ever evaluating JavaScript."""
    match = re.search(r"\bvar\s+" + re.escape(name) + r"\s*=\s*", source)
    if not match:
        raise ValueError(f"This JSX has no supported {name} data array.")
    try:
        value, _ = json.JSONDecoder().raw_decode(source[match.end():])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a literal JSON array in this JSX.") from exc
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array.")
    return value


def read_layout(path: Path) -> Layout:
    source = path.read_text(encoding="utf-8-sig")
    dimensions = re.search(
        r"\bvar\s+WIDTH\s*=\s*(\d+)\s*,\s*HEIGHT\s*=\s*(\d+)\s*;", source
    )
    names_match = re.search(r"\bvar\s+names\s*=\s*(\{.*?\})\s*;", source, re.S)
    if dimensions is None or names_match is None:
        raise ValueError("Expected the file JSX canvas dimensions and region names.")
    width, height = map(int, dimensions.groups())
    if not (1 <= width <= 30000 and 1 <= height <= 30000):
        raise ValueError("PSD dimensions must be between 1 and 30000 pixels.")
    # The source names object has numeric property names (legal JS, not JSON).
    names_json = re.sub(r"([,{]\s*)(\d+)\s*:", r'\1"\2":', names_match[1])
    names = {int(key): value for key, value in json.loads(names_json).items()}
    if not names or any(not isinstance(v, str) or not v for v in names.values()):
        raise ValueError("Invalid region names.")
    order = read_json_variable(source, "order")
    if any(type(i) is not int for i in order):
        raise ValueError("Region order must contain integer IDs.")
    if len(set(order)) != len(order) or set(order) != set(names):
        raise ValueError("Region order must contain every region exactly once.")
    records = read_json_variable(source, "DATA")
    if not records:
        raise ValueError("The JSX contains no words.")
    words = []
    for index, row in enumerate(records, 1):
        try:
            if not isinstance(row, list) or len(row) != 11:
                raise ValueError("expected 11 fields")
            region, text, x, y, w, h, size, angle, rgb, quote, colors = row
            if any(type(v) is not int for v in (region, x, y, w, h, quote)):
                raise ValueError("region, bounds and quote index must be integers")
            if region not in names or quote < 0:
                raise ValueError("invalid region or quote index")
            if not isinstance(text, str) or not text.strip() or "\n" in text or "\r" in text:
                raise ValueError("expected nonempty single-line text")
            if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
                raise ValueError("word is outside the canvas")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in (size, angle)):
                raise ValueError("invalid font size or angle")
            if not 0 < size <= 1000:
                raise ValueError("font size must be between 0 and 1000")
            if not isinstance(rgb, list) or len(rgb) != 3 or any(
                type(c) is not int or not 0 <= c <= 255 for c in rgb
            ):
                raise ValueError("invalid RGB color")
            if not isinstance(colors, list) or (colors and len(colors) != len(text)):
                raise ValueError("character color count does not match the text")
            if any(type(c) is not int or not 0 <= c <= 0xFFFFFF for c in colors):
                raise ValueError("invalid packed character color")
            words.append(Word(region, text, x, y, w, h, size, angle, tuple(rgb), quote, tuple(colors)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid word {index}: {exc}") from exc
    if len(words) + 2 * (len(names) + 1) + 2 > 32767:
        raise ValueError("Too many layer records for a PSD file.")
    return Layout(width, height, names, order, words)


def find_font(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise ValueError(f"Font does not exist: {explicit}")
        return explicit.resolve()
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = [
        windows / "arialbd.ttf",
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise ValueError("No supported bold font found. Use --font path/to/a-bold-font.ttf")


@lru_cache(maxsize=256)
def load_font(path: str, size: int):
    return ImageFont.truetype(path, size)


def crop_ink(image: Image.Image) -> Image.Image:
    bounds = image.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("Font produced no visible characters.")
    return image.crop(bounds)


def render_word(word: Word, font_path: Path, supersample: int = 4) -> Image.Image:
    font = load_font(str(font_path), max(1, round(word.font_size * supersample)))
    left, top, right, bottom = font.getbbox(word.text, anchor="ls")
    pad = max(4, font.size)
    tile = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad))
    draw = ImageDraw.Draw(tile)
    for index, character in enumerate(word.text):
        # Include the preceding kerning pair while keeping each letter's color.
        advance = font.getlength(word.text[:index + 1]) - font.getlength(character)
        if word.colors:
            packed = word.colors[index]
            rgb = ((packed >> 16) & 255, (packed >> 8) & 255, packed & 255)
        else:
            rgb = word.color
        draw.text((pad - left + advance, pad - top), character, font=font,
                  fill=(*rgb, 255), anchor="ls")
    tile = crop_ink(tile)
    # Photoshop scales to 83% horizontally BEFORE rotating. Pillow's positive
    # angle is counterclockwise; the source JSX's positive angle is clockwise.
    tile = tile.resize((max(1, round(tile.width * 0.83)), tile.height), Image.Resampling.LANCZOS)
    if word.angle:
        tile = tile.rotate(-word.angle, resample=Image.Resampling.BICUBIC, expand=True)
    tile = crop_ink(tile)
    return tile.resize((word.width, word.height), Image.Resampling.LANCZOS)


def add_metadata(psd: PSDImage) -> bytes:
    psd.image_resources[Resource.RESOLUTION_INFO] = ImageResource(
        key=Resource.RESOLUTION_INFO,
        data=ResoulutionInfo(horizontal=300 * 65536, horizontal_unit=1, width_unit=1,
                            vertical=300 * 65536, vertical_unit=1, height_unit=1),
    )
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    psd.image_resources[Resource.ICC_PROFILE] = ImageResource(key=Resource.ICC_PROFILE, data=profile)
    return profile


def write_psd(psd: PSDImage, merged: Image.Image, destination: Path) -> None:
    """Save both the layer tree and an accurate merged image without SciPy.

    All layers use normal alpha blending, already composed below with Pillow.
    These psd-tools low-level calls are tested with the pinned version in
    requirements.txt. Bypass save()'s optional, much heavier compositor so the
    file never gets a blank/stale preview when that optional extra is absent.
    """
    psd._update_record()
    psd._record.image_data.compression = Compression.RLE
    psd._record.image_data.set_data([band.tobytes() for band in merged.split()], psd._record.header)
    # A negative count identifies the merged fourth channel as transparency.
    layer_info = psd._record.layer_and_mask_information.layer_info
    layer_info.layer_count = -abs(layer_info.layer_count)
    with destination.open("wb") as stream:
        psd._record.write(stream)


def convert(source: Path, output: Path, font: Path | None = None,
            supersample: int = 4, force: bool = False) -> tuple[Path, Path, int]:
    source, output = source.resolve(), output.resolve()
    preview_path = output.with_suffix(".png")
    if output.suffix.lower() != ".psd":
        raise ValueError("The output filename must end in .psd.")
    if not 1 <= supersample <= 8:
        raise ValueError("Supersample must be between 1 and 8.")
    reference_path = source.with_name("file_Reference.png")
    if source in (output, preview_path) or reference_path in (output, preview_path):
        raise ValueError("Output must not overwrite the JSX or reference image.")
    if not force:
        for path in (output, preview_path):
            if path.exists():
                raise ValueError(f"Already exists: {path}\nUse --force to replace generated files.")
    layout = read_layout(source)
    font_path = find_font(font)
    print(f"Canvas: {layout.width} x {layout.height}, 300 ppi; words: {len(layout.words)}", flush=True)
    print(f"Font: {font_path}", flush=True)
    psd = PSDImage.new("RGBA", (layout.width, layout.height), depth=8)
    profile = add_metadata(psd)
    merged = Image.new("RGBA", psd.size, (0, 0, 0, 255))
    PixelLayer.frompil(merged, parent=psd, name="00 Solid black background")
    if reference_path.is_file():
        with Image.open(reference_path) as reference:
            reference = reference.convert("RGBA").resize(psd.size, Image.Resampling.LANCZOS)
            layer = PixelLayer.frompil(reference, parent=psd, name="REFERENCE - original image - hidden")
            layer.visible = False
    root = Group.new(parent=psd, name="file - typography (paint layers)", open_folder=True)
    root.blend_mode = BlendMode.PASS_THROUGH
    by_region = defaultdict(list)
    for index, word in enumerate(layout.words, 1):
        by_region[word.region].append((index, word))
    done = 0
    # Both PSD groups and words are stored bottom-to-top, matching JSX add().
    # Drawing region-by-region also makes the merged preview match the PSD.
    for region in layout.order:
        group = Group.new(parent=root, name=layout.names[region], open_folder=False)
        group.blend_mode = BlendMode.PASS_THROUGH
        for index, word in by_region[region]:
            tile = render_word(word, font_path, supersample)
            PixelLayer.frompil(tile, parent=group,
                              name=f"{index} | {word.text} | Q{word.quote + 1}",
                              left=word.x, top=word.y)
            merged.alpha_composite(tile, (word.x, word.y))
            done += 1
        print(f"  {done}/{len(layout.words)}: {layout.names[region]}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Finish and verify temporary files before replacing any existing output.
    with tempfile.TemporaryDirectory(prefix="jsx_to_psd_", dir=output.parent) as temp:
        psd_temp = Path(temp) / "portrait.psd"
        png_temp = Path(temp) / "preview.png"
        print("Writing and checking PSD...", flush=True)
        write_psd(psd, merged, psd_temp)
        merged.convert("RGB").save(png_temp, dpi=(300, 300), icc_profile=profile)
        reopened = PSDImage.open(psd_temp)
        word_layers = [layer for layer in reopened.descendants() if " | " in layer.name]
        if reopened.size != psd.size or len(word_layers) != len(layout.words):
            raise RuntimeError("PSD validation failed: dimensions or word layers do not match.")
        if reopened.topil(apply_icc=False).convert("RGBA").tobytes() != merged.tobytes():
            raise RuntimeError("PSD validation failed: merged preview does not match.")
        os.replace(psd_temp, output)
        os.replace(png_temp, preview_path)
    print(f"PSD: {output}\nPreview: {preview_path}", flush=True)
    print("Words are separate paint layers, not editable text. Open the PSD in Krita.", flush=True)
    return output, preview_path, len(layout.words)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path,
                        default=Path(__file__).resolve().with_name("file.jsx"))
    parser.add_argument("-o", "--output", type=Path, help="output PSD (default: beside the JSX)")
    parser.add_argument("--font", type=Path, help="bold TTF/OTF font; Arial Bold is preferred")
    parser.add_argument("--supersample", type=int, choices=range(1, 9), default=4, metavar="1-8")
    parser.add_argument("--force", action="store_true", help="replace existing output PSD and PNG")
    args = parser.parse_args()
    output = args.output or args.source.with_name(args.source.stem.removesuffix("_Photoshop") + ".psd")
    try:
        convert(args.source, output, args.font, args.supersample, args.force)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
