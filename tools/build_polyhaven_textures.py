#!/usr/bin/env python3
"""Build the 512px PBR block texture set from Poly Haven CC0 materials."""

from __future__ import annotations

import hashlib
import io
import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "textures"
CACHE = ROOT / "target" / "polyhaven-source"
SIZE = 512
SOURCES = {
    "leafy_grass",
    "dirt_aerial_02",
    "rock_ground_02",
    "cobblestone_02",
    "sand_03",
    "sandstone_blocks_04",
    "gravel_stones",
    "clay_plaster",
    "bark_brown_02",
    "brown_planks_04",
    "forest_leaves_02",
    "volcanic_rock_tiles",
    "metal_plate_02",
    "fine_grained_wood",
}


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "mc-texture-builder/0.1"})
    return urllib.request.urlopen(request).read()


def manifest(slug: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{slug}.json"
    if not path.exists():
        data = download(f"https://api.polyhaven.com/files/{slug}")
        path.write_bytes(data)
    return json.loads(path.read_text())


def fetch(slug: str, map_name: str) -> Image.Image:
    key = next(k for k in manifest(slug) if k.lower() == map_name.lower())
    entry = manifest(slug)[key]["1k"]["jpg"]
    path = CACHE / f"{slug}_{map_name.lower()}.jpg"
    if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != entry["md5"]:
        data = download(entry["url"])
        if hashlib.md5(data).hexdigest() != entry["md5"]:
            raise RuntimeError(f"Poly Haven checksum mismatch: {slug}/{map_name}")
        path.write_bytes(data)
    return Image.open(io.BytesIO(path.read_bytes())).convert("RGB")


def material(slug: str, *, centering=(0.5, 0.5)) -> tuple[Image.Image, Image.Image, Image.Image]:
    def fit(image: Image.Image, *, sharpen=False) -> Image.Image:
        result = ImageOps.fit(image, (SIZE, SIZE), Image.Resampling.LANCZOS, centering=centering)
        if sharpen:
            result = result.filter(ImageFilter.UnsharpMask(radius=1.15, percent=85, threshold=2))
        return result

    albedo = fit(fetch(slug, "Diffuse"), sharpen=True)
    normal = fit(fetch(slug, "nor_gl"))
    roughness = fit(fetch(slug, "Rough").convert("L")).convert("RGBA")
    return albedo.convert("RGBA"), normal.convert("RGBA"), roughness


def tint(image: Image.Image, color: tuple[int, int, int], amount: float) -> Image.Image:
    return Image.blend(image.convert("RGBA"), Image.new("RGBA", image.size, (*color, 255)), amount)


def adjusted(image: Image.Image, *, contrast=1.0, color=1.0, brightness=1.0) -> Image.Image:
    rgb = image.convert("RGB")
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Color(rgb).enhance(color)
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    return rgb.convert("RGBA")


def with_alpha(image: Image.Image, alpha: Image.Image) -> Image.Image:
    result = image.convert("RGBA")
    result.putalpha(alpha)
    return result


def save_set(name: str, albedo: Image.Image, normal: Image.Image, roughness: Image.Image) -> None:
    alpha = albedo.getchannel("A")
    stem = name.removesuffix(".png")
    albedo.save(OUT / name, optimize=True)
    with_alpha(normal, alpha).save(OUT / f"{stem}_normal.png", optimize=True)
    with_alpha(roughness, alpha).save(OUT / f"{stem}_roughness.png", optimize=True)


def flat_normal(alpha=255) -> Image.Image:
    return Image.new("RGBA", (SIZE, SIZE), (128, 128, 255, alpha))


def constant_roughness(value: int, alpha=255) -> Image.Image:
    return Image.new("RGBA", (SIZE, SIZE), (value, value, value, alpha))


def px(value: int) -> int:
    """Scale authored face coordinates from their 256px design grid."""
    return value * SIZE // 256


def ore_set(base, color: tuple[int, int, int], roughness_value: int):
    albedo, normal, roughness = (part.copy() for part in base)
    mask = Image.new("L", (SIZE, SIZE), 0)
    mask_draw = ImageDraw.Draw(mask)
    # One deterministic vein in every 256px world-tile keeps ore readable while
    # the larger 2x2 photograph remains continuous across neighboring blocks.
    for gy in range(2):
        for gx in range(2):
            ox, oy = gx * 256, gy * 256
            cx = ox + 70 + (gx * 71 + gy * 43) % 120
            cy = oy + 62 + (gx * 31 + gy * 89) % 132
            points = [
                (cx - 52, cy + 12),
                (cx - 20, cy - 28),
                (cx + 4, cy - 16),
                (cx + 32, cy - 52),
                (cx + 48, cy - 32),
                (cx + 28, cy + 4),
                (cx + 56, cy + 32),
                (cx + 20, cy + 44),
                (cx - 8, cy + 24),
                (cx - 40, cy + 48),
            ]
            mask_draw.polygon(points, fill=205)
            mask_draw.ellipse((cx - 20, cy - 24, cx + 20, cy + 20), fill=245)
    mask = mask.filter(ImageFilter.GaussianBlur(3.2))
    outline = mask.filter(ImageFilter.MaxFilter(15))
    mineral = Image.blend(albedo, Image.new("RGBA", albedo.size, (*color, 255)), 0.64)
    shadow_color = tuple(int(c * 0.46) for c in color)
    shadow = Image.blend(albedo, Image.new("RGBA", albedo.size, (*shadow_color, 255)), 0.34)
    albedo = Image.composite(shadow, albedo, outline)
    albedo = Image.composite(mineral, albedo, mask)
    roughness = Image.composite(constant_roughness(roughness_value), roughness, mask)
    return albedo, normal, roughness


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Fetch in parallel; the public API publishes both URLs and checksums.
    for slug in SOURCES:
        manifest(slug)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: fetch(*item), ((s, m) for s in SOURCES for m in ("Diffuse", "nor_gl", "Rough"))))

    grass = material("leafy_grass", centering=(0.38, 0.62))
    grass = (adjusted(tint(grass[0], (54, 108, 43), 0.16), contrast=1.08, color=1.08), grass[1], grass[2])
    dirt = material("dirt_aerial_02", centering=(0.36, 0.64))
    stone = material("rock_ground_02", centering=(0.64, 0.40))
    stone = (adjusted(stone[0], color=0.78), stone[1], stone[2])
    cobble = material("cobblestone_02")
    sand = material("sand_03", centering=(0.62, 0.44))
    sandstone = material("sandstone_blocks_04")
    gravel = material("gravel_stones")
    clay = material("clay_plaster")
    bark = material("bark_brown_02")
    planks = material("brown_planks_04")
    leaves_source = material("forest_leaves_02")
    volcanic = material("volcanic_rock_tiles")
    metal = material("metal_plate_02")
    wood = material("fine_grained_wood")

    save_set("grass_block_top.png", *grass)
    mask = Image.new("L", (SIZE, SIZE), 0)
    mask_draw = ImageDraw.Draw(mask)
    for x in range(SIZE):
        edge = 76 + int(20 * math.sin(x * 0.045) + 10 * math.sin(x * 0.155))
        mask_draw.line((x, 0, x, max(24, edge)), fill=255)
        mask_draw.line((x, edge, x, edge + 32), fill=150)
    save_set("grass_block_side_overlay.png", *(with_alpha(part, mask) for part in grass))
    save_set("dirt.png", *dirt)
    save_set("stone.png", *stone)
    save_set("cobblestone.png", *cobble)
    save_set("sand.png", *sand)
    save_set("sandstone.png", *sandstone)
    save_set("sandstone_top.png", *sand)
    save_set("sandstone_bottom.png", tint(sandstone[0], (119, 94, 59), 0.12), sandstone[1], sandstone[2])
    save_set("gravel.png", *gravel)
    save_set("clay.png", tint(clay[0], (150, 162, 171), 0.18), clay[1], clay[2])
    save_set("oak_log.png", *bark)
    save_set("oak_planks.png", *planks)

    log_top = wood[0].copy()
    ring = ImageDraw.Draw(log_top)
    for inset, c in ((16, (84, 49, 25, 255)), (48, (181, 124, 69, 255)), (90, (105, 65, 34, 255)), (138, (198, 143, 78, 255)), (188, (112, 72, 39, 255))):
        ring.ellipse((inset, inset, SIZE - inset, SIZE - inset), outline=c, width=10)
    save_set("oak_log_top.png", log_top, wood[1], wood[2])

    leaves_albedo = adjusted(tint(leaves_source[0], (40, 112, 42), 0.48), contrast=1.15, color=1.1)
    noise = leaves_source[0].convert("L").filter(ImageFilter.GaussianBlur(2.4))
    leaves_mask = noise.point(lambda p: 255 if p > 82 else 0)
    save_set("oak_leaves.png", with_alpha(leaves_albedo, leaves_mask), leaves_source[1], leaves_source[2])

    glass_alpha = metal[0].convert("L").point(lambda p: 28 + p // 7)
    glass_alpha_draw = ImageDraw.Draw(glass_alpha)
    glass_alpha_draw.rectangle((0, 0, SIZE - 1, SIZE - 1), outline=145, width=px(4))
    save_set("glass.png", with_alpha(tint(metal[0], (165, 221, 230), 0.78), glass_alpha), flat_normal(), constant_roughness(38))
    save_set("bedrock.png", adjusted(volcanic[0], brightness=0.68, color=0.55), volcanic[1], volcanic[2])
    save_set("obsidian.png", tint(adjusted(volcanic[0], brightness=0.52), (24, 10, 44), 0.56), volcanic[1], constant_roughness(54))

    save_set("coal_ore.png", *ore_set(stone, (38, 41, 43), 185))
    save_set("iron_ore.png", *ore_set(stone, (177, 111, 72), 135))
    save_set("gold_ore.png", *ore_set(stone, (226, 174, 42), 72))
    save_set("diamond_ore.png", *ore_set(stone, (50, 205, 201), 45))
    save_set("redstone_ore.png", *ore_set(stone, (198, 31, 28), 92))

    water_luma = stone[0].convert("L").filter(ImageFilter.GaussianBlur(px(4)))
    water = ImageOps.colorize(water_luma, (11, 57, 107), (65, 163, 202)).convert("RGBA")
    save_set("water_overlay.png", with_alpha(water, Image.new("L", (SIZE, SIZE), 155)), flat_normal(), constant_roughness(32))

    shelf = planks[0].copy()
    shelf_draw = ImageDraw.Draw(shelf)
    colors = ((133, 43, 39), (43, 76, 119), (176, 128, 42), (52, 104, 64), (98, 56, 108))
    for row_base in (18, 137):
        row = px(row_base)
        shelf_draw.rectangle((0, row - px(10), SIZE, row), fill=(68, 40, 22, 255))
        for i, x_base in enumerate(range(12, 244, 25)):
            x = px(x_base)
            width = px(13 + (i * 7) % 9)
            shelf_draw.rounded_rectangle((x, row, x + width, row + px(92)), px(3), fill=colors[(i + row_base) % len(colors)] + (255,))
    save_set("bookshelf.png", shelf, planks[1], planks[2])
    save_set("bookshelf_top.png", *planks)

    hive = tint(wood[0], (219, 151, 58), 0.38)
    hive_draw = ImageDraw.Draw(hive)
    for y in map(px, (48, 101, 154, 207)):
        hive_draw.line((0, y, SIZE, y), fill=(95, 61, 28, 255), width=px(5))
    hive_draw.ellipse(tuple(map(px, (94, 108, 162, 148))), fill=(48, 34, 23, 255))
    save_set("beehive_side.png", hive, wood[1], wood[2])
    save_set("beehive_end.png", tint(wood[0], (222, 161, 77), 0.24), wood[1], wood[2])

    furnace = cobble[0].copy()
    furnace_draw = ImageDraw.Draw(furnace)
    furnace_draw.rounded_rectangle(tuple(map(px, (45, 55, 211, 218))), px(10), fill=(43, 46, 46, 255), outline=(20, 22, 23, 255), width=px(9))
    furnace_draw.rectangle(tuple(map(px, (75, 153, 181, 200))), fill=(135, 57, 24, 255))
    save_set("furnace_side.png", furnace, cobble[1], cobble[2])
    save_set("furnace_top.png", *cobble)

    piston_side = metal[0].copy()
    piston_draw = ImageDraw.Draw(piston_side)
    piston_draw.rounded_rectangle(tuple(map(px, (50, -8, 206, 264))), px(8), fill=(91, 96, 96, 255), outline=(40, 43, 44, 255), width=px(8))
    piston_draw.rectangle(tuple(map(px, (78, 25, 178, 231))), fill=(151, 103, 56, 255))
    save_set("piston_side.png", piston_side, metal[1], metal[2])
    piston_top = planks[0].copy()
    ImageDraw.Draw(piston_top).rounded_rectangle(tuple(map(px, (25, 25, 230, 230))), px(9), outline=(69, 72, 68, 255), width=px(14))
    save_set("piston_top.png", piston_top, planks[1], planks[2])
    save_set("piston_inner.png", *metal)
    print(f"wrote 33 albedo + normal + roughness sets to {OUT}")


if __name__ == "__main__":
    main()
