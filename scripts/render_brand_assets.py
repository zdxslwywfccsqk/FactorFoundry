from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
FONT_CANDIDATES = {
    "regular": [
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ],
    "bold": [
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ],
}

INK = (5, 14, 18, 255)
INK_2 = (8, 25, 31, 255)
WHITE = (239, 249, 247, 255)
MUTED = (143, 173, 174, 255)
MINT = (57, 242, 193, 255)
MINT_SOFT = (126, 255, 220, 255)
BLUE = (92, 151, 255, 255)
AMBER = (255, 184, 92, 255)
LINE = (62, 105, 111, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES["bold" if bold else "regular"]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius, fill=255)
    return mask


def vertical_gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)
    for y in range(size[1]):
        t = y / max(1, size[1] - 1)
        color = tuple(round(top[i] * (1 - t) + bottom[i] * t) for i in range(3)) + (255,)
        draw.line((0, y, size[0], y), fill=color)
    return image


def glow_layer(size: tuple[int, int], blobs: list[tuple[int, int, int, tuple[int, int, int, int]]]) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for x, y, radius, color in blobs:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return layer.filter(ImageFilter.GaussianBlur(80))


def draw_grid(image: Image.Image, spacing: int = 56, alpha: int = 22) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x in range(0, image.width, spacing):
        draw.line((x, 0, x, image.height), fill=(97, 143, 148, alpha), width=1)
    for y in range(0, image.height, spacing):
        draw.line((0, y, image.width, y), fill=(97, 143, 148, alpha), width=1)
    image.alpha_composite(overlay)


def draw_spotlights(image: Image.Image) -> None:
    rng = random.Random(42)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(70):
        x = rng.randrange(0, image.width)
        y = rng.randrange(0, image.height)
        radius = rng.choice((1, 1, 1, 2))
        alpha = rng.randrange(28, 95)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*MINT_SOFT[:3], alpha))
    image.alpha_composite(overlay)


def draw_pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int = 22,
    color: tuple[int, int, int, int] = MINT,
) -> int:
    fnt = font(size, bold=True)
    padding_x = round(size * 0.85)
    padding_y = round(size * 0.45)
    left, top = xy
    bbox = draw.textbbox((0, 0), text, font=fnt)
    width = bbox[2] - bbox[0] + padding_x * 2
    height = bbox[3] - bbox[1] + padding_y * 2
    fill = mix((8, 25, 29), color, 0.18)
    outline = mix((8, 25, 29), color, 0.60)
    draw.rounded_rectangle(
        (left, top, left + width, top + height),
        radius=height // 2,
        fill=fill,
        outline=outline,
        width=1,
    )
    draw.text((left + padding_x, top + padding_y - bbox[1]), text, font=fnt, fill=color)
    return width


def draw_mark(draw: ImageDraw.ImageDraw, xy: tuple[int, int], size: int) -> None:
    x, y = xy
    center = (x + size / 2, y + size / 2)
    points = []
    for index in range(6):
        angle = math.radians(60 * index - 30)
        points.append((center[0] + size * 0.46 * math.cos(angle), center[1] + size * 0.46 * math.sin(angle)))
    draw.polygon(points, fill=(*MINT[:3], 18), outline=(*MINT[:3], 150))
    draw.line(points + [points[0]], fill=(*MINT[:3], 220), width=max(2, size // 28))
    inner = size * 0.26
    draw.ellipse((center[0] - inner, center[1] - inner, center[0] + inner, center[1] + inner), outline=(*BLUE[:3], 210), width=max(2, size // 34))
    draw.text((center[0] - size * 0.16, center[1] - size * 0.24), "α", font=font(round(size * 0.47), bold=True), fill=WHITE)


def mix(base: tuple[int, int, int], accent: tuple[int, int, int, int], weight: float) -> tuple[int, int, int, int]:
    return tuple(round(base[i] * (1 - weight) + accent[i] * weight) for i in range(3)) + (255,)


def base_canvas(size: tuple[int, int]) -> Image.Image:
    image = vertical_gradient(size, (4, 12, 16), (7, 28, 33)).convert("RGBA")
    image.alpha_composite(
        glow_layer(
            size,
            [
                (round(size[0] * 0.86), round(size[1] * 0.08), 190, (*MINT[:3], 42)),
                (round(size[0] * 0.16), round(size[1] * 0.94), 210, (*BLUE[:3], 34)),
                (round(size[0] * 0.58), round(size[1] * 0.68), 140, (*AMBER[:3], 12)),
            ],
        )
    )
    draw_grid(image, alpha=18)
    draw_spotlights(image)
    return image


def draw_hero() -> Image.Image:
    width, height = 1600, 680
    image = base_canvas((width, height))
    draw = ImageDraw.Draw(image)

    draw_mark(draw, (72, 54), 72)
    draw.text((162, 62), "FACTORFOUNDRY", font=font(27, bold=True), fill=WHITE)
    draw.text((163, 97), "OPEN-SOURCE ALPHA RESEARCH INFRASTRUCTURE", font=font(15, bold=True), fill=(*MINT[:3], 210))

    kicker = "MULTI-AGENT  ·  DETERMINISTIC  ·  AUDITABLE"
    draw.text((74, 194), kicker, font=font(20, bold=True), fill=(*MINT[:3], 225))

    title = "Turn hypotheses into"
    title_2 = "evidence, not stories."
    draw.text((72, 230), title, font=font(65, bold=True), fill=WHITE)
    draw.text((72, 304), title_2, font=font(65, bold=True), fill=(*MINT_SOFT[:3], 255))

    subtitle = "An open-source research lab where LLM agents propose and critique,"
    subtitle_2 = "while deterministic tools compute, audit, and lock the holdout."
    draw.text((75, 399), subtitle, font=font(25), fill=MUTED)
    draw.text((75, 437), subtitle_2, font=font(25), fill=MUTED)

    x = 75
    for label in ("HOLDOUT LOCK", "FDR · DSR · PBO", "COST-AWARE", "MINI-AGENT"):
        x += draw_pill(draw, (x, 510), label, size=17, color=MINT if label != "COST-AWARE" else AMBER) + 12

    draw.text((76, 602), "github.com/zdxslwywfccsqk/FactorFoundry", font=font(19, bold=True), fill=(*BLUE[:3], 220))

    panel = (1050, 110, 1518, 578)
    panel_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel_layer)
    panel_draw.rounded_rectangle(panel, radius=28, fill=(6, 20, 25, 210), outline=(*LINE[:3], 120), width=1)
    image.alpha_composite(panel_layer)

    draw = ImageDraw.Draw(image)
    draw.text((1082, 146), "RESEARCH LOOP", font=font(16, bold=True), fill=(*MUTED[:3], 220))
    draw.text((1082, 176), "Agentic surface", font=font(33, bold=True), fill=WHITE)
    draw.text((1082, 219), "Six bounded roles. One evidence chain.", font=font(17), fill=MUTED)

    roles = [
        ("01", "HYPOTHESIS", "economic mechanism"),
        ("02", "FEASIBILITY", "field + coverage"),
        ("03", "ENGINEER", "DSL implementation"),
        ("04", "EVALUATOR", "cost-aware metrics"),
        ("05", "CRITIC", "veto + evidence"),
        ("06", "PORTFOLIO", "low-correlation mix"),
    ]
    for index, (number, name, detail) in enumerate(roles):
        row = index // 2
        col = index % 2
        left = 1082 + col * 204
        top = 272 + row * 86
        color = MINT if index in (0, 4) else BLUE if index in (1, 3, 5) else AMBER
        card_fill = mix((7, 24, 28), color, 0.13)
        card_outline = mix((7, 24, 28), color, 0.48)
        draw.rounded_rectangle((left, top, left + 188, top + 68), radius=12, fill=card_fill, outline=card_outline)
        draw.text((left + 13, top + 10), number, font=font(12, bold=True), fill=(*color[:3], 190))
        draw.text((left + 42, top + 8), name, font=font(15, bold=True), fill=WHITE)
        draw.text((left + 13, top + 38), detail, font=font(13), fill=MUTED)

    gate = (1082, 536, 1484, 558)
    draw.rounded_rectangle(gate, radius=11, fill=mix((7, 24, 28), MINT, 0.18), outline=mix((7, 24, 28), MINT, 0.68))
    draw.text((1110, 539), "DETERMINISTIC GATE  ·  test locked", font=font(13, bold=True), fill=(*MINT_SOFT[:3], 245))

    return image.convert("RGB")


def draw_social_preview() -> Image.Image:
    width, height = 1280, 640
    image = base_canvas((width, height))
    draw = ImageDraw.Draw(image)

    draw_mark(draw, (62, 48), 64)
    draw.text((142, 55), "FACTORFOUNDRY", font=font(24, bold=True), fill=WHITE)
    draw.text((143, 86), "OPEN-SOURCE ALPHA RESEARCH", font=font(13, bold=True), fill=(*MINT[:3], 210))

    draw.text((64, 168), "Agents propose.", font=font(62, bold=True), fill=WHITE)
    draw.text((64, 238), "Evidence decides.", font=font(62, bold=True), fill=(*MINT_SOFT[:3], 255))
    draw.text((66, 326), "A multi-agent lab for auditable alpha discovery on A-shares.", font=font(23), fill=MUTED)

    x = 66
    for label in ("HOLDOUT LOCK", "FDR · DSR · PBO", "COST-AWARE", "MINI-AGENT"):
        x += draw_pill(draw, (x, 394), label, size=15, color=MINT if label != "COST-AWARE" else AMBER) + 10

    draw.text((66, 535), "github.com/zdxslwywfccsqk/FactorFoundry", font=font(18, bold=True), fill=(*BLUE[:3], 225))

    center = (1050, 320)
    for radius, alpha in ((196, 28), (150, 42), (104, 62)):
        draw.ellipse(
            (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
            outline=(*MINT[:3], alpha),
            width=1,
        )
    nodes = [
        (-136, -106, "H"),
        (104, -144, "F"),
        (154, 22, "E"),
        (46, 152, "C"),
        (-146, 110, "P"),
    ]
    for nx, ny, label in nodes:
        point = (center[0] + nx, center[1] + ny)
        draw.ellipse((point[0] - 31, point[1] - 31, point[0] + 31, point[1] + 31), fill=(6, 24, 29, 245), outline=(*MINT[:3], 150), width=2)
        bbox = draw.textbbox((0, 0), label, font=font(17, bold=True))
        draw.text((point[0] - (bbox[2] - bbox[0]) / 2, point[1] - (bbox[3] - bbox[1]) / 2 - 2), label, font=font(17, bold=True), fill=WHITE)
    draw.rounded_rectangle(
        (center[0] - 75, center[1] - 39, center[0] + 75, center[1] + 39),
        radius=18,
        fill=mix((7, 24, 28), MINT, 0.28),
        outline=mix((7, 24, 28), MINT, 0.82),
        width=2,
    )
    draw.text((center[0] - 49, center[1] - 14), "GATE", font=font(19, bold=True), fill=MINT_SOFT)

    return image.convert("RGB")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    draw_hero().save(ASSETS / "hero.png", optimize=True)
    draw_social_preview().save(ASSETS / "social-preview.png", optimize=True)
    print(f"wrote {ASSETS / 'hero.png'}")
    print(f"wrote {ASSETS / 'social-preview.png'}")


if __name__ == "__main__":
    main()
