"""Generate test/synthetic_captions.png: outlined multi-speaker vertical captions.

A stress test for pipeline/core/detection/captions.py: several speakers' columns side
by side in different fill colours (white, orange, blues), with a blue column squeezed
tightly between orange and white ones, outlines touching, over a busy background
(dark blobs, light patches, gradients and noise). A third group tests colours close to
each other (white / pale lavender, pink / light orange), shaded fill and a dark-grey
background. A fourth is packed tight (outlines fully merged) with a longer, higher
rightmost column, alternating colours, a speaker continuing over two columns, a
skin-tone patch behind it and pink sound-effect lettering beside it.

    python tools/make_synthetic_captions.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "test", "synthetic_captions.png")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_INDEX = 0  # Noto Sans CJK JP

W, H = 1131, 1600
SIZE = 44
STROKE = 6

WHITE = (255, 255, 255)
ORANGE = (255, 140, 80)
BLUE = (150, 150, 225)        # light lavender-blue
DIM_BLUE = (95, 110, 190)     # darker blue, close to the outline-darkness cut-off
PINK = (250, 155, 175)
LAVENDER = (215, 205, 245)    # pale, close to white
LIGHT_ORANGE = (255, 185, 140)  # close to pink
CREAM = (250, 240, 215)
SHADED_BLUE = ((190, 200, 255), (110, 125, 215))  # fill shaded from light (top) to dark (bottom)

# (x of column centre, top y, fill colour, text). Columns are listed right to left.
CAPTIONS = [
    # Right group: white / orange / squeezed blue / white, columns 50px apart (touching outlines).
    # The second white column runs far below the rest, which detectors tend to cut short.
    [(1070, 50, WHITE, "ある日の放課後、駅前の本屋に寄った。"),
     (1020, 60, ORANGE, "「この新刊、もう出てたんだ！」"),
     (970, 60, BLUE, "「ほんとだ、表紙きれいだね」"),
     (920, 60, WHITE, "二人はしばらく棚の前で悩んでいたが、結局どちらも買わずに帰った。"),
     (870, 60, ORANGE, "「やっぱり両方買っちゃおうかな」")],
    # Left group: dim blue squeezed between orange and white, over dark background.
    [(290, 60, ORANGE, "「明日は雨らしいよ」"),
     (240, 60, DIM_BLUE, "「えっ、傘持ってないのに…」"),
     (190, 60, WHITE, "空はまだ晴れていたが、風が冷たくなってきた。"),
     (140, 60, PINK, "「コンビニで買えばいいじゃない」"),
     (90, 60, BLUE, "「それもそうだね」")],
    # Bottom-middle group, over a dark-grey area: colours close to each other (white /
    # lavender, pink / light orange), a column with shaded fill, and cream.
    [(660, 1000, WHITE, "夕方、雨が降り始めた。"),
     (610, 1000, LAVENDER, "「降ってきたね」"),
     (560, 1000, PINK, "「早く帰ろうよ」"),
     (510, 1000, LIGHT_ORANGE, "「待って、靴ひもが…」"),
     (460, 1000, SHADED_BLUE, "「走れば間に合う」"),
     (410, 1000, CREAM, "二人は駅まで走った。")],
    # Top-middle group, packed tight: columns 44px apart (= the font size, outlines fully
    # merged). The rightmost narration column starts higher and runs longer; colours
    # alternate; two white narration columns in a row are one speaker continuing.
    # Over a skin-tone patch, next to large pink outlined sound-effect lettering.
    [(770, 20, WHITE, "その日の夕方、二人は公園で話していた。"),
     (726, 40, ORANGE, "「明日の試合、見に来てくれる？」"),
     (682, 40, LAVENDER, "「もちろん行くよ」"),
     (638, 40, ORANGE, "「よかった、がんばるね！」"),
     (594, 40, WHITE, "空にはもう星が見え始めていた。"),
     (550, 40, WHITE, "風が少し冷たくなってきた。"),
     (506, 40, PINK, "「そろそろ帰ろうか」"),
     (462, 40, ORANGE, "「うん、また明日ね」")],
]

# Art lettering (sound effects) drawn before the captions: (x centre, top y, size, colour, text).
SOUND_EFFECTS = [(395, 560, 64, (235, 60, 200), "ドキッ")]


def background() -> Image.Image:
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:H, 0:W]
    base = np.stack([200 + 30 * np.sin(x / 170), 150 + 40 * np.cos(y / 230), 110 + 30 * np.sin((x + y) / 300)], -1)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    # Dark "hair"-like blobs behind the left captions, a light "sleeve" patch, mid-tone shapes.
    d.ellipse((-150, -100, 380, 700), fill=(25, 30, 60))
    d.polygon([(0, 650), (330, 520), (360, 900), (0, 1000)], fill=(40, 45, 90))
    d.rectangle((200, 350, 460, 520), fill=(245, 245, 240))
    d.ellipse((820, 500, 1131, 900), fill=(230, 190, 170))
    d.rectangle((960, 0, 1131, 1600), fill=(100, 70, 50))
    d.rectangle((370, 960, 720, 1600), fill=(58, 62, 60))  # dark grey behind the bottom group
    d.ellipse((560, 300, 760, 700), fill=(245, 210, 190))  # skin-tone patch behind the top-middle group
    for _ in range(60):
        cx, cy, r = rng.integers(0, W), rng.integers(0, H), rng.integers(5, 40)
        d.line((cx, cy, cx + rng.integers(-200, 200), cy + rng.integers(-200, 200)),
               fill=tuple(int(v) for v in rng.integers(20, 240, 3)), width=int(r // 5) + 1)
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    noise = rng.normal(0, 6, (H, W, 3))
    return Image.fromarray(np.clip(np.asarray(img) + noise, 0, 255).astype(np.uint8))


def main():
    img = background()
    d = ImageDraw.Draw(img)
    for cx, top, size, color, text in SOUND_EFFECTS:
        sfx_font = ImageFont.truetype(FONT, size, index=FONT_INDEX)
        y = top
        for ch in text:
            w = d.textlength(ch, font=sfx_font)
            d.text((cx - w / 2, y), ch, font=sfx_font, fill=color, stroke_width=STROKE + 2, stroke_fill=(0, 0, 0))
            y += size
    font = ImageFont.truetype(FONT, SIZE, index=FONT_INDEX)
    for group in CAPTIONS:
        for cx, top, color, text in group:
            y = top
            for ch in text:
                w = d.textlength(ch, font=font)
                if isinstance(color[0], tuple):
                    # Shaded fill: draw the outline, then paint a vertical gradient
                    # through the letter's fill mask.
                    d.text((cx - w / 2, y), ch, font=font, fill=(0, 0, 0), stroke_width=STROKE, stroke_fill=(0, 0, 0))
                    mask = Image.new("L", img.size, 0)
                    ImageDraw.Draw(mask).text((cx - w / 2, y), ch, font=font, fill=255)
                    t = np.clip((np.arange(H) - top) / max(1, len(text) * (SIZE + 2)), 0, 1)[:, None, None]
                    gradient = (np.array(color[0]) * (1 - t) + np.array(color[1]) * t).repeat(W, axis=1)
                    img.paste(Image.fromarray(gradient.astype(np.uint8)), (0, 0), mask)
                else:
                    d.text((cx - w / 2, y), ch, font=font, fill=color, stroke_width=STROKE, stroke_fill=(0, 0, 0))
                y += SIZE + 2
    img.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
