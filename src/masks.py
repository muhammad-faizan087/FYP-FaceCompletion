import numpy as np
import cv2

MASK_TYPES = ['free_form', 'wide', 'box', 'center', 'half', 'occlusion']
MASK_WEIGHTS = [0.30, 0.20, 0.15, 0.05, 0.05, 0.25]


def _free_form(h, w, rng, target, brush_range, max_vertex=8, max_len=60):
    # Random brush strokes (idea from Gated Convolution): each stroke is a few connected line segments
    s = h / 256.0
    m = np.zeros((h, w), np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        angle = rng.uniform(0, 2 * np.pi)
        width = max(1, int(rng.integers(brush_range[0], brush_range[1] + 1) * s))
        for _ in range(int(rng.integers(1, max_vertex + 1))):
            angle += rng.uniform(-1.2, 1.2)
            length = int(rng.integers(10, max_len + 1) * s) + 1
            nx = int(np.clip(x + length * np.cos(angle), 0, w - 1))
            ny = int(np.clip(y + length * np.sin(angle), 0, h - 1))
            cv2.line(m, (x, y), (nx, ny), 1, width)
            cv2.circle(m, (nx, ny), width // 2, 1, -1)
            x, y = nx, ny
        if m.mean() >= target:
            break
    return m


def _boxes(h, w, rng, target, n_boxes):
    # One to three rectangles that together cover about `target` of the image
    m = np.zeros((h, w), np.uint8)
    for _ in range(n_boxes):
        area = target * h * w / n_boxes
        aspect = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
        bh = int(np.clip(np.sqrt(area / aspect), 4, h))
        bw = int(np.clip(area / bh, 4, w))
        y0 = int(rng.integers(0, h - bh + 1))
        x0 = int(rng.integers(0, w - bw + 1))
        m[y0:y0 + bh, x0:x0 + bw] = 1
    return m


def _center(h, w, rng, target):
    # Square hole in the middle (Context Encoder style)
    side = int(np.sqrt(target * h * w))
    y0, x0 = (h - side) // 2, (w - side) // 2
    m = np.zeros((h, w), np.uint8)
    m[y0:y0 + side, x0:x0 + side] = 1
    return m


def _half(h, w, rng):
    # Left, right, top or bottom half is missing (ratio is always 0.5)
    m = np.zeros((h, w), np.uint8)
    side = int(rng.integers(0, 4))
    if side == 0:
        m[:, :w // 2] = 1
    elif side == 1:
        m[:, w // 2:] = 1
    elif side == 2:
        m[:h // 2, :] = 1
    else:
        m[h // 2:, :] = 1
    return m


def _occlusion(h, w, rng, target):
    # Object-like occluders: a band over the eyes (sunglasses/mask-like) or random ellipses (hand/object-like)
    m = np.zeros((h, w), np.uint8)
    if rng.random() < 0.4:
        bh = int(np.clip(target * h * 0.9, 6, h))
        bw = int(np.clip(target * h * w / bh, 6, w))
        cy = int(rng.uniform(0.35, 0.55) * h)
        cx = int(w / 2 + rng.uniform(-0.05, 0.05) * w)
        cv2.rectangle(m, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), 1, -1)
    else:
        for _ in range(12):
            c = (int(rng.uniform(0.2, 0.8) * w), int(rng.uniform(0.2, 0.8) * h))
            ax = (int(rng.uniform(0.08, 0.3) * w), int(rng.uniform(0.08, 0.3) * h))
            cv2.ellipse(m, c, ax, float(rng.uniform(0, 180)), 0, 360, 1, -1)
            if m.mean() >= target:
                break
    return m


def make_mask(h=256, w=256, rng=None, mask_type=None, ratio_range=(0.10, 0.50), tries=20):
    # Returns a float32 (h, w) array: 1 = hole, 0 = valid pixel.
    # rng=None -> fresh random mask every call (dynamic masking for training)
    # rng=np.random.default_rng(seed) -> reproducible mask (use for val/test)
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = ratio_range
    types = [t for t in MASK_TYPES if t != 'half' or lo <= 0.5 <= hi]
    wts = np.array([MASK_WEIGHTS[MASK_TYPES.index(t)] for t in types])
    wts = wts / wts.sum()
    m = None
    for _ in range(tries):
        t = mask_type or str(rng.choice(types, p=wts))
        target = rng.uniform(lo, hi)
        if t == 'free_form':
            m = _free_form(h, w, rng, target, (6, 16))
        elif t == 'wide':
            m = _free_form(h, w, rng, target, (24, 48), max_vertex=4, max_len=80)
        elif t == 'box':
            m = _boxes(h, w, rng, target, int(rng.integers(1, 4)))
        elif t == 'center':
            m = _center(h, w, rng, target)
        elif t == 'half':
            m = _half(h, w, rng)
        else:
            m = _occlusion(h, w, rng, target)
        if lo <= m.mean() <= hi:
            break
    return m.astype(np.float32)
