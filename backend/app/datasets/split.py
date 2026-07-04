"""Group-aware, stratified train/val split (DESIGN.md §8).

Split assignment operates on capture groups, never individual images, so
near-duplicates can't straddle the split. Per class: groups are ordered
deterministically and assigned to validation until the target image fraction
is reached, always leaving at least one group in train. A class with a single
group goes entirely to train and produces the §8 warning ("all N photos come
from one session — validation will be optimistic" — here: absent).
"""

VAL_FRACTION = 0.2
RECOMMENDED_MIN_PER_CLASS = 30


def assign_split(images: list[dict], val_fraction: float = VAL_FRACTION):
    """images: labeled, non-excluded rows. Returns (split: image_id -> 'train'|'val',
    warnings: [str])."""
    by_class: dict[str, dict[str, list[dict]]] = {}
    for img in images:
        by_class.setdefault(img["label"], {}).setdefault(img["capture_group"], []).append(img)

    split: dict[str, str] = {}
    warnings: list[str] = []

    for label in sorted(by_class):
        groups = by_class[label]
        n_images = sum(len(v) for v in groups.values())
        if n_images < RECOMMENDED_MIN_PER_CLASS:
            warnings.append(
                f'"{label}": only {n_images} images — aim for '
                f"{RECOMMENDED_MIN_PER_CLASS}+ for a usable model")

        ordered = sorted(groups)  # deterministic: by group id
        if len(ordered) == 1:
            warnings.append(
                f'"{label}": all {n_images} images come from one capture session — '
                f"none can be held out, so validation cannot measure this class")
            for img in groups[ordered[0]]:
                split[img["id"]] = "train"
            continue
        if len(ordered) < 4:
            warnings.append(
                f'"{label}": images come from only {len(ordered)} capture sessions — '
                f"validation will be optimistic; add photos from new sessions/lighting")

        target = max(1, round(n_images * val_fraction))
        val_count = 0
        # Smaller groups fill val first (finer fraction granularity); the
        # largest group is processed last and always lands in train.
        size_sorted = sorted(ordered, key=lambda g: (len(groups[g]), g))
        for idx, group_id in enumerate(size_sorted):
            is_largest = idx == len(size_sorted) - 1
            to_val = val_count < target and not is_largest
            for img in groups[group_id]:
                split[img["id"]] = "val" if to_val else "train"
            if to_val:
                val_count += len(groups[group_id])

    return split, warnings
