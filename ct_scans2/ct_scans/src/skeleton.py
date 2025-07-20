import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, closing, square
from typing import List, Tuple
from scipy import ndimage
from skimage import measure


def get_skeleton(mask: np.ndarray) -> np.ndarray:
    """Preprocesare înainte de skeletonizare"""
    mask = closing(mask.astype(bool), square(3))
    mask = remove_small_objects(mask, min_size=200)
    return skeletonize(mask)


def is_valid_y_structure(skel: np.ndarray, r: int, c: int, min_branch_length: int = 6) -> bool:
    region_size = 15
    r_start = max(0, r - region_size)
    r_end = min(skel.shape[0], r + region_size + 1)
    c_start = max(0, c - region_size)
    c_end = min(skel.shape[1], c + region_size + 1)

    region = skel[r_start:r_end, c_start:c_end]
    center_r, center_c = r - r_start, c - c_start

    if region.shape[0] < 10 or region.shape[1] < 10:
        return False

    directions = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = center_r + dr, center_c + dc
            if (0 <= nr < region.shape[0] and 0 <= nc < region.shape[1] and region[nr, nc]):
                directions.append((dr, dc))

    if len(directions) < 3:
        return False

    valid_branches = 0
    for dr, dc in directions:
        branch_length = 0
        curr_r, curr_c = center_r + dr, center_c + dc

        while (0 <= curr_r < region.shape[0] and 0 <= curr_c < region.shape[1] and
               region[curr_r, curr_c]):
            branch_length += 1

            next_found = False
            for next_dr in range(-1, 2):
                for next_dc in range(-1, 2):
                    if next_dr == 0 and next_dc == 0:
                        continue
                    next_r, next_c = curr_r + next_dr, curr_c + next_dc
                    if (0 <= next_r < region.shape[0] and 0 <= next_c < region.shape[1] and
                            region[next_r, next_c] and
                            (next_dr * dr >= 0 and next_dc * dc >= 0)):
                        curr_r, curr_c = next_r, next_c
                        next_found = True
                        break
                if next_found:
                    break

            if not next_found or branch_length > 50:
                break

        if branch_length >= min_branch_length:
            valid_branches += 1

    return valid_branches >= 2


def detect_y_points(skel: np.ndarray, min_distance: int = 25) -> List[Tuple[int, int]]:
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    rows, cols = skel.shape
    candidate_points = []

    for r in range(5, rows - 5):
        for c in range(5, cols - 5):
            if not skel[r, c]:
                continue

            neigh = 0
            for dr, dc in offsets:
                if skel[r + dr, c + dc]:
                    neigh += 1

            if 3 <= neigh <= 5:
                candidate_points.append((r, c))

    valid_points = []
    for r, c in candidate_points:
        if is_valid_y_structure(skel, r, c, min_branch_length=6):
            valid_points.append((r, c))

    if len(valid_points) > 1:
        final_points = []
        center_r, center_c = rows // 2, cols // 2

        valid_points.sort(key=lambda p: (p[0] - center_r) ** 2 + (p[1] - center_c) ** 2)

        for r, c in valid_points:
            too_close = False
            for existing_r, existing_c in final_points:
                distance = ((r - existing_r) ** 2 + (c - existing_c) ** 2) ** 0.5
                if distance < min_distance:
                    too_close = True
                    break

            if not too_close:
                final_points.append((r, c))

        valid_points = final_points

    if len(valid_points) > 2:
        center_r, center_c = rows // 2, cols // 2
        valid_points.sort(key=lambda p: (p[0] - center_r) ** 2 + (p[1] - center_c) ** 2)
        valid_points = valid_points[:2]

    return valid_points


def analyze_skeleton_structure(skel: np.ndarray) -> dict:
    labeled, num_components = ndimage.label(skel)

    endpoints = 0
    junction_points = 0

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for r in range(1, skel.shape[0] - 1):
        for c in range(1, skel.shape[1] - 1):
            if skel[r, c]:
                neighbors = sum(1 for dr, dc in offsets if skel[r + dr, c + dc])
                if neighbors == 1:
                    endpoints += 1
                elif neighbors > 2:
                    junction_points += 1

    return {
        'total_skeleton_pixels': int(np.sum(skel)),
        'connected_components': num_components,
        'endpoints': endpoints,
        'junction_points': junction_points,
        'skeleton_density': np.sum(skel) / (skel.shape[0] * skel.shape[1])
    }


def analyze_vertebra_structure(mask: np.ndarray) -> dict:
    labeled = measure.label(mask)
    num_components = np.max(labeled)
    regions = measure.regionprops(labeled)

    if not regions:
        return {'is_valid': False, 'reason': 'No components found'}

    main_region = max(regions, key=lambda x: x.area)

    area = main_region.area
    solidity = main_region.solidity
    extent = main_region.extent
    eccentricity = main_region.eccentricity

    checks = {
        'area_ok': 8000 < area < 40000,
        'solidity_ok': solidity > 0.5,
        'extent_ok': extent > 0.4,
        'eccentricity_ok': 0.2 < eccentricity < 0.8,
        'components_ok': num_components <= 2
    }

    bbox_area = (main_region.bbox[2] - main_region.bbox[0]) * (main_region.bbox[3] - main_region.bbox[1])
    area_ratio = area / bbox_area if bbox_area > 0 else 0
    checks['shape_complexity'] = 0.3 < area_ratio < 0.8

    is_valid = all(checks.values())

    return {
        'is_valid': is_valid,
        'area': area,
        'solidity': solidity,
        'extent': extent,
        'eccentricity': eccentricity,
        'components': num_components,
        'checks': checks,
        'main_region': main_region
    }


def has_vertebral_anatomy(mask: np.ndarray, y_points: List[Tuple[int, int]]) -> bool:
    if len(y_points) == 0:
        return False

    height, width = mask.shape

    anterior_third = mask[:, :width // 3]
    anterior_area = np.sum(anterior_third)

    posterior_third = mask[:, 2 * width // 3:]
    posterior_area = np.sum(posterior_third)

    y_positions = []
    for y, x in y_points:
        if x < width // 3:
            y_positions.append('anterior')
        elif x > 2 * width // 3:
            y_positions.append('posterior')
        else:
            y_positions.append('middle')

    has_anterior = anterior_area > 5000
    has_posterior = posterior_area > 2000
    y_distribution_ok = 'middle' in y_positions or 'posterior' in y_positions

    return has_anterior and has_posterior and y_distribution_ok
