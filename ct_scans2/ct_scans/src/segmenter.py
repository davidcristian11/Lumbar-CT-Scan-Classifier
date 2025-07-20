import numpy as np
from skimage.filters import threshold_otsu, threshold_li, threshold_yen
from skimage.morphology import remove_small_objects, remove_small_holes, opening, closing, disk
from pydicom import Dataset
from skimage import filters, morphology


def window_image(ds: Dataset, center: float = 300, width: float = 1700) -> np.ndarray:
    """Standard bone windowing"""
    img = ds.pixel_array.astype(np.int16)
    hu = img * ds.RescaleSlope + ds.RescaleIntercept
    low, high = center - width / 2, center + width / 2
    clipped = np.clip(hu, low, high)
    return (clipped - low) / (high - low)


def adaptive_window_image(ds: Dataset) -> np.ndarray:
    """Adaptive windowing that tries multiple approaches"""
    img = ds.pixel_array.astype(np.int16)
    hu = img * ds.RescaleSlope + ds.RescaleIntercept

    hu_mean = np.mean(hu)
    hu_std = np.std(hu)

    windows = [
        (300, 1700),
        (400, 1500),
        (200, 2000),
        (hu_mean + hu_std, 4 * hu_std),
        (350, 1200),
    ]

    best_windowed = None
    best_score = 0

    for center, width in windows:
        try:
            low, high = center - width / 2, center + width / 2
            clipped = np.clip(hu, low, high)
            windowed = (clipped - low) / (high - low)

            grad_x = np.abs(np.gradient(windowed, axis=1))
            grad_y = np.abs(np.gradient(windowed, axis=0))
            contrast_score = np.mean(grad_x + grad_y)

            bone_content = np.sum(windowed > 0.5) / windowed.size

            if 0.15 <= bone_content <= 0.40:
                score = contrast_score * (1.0 - abs(bone_content - 0.25))

                if score > best_score:
                    best_score = score
                    best_windowed = windowed

        except Exception:
            continue

    if best_windowed is None:
        best_windowed = window_image(ds)

    return best_windowed


def enhanced_binarize(image: np.ndarray, min_size: int = 64, min_hole: int = 64) -> np.ndarray:
    """Enhanced binarization with multiple thresholding methods"""

    thresholding_methods = []

    try:
        thresh_otsu = threshold_otsu(image)
        mask_otsu = image > thresh_otsu
        thresholding_methods.append(('otsu', mask_otsu, thresh_otsu))
    except:
        pass

    try:
        thresh_li = threshold_li(image)
        mask_li = image > thresh_li
        thresholding_methods.append(('li', mask_li, thresh_li))
    except:
        pass

    try:
        thresh_yen = threshold_yen(image)
        mask_yen = image > thresh_yen
        thresholding_methods.append(('yen', mask_yen, thresh_yen))
    except:
        pass

    for thresh_val in [0.3, 0.4, 0.5, 0.6]:
        mask_manual = image > thresh_val
        thresholding_methods.append((f'manual_{thresh_val}', mask_manual, thresh_val))

    best_mask = None
    best_score = 0

    for method_name, mask, thresh_val in thresholding_methods:
        try:
            mask_clean = remove_small_objects(mask, min_size=min_size)
            mask_clean = remove_small_holes(mask_clean, area_threshold=min_hole)

            mask_clean = opening(mask_clean, disk(2))
            mask_clean = closing(mask_clean, disk(3))

            bone_ratio = np.sum(mask_clean) / mask_clean.size

            if 0.1 <= bone_ratio <= 0.4:
                from skimage import measure
                labeled = measure.label(mask_clean)
                num_components = np.max(labeled)

                if num_components > 0:
                    regions = measure.regionprops(labeled)
                    largest_area = max(region.area for region in regions)

                    bone_score = 1.0 - abs(bone_ratio - 0.25)

                    component_score = largest_area / np.sum(mask_clean) if np.sum(mask_clean) > 0 else 0

                    total_score = bone_score * component_score

                    if total_score > best_score:
                        best_score = total_score
                        best_mask = mask_clean

        except Exception:
            continue

    if best_mask is None:
        best_mask = binarize(image, min_size, min_hole)

    return best_mask


def binarize(image: np.ndarray, min_size: int = 64, min_hole: int = 64) -> np.ndarray:
    """Original binarization method"""
    thresh = threshold_otsu(image)
    mask = image > thresh
    mask = remove_small_objects(mask, min_size)
    mask = remove_small_holes(mask, min_hole)
    return mask


def multi_scale_binarize(image: np.ndarray) -> np.ndarray:
    """Multi-scale approach for challenging images"""
    masks = []

    scales = [(64, 64), (32, 32), (128, 128), (16, 16)]

    for min_size, min_hole in scales:
        try:
            mask = enhanced_binarize(image, min_size, min_hole)
            masks.append(mask)
        except:
            continue

    if not masks:
        return binarize(image)

    if len(masks) == 1:
        return masks[0]

    mask_stack = np.stack(masks, axis=2)
    combined_mask = np.sum(mask_stack, axis=2) > (len(masks) // 2)

    return combined_mask.astype(bool)


def preprocess_for_vertebra_detection(ds: Dataset) -> tuple:
    """Complete preprocessing pipeline optimized for vertebra detection"""

    windowed = adaptive_window_image(ds)

    mask = enhanced_binarize(windowed)

    mask = remove_small_objects(mask, min_size=100)

    mask = remove_small_holes(mask, area_threshold=200)

    mask = opening(mask, disk(2))
    mask = closing(mask, disk(4))

    return windowed, windowed, mask


def get_vertebra_region_of_interest(image: np.ndarray, mask: np.ndarray) -> tuple:
    """Extract region of interest around the main vertebral structure"""

    from skimage import measure

    labeled = measure.label(mask)

    if np.max(labeled) == 0:
        return image, mask

    regions = measure.regionprops(labeled)
    largest_region = max(regions, key=lambda x: x.area)

    min_row, min_col, max_row, max_col = largest_region.bbox

    height, width = image.shape
    padding = 20

    min_row = max(0, min_row - padding)
    min_col = max(0, min_col - padding)
    max_row = min(height, max_row + padding)
    max_col = min(width, max_col + padding)

    roi_image = image[min_row:max_row, min_col:max_col]
    roi_mask = mask[min_row:max_row, min_col:max_col]

    return roi_image, roi_mask