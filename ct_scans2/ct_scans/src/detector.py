import numpy as np
from typing import Optional, Tuple, Dict, List
import pydicom
from scipy import ndimage
from skimage import measure, morphology, filters
from .loader import load_dicom_series, sort_by_position
from .segmenter import window_image, binarize
from .skeleton import (get_skeleton, detect_y_points, analyze_vertebra_structure,
                       has_vertebral_anatomy)
import os
import matplotlib.pyplot as plt


class ThirdYDetector:
    def __init__(self, dicom_folder: str):
        self.folder = dicom_folder
        self.datasets = []
        self.vertebra_positions = []

    def load_and_sort(self):
        self.datasets = load_dicom_series(self.folder)
        self.datasets = sort_by_position(self.datasets)
        print(f"Loaded {len(self.datasets)} DICOM images")

    def enhanced_windowing(self, ds: pydicom.Dataset) -> np.ndarray:
        """Try multiple windowing approaches and pick the best one"""
        img = ds.pixel_array.astype(np.int16)
        hu = img * ds.RescaleSlope + ds.RescaleIntercept

        windows = [
            (300, 1700),
            (400, 1500),
            (200, 2000),
            (350, 1200),
        ]

        best_windowed = None
        best_score = 0

        for center, width in windows:
            low, high = center - width / 2, center + width / 2
            clipped = np.clip(hu, low, high)
            windowed = (clipped - low) / (high - low)

            bone_pixels = np.sum(windowed > 0.5)
            total_pixels = windowed.size
            bone_ratio = bone_pixels / total_pixels

            if 0.15 <= bone_ratio <= 0.35:
                score = 1.0 - abs(bone_ratio - 0.25)
                if score > best_score:
                    best_score = score
                    best_windowed = windowed

        return best_windowed if best_windowed is not None else window_image(ds)

    def enhanced_binarization(self, windowed: np.ndarray) -> np.ndarray:
        """Enhanced binarization with multiple thresholding approaches"""
        methods = []

        try:
            thresh_otsu = filters.threshold_otsu(windowed)
            mask_otsu = windowed > thresh_otsu
            methods.append(('otsu', mask_otsu, thresh_otsu))
        except:
            pass

        try:
            thresh_li = filters.threshold_li(windowed)
            mask_li = windowed > thresh_li
            methods.append(('li', mask_li, thresh_li))
        except:
            pass

        try:
            thresh_yen = filters.threshold_yen(windowed)
            mask_yen = windowed > thresh_yen
            methods.append(('yen', mask_yen, thresh_yen))
        except:
            pass

        mask_manual = windowed > 0.4
        methods.append(('manual', mask_manual, 0.4))

        best_mask = None
        best_score = 0

        for name, mask, thresh in methods:
            mask_clean = morphology.remove_small_objects(mask, min_size=100)
            mask_clean = morphology.remove_small_holes(mask_clean, area_threshold=100)

            bone_ratio = np.sum(mask_clean) / mask_clean.size

            if 0.1 <= bone_ratio <= 0.4:
                score = 1.0 - abs(bone_ratio - 0.25)
                if score > best_score:
                    best_score = score
                    best_mask = mask_clean

        if best_mask is None:
            best_mask = binarize(windowed)

        return best_mask

    def simple_vertebra_check(self, mask: np.ndarray) -> bool:
        """Simplified vertebra detection - less restrictive"""
        if mask is None or np.sum(mask) == 0:
            return False

        area = np.sum(mask)
        if area < 2000 or area > 60000:
            return False

        labeled = measure.label(mask)
        regions = measure.regionprops(labeled)

        if not regions:
            return False

        main_region = max(regions, key=lambda x: x.area)

        if main_region.area < 1500:
            return False

        if main_region.solidity < 0.3:
            return False

        return True

    def detect_vertebra_with_fallbacks(self, ds: pydicom.Dataset) -> Tuple[bool, Dict]:
        """Multiple detection strategies with fallbacks"""
        try:
            windowed = self.enhanced_windowing(ds)
            mask = self.enhanced_binarization(windowed)

            if self.simple_vertebra_check(mask):
                skeleton = get_skeleton(mask)
                y_points = detect_y_points(skeleton, min_distance=10)

                analysis = {
                    'method': 'enhanced',
                    'area': np.sum(mask),
                    'y_points_count': len(y_points),
                    'y_points': y_points,
                    'is_valid': True
                }
                return True, analysis

            windowed_orig = window_image(ds)
            mask_orig = binarize(windowed_orig)

            if self.simple_vertebra_check(mask_orig):
                skeleton = get_skeleton(mask_orig)
                y_points = detect_y_points(skeleton, min_distance=8)

                analysis = {
                    'method': 'original_relaxed',
                    'area': np.sum(mask_orig),
                    'y_points_count': len(y_points),
                    'y_points': y_points,
                    'is_valid': True
                }
                return True, analysis

            for thresh_val in [0.3, 0.4, 0.5, 0.6]:
                mask_permissive = windowed_orig > thresh_val
                mask_permissive = morphology.remove_small_objects(mask_permissive, min_size=50)

                area = np.sum(mask_permissive)
                if 1000 < area < 80000:
                    analysis = {
                        'method': 'permissive',
                        'area': area,
                        'threshold': thresh_val,
                        'y_points_count': 0,
                        'y_points': [],
                        'is_valid': True
                    }
                    return True, analysis

            return False, {'method': 'failed', 'error': 'No valid vertebra detected'}

        except Exception as e:
            return False, {'method': 'error', 'error': str(e)}

    def find_third_y_enhanced(self, start_id: str, output_dir: str, debug: bool = True) -> Optional[str]:
        """Enhanced third Y detection with multiple strategies"""
        start_idx = None
        for i, ds in enumerate(self.datasets):
            if ds.SOPInstanceUID == start_id or str(ds.InstanceNumber) == start_id:
                start_idx = i
                break

        if start_idx is None:
            if debug:
                print(f"Start ID {start_id} not found")
                start_idx = 0
                print(f"Using first image as start (index 0)")

        if debug:
            print(f"Starting enhanced analysis from image {start_idx}")
            print(f"Total images available: {len(self.datasets)}")

        os.makedirs(output_dir, exist_ok=True)
        vertebra_count = 0

        search_range = min(30, len(self.datasets) - start_idx - 1)

        for i in range(start_idx + 1, start_idx + search_range + 1):
            ds = self.datasets[i]

            if debug:
                print(f"\nAnalyzing image {i} (Instance {getattr(ds, 'InstanceNumber', 'N/A')})")

            is_vertebra, analysis = self.detect_vertebra_with_fallbacks(ds)

            if is_vertebra:
                vertebra_count += 1
                if debug:
                    print(f"  ✓ VERTEBRA #{vertebra_count} detected!")
                    print(f"    Method: {analysis.get('method', 'unknown')}")
                    print(f"    Area: {analysis.get('area', 0)}")
                    print(f"    Y-points: {analysis.get('y_points_count', 0)}")

                if vertebra_count == 3:
                    if debug:
                        print(f"\n🎯 THIRD LUMBAR VERTEBRA FOUND in image {i}!")

                    output_path = os.path.join(output_dir, f"third_Y_{ds.SOPInstanceUID}.dcm")
                    ds.save_as(output_path)

                    if debug:
                        self.save_debug_image(ds, output_dir, f"third_Y_debug_{ds.SOPInstanceUID}")

                    return ds.SOPInstanceUID
            else:
                if debug:
                    print(f"  ✗ No vertebra detected")
                    if 'error' in analysis:
                        print(f"    Error: {analysis['error']}")

        if debug:
            print(
                f"\n❌ Third lumbar vertebra not found. Only {vertebra_count} vertebrae detected in {search_range} images.")

        return None

    def save_debug_image(self, ds: pydicom.Dataset, output_dir: str, filename: str):
        """Save debug visualization"""
        try:
            windowed = self.enhanced_windowing(ds)
            mask = self.enhanced_binarization(windowed)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(ds.pixel_array, cmap='gray')
            axes[0].set_title('Original DICOM')
            axes[0].axis('off')

            axes[1].imshow(windowed, cmap='gray')
            axes[1].set_title('Windowed')
            axes[1].axis('off')

            axes[2].imshow(mask, cmap='gray')
            axes[2].set_title('Segmented Vertebra')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Could not save debug image: {e}")

    def analyze_vertebral_shape(self, binary_mask: np.ndarray) -> Dict:
        return analyze_vertebra_structure(binary_mask)

    def check_vertebral_body(self, binary_mask: np.ndarray) -> bool:
        height, width = binary_mask.shape
        anterior_region = binary_mask[:, :width // 3]
        labeled_anterior = measure.label(anterior_region)
        regions_anterior = measure.regionprops(labeled_anterior)

        if not regions_anterior:
            return False

        for region in regions_anterior:
            if (region.area > 1000 and region.solidity > 0.4 and 0.2 < region.eccentricity < 0.9):
                return True
        return False

    def check_spinous_process(self, binary_mask: np.ndarray) -> bool:
        height, width = binary_mask.shape
        posterior_region = binary_mask[:, 2 * width // 3:]
        labeled_posterior = measure.label(posterior_region)
        regions_posterior = measure.regionprops(labeled_posterior)

        if not regions_posterior:
            return False

        for region in regions_posterior:
            if (region.area > 100 and
                    region.eccentricity > 0.3):
                return True
        return False

    def check_transverse_processes(self, binary_mask: np.ndarray) -> bool:
        height, width = binary_mask.shape
        middle_height = height // 2
        horizontal_profile = binary_mask[middle_height, :]

        segments = []
        in_segment = False
        start = 0

        for i, pixel in enumerate(horizontal_profile):
            if pixel and not in_segment:
                start = i
                in_segment = True
            elif not pixel and in_segment:
                segments.append((start, i))
                in_segment = False

        if in_segment:
            segments.append((start, len(horizontal_profile)))

        return len(segments) >= 2

    def find_third_y(self, start_id: str, output_dir: str, debug: bool = True) -> Optional[str]:
        """Main method - now uses enhanced detection"""
        return self.find_third_y_enhanced(start_id, output_dir, debug)

    def debug_analysis(self, start_id: str, max_images: int = 10):
        """Enhanced debug analysis"""
        start_idx = None
        for i, ds in enumerate(self.datasets):
            if ds.SOPInstanceUID == start_id or str(ds.InstanceNumber) == start_id:
                start_idx = i
                break

        if start_idx is None:
            print(f"Start ID {start_id} not found, using first image")
            start_idx = 0

        print(f"DEBUG: Starting analysis from slice {start_idx}")
        print("=" * 60)

        for i in range(start_idx + 1, min(start_idx + max_images + 1, len(self.datasets))):
            ds = self.datasets[i]
            print(f"\nSLICE {i} (Instance {getattr(ds, 'InstanceNumber', 'N/A')})")
            print("-" * 40)

            is_valid, analysis = self.detect_vertebra_with_fallbacks(ds)

            print(f"Detection Result: {'✓ VALID VERTEBRA' if is_valid else '✗ INVALID'}")
            print(f"Method: {analysis.get('method', 'unknown')}")
            print(f"Area: {analysis.get('area', 0)}")
            print(f"Y Points: {analysis.get('y_points_count', 0)}")

            if 'error' in analysis:
                print(f"Error: {analysis['error']}")


def analyze_vertebral_shape(binary_mask: np.ndarray) -> Dict:
    return analyze_vertebra_structure(binary_mask)