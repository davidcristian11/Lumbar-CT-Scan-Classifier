import os
import pydicom
from typing import List

def load_dicom_series(folder: str) -> List[pydicom.Dataset]:
    files = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".dcm"):
            filepath = os.path.join(folder, filename)
            files.append(filepath)

    datasets = []
    for filepath in files:
        ds = pydicom.dcmread(filepath)
        datasets.append(ds)

    return datasets


def sort_by_position(datasets: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
    try:
        return sorted(datasets, key=lambda ds: int(ds.InstanceNumber))
    except Exception:
        return sorted(datasets, key=lambda ds: float(ds.ImagePositionPatient[2]))

