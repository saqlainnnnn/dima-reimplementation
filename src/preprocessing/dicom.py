"""
DICOM Reader for RSNA Mammography.

Responsibilities
----------------
- Read DICOM images
- Apply VOI LUT (if available)
- Handle MONOCHROME1 inversion
- Normalize intensities to [0, 1]
- Return float32 NumPy arrays
"""

from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


class DicomReader:
    """
    Utility class for reading mammography DICOM images.
    """

    def __init__(self):
        pass

    def read(self, path: str | Path) -> np.ndarray:
        """
        Read a DICOM file.

        Parameters
        ----------
        path : str | Path
            Path to a .dcm file.

        Returns
        -------
        np.ndarray
            Float32 image normalized to [0, 1].
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        ds = pydicom.dcmread(path)

        image = self._get_pixels(ds)

        image = self._normalize(image)

        return image

    ##################################################################

    def _get_pixels(self, ds) -> np.ndarray:
        """
        Extract pixel array and apply VOI LUT if present.
        """

        try:
            image = apply_voi_lut(ds.pixel_array, ds)
        except Exception:
            image = ds.pixel_array

        image = image.astype(np.float32)

        # Mammography images are often MONOCHROME1
        # meaning black/white are inverted.
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            image = image.max() - image

        return image

    ##################################################################

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """
        Min-max normalize image into [0,1].
        """

        image = image.astype(np.float32)

        minimum = image.min()
        maximum = image.max()

        if maximum == minimum:
            return np.zeros_like(image, dtype=np.float32)

        image = (image - minimum) / (maximum - minimum)

        return image.astype(np.float32)

    ##################################################################

    @staticmethod
    def metadata(path: str | Path) -> dict:
        """
        Return selected DICOM metadata.
        """

        ds = pydicom.dcmread(path, stop_before_pixels=True)

        return {

            "PatientID": getattr(ds, "PatientID", None),

            "Rows": getattr(ds, "Rows", None),

            "Columns": getattr(ds, "Columns", None),

            "PhotometricInterpretation":
                getattr(ds, "PhotometricInterpretation", None),

            "BitsStored":
                getattr(ds, "BitsStored", None),

            "PixelSpacing":
                getattr(ds, "PixelSpacing", None),

            "WindowCenter":
                getattr(ds, "WindowCenter", None),

            "WindowWidth":
                getattr(ds, "WindowWidth", None),
        }

    ##################################################################

    @staticmethod
    def info(path: str | Path):

        meta = DicomReader.metadata(path)

        print("\nDICOM Metadata")
        print("-" * 40)

        for key, value in meta.items():
            print(f"{key:28}: {value}")