import abc
import cv2
import numpy as np
import re
import warnings
import xml.etree.ElementTree as ET

from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

try:
    import openslide
except ImportError as e:
    warnings.warn(f"module {e.name} not found (required by OpenSlideReader)")

try:
    import tifffile
    import zarr
except ImportError as e:
    warnings.warn(f"module {e.name} not found (required by TiffReader)")

try:
    from pixelengine import PixelEngine
    from softwarerendercontext import SoftwareRenderContext
    from softwarerenderbackend import SoftwareRenderBackend
except ImportError as e:
    warnings.warn(f"module {e.name} not found (required by IsyntaxReader)")


class WSIReader(metaclass=abc.ABCMeta):
    """Interface class for a WSI reader."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Close the slide.

        Returns:
            None
        """
        raise NotImplementedError

    def __enter__(self) -> "WSIReader":
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        self.close()

    def read_region(
        self,
        x_y: Tuple[int, int],
        level: int,
        tile_size: Union[Tuple[int, int], int],
        normalize: bool = True,
        downsample_level_0: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reads the contens of the specified region in the slide from the given level.

        Args:
            x_y (Tuple[int, int]): coordinates of the top left pixel of the region in the given level reference frame.
            level (int): the desired level.
            tile_size (Union[Tuple[int, int], int]): size of the region. Can be a tuple in the format (width, height) or a single scalar to specify a square region.
            normalize (bool, optional): True to normalize the pixel values in therange [0,1]. Defaults to True.
            downsample_level_0 (bool, optional): True to render the region by downsampling from level 0. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of pixel data and alpha mask of the specified region.
        """
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)

        x, y = x_y
        if downsample_level_0 and level > 0:
            downsample = round(
                self.level_dimensions[0][0] / self.level_dimensions[level][0]
            )
            x, y = x * downsample, y * downsample
            tile_w, tile_h = tile_size[0] * downsample, tile_size[1] * downsample
            width, height = self.level_dimensions[0]
        else:
            tile_w, tile_h = tile_size
            width, height = self.level_dimensions[level]

        tile_w = tile_w + x if x < 0 else tile_w
        tile_h = tile_h + y if y < 0 else tile_h
        x = max(x, 0)
        y = max(y, 0)
        tile_w = width - x if (x + tile_w > width) else tile_w
        tile_h = height - y if (y + tile_h > height) else tile_h

        tile, alfa_mask = self._read_region(
            (x, y), 0 if downsample_level_0 else level, (tile_w, tile_h)
        )
        if downsample_level_0 and level > 0:
            tile_w = tile_w // downsample
            tile_h = tile_h // downsample
            x = x // downsample
            y = y // downsample
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
            alfa_mask = cv2.resize(
                alfa_mask.astype(np.uint8),
                (tile_w, tile_h),
                interpolation=cv2.INTER_CUBIC,
            ).astype(np.bool)

        if normalize:
            tile = self._normalize(tile)

        padding = [
            (y - x_y[1], tile_size[1] - tile_h + min(x_y[1], 0)),
            (x - x_y[0], tile_size[0] - tile_w + min(x_y[0], 0)),
        ]
        tile = np.pad(
            tile,
            padding + [(0, 0)] * (len(tile.shape) - 2),
            "constant",
            constant_values=0,
        )
        alfa_mask = np.pad(alfa_mask, padding, "constant", constant_values=0)

        return tile, alfa_mask

    def read_region_ds(
        self,
        x_y: Tuple[int, int],
        downsample: float,
        tile_size: Union[Tuple[int, int], int],
        normalize: bool = True,
        downsample_level_0: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reads the contens of the specified region in the slide for the given downsample factor.

        Args:
            x_y (Tuple[int, int]): coordinates of the top left pixel of the region in the given downsample factor reference frame.
            downsample (float): the desired downsample factor.
            tile_size (Union[Tuple[int, int], int]): size of the region. Can be a tuple in the format (width, height) or a single scalar to specify a square region.
            normalize (bool, optional): True to normalize the pixel values in therange [0,1]. Defaults to True.
            downsample_level_0 (bool, optional): True to render the region by downsampling from level 0. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of pixel data and alpha mask of the specified region.
        """
        if downsample <= 0:
            raise RuntimeError("Downsample factor must be positive")

        if not isinstance(tile_size, tuple):
            tile_size = (tile_size, tile_size)

        if downsample == 1:
            downsample_level_0 = False

        if downsample in self.level_downsamples and not downsample_level_0:
            level = self.get_best_level_for_downsample(downsample)
            tile, alfa_mask = self.read_region(x_y, level, tile_size, False, False)
        else:
            level = (
                0
                if downsample_level_0
                else self.get_best_level_for_downsample(downsample)
            )
            x_y_level = (
                round(x_y[0] * downsample / self.level_downsamples[level]),
                round(x_y[1] * downsample / self.level_downsamples[level]),
            )
            tile_size_level = (
                round(tile_size[0] * downsample / self.level_downsamples[level]),
                round(tile_size[1] * downsample / self.level_downsamples[level]),
            )
            tile, alfa_mask = self.read_region(
                x_y_level, level, tile_size_level, False, downsample_level_0
            )
            tile = cv2.resize(tile, tile_size, interpolation=cv2.INTER_CUBIC)
            alfa_mask = cv2.resize(
                alfa_mask.astype(np.uint8), tile_size, interpolation=cv2.INTER_CUBIC
            ).astype(np.bool)

        if normalize:
            tile = self._normalize(tile)

        return tile, alfa_mask

    @abc.abstractmethod
    def _read_region(
        self, x_y: Tuple[int, int], level: int, tile_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Return the best level for the given downsample factor.

        Args:
            downsample (float): the downsample factor.

        Returns:
            int: the level.
        """
        if downsample < self.level_downsamples[0]:
            return 0

        for i in range(1, self.level_count):
            if downsample < self.level_downsamples[i]:
                return i - 1

        return self.level_count - 1

    def get_downsampled_slide(
        self, dims: Tuple[int, int], normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a downsampled version of the slide with the given dimensions.

        Args:
            dims (Tuple[int, int]): size of the downsampled slide asa (width, height) tuple.
            normalize (bool, optional): True to normalize the pixel values in therange [0,1]. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of pixel data and alpha mask of the downsampled slide.
        """
        downsample = min(a / b for a, b in zip(self.level_dimensions[0], dims))
        slide_downsampled, alfa_mask = self.read_region_ds(
            (0, 0),
            downsample,
            self.get_dimensions_for_downsample(downsample),
            normalize=normalize,
        )
        return slide_downsampled, alfa_mask

    def get_dimensions_for_downsample(self, downsample: float) -> Tuple[int, int]:
        """Return the slide dimensions for for a given fownsample factor.

        Args:
            downsample (float): downsample factor.

        Returns:
            Tuple[int, int]: slide dimensions for the given downsample factor as a tuple (width, height).
        """
        if downsample <= 0:
            raise RuntimeError("Downsample factor must be positive")
        if downsample in self.level_downsamples:
            level = self.level_downsamples.index(downsample)
            dims = self.level_dimensions[level]
        else:
            w, h = self.level_dimensions[0]
            dims = round(w / downsample), round(h / downsample)
        return dims

    @property
    @abc.abstractmethod
    def level_count(self) -> int:
        """Number of levels in the slide."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def level_dimensions(self) -> List[Tuple[int, int]]:
        """Slide dimensions for each slide level as a list of (width, height) tuples."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tile_dimensions(self) -> List[Tuple[int, int]]:
        """Tile dimensions for each slide level as a list of (width, height) tuples."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mpp(self) -> Tuple[Optional[float], Optional[float]]:
        """A tuple containing the number of microns per pixel of level 0 in the X and Y dimensions respectively, if known."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """Numpy data type of the slide pixels."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_channels(self) -> int:
        """Number of channels in the slide."""
        raise NotImplementedError

    @property
    def level_downsamples(self) -> List[float]:
        """Return a list of downsample factors for each level of the slide.

        Returns:
            List[float]: The list of downsample factors.
        """
        if not hasattr(self, "_level_downsamples"):
            self._level_downsamples = []
            width, height = self.level_dimensions[0]
            for level in range(self.level_count):
                w, h = self.level_dimensions[level]
                ds = float(round(width / w))
                self._level_downsamples.append(ds)
        return self._level_downsamples

    @staticmethod
    def _normalize(pixels: np.ndarray) -> np.ndarray:
        if np.issubdtype(pixels.dtype, np.integer):
            pixels = (pixels / 255).astype(np.float32)
        return pixels

    @staticmethod
    def _round(x: float, base: int) -> int:
        return base * round(x / base)


class OpenSlideReader(WSIReader):
    """Implementation of the WSIReader interface backed by openslide"""

    def __init__(self, slide_path: Path, **kwargs) -> None:
        """Open a slide. The object may be used as a context manager, in which case it will be closed upon exiting the context.

        Args:
            slide_path (Path): Path of the slide to open.
        """
        self.slide_path = slide_path
        self._slide = openslide.open_slide(str(slide_path))

    def close(self) -> None:
        self._slide.close()
        if hasattr(self, "_tile_dimensions"):
            delattr(self, "_tile_dimensions")

    def _read_region(
        self, x_y: Tuple[int, int], level: int, tile_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        tile = np.array(self._slide.read_region(x_y, level, tile_size), dtype=np.uint8)
        alfa_mask = tile[:, :, 3] > 0
        tile = tile[:, :, :3]
        return tile, alfa_mask

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return self._slide.get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        return self._slide.level_dimensions

    @property
    def level_count(self) -> int:
        return self._slide.level_count

    @property
    def mpp(self) -> Tuple[Optional[float], Optional[float]]:
        mpp_x = self._slide.properties["openslide.mpp-x"]
        mpp_x = float(mpp_x) if mpp_x else mpp_x
        mpp_y = self._slide.properties["openslide.mpp-y"]
        mpp_y = float(mpp_y) if mpp_y else mpp_y
        return mpp_x, mpp_y

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint8)

    @property
    def n_channels(self) -> int:
        return 3

    @property
    def level_downsamples(self) -> List[float]:
        """Return a list of downsample factors for each level of the slide.

        Returns:
            List[float]: The list of downsample factors.
        """
        return self._slide.level_downsamples

    @property
    def tile_dimensions(self) -> List[Tuple[int, int]]:
        if not hasattr(self, "_tile_dimensions"):
            self._tile_dimensions = []
            for level in range(self.level_count):
                tile_width = int(
                    self._slide.properties[f"openslide.level[{level}].tile-width"]
                )
                tile_height = int(
                    self._slide.properties[f"openslide.level[{level}].tile-height"]
                )
                self._tile_dimensions.append((tile_width, tile_height))
        return self._tile_dimensions


class TiffReader(WSIReader):
    """Implementation of the WSIReader interface backed by tifffile."""

    def __init__(self, slide_path: Path, series: int = 0, **kwargs) -> None:
        """Open a slide. The object may be used as a context manager, in which case it will be closed upon exiting the context.

        Args:
            slide_path (Path): Path of the slide to open.
            series (int, optional): For multi-series formats, image series to open. Defaults to 0.
        """
        self.slide_path = slide_path
        self.series = series
        self._store = tifffile.imread(slide_path, aszarr=True, series=series)
        self._z = zarr.open(self._store, mode="r")

    def close(self) -> None:
        """Close the slide.

        Returns:
            None
        """
        self._store.close()
        if hasattr(self, "_mpp"):
            delattr(self, "_mpp")
        if hasattr(self, "_tile_dimensions"):
            delattr(self, "_tile_dimensions")
        if hasattr(self, "_level_dimensions"):
            delattr(self, "_level_dimensions")
        if hasattr(self, "_level_downsamples"):
            delattr(self, "_level_downsamples")

    @property
    def tile_dimensions(self) -> List[Tuple[int, int]]:
        if not hasattr(self, "_tile_dimensions"):
            self._tile_dimensions = []
            for level in range(self.level_count):
                page = self._store._data[level].pages[0]
                self._tile_dimensions.append((page.tilewidth, page.tilelength))
        return self._tile_dimensions

    def _read_region(
        self, x_y: Tuple[int, int], level: int, tile_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x, y = x_y
        tile_w, tile_h = tile_size
        return self._z[level][y : y + tile_h, x : x + tile_w], np.ones(
            (tile_h, tile_w), np.bool
        )

    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        if not hasattr(self, "_level_dimensions"):
            self._level_dimensions = []
            for level in range(self.level_count):
                page = self._store._data[level].pages[0]
                self._level_dimensions.append((page.imagewidth, page.imagelength))
        return self._level_dimensions

    @property
    def level_count(self) -> int:
        return len(self._z)

    @property
    def mpp(self) -> Tuple[Optional[float], Optional[float]]:
        if not hasattr(self, "_mpp"):
            self._mpp: Tuple[Optional[float], Optional[float]] = (None, None)
            page = self._store._data[0].pages[0]
            if page.is_svs:
                metadata = tifffile.tifffile.svs_description_metadata(page.description)
                self._mpp = (metadata["MPP"], metadata["MPP"])
            elif page.is_ome:
                root = ET.fromstring(page.description)
                namespace_match = re.search("^{.*}", root.tag)
                namespace = namespace_match.group() if namespace_match else ""
                pixels = list(root.findall(namespace + "Image"))[self.series].find(
                    namespace + "Pixels"
                )
                mpp_x = pixels.get("PhysicalSizeX") if pixels else None
                mpp_y = pixels.get("PhysicalSizeY") if pixels else None
                self._mpp = (
                    float(mpp_x) if mpp_x else None,
                    float(mpp_y) if mpp_y else None,
                )
            elif page.is_philips:
                root = ET.fromstring(page.description)
                mpp_attribute = root.find(
                    "./Attribute/[@Name='PIM_DP_SCANNED_IMAGES']/Array/DataObject/[@ObjectType='DPScannedImage']/Attribute/[@Name='PIM_DP_IMAGE_TYPE'][.='WSI']/Attribute[@Name='PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE']/Array/DataObject[@ObjectType='PixelDataRepresentation']/Attribute[@Name='DICOM_PIXEL_SPACING']"
                )
                mpp = (
                    float(mpp_attribute.text)
                    if mpp_attribute and mpp_attribute.text
                    else None
                )
                self._mpp = (mpp, mpp)
            elif page.is_ndpi or page.is_scn or page.is_qpi or True:
                page = self._store._data[0].pages[0]
                if (
                    "ResolutionUnit" in page.tags
                    and page.tags["ResolutionUnit"].value == 3
                    and "XResolution" in page.tags
                    and "YResolution" in page.tags
                ):
                    self._mpp = (
                        1e4 / float(Fraction(*page.tags["XResolution"].value)),
                        1e4 / float(Fraction(*page.tags["YResolution"].value)),
                    )
        return self._mpp

    @property
    def dtype(self) -> np.dtype:
        return self._z[0].dtype

    @property
    def n_channels(self) -> int:
        page = self._store._data[0].pages[0]
        return page.samplesperpixel


class IsyntaxReader(WSIReader):
    """Implementation of the WSIReader interface for the isyntax format backed by the Philips pathology SDK."""

    def __init__(self, slide_path: Path, **kwargs) -> None:
        """Open a slide. The object may be used as a context manager, in which case it will be closed upon exiting the context.

        Args:
            slide_path (Path): Path of the slide to open.
        """
        self.slide_path = slide_path
        self._pe = PixelEngine(SoftwareRenderBackend(), SoftwareRenderContext())
        self._pe["in"].open(str(slide_path), "ficom")
        self._view = self._pe["in"]["WSI"].source_view
        trunc_bits = {0: [0, 0, 0]}
        self._view.truncation(False, False, trunc_bits)

    def close(self) -> None:
        """Close the slide.

        Returns:
            None
        """
        self._pe["in"].close()
        if hasattr(self, "_tile_dimensions"):
            delattr(self, "_tile_dimensions")
        if hasattr(self, "_level_dimensions"):
            delattr(self, "_level_dimensions")
        if hasattr(self, "_level_downsamples"):
            delattr(self, "_level_downsamples")

    @property
    def tile_dimensions(self) -> List[Tuple[int, int]]:
        if not hasattr(self, "_tile_dimensions"):
            tile_w, tile_h = self._pe["in"]["WSI"].block_size()[:2]
            self._tile_dimensions = [(tile_w, tile_h)] * self.level_count
        return self._tile_dimensions

    def _read_region(
        self, x_y: Tuple[int, int], level: int, tile_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_start, y_start = x_y
        ds = self.level_downsamples[level]
        x_start = round(x_start * ds)
        y_start = round(y_start * ds)
        tile_w, tile_h = tile_size
        x_end, y_end = round(x_start + (tile_w - 1) * ds), round(y_start + (tile_h - 1) * ds)
        view_range = [x_start, x_end, y_start, y_end, level]
        regions = self._view.request_regions(
            [view_range],
            self._view.data_envelopes(level),
            True,
            [255, 255, 255],
            self._pe.BufferType(1),
        )
        (region,) = self._pe.wait_any(regions)
        tile = np.empty(np.prod(tile_size) * 4, dtype=np.uint8)
        region.get(tile)
        tile.shape = (tile_h, tile_w, 4)
        return tile[:, :, :3], tile[:, :, 3] > 0

    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        if not hasattr(self, "_level_dimensions"):
            self._level_dimensions = []
            for level in range(self.level_count):
                x_step, x_end = self._view.dimension_ranges(level)[0][1:]
                y_step, y_end = self._view.dimension_ranges(level)[1][1:]
                range_x = (x_end + 1) // x_step
                range_y = (y_end + 1) // y_step
                self._level_dimensions.append((range_x, range_y))
        return self._level_dimensions

    @property
    def level_count(self) -> int:
        return self._view.num_derived_levels + 1

    @property
    def mpp(self) -> Tuple[Optional[float], Optional[float]]:
        return self._view.scale[0], self._view.scale[1]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint8)

    @property
    def n_channels(self) -> int:
        return 3

    @property
    def level_downsamples(self) -> List[float]:
        if not hasattr(self, "_level_downsamples"):
            self._level_downsamples = [
                float(self._view.dimension_ranges(level)[0][1])
                for level in range(self.level_count)
            ]
        return self._level_downsamples


def get_wsi_reader(slide_path: Path) -> Type[WSIReader]:
    """Return a class implementing WSIReader interface based on the image file extension.
    If file extension is .isyntax the class IsyntaxReader is returned, else the class TiffReader.
    Args:
        slide_path (Path): Path of the image file.

    Returns:
        Type[WSIReader]: [description]
    """
    if slide_path.suffix == ".isyntax":
        return IsyntaxReader
    else:
        return TiffReader
