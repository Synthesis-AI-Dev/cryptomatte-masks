import colorsys
import enum
import json
import math
import random
import struct
from collections import OrderedDict
from typing import Optional, Tuple, Dict

import Imath
import OpenEXR
import numpy as np

import exr_info

MASK_THRESHOLD = 0.48 * 255
MANIFEST_IDENTIFIER = '/manifest'


class ExrDtype(enum.Enum):
    FLOAT32 = 0
    FLOAT16 = 1


def get_imsize(exr_file: OpenEXR.InputFile) -> Tuple[int, int]:
    """Get the height and width of image within and EXR file

    Args:
        exr_file (OpenEXR.InputFile): The opened EXR file object.

    Returns:
        int, int: Height, Width of image
    """
    header = exr_file.header()
    dw = header['dataWindow']
    height = int(dw.max.y - dw.min.y + 1)
    width = int(dw.max.x - dw.min.x + 1)

    return height, width


def exr_channel_to_numpy(exr_file: OpenEXR.InputFile, channel_name: str, reshape: Optional[Tuple[int, int]],
                         dtype: ExrDtype = ExrDtype.FLOAT32) -> np.ndarray:
    """Extracts a channel in an EXR file into a numpy array

    Args:
        exr_file (OpenEXR.InputFile): The opened EXR file object
        channel_name (str): The name of the channel to be converted to numpy
        reshape (None or Tuple(height, width): If given, will reshape the 2D numpy array into (height, width)
        dtype (ExrDtype): Whether the data in channel is of float32 or float16 type

    Returns:
        numpy.ndarray: The extracted channel in form of numpy array
    """
    if dtype == ExrDtype.FLOAT32:
        point_type = Imath.PixelType(Imath.PixelType.FLOAT)
        np_type = np.float32
    else:
        point_type = Imath.PixelType(Imath.PixelType.HALF)
        np_type = np.float16

    channel_arr = np.frombuffer(exr_file.channel(channel_name, point_type), dtype=np_type)
    if reshape:
        channel_arr = channel_arr.reshape(reshape)

    return channel_arr


def get_crypto_layer(exr_file: OpenEXR.InputFile, layer_mapping: exr_info.CryptoLayerMapping) -> np.ndarray:
    """Extract one of the cryptomatte layers as a 4-channel numpy array.
    Each cryptomatte layer has 4 channels for RGBA.

    Note:
        - By default, there are 3 crypto layers in an EXR, corresponding to a Cryptomatte of Level 6.
          Example: ['cryptomatte_00', 'cryptomatte_01', 'cryptomatte_02'].
        - The "Level" of a cryptomatte corresponds to the max number of unique objects it can store masks for.
        - Each cryptomatte layer has 4 channels for RGBA.

    Args:
        exr_file (OpenEXR.InputFile): The opened EXR file object
        layer_mapping (exr_info.CryptoLayerMapping): A dataclass containing the exact exr channel name for each of the
                                                     'RGBA' channels in a cryptomatte layer.

    Returns:
        numpy.ndarray: The cryptomatte layer. Shape: (H, W, 4)
    """
    height, width = get_imsize(exr_file)
    cr_00 = np.zeros((height, width, 4), dtype=np.float32)

    cr_00[:, :, 0] = exr_channel_to_numpy(exr_file, channel_name=layer_mapping.R, reshape=(height, width))
    cr_00[:, :, 1] = exr_channel_to_numpy(exr_file, channel_name=layer_mapping.G, reshape=(height, width))
    cr_00[:, :, 2] = exr_channel_to_numpy(exr_file, channel_name=layer_mapping.B, reshape=(height, width))
    cr_00[:, :, 3] = exr_channel_to_numpy(exr_file, channel_name=layer_mapping.A, reshape=(height, width))

    return cr_00


def get_crypto_layers_from_exr(exr_file: OpenEXR.InputFile, level: int = 6) -> np.ndarray:
    """Extracts all the cryptomatte layers from an EXR file

    Note:
        - By default, there are 3 crypto layers in an EXR, corresponding to a Cryptomatte of Level 6.
          Example: ['cryptomatte_00', 'cryptomatte_01', 'cryptomatte_02'].
        - The "Level" of a cryptomatte corresponds to the max number of unique objects it can store masks for.
        - Each cryptomatte layer has 4 channels for RGBA.

    Args:
        exr_file (OpenEXR.InputFile): Opened EXR file object
        level (int): Default is 6. The Level of the cryptomatte limits the max. num of unique masks in the cryptomatte.
                     The number of cruptomatte layers depends on the level.

    Returns:
        numpy.ndarray: Combined cryptomatte layer. Shape: (H, W, 4 * ceil(level/2))
    """
    exr_channels = exr_info.ExrChannels(exr_info.Renderer.VRAY)

    num_layers = math.ceil(level / 2)
    cr_list = []
    for layer_num in range(num_layers):
        cr = get_crypto_layer(exr_file, exr_channels.cryptomatte[f'{layer_num:02d}'])
        cr_list.append(cr)

    cr_combined = np.concatenate(cr_list, axis=2)

    return cr_combined


def get_coverage_for_rank(float_id: float, cr_combined: np.ndarray, rank: int) -> np.ndarray:
    """Get the coverage mask for a given rank from cryptomatte layers
    Args:
        float_id (float32): The ID of the object
        cr_combined (numpy.ndarray): The cryptomatte layers combined into a single array along the channels axis.
                                     By default, there are 3 layers, corresponding to a level of 6.
        rank (int): The rank, or level, of the coverage to be calculated

    Returns:
        numpy.ndarray: Mask for given coverage rank. Dtype: np.float32, Range: [0, 1]
    """
    id_rank = (cr_combined[:, :, rank * 2] == float_id)
    coverage_rank = cr_combined[:, :, rank * 2 + 1] * id_rank

    return coverage_rank


def get_mask_for_id(float_id: float, cr_combined: np.ndarray, level: int = 6) -> np.ndarray:
    """Extract mask corresponding to a float id from the cryptomatte layers
    Args:
        float_id (float32): The ID of the object
        cr_combined (numpy.ndarray): The cryptomatte layers combined into a single array along the channels axis.
                                     By default, there are 3 layers, corresponding to a level of 6.
        level (int): The Level of the Cryptomatte. Default is 6 for most rendering engines. The level dictates the
                     max num of objects that the crytomatte can represent. The number of cryptomatte layers in EXR
                     will change depending on level.

    Returns:
        numpy.ndarray: Mask from cryptomatte for a given id. Dtype: np.uint8, Range: [0, 255]
    """
    coverage_list = []
    for rank in range(level):
        coverage_rank = get_coverage_for_rank(float_id, cr_combined, rank)
        coverage_list.append(coverage_rank)
    coverage = sum(coverage_list)
    coverage = np.clip(coverage, 0.0, 1.0)

    mask = (coverage * 255).astype(np.uint8)

    return mask


def extract_mask(exr_file: OpenEXR.InputFile,
                 extract_id_mapping_from_manifest: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Get a mask of all the objects in an EXR image from the cryptomatte
    Args:
        exr_file (OpenEXR.InputFile): The opened EXR file object
        extract_id_mapping_from_manifest (bool): In latest renders, the ID that each object in cryptomatte maps to
                                                 in the output mask is present in the object's name in the manifest.
                                                 Eg: cords_2, means that the object is of type "cords" and it maps to
                                                     a value of 2 in the output mask

    Returns:
        numpy.ndarray: Mask of all objects in scene. Each object has a unique value. Dtype: np.float16, Shape: (H, W)
        numpy.ndarray: RGB visualization of the mask. Dtype: np.uint8, Shape: (H, W, 3)
        dict[]
    """
    # Get the manifest (mapping of object names to hash ids)
    header = exr_file.header()
    manifest = None
    for key in header:
        if MANIFEST_IDENTIFIER in key:
            manifest = json.loads(header[key], object_pairs_hook=OrderedDict)
            break
    if manifest is None:
        raise RuntimeError('The EXR file\'s header does not contain the manifest for cryptomattes')

    # Convert hash ids to float ids
    float_ids = []
    obj_ids = []
    for ii, obj_name in enumerate(sorted(manifest)):
        hex_id = manifest[obj_name]
        bytes_val = bytes.fromhex(hex_id)
        float_val = struct.unpack('>f', bytes_val)[0]
        float_ids.append(float_val)
        obj_ids.append(ii)

    # Extract the crypto layers from EXR file
    cr_combined = get_crypto_layers_from_exr(exr_file)

    # Extract mask of each object in manifest
    mask_list = []
    id_list = []
    id_mapping = OrderedDict()  # Mapping the name of each obj to obj id
    for float_id, obj_id, obj_name in zip(float_ids, obj_ids, sorted(manifest)):
        # Ignore the vrayLightDome.
        if obj_name == 'vrayLightDome':
            continue

        mask = get_mask_for_id(float_id, cr_combined)
        mask_list.append(mask)

        if extract_id_mapping_from_manifest:
            # The object type and ID is encoded in the object's name in manifest
            obj_type, obj_id_manifest = obj_name.split('_')
            id_mapping[obj_name] = int(obj_id_manifest)
            id_list.append(int(obj_id_manifest))
        else:
            # The ID will be same as index of object in manifest. IDs extracted like should should be saved in a
            # separate file so that the mapping from IDs to objects is available
            id_mapping[obj_name] = obj_id
            id_list.append(obj_id)

    # Combine all the masks into single mask
    height, width = get_imsize(exr_file)
    mask_combined = np.zeros((height, width), dtype=np.uint16)
    mask_combined_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for mask, obj_id in zip(mask_list, id_list):
        mask_combined[mask > MASK_THRESHOLD] = obj_id

        hue = random.random()
        sat, val = 0.7, 0.7
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        rgb = []
        for col in [r, g, b]:
            col_np = np.array(col, dtype=np.float32)
            col_np = (np.clip(col_np * 255, 0, 255)).astype(np.uint8)
            rgb.append(col_np)
        mask_combined_rgb[mask > MASK_THRESHOLD, :] = rgb

    return mask_combined, mask_combined_rgb, id_mapping


def extract_mask_from_file(path_exr: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """From an EXR file on disk, extract the masks"""
    exr_file = OpenEXR.InputFile(str(path_exr))
    mask_combined, mask_combined_rgb, id_mapping = extract_mask(exr_file)
    return mask_combined, mask_combined_rgb, id_mapping