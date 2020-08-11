"""Use the cryptomattes from Vray/Blender and extract masks from them"""
import argparse
import enum
import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import Imath
import OpenEXR
import imageio
import numpy as np

from exr_info import exr_info

MASK_THRESHOLD = 0.55 * 255


class ExrDtype(enum.Enum):
    FLOAT32 = 0
    FLOAT16 = 1


def get_imsize(exr_file: OpenEXR.InputFile):
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
                         dtype: ExrDtype = ExrDtype.FLOAT32):
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


def get_crypto_layer(exr_file: OpenEXR.InputFile, layer_mapping: exr_info.CryptoLayerMapping):
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


def get_crypto_layers_from_exr(exr_file: OpenEXR.InputFile):
    """Extracts all the cryptomatte layers from an EXR file

    Note:
        - By default, there are 3 crypto layers in an EXR, corresponding to a Cryptomatte of Level 6.
          Example: ['cryptomatte_00', 'cryptomatte_01', 'cryptomatte_02'].
        - The "Level" of a cryptomatte corresponds to the max number of unique objects it can store masks for.
        - Each cryptomatte layer has 4 channels for RGBA.

    Args:
        exr_file (OpenEXR.InputFile): Opened EXR file object

    Returns:
        numpy.ndarray: First cryptomatte layer. Shape: (H, W, 4)
        numpy.ndarray: Second cryptomatte layer. Shape: (H, W, 4)
        numpy.ndarray: Third cryptomatte layer. Shape: (H, W, 4)
    """
    # TODO: Change the number of cryptomatte layers depending on the level of the cryptomatte.
    exr_channels = exr_info.ExrChannels(exr_info.Renderer.VRAY)
    cr_00 = get_crypto_layer(exr_file, exr_channels.cryptomatte_00)
    cr_01 = get_crypto_layer(exr_file, exr_channels.cryptomatte_00)
    cr_02 = get_crypto_layer(exr_file, exr_channels.cryptomatte_00)

    return cr_00, cr_01, cr_02


def get_mask_for_id(float_id, cr_00, cr_01, cr_02):
    id_rank0 = (cr_00[:, :, 0] == float_id)
    coverage0 = cr_00[:, :, 1] * id_rank0
    id_rank1 = (cr_00[:, :, 2] == float_id)
    coverage1 = cr_00[:, :, 3] * id_rank1

    id_rank2 = (cr_01[:, :, 0] == float_id)
    coverage2 = cr_01[:, :, 1] * id_rank2
    id_rank3 = (cr_01[:, :, 2] == float_id)
    coverage3 = cr_01[:, :, 3] * id_rank3

    id_rank4 = (cr_02[:, :, 0] == float_id)
    coverage4 = cr_02[:, :, 1] * id_rank4
    id_rank5 = (cr_02[:, :, 2] == float_id)
    coverage5 = cr_02[:, :, 3] * id_rank5

    coverage = coverage0 + coverage1 + coverage2 + coverage3 + coverage4 + coverage5
    mask = (coverage * 255).astype(np.uint8)

    return mask


def extract_mask(exr_file):
    # Get the manifest (mapping of object names to hash ids)
    header = exr_file.header()
    for key in header:
        if '/manifest' in key:
            manifest = json.loads(header[key])
            break

    # Convert hash ids to float ids
    float_ids = {}
    for name, value in manifest.items():
        bytes_val = bytes.fromhex(value)
        float_val = struct.unpack('>f', bytes_val)[0]
        float_ids[name] = float_val

    # Extract the crypto layers from EXR file
    cr_00, cr_01, cr_02 = get_crypto_layers_from_exr(exr_file)

    # Extract mask of each object in manifest
    mask_list = []
    id_list = []
    id_mapping = {}  # Mapping the name of each obj to obj id
    for obj_id, (obj_name, float_id) in enumerate(float_ids.items()):
        mask = get_mask_for_id(float_id, cr_00, cr_01, cr_02)
        mask_list.append(mask)
        id_list.append(obj_id)
        id_mapping[obj_name] = obj_id

    # Combine all the masks into single mask
    height, width = get_imsize(exr_file)
    mask_combined = np.zeros((height, width), dtype=np.uint16)
    mask_combined_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for mask, obj_id in zip(mask_list, id_list):
        mask_combined[mask > MASK_THRESHOLD] = obj_id

        # TODO: Generate random hue, keeping saturation and value constant. Then convert HSV to RGB.
        rand_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        mask_combined_rgb[mask > MASK_THRESHOLD, :] = rand_color

    return mask_combined, mask_combined_rgb, id_mapping


def main(args):
    path_exr = args.file
    if not path_exr.exists():
        raise ValueError(f'The file does not exist: {path_exr}')

    exr_file = OpenEXR.InputFile(str(path_exr))
    mask_combined, mask_combined_rgb, id_mapping = extract_mask(exr_file)

    out_dir = path_exr.parent
    out_file = out_dir / f"{path_exr.stem}.mask.png"
    imageio.imwrite(out_file, mask_combined)

    out_file = out_dir / f"{path_exr.stem}.mask_rgb.png"
    imageio.imwrite(out_file, mask_combined_rgb)

    out_file = out_dir / f"{path_exr.stem}.mask_id_mapping.json"
    with out_file.open('w') as json_file:
        json.dump(id_mapping, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect EXR header')
    parser.add_argument("-f", "--file", help="EXR file name", default='', type=Path)
    args = parser.parse_args()
    main(args)
