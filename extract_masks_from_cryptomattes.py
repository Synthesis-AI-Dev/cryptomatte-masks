"""Use the cryptomattes from Vray/Blender and extract masks from them"""
import argparse
import json
import random
import struct
from pathlib import Path

import Imath
import OpenEXR
import imageio
import numpy as np

from exr_info import exr_info

MASK_THRESHOLD = 0.55 * 255


def get_crypto_layer_from_exr(exr_file, imsize):
    exr_channels = exr_info.ExrChannels(exr_info.Renderer.VRAY)
    pt_f32 = Imath.PixelType(Imath.PixelType.FLOAT)

    cr_00 = np.zeros(imsize + (4,), dtype=np.float32)
    cr_00[:, :, 0] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_00['R'], pt_f32),
                                   dtype=np.float32).reshape(imsize)
    cr_00[:, :, 1] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_00['G'], pt_f32),
                                   dtype=np.float32).reshape(imsize)
    cr_00[:, :, 2] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_00['B'], pt_f32),
                                   dtype=np.float32).reshape(imsize)
    cr_00[:, :, 3] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_00['A'], pt_f32),
                                   dtype=np.float32).reshape(imsize)

    cr_01 = np.zeros(imsize + (4,), dtype=np.float32)
    cr_01[:, :, 0] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_01['R'], pt_f32),
                                   dtype=np.float32).reshape(imsize)
    cr_01[:, :, 1] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_01['G'], pt_f32),
                                   dtype=np.float32).reshape(imsize)
    cr_01[:, :, 2] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_01['B'], pt_f32),
                                   dtype=np.float32).reshape(imsize)
    cr_01[:, :, 3] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_01['A'], pt_f32),
                                   dtype=np.float32).reshape(imsize)

    cr_02 = np.zeros(imsize + (4,), dtype=np.float32)
    cr_02[:, :, 0] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_02['R'], pt_f32),
                                   dtype=np.float32).reshape(imsize)
    cr_02[:, :, 1] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_02['G'], pt_f32),
                                   dtype=np.float32).reshape(imsize)
    cr_02[:, :, 2] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_02['B'], pt_f32),
                                   dtype=np.float32).reshape(imsize)
    cr_02[:, :, 3] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_02['A'], pt_f32),
                                   dtype=np.float32).reshape(imsize)

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

    header = exr_file.header()
    dw = header['dataWindow']
    width, height = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    imsize = (height, width)

    for key in header:
        if '/manifest' in key:
            manifest = json.loads(header[key])
            break

    float_ids = {}
    for name, value in manifest.items():
        bytes_val = bytes.fromhex(value)
        float_val = struct.unpack('>f', bytes_val)[0]
        float_ids[name] = float_val

    cr_00, cr_01, cr_02 = get_crypto_layer_from_exr(exr_file, imsize)

    mask_list = {}
    id_mapping = {}
    for ii, (obj_name, float_id) in enumerate(float_ids.items()):
        mask = get_mask_for_id(float_id, cr_00, cr_01, cr_02)
        mask_list[obj_name] = mask
        id_mapping[obj_name] = ii

    mask_combined = np.zeros(imsize, dtype=np.uint16)
    mask_combined_rgb = np.zeros(imsize + (3, ), dtype=np.uint8)
    for obj_name in mask_list:
        mask = mask_list[obj_name]
        obj_id = id_mapping[obj_name]

        mask_combined[mask > MASK_THRESHOLD] = obj_id
        # TODO: Generate random hue, keeping saturation and value constant. Then convert HSV to RGB.
        color = lambda: [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        rand_color = color()
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
