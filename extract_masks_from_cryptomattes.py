"""Use the cryptomattes from Vray/Blender and extract masks from them"""
import argparse
import struct
import json
from pathlib import Path

import OpenEXR
import Imath
import numpy as np
from PIL import Image

from exr_info import exr_info

parser = argparse.ArgumentParser(description='Inspect EXR header')
parser.add_argument("-f", "--file", help="EXR file name", default='', type=str)
args = parser.parse_args()

path_exr = Path(args.file)
if not path_exr.exists():
    raise ValueError('The file does not exist: {}'.format(path_exr))

exr_file = OpenEXR.InputFile(str(path_exr))
header = exr_file.header()
dw = header['dataWindow']
width, height = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
imsize = (height, width)
pt_f32 = Imath.PixelType(Imath.PixelType.FLOAT)

exr_channels = exr_info.ExrChannels('vray')

# Extract RGB Image
im = np.zeros(imsize + (3,), dtype=np.float32)
im[:, :, 0] = np.frombuffer(exr_file.channel('R', pt_f32), dtype=np.float32).reshape(imsize)
im[:, :, 1] = np.frombuffer(exr_file.channel('G', pt_f32), dtype=np.float32).reshape(imsize)
im[:, :, 2] = np.frombuffer(exr_file.channel('B', pt_f32), dtype=np.float32).reshape(imsize)
im = exr_info.lin_rgb_to_srgb_colorspace(im)
img_rgb = (im.clip(0, 1) * 255).astype(np.uint8)
img_rgb = Image.fromarray(img_rgb)
# img_rgb.show()
img_rgb.save('output/rgb_out.png')

# Extract Cryptomatte
manifest = json.loads(header['cryptomatte/881c23b/manifest'])
print('manifest: ', manifest)

float_ids = {}
for name, value in manifest.items():
    bytes_val = bytes.fromhex(value)
    float_val = struct.unpack('>f', bytes_val)[0]
    float_ids[name] = float_val
print('float_ids: ', float_ids)


cr_00 = np.zeros(imsize + (4,), dtype=np.float32)
cr_00[:, :, 0] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_00['R'], pt_f32), dtype=np.float32).reshape(imsize)
cr_00[:, :, 1] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_00['G'], pt_f32), dtype=np.float32).reshape(imsize)
cr_00[:, :, 2] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_00['B'], pt_f32), dtype=np.float32).reshape(imsize)
cr_00[:, :, 3] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_00['A'], pt_f32), dtype=np.float32).reshape(imsize)
cr_01 = np.zeros(imsize + (4,), dtype=np.float32)
cr_01[:, :, 0] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_01['R'], pt_f32), dtype=np.float32).reshape(imsize)
cr_01[:, :, 1] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_01['G'], pt_f32), dtype=np.float32).reshape(imsize)
cr_01[:, :, 2] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_01['B'], pt_f32), dtype=np.float32).reshape(imsize)
cr_01[:, :, 3] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_01['A'], pt_f32), dtype=np.float32).reshape(imsize)
cr_02 = np.zeros(imsize + (4,), dtype=np.float32)
cr_02[:, :, 0] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_02['R'], pt_f32), dtype=np.float32).reshape(imsize)
cr_02[:, :, 1] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_02['G'], pt_f32), dtype=np.float32).reshape(imsize)
cr_02[:, :, 2] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_02['B'], pt_f32), dtype=np.float32).reshape(imsize)
cr_02[:, :, 3] = np.frombuffer(exr_file.channel(exr_channels.cryptomatte_02['A'], pt_f32), dtype=np.float32).reshape(imsize)

id_tmp = 'walls'
id_rank0 = (cr_00[:, :, 0] == float_ids[id_tmp])
coverage0 = cr_00[:, :, 1] * id_rank0
id_rank1 = (cr_00[:, :, 2] == float_ids[id_tmp])
coverage1 = cr_00[:, :, 3] * id_rank1
# mask1 = Image.fromarray(id_rank0.astype(np.uint8) * 255)
# mask1.show(title='id_rank0')
# mask_c1 = Image.fromarray(coverage0.astype(np.uint8) * 255)
# mask_c1.show(title='coverage0')

id_rank2 = (cr_01[:, :, 0] == float_ids[id_tmp])
coverage2 = cr_01[:, :, 1] * id_rank2
id_rank3 = (cr_01[:, :, 2] == float_ids[id_tmp])
coverage3 = cr_01[:, :, 3] * id_rank3

id_rank4 = (cr_02[:, :, 0] == float_ids[id_tmp])
coverage4 = cr_02[:, :, 1] * id_rank4
id_rank5 = (cr_02[:, :, 2] == float_ids[id_tmp])
coverage5 = cr_02[:, :, 3] * id_rank5

coverage = coverage0 + coverage1 + coverage2 + coverage3 + coverage4 + coverage5
mask_c2 = Image.fromarray((coverage * 255).astype(np.uint8))
# mask_c2.show(title='id_rank0')
mask_c2.save('output/mask_walls.png')



