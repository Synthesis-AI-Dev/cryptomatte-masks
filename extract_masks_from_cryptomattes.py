"""Use the cryptomattes from Vray/Blender and extract masks from them"""
import json
import random
from pathlib import Path
from typing import Optional

import imageio
from omegaconf import OmegaConf, DictConfig

import cryptomatte_masks

CONFIG_FILE = 'config.yaml'


def validate_inputs(conf: DictConfig):
    if not Path(conf.dir_input).is_dir():
        raise ValueError(f'Not a directory: {conf.dir_input}')

    valid_img_formats = ['jpg', 'png']
    if conf.output_mask_ext.split('.')[-1] not in valid_img_formats:
        raise ValueError(f'Invalid format for output mask file ({conf.output_mask_ext}). '
                         f'Valid formats: {valid_img_formats}')
    if conf.output_mask_rgb_ext.split('.')[-1] not in valid_img_formats:
        raise ValueError(f'Invalid format for output mask file ({conf.output_mask_rgb_ext}). '
                         f'Valid formats: {valid_img_formats}')

    if conf.input_exr_ext.split('.')[-1] != 'exr':
        raise ValueError(f'Invalid file ext "{conf.input_exr_ext}". Input must be in .exr format.')

    if conf.id_map_ext.split('.')[-1] != 'json':
        raise ValueError(f'Invalid file ext "{conf.id_map_ext}". Input must be in .json format.')

    file_ext = {
        'input': conf.input_exr_ext,
        'mask': conf.output_mask_ext,
        'mask_rgb': conf.output_mask_rgb_ext,
        'id_map': conf.id_map_ext
    }

    return file_ext


def process_file(path_exr: Path, mask_ext: str, mask_rgb_ext: str, id_map_ext: str, output_dir: Optional[Path] = None):
    """Extract mask from an EXR and save to disk
    The mapping from object names to ids in mask is saved as a json. We perform checks to make sure the manifest
    is the same for all images in the dir.

    Args:
        path_exr (pathlib.Path): The input EXR file
        output_dir (None or pathlib.Path): The dir to create output files in
        mask_ext (str): The extention to give to filenames of output masks
        mask_rgb_ext (str): The extention to give to filenames of RGB visualization of output masks
        id_map_ext (str): The extention to give to filenames of JSONs mapping object names to IDs in mask.
    """
    if not path_exr.exists():
        raise ValueError(f'The file does not exist: {path_exr}')
    print(f'Extracting masks from: {path_exr}')

    mask_combined, mask_combined_rgb, id_mapping = cryptomatte_masks.extract_mask_from_file(str(path_exr))

    if output_dir is None:
        output_dir = path_exr.parent

    out_file = output_dir / f"{path_exr.stem}{mask_ext}"
    imageio.imwrite(out_file, mask_combined)

    out_file = output_dir / f"{path_exr.stem}{mask_rgb_ext}"
    imageio.imwrite(out_file, mask_combined_rgb)

    id_map_file = output_dir / f"{path_exr.stem}{id_map_ext}"
    with id_map_file.open('w') as json_file:
        json.dump(id_mapping, json_file, indent=4)
        json_file.write('\n')


def main():
    base_conf = OmegaConf.load(CONFIG_FILE)
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(base_conf, cli_conf)

    file_ext = validate_inputs(conf)
    input_exr_ext = file_ext['input']
    output_mask_ext = file_ext['mask']
    output_mask_rgb_ext = file_ext['mask_rgb']
    id_map_ext = file_ext['id_map']

    dir_input = Path(conf.dir_input)
    dir_output = Path(conf.dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)
    random_seed = int(conf.random_seed)
    random.seed(random_seed)

    exr_filenames = sorted(dir_input.glob('*' + input_exr_ext))
    for f_exr in exr_filenames:
        process_file(f_exr, output_mask_ext, output_mask_rgb_ext, id_map_ext, dir_output)


if __name__ == "__main__":
    main()
