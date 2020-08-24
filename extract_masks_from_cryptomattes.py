"""Use the cryptomattes from Vray/Blender and extract masks from them"""
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import imageio
from omegaconf import OmegaConf

import cryptomatte_masks

CONFIG_FILE = 'config.yaml'


def process_file(path_exr: Path, output_dir: Optional[Path], mask_ext: str, mask_rgb_ext: str, mask_mapping_json: str):
    """Extract mask from an EXR and save to disk
    The mapping from object names to ids in mask is saved as a json. We perform checks to make sure the manifest
    is the same for all images in the dir.

    Args:
        path_exr (pathlib.Path): The input EXR file
        output_dir (None or pathlib.Path): The dir to create output files in
        mask_ext (str): The extention to give to filenames of output masks
        mask_rgb_ext (str): The extention to give to filenames of RGB visualization of output masks
        mask_mapping_json (str): The name of the output file containing mappings from object names to IDs in mask.
    """
    if not path_exr.exists():
        raise ValueError(f'The file does not exist: {path_exr}')
    print(f'Extracting masks from: {path_exr}')

    mask_mapping_file = output_dir / mask_mapping_json
    if mask_mapping_file.exists():
        with mask_mapping_file.open() as fd:
            id_mapping_saved = json.load(fd, object_pairs_hook=OrderedDict)

    mask_combined, mask_combined_rgb, id_mapping = cryptomatte_masks.extract_mask_from_file(str(path_exr))

    if output_dir is None:
        output_dir = path_exr.parent

    out_file = output_dir / f"{path_exr.stem}{mask_ext}"
    imageio.imwrite(out_file, mask_combined)

    out_file = output_dir / f"{path_exr.stem}{mask_rgb_ext}"
    imageio.imwrite(out_file, mask_combined_rgb)

    if not mask_mapping_file.exists():
        with mask_mapping_file.open('w') as json_file:
            json.dump(id_mapping, json_file)
    else:
        if id_mapping != id_mapping_saved:
            print(f'manifest_saved: {id_mapping_saved}, conflicting_manifest: {id_mapping}')
            raise ValueError(f'The manifest in EXR file {path_exr} changed from manifest in previous images in same dir!')


def main():
    base_conf = OmegaConf.load(CONFIG_FILE)
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(base_conf, cli_conf)

    dir_input = Path(conf.dir_input)
    dir_output = Path(conf.dir_output)
    input_exr_ext = conf.input_exr_ext
    mask_mapping_json = conf.mask_mapping_json
    random_seed = conf.random_seed
    random.seed(random_seed)

    output_mask_ext = conf.output_mask_ext
    output_mask_rgb_ext = conf.output_mask_rgb_ext

    if not dir_input.is_dir():
        raise ValueError(f'Not a directory: {dir_input}')
    dir_output.mkdir(parents=True, exist_ok=True)

    exr_filenames = sorted(dir_input.glob('*' + input_exr_ext))

    for f_exr in exr_filenames:
        process_file(f_exr, dir_output, output_mask_ext, output_mask_rgb_ext, mask_mapping_json)


if __name__ == "__main__":
    main()
