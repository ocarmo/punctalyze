"""
Import data as numpy array
"""

import re
from pathlib import Path
import numpy as np
from loguru import logger
from bioio import BioImage
from aicspylibczi import CziFile
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# ====================== USER CONFIGURATION ======================
# Paths and experiments
input_path = Path('M:/Olivia')
output_folder = Path('results/initial_cleanup')
experiments = ['2026-04-10']  # List of folders; None = all
image_extensions = {'.czi', '.tif', '.tiff', '.lif', '.nd2'}

# Well-to-sample mappings (for multi-well acquisitions)
set1 = {
    'B2-B2': 'jetprime_PopTag-co-mCherry_LIP5_0',
    'B3-B3': 'jetprime_FREE1-co-mCherry_LIP5_0',
    'B4-B4': 'jetprime_FLOE2-co-mCherry_LIP5_0',
    'B5-B5': 'jetprime_FLOE3-co-mCherry_LIP5_0',
}
set2 = {
    'B2-B2': 'jetprime_PopTag-co-mCherry_0',
    'B3-B3': 'jetprime_FREE1-co-mCherry_0',
    'B4-B4': 'jetprime_FLOE2-co-mCherry_0',
    'B5-B5': 'jetprime_FLOE3-co-mCherry_0',
}
experiment_mappings = {
    '2026-04-10': set1,
    '2026-04-11': set1,
    '2026-04-14': set2,
    '2026-04-15': set2,
}
default_mapping = None  # Fallback for unlisted experiments

# Image processing
scene_mode = 'mosaic'     # 'auto', 'mosaic', 'split', 'single'
MIP = True               # Maximum Z projection
experiment_prefix = True # Prefix output filenames with experiment
skip_existing = True     # False = overwrite existing files

# File filtering
do_not_quantitate = ['_no-', 'noDNA', 'UT', 'test', 'fewcells', 'poorfocus']
required_name_parts = ['(', 'co_']  # [] = no restrictions
# ================================================================


def squarify(image):
    """Zero-pad final Y/X axes to square; expects CYX or CZYX."""
    if image.ndim not in (3, 4):
        raise ValueError(f'Expected CYX or CZYX, got {image.shape}')
    height, width = image.shape[-2:]
    pads = [(0, 0)] * (image.ndim - 2) + [(0, max(0, width - height)),
                                             (0, max(0, height - width))]
    return np.pad(image, pads, mode='constant')


def finish_image(data, mip):
    """Input CZYX; output CYX (MIP) or CZYX (no MIP)."""
    if data.ndim != 4:
        raise ValueError(f'Expected CZYX, got {data.shape}')
    return squarify(data.max(axis=1) if mip else data)


def experiment_label(path, root):
    rel = path.resolve().relative_to(root.resolve())
    return rel.parts[0] if len(rel.parts) > 1 else root.name


def safe_name(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(name)).strip(' .')
    if not name:
        raise ValueError('Empty output name')
    return name


def save_array(data, out, skip=True):
    if out.exists():
        if skip:
            logger.info(f'Skipping existing {out}')
            return out
        raise FileExistsError(f'Output already exists: {out}')
    np.save(out, data)
    logger.info(f'Saved {out.name}: {data.shape}, {data.dtype}')
    return out


def mapped_name(scene, mapping, fallback):
    if mapping is None:
        return fallback
    if scene not in mapping:
        logger.warning(f'No mapping for scene {scene!r}; skipping')
        return None
    return mapping[scene]


def output_name(path, label, scene, mapping, prefix, *, single=False):
    # Include acquisition filename for non-mapped scenes to avoid identical scene IDs
    # across files; mapped names intentionally preserve the established convention.
    fallback = path.stem if single else f'{path.stem}_{scene}'
    name = mapped_name(scene, mapping, fallback)
    if name is None:
        return None
    return safe_name(f'{label}_{name}' if prefix else name)


def scene_splitter(path, outdir, label, mapping=None, mip=False, prefix=True,
                   skip=True):
    """Save each independent scene without mosaic stitching."""
    bio = BioImage(str(path))
    outputs = []
    for scene in bio.scenes:
        name = output_name(path, label, scene, mapping, prefix)
        if name is None:
            continue
        out = outdir / f'{name}.npy'
        if skip and out.exists():
            logger.info(f'Skipping existing {out}')
            continue
        bio.set_scene(scene)
        # T=0 matches the original workflow; other timepoints require explicit handling.
        data = bio.get_image_data('CZYX', T=0)
        outputs.append(save_array(finish_image(data, mip), out, skip))
    return outputs


def _mosaic_plane(czi, c, z, scene_index=0):
    """Read one stitched C/Z plane; require unambiguous 2D output."""
    kwargs = {'C': c, 'Z': z, 'scale_factor': 1.0}
    # A single-scene CZI generally needs no S argument.
    # Multi-scene mosaics require explicit scene handling, not silent mixing.
    plane = np.asarray(czi.read_mosaic(**kwargs))
    # Verified Zeiss example: (1, 1, 3892, 3892) -> (3892, 3892).
    plane = np.squeeze(plane)
    if plane.ndim != 2:
        raise ValueError(f'read_mosaic(C={c}, Z={z}) returned {plane.shape}; expected 2D')
    return plane


def scene_finder(path, outdir, label, mapping=None, mip=False, prefix=True,
                 skip=True):
    """Stitch a single-scene Zeiss CZI mosaic across all C and Z planes."""
    bio = BioImage(str(path))
    if len(bio.scenes) != 1:
        raise ValueError(f'Mosaic file has {len(bio.scenes)} scenes; '
                         'this reader currently supports one scene per file')
    scene = bio.scenes[0]
    name = output_name(path, label, scene, mapping, prefix)
    if name is None:
        return []
    out = outdir / f'{name}.npy'
    if skip and out.exists():
        logger.info(f'Skipping existing {out}')
        return [out]
    czi = CziFile(str(path))
    if not czi.is_mosaic():
        raise ValueError(f'Not a Zeiss mosaic: {path}')
    shapes = czi.get_dims_shape()
    if len(shapes) != 1:
        raise ValueError(f'Ambiguous CZI dimension ranges: {shapes}')
    dims = shapes[0]
    for axis in ('C', 'Z'):
        if axis not in dims:
            raise ValueError(f'Missing {axis} dimension: {dims}')
    # Explicitly reject unhandled timepoints/other variable dimensions.
    for axis in ('T', 'S', 'H'):
        start, end = dims.get(axis, (0, 1))
        if end - start > 1:
            raise ValueError(f'Unsupported multi-{axis} mosaic: {dims}')
    c0, c1 = dims['C']
    z0, z1 = dims['Z']
    channels = []
    for c in range(c0, c1):
        if mip:
            projected = None
            for z in range(z0, z1):
                plane = _mosaic_plane(czi, c, z)
                if projected is None:
                    projected = plane.copy()
                else:
                    if plane.shape != projected.shape:
                        raise ValueError(f'Mosaic plane shapes differ in channel {c}')
                    np.maximum(projected, plane, out=projected)
            channels.append(projected)
        else:
            planes = [_mosaic_plane(czi, c, z) for z in range(z0, z1)]
            if len({p.shape for p in planes}) != 1:
                raise ValueError(f'Mosaic plane shapes differ in channel {c}')
            channels.append(np.stack(planes, axis=0))
    if len({arr.shape for arr in channels}) != 1:
        raise ValueError('Mosaic channel shapes differ')
    data = np.stack(channels, axis=0)  # CYX if MIP; otherwise CZYX
    return [save_array(squarify(data), out, skip)]


def image_converter(path, outdir, label, mapping=None, mode='auto', mip=False,
                    prefix=True, skip=True):
    """Dispatch to mosaic stitching, independent-scene splitting, or normal reading."""
    path = Path(path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if mode not in {'auto', 'mosaic', 'split', 'single'}:
        raise ValueError(f'Invalid scene_mode: {mode}')
    if mode == 'auto':
        if path.suffix.lower() == '.czi' and CziFile(str(path)).is_mosaic():
            mode = 'mosaic'
        else:
            bio = BioImage(str(path))
            mode = 'split' if len(bio.scenes) > 1 else 'single'
    if mode == 'mosaic':
        if path.suffix.lower() != '.czi':
            raise ValueError('Mosaic mode requires a .czi file')
        return scene_finder(path, outdir, label, mapping, mip, prefix, skip)
    if mode == 'split':
        return scene_splitter(path, outdir, label, mapping, mip, prefix, skip)
    bio = BioImage(str(path))
    if len(bio.scenes) != 1:
        raise ValueError(f'Single mode requires one scene, found {len(bio.scenes)}')
    scene = bio.scenes[0]
    name = output_name(path, label, scene, mapping, prefix, single=True)
    if name is None:
        return []
    out = outdir / f'{name}.npy'
    if skip and out.exists():
        logger.info(f'Skipping existing {out}')
        return [out]
    data = bio.get_image_data('CZYX', T=0)
    return [save_array(finish_image(data, mip), out, skip)]


def discover_images(root, selected, extensions, exclude, required):
    """Discover experiment images, avoiding recursive duplicate scans."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if selected is None:
        folders = sorted(p for p in root.iterdir() if p.is_dir())
        # Allow a flat raw_data directory as well.
        if not folders:
            folders = [root]
    else:
        folders = [root / name for name in selected]
    found = {}
    for folder in folders:
        if not folder.is_dir():
            logger.warning(f'Missing experiment folder: {folder}')
            continue
        for path in folder.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if any(term in str(path) for term in exclude):
                continue
            if not all(term in path.name for term in required):
                continue
            found[path.resolve()] = (experiment_label(path, root), path)
    return [found[key] for key in sorted(found)]


if __name__ == '__main__':
    images = discover_images(input_path, experiments, image_extensions,
                             do_not_quantitate, required_name_parts)
    logger.info(f'Found {len(images)} images')
    errors = []
    for label, path in images:
        mapping = experiment_mappings.get(label, default_mapping)
        try:
            logger.info(f'Processing {label}: {path.name}')
            image_converter(path, output_folder, label, mapping=mapping,
                            mode=scene_mode, mip=MIP, prefix=experiment_prefix,
                            skip=skip_existing)
        except Exception as exc:
            logger.exception(f'Failed {path}: {exc}')
            errors.append(str(path))
    logger.info(f'Complete: {len(images) - len(errors)}/{len(images)} files without errors')
    if errors:
        raise RuntimeError(f'{len(errors)} image(s) failed; inspect log')
