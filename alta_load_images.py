import os
import hashlib
import numpy as np
import torch
from PIL import Image, ImageOps
import itertools

from comfy.k_diffusion.utils import FolderOfImages
from comfy.utils import common_upscale, ProgressBar
# from .logger import logger
from .utils import BIGMAX, calculate_file_hash, get_sorted_dir_files_from_directory, validate_path, strip_path


def is_changed_load_images(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1, **kwargs):
    if not os.path.isdir(directory):
            return False
        
    dir_files = get_sorted_dir_files_from_directory(directory, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)
    if image_load_cap != 0:
        dir_files = dir_files[:image_load_cap]

    m = hashlib.sha256()
    for filepath in dir_files:
        m.update(calculate_file_hash(filepath).encode()) # strings must be encoded before hashing
    return m.digest().hex()


def validate_load_images(directory: str):
    if not os.path.isdir(directory):
            return f"Directory '{directory}' cannot be found."
    dir_files = os.listdir(directory)
    if len(dir_files) == 0:
        return f"No files in directory '{directory}'."

    return True

def images_generator(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1, meta_batch=None, unique_id=None):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory} cannot be found.")
    dir_files = get_sorted_dir_files_from_directory(directory, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)

    if len(dir_files) == 0:
        raise FileNotFoundError(f"No files in directory '{directory}'.")
    if image_load_cap > 0:
        dir_files = dir_files[:image_load_cap]
    sizes = {}
    has_alpha = False
    for image_path in dir_files:
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        has_alpha |= 'A' in i.getbands()
        count = sizes.get(i.size, 0)
        sizes[i.size] = count +1
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1], has_alpha
    if meta_batch is not None:
        yield min(image_load_cap, len(dir_files)) or len(dir_files)

    iformat = "RGBA" if has_alpha else "RGB"
    def load_image(file_path):
        i = Image.open(file_path)
        i = ImageOps.exif_transpose(i)
        i = i.convert(iformat)
        i = np.array(i, dtype=np.float32)
        torch.from_numpy(i).div_(255)  # normalize
        if i.shape[0] != size[1] or i.shape[1] != size[0]:
            i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
            i = common_upscale(i, size[0], size[1], "lanczos", "center")
            i = i.squeeze(0).movedim(0, -1).numpy()
        if has_alpha:
            i[:,:,-1] = 1 - i[:,:,-1]
        return i

    total_images = len(dir_files)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, dir_files)

    try:
        prev_image = next(images)
        prev_filename = os.path.basename(dir_files[0])  # 首个文件名
        for file_path, next_image in zip(dir_files[1:], images):
            yield prev_image, prev_filename
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image, prev_filename = next_image, os.path.basename(file_path)
    except StopIteration:
        pass

    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id)
        meta_batch.has_closed_inputs = True

    if prev_image is not None:
        yield prev_image, prev_filename


def load_images(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1, meta_batch=None, unique_id=None):
    if meta_batch is None or unique_id not in meta_batch.inputs:
        gen = images_generator(directory, image_load_cap, skip_first_images, select_every_nth, meta_batch, unique_id)
        (width, height, has_alpha) = next(gen)
        if meta_batch is not None:
            meta_batch.inputs[unique_id] = (gen, width, height, has_alpha)
            meta_batch.total_frames = min(meta_batch.total_frames, next(gen))
    else:
        gen, width, height, has_alpha = meta_batch.inputs[unique_id]

    if meta_batch is not None:
        gen = itertools.islice(gen, meta_batch.frames_per_batch)

    # 收集图像和文件名
    results = list(gen)
    if len(results) == 0:
        raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

    images_np, filenames = zip(*results)
    images = torch.from_numpy(np.array(images_np, dtype=np.float32))

    if has_alpha:
        masks = images[:,:,:,3]
        images = images[:,:,:,:3]
    else:
        masks = torch.zeros((images.size(0), 64, 64), dtype=torch.float32, device="cpu")

    return images, masks, images.size(0), list(filenames)


class LoadImagesFromDirectoryUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        directories = []
        for item in os.listdir(input_dir):
            if not os.path.isfile(os.path.join(input_dir, item)) and item != "clipspace":
                directories.append(item)
        return {
            "required": {
                "directory": (directories,),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count")
    FUNCTION = "load_images"

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    def load_images(self, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(strip_path(directory))
        return load_images(directory, **kwargs)
    
    @classmethod
    def IS_CHANGED(s, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(strip_path(directory))
        return is_changed_load_images(directory, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(s, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(strip_path(directory))
        return validate_load_images(directory)



class LoadImagesFromDirectoryPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"placeholder": "X://path/to/images", "vhs_path_extensions": []}),
                "output_folder": ("STRING", {"placeholder": "X://path/to/output", "default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "LIST")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count", "filepaths")
    FUNCTION = "load_images"

    CATEGORY = "Alta"

    def load_images(self, directory: str, output_folder: str, **kwargs):
        directory = strip_path(directory)
        if directory is None or validate_load_images(directory) != True:
            raise Exception("directory is not valid: " + str(directory))

        images, masks, frame_count, filenames = load_images(directory, **kwargs)

        # 拼接 output folder + filename 得到完整路径
        if output_folder and os.path.isdir(output_folder):
            filepaths = [os.path.join(output_folder, fn) for fn in filenames]
        else:
            # 如果没传 output_folder 就直接返回原始 filenames
            filepaths = filenames

        return images, masks, frame_count, filepaths
    
    @classmethod
    def IS_CHANGED(s, directory: str, **kwargs):
        if directory is None:
            return "input"
        return is_changed_load_images(directory, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(s, directory: str, **kwargs):
        if directory is None:
            return True
        return validate_load_images(strip_path(directory))
    

# 节点映射
NODE_CLASS_MAPPINGS = {
    "Alta:LoadImagesPath": LoadImagesFromDirectoryPath
}
