import os
import hashlib
import numpy as np
import torch
from PIL import Image, ImageOps
import itertools
from .logger import logger
from comfy.k_diffusion.utils import FolderOfImages
from comfy.utils import common_upscale, ProgressBar
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


import os
from PIL import Image, ImageOps
import numpy as np
import torch
import folder_paths

class LoadImageFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"multiline": False}),
            }
        }

    CATEGORY = "Alta"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    FUNCTION = "load_image"

    def load_image(self, path):
        # 找到绝对路径
        if os.path.exists(path):
            full_path = path
        else:
            full_path = os.path.join(folder_paths.get_input_directory(), path)

        img_obj = Image.open(full_path)

        # 转 RGB 并处理 EXIF
        img = ImageOps.exif_transpose(img_obj).convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0

        # 转成 (1, H, W, 3)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        filename = os.path.basename(full_path)
        return (img_tensor, filename)
    
class ReadImageFromFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 图片所在目录
                "folder": ("STRING", {
                    "default": "inputs", 
                    "multiline": False
                }),
                # 图片文件名（带或不带扩展名）
                "filename": ("STRING", {
                    "default": "image1",
                    "multiline": False
                }),
                # 可选扩展名（默认为 jpg）
                "extension": ("STRING", {
                    "default": "jpg",
                    "multiline": False
                }),
            }
        }

    CATEGORY = "Alta"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    FUNCTION = "read_image"

    def read_image(self, folder, filename, extension="jpg"):
        # 如果文件名里已经带了扩展名，就不再重复加
        if not filename.lower().endswith(f".{extension.lower()}"):
            filename_with_ext = f"{filename}.{extension}"
        else:
            filename_with_ext = filename

        # 拼接完整路径
        if os.path.isabs(folder):
            full_path = os.path.join(folder, filename_with_ext)
        else:
            # 相对路径：从 ComfyUI 输入目录开始
            full_path = os.path.join(folder_paths.get_input_directory(), folder, filename_with_ext)

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        # 打开图片
        img_obj = Image.open(full_path)
        img = ImageOps.exif_transpose(img_obj).convert("RGB")

        # 转 numpy 并归一化
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W, 3)

        return (img_tensor, filename_with_ext)


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
    
class LoadImageWithPath:
    """
    Load a single image and output (image, filename).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"placeholder": "X://path/image.jpg","default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    FUNCTION = "load_image"
    CATEGORY = "Alta"

    def load_image(self, image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None,]  # (1,H,W,C)
        fname = os.path.basename(image_path)
        return (tensor, fname)


import torch

class GetImageByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),   # 批量图片 (batch,h,w,c)
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "optional": {
                "pre_value": ("IMAGE", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "get_image"
    CATEGORY = "InspirePack/List"

    def get_image(self, images: torch.Tensor, index: int, pre_value: torch.Tensor = None):
        batch_size = images.shape[0]
        if index < 0 or index >= batch_size:
            raise IndexError(f"Index {index} out of range (0~{batch_size-1})")

        # 取出 batch 里的某一张，保持维度一致 (1,h,w,c)
        single = images[index:index+1].clone()
        return (single,)

class GetStringByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strings": ("LIST",),   # 输入是一个 list
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "optional": {
                "pre_value": ("ANY", {"default": None}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "get_string"
    CATEGORY = "InspirePack/List"

    def get_string(self, strings, index: int, pre_value: str = None):
        if not isinstance(strings, (list, tuple)):
            raise TypeError(f"Expected list/tuple, got {type(strings)}")

        total = len(strings)
        if total == 0:
            raise ValueError("Input list is empty.")

        if index < 0 or index >= total:
            raise IndexError(f"Index {index} out of range (0~{total-1})")

        return (strings[index],)
    
import torch

class GetImageAndPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),   # 批量图像 (batch,h,w,c)
                "paths": ("LIST",),     # 对应的路径列表
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "optional": {
                "pre_value": ("IMAGE", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "path")
    FUNCTION = "get_image_and_path"
    CATEGORY = "InspirePack/List"

    def get_image_and_path(self, images: torch.Tensor, paths, index: int, pre_value: torch.Tensor = None):
        # 校验 batch
        batch_size = images.shape[0]
        if index < 0 or index >= batch_size:
            raise IndexError(f"Index {index} out of range for images (0~{batch_size-1})")

        # 校验 list
        if not isinstance(paths, (list, tuple)):
            raise TypeError(f"Expected list/tuple for paths, got {type(paths)}")
        if len(paths) != batch_size:
            raise ValueError(f"Number of paths ({len(paths)}) does not match images batch size ({batch_size})")

        # 取出单张图
        single_image = images[index:index+1].clone()
        single_path = paths[index]

        return (single_image, single_path)


from PIL import Image, ImageOps, ImageSequence

import node_helpers

class LoadImage:
    def __init__(self):
        # 每个节点实例单独存储
        self.original_filename = None

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "Alta"

    RETURN_TYPES = ("IMAGE", "MASK","STRING")
    FUNCTION = "load_image"
    def load_image(self, image):
        if "clipspace" in image or "[input]" in image:
            # 这是从编辑/Mask 过来的临时文件
            is_edit = True
        else:
            # 这是用户直接在 UI 下拉框里选择的原始文件
            is_edit = False
        if not is_edit:
            self.original_filename = os.path.basename(image)
        logger.info(f"Loading image: {self.original_filename}")
                    
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        
        return (output_image, output_mask,self.original_filename)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

class ExtractFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "extract"
    CATEGORY = "Alta"

    def extract(self, image):
        filename = image.metadata.get("filename", "")
        return (filename,)
# 节点映射
NODE_CLASS_MAPPINGS = {
    "Alta:GetImageAndPath": GetImageAndPath,
    "Alta:GetStringByIndex": GetStringByIndex,
    "Alta:GetImageByIndex": GetImageByIndex,
    "Alta:LoadImage": LoadImage,
    "Alta:LoadImageFromPath": LoadImageFromPath,
    "Alta:LoadImagesPath": LoadImagesFromDirectoryPath,
    "Alta:LoadImageWithPath": LoadImageWithPath,
    "Alta:ReadImageFromFile": ReadImageFromFile,
}
