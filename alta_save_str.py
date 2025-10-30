import os


class GetFilenameNoExt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "/path/to/video.mp4",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_name"
    CATEGORY = "alta/Utils"

    def get_name(self, file_path):
        # 取出文件名（不含路径和后缀）
        name = os.path.splitext(os.path.basename(file_path))[0]
        return (name,)
    
class GetFileFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "/path/to/video.mp4",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_name"
    CATEGORY = "alta/Utils"

    def get_name(self, file_path):
        # 取出文件夹路径
        name = os.path.dirname(file_path)
        return (name,)
    
class GetFilenameWithExt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "/path/to/video.mp4",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_name"
    CATEGORY = "alta/Utils"

    def get_name(self, file_path):
        # 取出文件名（不含路径和后缀）
        name = os.path.basename(file_path)
        return (name,)


class WriteStringToFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 主内容，多行输入框
                "content": ("STRING", {
                    "multiline": True,
                    "default": "Hello ComfyUI!"
                }),
                # 文件名，可输入或连线
                "filename": ("STRING", {
                    "default": "output_file"
                }),
                # 后缀，默认 txt
                "extension": ("STRING", {
                    "default": "txt"
                }),
                # 输出目录
                "output_dir": ("STRING", {
                    "default": "outputs"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "write_file"
    CATEGORY = "Utils"

    def write_file(self, content, filename, extension, output_dir):
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 文件路径
        file_path = os.path.join(output_dir, f"{filename}.{extension}")

        # 写入文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return (file_path,)


class ReadStringFromFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 输入文件名（不带扩展名）
                "filename": ("STRING", {
                    "default": "output_file"
                }),
                # 文件所在目录
                "folder": ("STRING", {
                    "default": "outputs"
                }),
                # 扩展名
                "extension": ("STRING", {
                    "default": "txt"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "read_file"
    CATEGORY = "Alta"

    def read_file(self, filename, folder, extension):
        # 拼接路径
        file_path = os.path.join(folder, f"{filename}.{extension}")

        # 检查文件是否存在
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return (content,)
    
import os

class BuildFilePath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 输入文件名（不带扩展名）
                "filename": ("STRING", {
                    "default": "filename"
                }),
                # 文件所在目录
                "folder": ("STRING", {
                    "default": "folder"
                }),
                # 新扩展名
                "extension": ("STRING", {
                    "default": "txt"
                }),
            },
            "optional": {
                
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "build_path"
    CATEGORY = "Utils"

    def build_path(self, filename, folder, extension):
        # 去掉多余空格与引号
        folder = folder.strip("\"' ")
        filename = filename.strip("\"' ")
        extension = extension.strip(". \"' ")

        # 拼接路径
        file_path = os.path.join(folder, f"{filename}.{extension}")

        return (file_path,)

    @classmethod
    def VALIDATE_INPUTS(cls, filename, folder, extension):
        # 只检查 folder 是否存在
        return os.path.isdir(folder)
    
    import os

import os
import hashlib

def simple_hash(value: str):
    """简单字符串哈希，用于 IS_CHANGED"""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()

def list_files_in_folder(folder, exts=None, recursive=True):
    filepaths = []
    walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
    for root, _, files in walker:
        for f in files:
            if exts:
                if any(f.lower().endswith(ext) for ext in exts):
                    filepaths.append(os.path.join(root, f))
            else:
                filepaths.append(os.path.join(root, f))
    filepaths.sort()
    return filepaths


# ======================================================
# ✅ 1. ListFilesByExt —— 带后缀筛选
# ======================================================
class ListFilesByExt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"placeholder": "X://path/to/folder"}),
            },
            "optional": {
                "extensions": ("STRING", {
                    "default": ".png,.jpg,.jpeg,.bmp,.tiff,.webp",
                    "tooltip": "Comma-separated file extensions"
                }),
                "recursive": ("BOOL", {"default": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "Alta"
    RETURN_TYPES = ("LIST", "INT")
    RETURN_NAMES = ("filepaths", "file_count")
    FUNCTION = "list_files"

    def list_files(self, folder: str, extensions: str = "", recursive: bool = True, **kwargs):
        folder = folder.strip("\"' ")
        if not os.path.isdir(folder):
            raise Exception(f"Invalid folder path: {folder}")

        exts = [e.strip().lower() for e in extensions.split(",") if e.strip()]
        if not exts:
            exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]

        filepaths = list_files_in_folder(folder, exts, recursive)
        return filepaths, len(filepaths)

    @classmethod
    def IS_CHANGED(cls, folder, extensions="", recursive=True, **kwargs):
        return simple_hash(f"{folder}-{extensions}-{recursive}")

    @classmethod
    def VALIDATE_INPUTS(cls, folder, **kwargs):
        return os.path.isdir(folder)


# ======================================================
# ✅ 2. ListAllFiles —— 不区分后缀
# ======================================================
class ListAllFiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"placeholder": "X://path/to/folder"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "Alta"
    RETURN_TYPES = ("LIST", "INT")
    RETURN_NAMES = ("filepaths", "file_count")
    FUNCTION = "list_all"

    def list_all(self, folder: str, **kwargs):
        folder = folder.strip("\"' ")
        if not os.path.isdir(folder):
            raise Exception(f"Invalid folder path: {folder}")

        filepaths = list_files_in_folder(folder, None, recursive)
        return filepaths, len(filepaths)

    @classmethod
    def IS_CHANGED(cls, folder, **kwargs):
        return simple_hash(f"{folder}")

    @classmethod
    def VALIDATE_INPUTS(cls, folder, **kwargs):
        return os.path.isdir(folder)


NODE_CLASS_MAPPINGS = {
    "Alta:SaveStringToFile": WriteStringToFile,
    "Alta:GetFilenameNoExt": GetFilenameNoExt,
    "Alta:GetFileFolder": GetFileFolder,
    "Alta:GetFilenameWithExt": GetFilenameWithExt,
    "Alta:ReadStringFromFile": ReadStringFromFile,  # 新增节点
    "Alta:BuildFilePath": BuildFilePath,  # 新增节点
    "Alta:ListFilesByExt": ListFilesByExt,  # 新增节点
    "Alta:ListAllFiles": ListAllFiles  # 新增节点
}
