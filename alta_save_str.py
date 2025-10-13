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
    CATEGORY = "Utils"

    def get_name(self, file_path):
        # 取出文件名（不含路径和后缀）
        name = os.path.splitext(os.path.basename(file_path))[0]
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

BIGMAX = 0xFFFFFFFF

class ListFilesByExt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "placeholder": "X://path/to/folder"
                }),
            },
            "optional": {
                "extensions": ("STRING", {
                    "default": ".png,.jpg,.jpeg,.bmp,.tiff,.webp",
                    "tooltip": "Comma-separated file extensions"
                }),
                "recursive": ("BOOL", {"default": True}),
            },
        }

    RETURN_TYPES = ("LIST", "INT")
    RETURN_NAMES = ("filepaths", "file_count")
    FUNCTION = "list_files"
    CATEGORY = "Alta"

    def list_files(self, folder: str, extensions: str = "", recursive: bool = True):
        folder = folder.strip("\"' ")
        if not os.path.isdir(folder):
            raise Exception(f"Invalid folder path: {folder}")

        # 处理后缀
        ext_list = [e.strip().lower() for e in extensions.split(",") if e.strip()]
        if not ext_list:
            ext_list = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]

        # 搜索文件
        filepaths = []
        walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
        for root, _, files in walker:
            for f in files:
                if any(f.lower().endswith(ext) for ext in ext_list):
                    filepaths.append(os.path.join(root, f))

        filepaths.sort()
        return (filepaths, len(filepaths))
    

class ListAllFiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "placeholder": "X://path/to/folder"
                }),
            },
            "optional": {
                "recursive": ("BOOL", {"default": True}),
            },
        }

    RETURN_TYPES = ("LIST", "INT")
    RETURN_NAMES = ("filepaths", "file_count")
    FUNCTION = "list_all"
    CATEGORY = "Alta"

    def list_all(self, folder: str, recursive: bool = True):
        folder = folder.strip("\"' ")
        if not os.path.isdir(folder):
            raise Exception(f"Invalid folder path: {folder}")

        filepaths = []
        walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
        for root, _, files in walker:
            for f in files:
                filepaths.append(os.path.join(root, f))

        filepaths.sort()
        return (filepaths, len(filepaths))


NODE_CLASS_MAPPINGS = {
    "Alta:SaveStringToFile": WriteStringToFile,
    "Alta:GetFilenameNoExt": GetFilenameNoExt,
    "Alta:ReadStringFromFile": ReadStringFromFile,  # 新增节点
    "Alta:BuildFilePath": BuildFilePath,  # 新增节点
    "Alta:ListFilesByExt": ListFilesByExt,  # 新增节点
    "Alta:ListAllFiles": ListAllFiles  # 新增节点
}
