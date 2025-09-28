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
    CATEGORY = "Utils"

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


NODE_CLASS_MAPPINGS = {
    "Alta:SaveStringToFile": WriteStringToFile,
    "Alta:GetFilenameNoExt": GetFilenameNoExt,
    "Alta:ReadStringFromFile": ReadStringFromFile,  # 新增节点
}
