from typing import Any, Tuple
import os
import time



class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

any_type = AlwaysEqualProxy("*")

class DynamicTupleNode:
    """
    动态输入组合节点
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 最多预留 5 个输入口
        return {
            "required": {
                "input_1": ("ANY", ),
            },
            "optional": {
                "input_2": ("ANY", ),
                "input_3": ("ANY", ),
                "input_4": ("ANY", ),
                "input_5": ("ANY", ),
            },
            "hidden": {
                # 用于 UI 控制
                "visible_inputs": (["1", "2", "3", "4", "5"], {"default": "1"})
            }
        }

    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "make_tuple"
    CATEGORY = "alta"

    def make_tuple(self, **kwargs) -> Tuple[Tuple[Any, ...]]:
        # 收集所有已连接的输入
        values = []
        for i in range(1, 6):
            key = f"input_{i}"
            if key in kwargs and kwargs[key] is not None:
                values.append(kwargs[key])
        return (tuple(values),)
    
class MultiRouteNode:
    """
    固定 5 输入（1 required + 4 optional），
    根据实际连线数量动态生成输出数量。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "in1": (any_type,),
            },
            "optional": {
                "in2": (any_type,),
                "in3": (any_type,),
                "in4": (any_type,),
                "in5": (any_type,),
            }
        }

    RETURN_TYPES = (any_type,any_type,any_type,any_type,any_type)
    RETURN_NAMES = ("out1","out2","out3","out4","out5")

    FUNCTION = "route"
    CATEGORY = "alta/utils"
    DESCRIPTION = "固定 5 输入（1必填4选填），根据连线自动生成输出数量"

    def route(self, in1, in2=None, in3=None, in4=None, in5=None):
        return (in1, in2, in3, in4, in5)
    


class ListLengthNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"list_input": ("LIST",)}}
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    FUNCTION = "get_length"
    CATEGORY = "alta/Utils/List"

    def get_length(self, list_input):
        return (len(list_input),)

class ListElementNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"list_input": ("LIST",), "index": ("INT", {"default": 0})}}
    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("element",)
    FUNCTION = "get_element"
    CATEGORY = "alta/Utils/List"

    def get_element(self, list_input, index):
        if not list_input:
            return (None,)
        index = max(0, min(index, len(list_input)-1))
        return (list_input[index],)


import json
from typing import Any, List

class JSONKeyExtractor:
    """
    Extract up to 5 keys from a JSON string.
    Each key becomes a separate output port.
    If key is missing, output None.
    Example:
        json_str = {"start":85.15,"end":85.17,"speaker":"speaker_SPEAKER_03"}
        keys = ["start","end","speaker"]
        outputs:
            out1 -> 85.15
            out2 -> 85.17
            out3 -> "speaker_SPEAKER_03"
            out4 -> None
            out5 -> None
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_str": ("STRING", {"multiline": False, "default": '{"start":85.15,"end":85.17,"speaker":"speaker_SPEAKER_03"}'}),
                "keys": ("STRING", {"multiline": False, "default": "start,end,speaker"}),
            }
        }

    RETURN_TYPES = ("ANY", "ANY", "ANY", "ANY", "ANY")
    RETURN_NAMES = ("out1", "out2", "out3", "out4", "out5")
    FUNCTION = "extract_values"
    CATEGORY = "alta/Utils/JSON Tools"

    def extract_values(self, json_str: str, keys: str):
        try:
            data = json.loads(json_str)

            # Parse keys input
            if keys.strip().startswith("["):
                keys_list = json.loads(keys)
            else:
                keys_list = [k.strip() for k in keys.split(",") if k.strip()]

            # Extract values
            values: List[Any] = [data.get(k, None) for k in keys_list]

            # Pad up to 5 outputs with None
            while len(values) < 5:
                values.append(None)

            # Trim if more than 5
            if len(values) > 5:
                values = values[:5]

            return tuple(values)

        except Exception as e:
            return (f"Error: {e}", None, None, None, None)

from typing import Any, List, Tuple

class Int2Str:
    """
    ComfyUI node to convert integer(s) to string(s)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT",),       # single int input
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str_value",)
    FUNCTION = "convert"
    CATEGORY = "Alta/Utils"
    DESCRIPTION = "Convert integer to string."

    def convert(self, value: int) -> Tuple[str]:
        try:
            return (str(value),)
        except Exception as e:
            return (f"Error: {e}",)


class StrToNum:
    """
    ComfyUI node to convert a string to int and float.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING",),  # input string
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("float_value", "int_value")
    FUNCTION = "convert"
    CATEGORY = "Alta/Utils"
    DESCRIPTION = "Convert string to float and int."

    def convert(self, value: str) -> Tuple[float, int]:
        try:
            f = float(value)
            i = int(f)
            return (f, i)
        except Exception as e:
            # Return 0 as default if conversion fails
            print(f"StrToNum conversion error: {e}")
            return (0.0, 0)
        

class AddNode:
    """
    ComfyUI node to add two numbers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT",),
                "b": ("FLOAT",),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("sum",)
    FUNCTION = "compute"
    CATEGORY = "Alta/Math"
    DESCRIPTION = "Add two numbers."

    def compute(self, a: float, b: float) -> Tuple[float]:
        return (a + b,)

class AddIntNode:
    """
    ComfyUI node to add two numbers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT",),
                "b": ("INT",),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("sum",)
    FUNCTION = "compute"
    CATEGORY = "Alta/Math"
    DESCRIPTION = "Add two numbers."

    def compute(self, a: int, b: int) -> Tuple[int]:
        return (a + b,)
# -------------------------
# Subtraction Node
# -------------------------
class SubIntNode:
    """
    ComfyUI node to subtract two numbers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT",),
                "b": ("INT",),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("difference",)
    FUNCTION = "compute"
    CATEGORY = "Alta/Math"
    DESCRIPTION = "Subtract b from a."

    def compute(self, a: int, b: int) -> Tuple[int]:
        return (a - b,)

class SubNode:
    """
    ComfyUI node to subtract two numbers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("Float",),
                "b": ("Float",),
            }
        }

    RETURN_TYPES = ("Float",)
    RETURN_NAMES = ("difference",)
    FUNCTION = "compute"
    CATEGORY = "Alta/Math"
    DESCRIPTION = "Subtract b from a."

    def compute(self, a: float, b: float) -> Tuple[float]:
        return (a - b,)

import shutil

class MoveFileNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src_path": ("STRING", {"multiline": False, "default": ""}),
                "dst_path": ("STRING", {"multiline": False, "default": ""}),
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    # 返回目标文件路径
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("moved_path",)
    FUNCTION = "move_file"
    CATEGORY = "Alta/File"
    DESCRIPTION = "Move a file from src_path to dst_path and return the destination path."

    def move_file(self, src_path, dst_path, overwrite):
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source file not found: {src_path}")

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if os.path.exists(dst_path):
            if overwrite:
                os.remove(dst_path)
            else:
                raise FileExistsError(f"File already exists: {dst_path}")

        shutil.move(src_path, dst_path)
        print(f"Moved {src_path} -> {dst_path}")
        return (dst_path,)
    


class DeleteFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("deleted_path",)
    FUNCTION = "delete_file"
    CATEGORY = "Utils"
    DESCRIPTION = "Delete file at given path"

    def delete_file(self, path: str) -> Tuple[str]:
        try:
            if os.path.exists(path):
                os.remove(path)
            return (path,)
        except Exception as e:
            return (f"Error deleting file: {e}",)

import re

class RegexMatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "pattern": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("STRING", "LIST")
    RETURN_NAMES = ("first_match", "all_matches")
    FUNCTION = "run"
    CATEGORY = "utils"

    def run(self, text, pattern):
        try:
            matches = re.findall(pattern, text)
        except Exception as e:
            return (f"Regex error: {e}", [])

        if matches:
            # flatten if group matches exists
            if isinstance(matches[0], tuple):
                matches = ["".join(m) for m in matches]

            return (matches[0], matches)

        return ("", [])


class CompareFoldersNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_a": ("STRING", {"multiline": False, "default": ""}),
                "folder_b": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "INT", "INT")
    RETURN_NAMES = ("files_not_in_b", "files_in_both", "len_files_not_in_b", "len_files_in_both")
    FUNCTION = "compare_folders"
    CATEGORY = "Alta/File"
    DESCRIPTION = "Compare two folders and return files in folder_a that are not in folder_b, and files that are in both."

    def IS_CHANGED(self, *args, **kwargs):
        return float("inf")

    def compare_folders(self, folder_a, folder_b):
        if not os.path.isdir(folder_a):
            raise FileNotFoundError(f"Folder A not found: {folder_a}")

        files_b_names = set()
        if os.path.isdir(folder_b):
            files_b_names = {os.path.splitext(f)[0] for f in os.listdir(folder_b) if os.path.isfile(os.path.join(folder_b, f))}

        files_a = [f for f in os.listdir(folder_a) if os.path.isfile(os.path.join(folder_a, f))]
        
        files_not_in_b = []
        files_in_both = []

        for file_a in files_a:
            file_a_name = os.path.splitext(file_a)[0]
            if file_a_name in files_b_names:
                files_in_both.append(os.path.join(folder_a, file_a))
            else:
                files_not_in_b.append(os.path.join(folder_a, file_a))
        
        return (files_not_in_b, files_in_both, len(files_not_in_b), len(files_in_both))
    



class IfOnlyNode:
    """
    If Only Node: Outputs value to true_output if condition is true, else to false_output.
    The output type will dynamically match the input type.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
                "value": (any_type,),  # Wildcard input, accepts any type
            }
        }

    RETURN_TYPES = (any_type, any_type, ) # Outputs will have the same type as the input
    RETURN_NAMES = ("true_output", "false_output")
    FUNCTION = "execute"
    CATEGORY = "Alta/Logic"
    DESCRIPTION = "Outputs value to true_output if condition is true, else to false_output."

    def execute(self, condition: bool, value):
        if condition:
            return (value, None)
        else:
            return (None, value)


class StringPlusNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"multiline": True, "default": ""}),
                "any": (any_type,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = "Alta/Utils"
    DESCRIPTION = "Return a string, but with an additional input to control execution flow."

    def execute(self, string: str, any):
        return (string,)

NODE_CLASS_MAPPINGS = {
    "Alta:MergeNodes": DynamicTupleNode,
    "Alta:MultiRoute": MultiRouteNode,
    "Alta:ListLength(Util)": ListLengthNode,
    "Alta:ListElement(Util)": ListElementNode,
    "Alta:JSONKeyExtractor(Util)": JSONKeyExtractor,
    "Alta:DeleteFile(Util)": DeleteFile,
    "Alta:RegexMatch(Util)": RegexMatchNode,
    "Alta:MoveFile(File)": MoveFileNode,
    "Alta:CompareFolders(File)": CompareFoldersNode,
    "Alta:Int2Str(Math)": Int2Str,
    "Alta:StrToNum(Math)": StrToNum,
    "Alta:Add(Math)": AddNode,
    "Alta:AddInt(Math)": AddIntNode,
    "Alta:Sub(Math)": SubNode,
    "Alta:SubInt(Math)": SubIntNode,
    "Alta:IfOnly(Logic)": IfOnlyNode,
    "Alta:StringPlus(Util)": StringPlusNode,
}

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Alta:MergeNodes": "Dynamic Tuple Builder"
# }