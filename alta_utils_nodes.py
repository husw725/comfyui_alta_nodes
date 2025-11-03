from typing import Any, Tuple
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
    多路路由节点：第1个输入必填，其余输入可选。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {         # 必须有的输入
                "in1": ("ANY", ),
            },
            "optional": {         # 可选输入
                "in2": ("ANY", ),
                "in3": ("ANY", ),
                "in4": ("ANY", ),
            }
        }

    RETURN_TYPES = ("ANY", "ANY", "ANY", "ANY")
    RETURN_NAMES = ("out1", "out2", "out3", "out4")
    FUNCTION = "route"
    CATEGORY = "Utils/Routing"

    def route(self, in1=None, in2=None, in3=None, in4=None):
        return (in1, in2, in3, in4)


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


NODE_CLASS_MAPPINGS = {
    "Alta:MergeNodes": DynamicTupleNode,
    "Alta:MultiRoute": MultiRouteNode,
    "Alta:ListLength(Util)": ListLengthNode,
    "Alta:ListElement(Util)": ListElementNode,
    "Alta:JSONKeyExtractor(Util)": JSONKeyExtractor,
    "Alta:Int2Str(Util)": Int2Str,
}

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Alta:MergeNodes": "Dynamic Tuple Builder"
# }