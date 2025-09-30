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


NODE_CLASS_MAPPINGS = {
    "Alta:MergeNodes": DynamicTupleNode,
    "Alta:MultiRoute": MultiRouteNode,
}

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Alta:MergeNodes": "Dynamic Tuple Builder"
# }