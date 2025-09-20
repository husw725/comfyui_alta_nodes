"""
Jimeng API Nodes
API Doc
https://www.volcengine.com/docs/85621/1544716
"""

from __future__ import annotations
from enum import Enum
import logging
import base64
import torch
import time
import json
from comfy_api_nodes.apinode_utils import (
    process_image_response
)
from comfy_api.latest import io as comfy_io
from comfy_api_nodes.apinode_utils import (
    tensor_to_base64_string,
    download_url_to_video_output,
    upload_video_to_comfyapi,
    upload_audio_to_comfyapi,
    download_url_to_image_tensor,
)
from comfy_api_nodes.mapper_utils import model_field_to_node_input
from comfy_api_nodes.util.validation_utils import (
    validate_image_dimensions,
    validate_image_aspect_ratio,
    validate_video_dimensions,
    validate_video_duration,
)
from comfy_api.input.basic_types import AudioInput
from comfy_api.input.video_types import VideoInput
from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from pydantic import BaseModel, Field

from volcengine.visual.VisualService import VisualService


JIMENG_API_VERSION = "v1"
# PATH_TEXT_TO_VIDEO = f"/proxy/jimeng/{JIMENG_API_VERSION}/videos/text2video"
# PATH_IMAGE_TO_VIDEO = f"/proxy/jimeng/{JIMENG_API_VERSION}/videos/image2video"
# PATH_VIDEO_EXTEND = f"/proxy/jimeng/{JIMENG_API_VERSION}/videos/video-extend"
# PATH_LIP_SYNC = f"/proxy/jimeng/{JIMENG_API_VERSION}/videos/lip-sync"
# PATH_VIDEO_EFFECTS = f"/proxy/jimeng/{JIMENG_API_VERSION}/videos/effects"
# PATH_CHARACTER_IMAGE = f"/proxy/jimeng/{JIMENG_API_VERSION}/images/generations"
# PATH_VIRTUAL_TRY_ON = f"/proxy/jimeng/{JIMENG_API_VERSION}/images/kolors-virtual-try-on"
# PATH_IMAGE_GENERATIONS = f"/proxy/jimeng/{JIMENG_API_VERSION}/images/generations"

# MAX_PROMPT_LENGTH_T2V = 2500
MAX_PROMPT_LENGTH_T2I = 800
MAX_PROMPT_LENGTH_I2V = 800
MAX_PROMPT_LENGTH_IMAGE_GEN = 500
MAX_NEGATIVE_PROMPT_LENGTH_IMAGE_GEN = 200
MAX_PROMPT_LENGTH_LIP_SYNC = 120

AVERAGE_DURATION_T2V = 319
AVERAGE_DURATION_I2V = 164
AVERAGE_DURATION_LIP_SYNC = 455
AVERAGE_DURATION_VIRTUAL_TRY_ON = 19
AVERAGE_DURATION_IMAGE_GEN = 32
AVERAGE_DURATION_VIDEO_EFFECTS = 320
AVERAGE_DURATION_VIDEO_EXTEND = 320


def validate_prompts(prompt: str, max_length: int) -> bool:
    """Verifies that the positive prompt is not empty and that neither promt is too long."""
    if not prompt:
        raise ValueError("Positive prompt is empty")
    if len(prompt) > max_length:
        raise ValueError(
            f"Positive prompt is too long: {len(prompt)} characters")
    return True


def validate_input_image(image: torch.Tensor) -> None:
    validate_image_dimensions(image, min_width=320,
                              min_height=320, max_width=4096, max_height=4096)
    validate_image_aspect_ratio(
        image, min_aspect_ratio=1 / 3, max_aspect_ratio=3)


class JimengApiError(Exception):
    """Base exception for Jimeng API errors."""

    pass


class JimengNodeBase(ComfyNodeABC):
    """Base class for Jimeng nodes."""

    FUNCTION = "api_call"
    API_NODE = True
    OUTPUT_NODE = True


class JimengVideoGenAspectRatio(str, Enum):
    field_16_9 = '16:9'
    field_4_3 = '4:3'
    field_1_1 = '1:1'
    field_3_4 = '3:4'
    field_9_16 = '9:16'
    field_21_9 = '21:9'


class JimengT2IReqType(str, Enum):
    t2i_v31 = 'jimeng_t2i_v31'


class JimengI2VReqType(str, Enum):
    i2v_720p_first_frame = 'jimeng_i2v_first_v30'
    i2v_720p_first_tail_frame = 'jimeng_i2v_first_tail_v30'
    # i2v_720p_recamera = 'jimeng_i2v_recamera_v30'
    i2v_1080p_first = 'jimeng_i2v_first_v30_1080'
    i2v_1080p_first_tail_frame = 'jimeng_i2v_first_tail_v30_1080'


class JimengT2VReqType(str, Enum):
    t2v_720p = 'jimeng_t2v_v30'
    t2v_1080p = 'jimeng_t2v_v30_1080p'


class JimengVideoGenDuration(str, Enum):
    field_5 = '5'
    field_10 = '10'

class JimengImage2VideoRequest(BaseModel):
    req_type: Optional[JimengI2VReqType] = JimengI2VReqType.i2v_720p_first_frame
    prompt: Optional[str] = Field(
        None, description='用于生成视频的提示词 ，中英文均可输入。建议在400字以内，不超过800字，prompt过长有概率出现效果异常或不生效', max_length=800
    )
    start_frame: Optional[str] = Field(
        None,
        description='图片文件base64编码，仅支持JPEG、PNG格式；',
    )
    tail_frame: Optional[str] = Field(
        None,
        description='图片文件base64编码，仅支持JPEG、PNG格式, 仅支持i2v_720p_first_tail_frame, i2v_1080p_first_tail_frame；',
    )
    seed: Optional[int] = Field(
        -1,
        description='随机种子,作为确定扩散初始状态的基础,默认-1(随机)。若随机种子为相同正整数且其他参数均一致,则生成视频极大概率效果一致',
    )
    duration: Optional[int] = Field(
        5,
        description='生成视频的时长，单位为秒，默认5秒，支持5s、10s',
    )
    frames: Optional[int] = Field(
        121,
        description='生成的总帧数（帧数 = 24 * n + 1，其中n为秒数，支持5s、10s）可选取值：[121, 241]默认值：121',
    )
    # aspect_ratio: Optional[JimengVideoGenAspectRatio] = JimengVideoGenAspectRatio.field_16_9

    def __init__(self, *args, **kwargs):
        self.visual_service = VisualService()
        return super().__init__(self, *args, **kwargs)


class JimengText2ImageRequest(BaseModel):
    """https://www.volcengine.com/docs/85621/1756900"""

    req_type: Optional[JimengI2VReqType] = JimengT2IReqType.t2i_v31
    prompt: Optional[str] = Field(
        None, description='用于生成图像的提示词 ，中英文均可输入。建议长度<=120字符，最长不超过800字符，prompt过长有概率出图异常或不生效', max_length=800
    )
    seed: Optional[int] = Field(
        -1,
        description='随机种子，作为确定扩散初始状态的基础，默认-1（随机）。若随机种子为相同正整数且其他参数均一致，则生成图片极大概率效果一致 默认值：-1',
    )
    width: Optional[int] = Field(
        512,
        description='1、生成图像宽高，系统默认生成1328 * 1328的图像；2、支持自定义生成图像宽高，宽高比在1:3到3:1之间，长度在[512, 2048]之间；',
    )
    height: Optional[int] = Field(
        121,
        description='1、生成图像宽高，系统默认生成1328 * 1328的图像；2、支持自定义生成图像宽高，宽高比在1:3到3:1之间，长度在[512, 2048]之间；',
    )
    use_pre_llm: Optional[bool] = Field(True, description="开启文本扩写，会针对输入prompt进行扩写优化，如果输入prompt较短建议开启，如果输入prompt较长建议关闭默认值：true")

    def __init__(self, *args, **kwargs):
        self.visual_service = VisualService()
        return super().__init__(self, *args, **kwargs)


# class JimengResponse(BaseModel):
#     code: Optional[int] = Field(None, description='Error code')
#     data: Optional[Data] = None
#     message: Optional[str] = Field(None, description='Error message')
#     request_id: Optional[str] = Field(None, description='Request ID')
#     status: Optional[int] = Field(None, description='Error code')
#     time_elapsed: Optional[str] = Field(None, description='Error code')


class JimengText2ImageNode(JimengNodeBase):
    """Jimeng Text to Image Node"""

    CATEGORY = "api node/image/Jimeng"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = "Jimeng Text to Image Node"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": (
                    IO.STRING, {
                        "tooltip": "Jimeng API Key"
                    }
                ),
                "api_secret": (
                    IO.STRING, {
                        "tooltip": "Jimeng API Secret"
                    }
                ),
                "req_type": model_field_to_node_input(
                    IO.COMBO,
                    JimengText2ImageRequest,
                    "req_type",
                    enum_type=JimengT2IReqType,
                ),
                "prompt": model_field_to_node_input(
                    IO.STRING, JimengText2ImageRequest, "prompt", multiline=True
                ),
            },
            "optional": {
                "seed": model_field_to_node_input(
                    IO.INT,
                    JimengText2ImageRequest,
                    "seed",
                    min=-1,
                    max=0xFFFFFFF,
                    default=-1
                ),
                "width": model_field_to_node_input(
                    IO.INT,
                    JimengText2ImageRequest,
                    "width",
                    min=256,
                    max=768,
                    default=512
                ),
                "height": model_field_to_node_input(
                    IO.INT,
                    JimengText2ImageRequest,
                    "height",
                    min=256,
                    max=768,
                    default=512
                ),
                "use_pre_llm": (
                    IO.BOOLEAN,
                    {
                        "default": True,
                        "tooltip": "开启文本扩写，会针对输入prompt进行扩写优化，如果输入prompt较短建议开启，如果输入prompt较长建议关闭默认值：true",
                    },
                ),

                # "use_sr": model_field_to_node_input(
                #     IO.BOOLEAN,
                #     JimengImage2VideoRequest,
                #     default=False
                # ),
                # "return_url": model_field_to_node_input(
                #     IO.BOOLEAN,
                #     JimengImage2VideoRequest,
                #     default=True
                # )
            },
        }



    async def api_call(
        self,
        api_key: str,
        api_secret: str,
        req_type: JimengT2IReqType,
        prompt: str,
        seed: int = -1,
        width: int = 512,
        height: int = 512,
        use_pre_llm: bool = True,
        **kwargs,
    ) -> str:
        validate_prompts(prompt, MAX_PROMPT_LENGTH_T2I)

        visual_service = VisualService()
        visual_service.set_ak(api_key)
        visual_service.set_sk(api_secret)

        form = {
            "req_key": req_type,
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "use_pre_llm": use_pre_llm
        }
        response = visual_service.cv_process(form)

        if response is None:
            error_msg = f"创建即梦图片生成任务失败."
            logging.error(error_msg)
            raise JimengApiError(error_msg)
        elif response['code'] != 10000:
            error_msg = f"创建即梦图片生成任务失败. Code: {response.get('code')}, Message: {response.get('message')}, Data: {response.get('data')}"
            logging.error(error_msg)
            raise JimengApiError(error_msg)

        data = response.get('data')
        image_data = data.get('binary_data_base64')[0]

        return comfy_io.NodeOutput(process_image_response(base64.b64decode(image_data)))


class JimengImage2VideoNode(JimengNodeBase):
    """Jimeng Image to Video Node"""

    CATEGORY = "api node/video/Jimeng"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": (
                    IO.STRING, {
                        "tooltip": "Jimeng API Key"
                    }
                ),
                "api_secret": (
                    IO.STRING, {
                        "tooltip": "Jimeng API Secret"
                    }
                ),
                "req_type": model_field_to_node_input(
                    IO.COMBO,
                    JimengImage2VideoRequest,
                    "req_type",
                    enum_type=JimengI2VReqType,
                ),
                "start_frame": model_field_to_node_input(
                    IO.IMAGE,
                    JimengImage2VideoRequest,
                    "start_frame",
                )
            },
            "optional": {
                "prompt": model_field_to_node_input(
                    IO.STRING, JimengImage2VideoRequest, "prompt", multiline=True
                ),
                "tail_frame": model_field_to_node_input(
                    IO.IMAGE,
                    JimengImage2VideoRequest,
                    "tail_frame",
                ),
                "seed": model_field_to_node_input(
                    IO.INT,
                    JimengImage2VideoRequest,
                    "seed",
                    min=-1,
                    max=0xFFFFFFF,
                    default=-1
                ),
                "duration": model_field_to_node_input(
                    IO.COMBO,
                    JimengImage2VideoRequest,
                    "duration",
                    enum_type=JimengVideoGenDuration,
                    default=JimengVideoGenDuration.field_5
                ),
                # "aspect_ratio": model_field_to_node_input(
                #     IO.COMBO,
                #     JimengImage2VideoRequest,
                #     "aspect_ratio",
                #     enum_type=JimengVideoGenAspectRatio,
                # )
            },
        }

    RETURN_TYPES = ("VIDEO", )
    RETURN_NAMES = ("VIDEO", )
    DESCRIPTION = "Jimeng Image to Video Node"

    async def api_call(
        self,
        api_key: str,
        api_secret: str,
        req_type: JimengI2VReqType,
        start_frame: torch.Tensor,
        tail_frame: torch.Tensor = None,
        prompt: str = '',
        seed: int = -1,
        duration: str = JimengVideoGenDuration.field_5,
        # aspect_ratio: str = JimengVideoGenAspectRatio.field_16_9,
        **kwargs,
    ) -> str:
        validate_prompts(prompt, MAX_PROMPT_LENGTH_I2V)
        validate_input_image(start_frame)
        if tail_frame is not None:
            validate_input_image(tail_frame)

        images = [tensor_to_base64_string(start_frame)]
        if tail_frame is not None:
            images.append(tensor_to_base64_string(tail_frame))
        form = {
            "req_key": req_type,
            "prompt": prompt,
            "binary_data_base64": images,
            "seed": seed,
            "frames": int(duration)*24+1,
            # "aspect_ratio": aspect_ratio
        }
        visual_service = VisualService()
        visual_service.set_ak(api_key)
        visual_service.set_sk(api_secret)
        response = visual_service.cv_process(form)

        with open("jimeng_responses.txt", "a+") as f:
            f.seek(0, 2)  # 移动到文件末尾
            f.write(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime()))
            f.write(f"{json.dumps(response, indent=4)}\n\n")  # 格式化 JSON 并附加内容

        if response is None:
            logging.error("创建即梦视频生成任务失败. ")
            raise JimengApiError(error_msg)
        elif response['code'] != 10000:
            error_msg = f"创建即梦视频生成任务失败. Code: {response.get('code')}, Message: {response.get('message')}, Data: {response.get('data')}"
            logging.error(error_msg)
            raise JimengApiError(error_msg)
        
        video_urls = response.get('data').get('urls')

        logging.info(f"视频已生成，URL： {video_urls}")

        if len(video_urls) > 0:
            return await download_url_to_video_output(video_urls[0])
        else:
            return None


NODE_CLASS_MAPPINGS = {
    "JimengText2Image": JimengText2ImageNode,
    "JiMengImage2Video": JimengImage2VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JimengText2Image": "即梦 文生图",
    "JiMengImage2Video": "即梦 图生视频"
}
