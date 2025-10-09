"""
Alta API Nodes
API Doc
https://www.volcengine.com/docs/85621/1544716
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
import base64
import torch
import time
import json
from comfy_api.latest import io as comfy_io
from comfy_api_nodes.apinode_utils import (
    tensor_to_base64_string,
    download_url_to_video_output,
)
from comfy_api_nodes.mapper_utils import model_field_to_node_input
from comfy_api_nodes.util.validation_utils import (
    validate_image_dimensions,
    validate_image_aspect_ratio
)
from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from pydantic import BaseModel, Field


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


def write_jiment_log(log):
    logging.info(log)
    with open("Alta_log.txt", "a+") as f:
        f.seek(0, 2)
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime()))
        f.write(log)
        f.write("\n")


class AltaApiError(Exception):
    """Base exception for Alta API errors."""
    pass


# class AltaVideoGenDuration(str, Enum):
#     field_5 = '5'
#     field_10 = '10'


class AltaImage2VideoRequest(BaseModel):
    prompt: Optional[str] = Field(
        None, description='用于生成视频的提示词 ，中英文均可输入。建议在400字以内，不超过800字，prompt过长有概率出现效果异常或不生效', max_length=800
    )
    start_frame: Optional[str] = Field(
        None,
        description='图片文件base64编码，仅支持JPEG、PNG格式；',
    )
    # tail_frame: Optional[str] = Field(
    #     None,
    #     description='图片文件base64编码，仅支持JPEG、PNG格式, 仅支持i2v_720p_first_tail_frame, i2v_1080p_first_tail_frame；',
    # )
    # seed: Optional[int] = Field(
    #     -1,
    #     description='随机种子,作为确定扩散初始状态的基础,默认-1(随机)。若随机种子为相同正整数且其他参数均一致,则生成视频极大概率效果一致',
    # )
    # duration: Optional[int] = Field(
    #     5,
    #     description='生成视频的时长，单位为秒，默认5秒，支持5s、10s',
    # )
    # frames: Optional[int] = Field(
    #     121,
    #     description='生成的总帧数（帧数 = 24 * n + 1，其中n为秒数，支持5s、10s）可选取值：[121, 241]默认值：121',
    # )


class AltaImage2VideoNode(ComfyNodeABC):
    """Alta Image to Video Node"""
    FUNCTION = "api_call"
    # API_NODE = True
    OUTPUT_NODE = True
    CATEGORY = "Alta"
    RETURN_TYPES = ("VIDEO", )
    RETURN_NAMES = ("VIDEO", )
    DESCRIPTION = "Alta Image to Video Node"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sid_tt": (
                    IO.STRING, {
                        "tooltip": "Jimemg cookie data"
                    }
                ),
                "start_frame": model_field_to_node_input(
                    IO.IMAGE,
                    AltaImage2VideoRequest,
                    "start_frame",
                )
            },
            "optional": {
                "prompt": model_field_to_node_input(
                    IO.STRING, AltaImage2VideoRequest, "prompt", multiline=True
                ),
                # "tail_frame": model_field_to_node_input(
                #     IO.IMAGE,
                #     AltaImage2VideoRequest,
                #     "tail_frame",
                # ),
                # "seed": model_field_to_node_input(
                #     IO.INT,
                #     AltaImage2VideoRequest,
                #     "seed",
                #     min=-1,
                #     max=0xFFFFFFF,
                #     default=-1
                # ),
                # "duration": model_field_to_node_input(
                #     IO.COMBO,
                #     AltaImage2VideoRequest,
                #     "duration",
                #     enum_type=AltaVideoGenDuration,
                #     default=AltaVideoGenDuration.field_5
                # ),
                # "aspect_ratio": model_field_to_node_input(
                #     IO.COMBO,
                #     AltaImage2VideoRequest,
                #     "aspect_ratio",
                #     enum_type=AltaVideoGenAspectRatio,
                # )
            },
        }

    async def api_call(
        self,
        sid_tt: str,
        start_frame: torch.Tensor,
        tail_frame: torch.Tensor = None,
        prompt: str = '',
        # seed: int = -1,
        # duration: str = AltaVideoGenDuration.field_5,
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
            "prompt": prompt,
            "binary_data_base64": images,
            # "seed": seed,
            # "frames": int(duration)*24+1,
        }
        # visual_service = VisualService()
        # visual_service.set_ak(api_key)
        # visual_service.set_sk(api_secret)
        # 同步调用
        # video_url = self.sync_api_call(visual_service, form)
        # 异步调用
        # video_url = await self.async_api_call(visual_service, form)
        video_url = ""

        return comfy_io.NodeOutput(await download_url_to_video_output(video_url))

    def sync_api_call(self, visual_service, form) -> str:
        response = visual_service.cv_process(form)

        if response is None:
            logging.error("创建即梦视频生成任务失败. ")
            raise AltaApiError(error_msg)
        elif response['code'] != 10000:
            error_msg = f"创建即梦视频生成任务失败. response:\n{json.dumps(response, indent=4)}"
            logging.error(error_msg)
            raise AltaApiError(error_msg)

        write_jiment_log('视频生成任务成功 response:\n' +
                         json.dumps(response, indent=4))

        video_urls = response.get('data').get('urls')

        if len(video_urls) > 0:
            logging.info(f"视频已生成，URL： {video_urls}")
            return video_urls[0]
        else:
            logging.error("视频生成失败")
            raise AltaApiError("视频生成失败")

    async def async_api_call(self, visual_service, form) -> str:
        response = visual_service.cv_sync2async_submit_task(form)

        if response is not None and response['code'] == 10000:
            write_jiment_log(
                f"视频生成异步任务创建成功 task_id: {response.get('data').get('task_id')}")

            task_id = response.get('data').get('task_id')
            query_form = {'req_key': form['req_key'], 'task_id': task_id}
            max_retry_count = 1000
            while max_retry_count > 0:
                response = visual_service.cv_sync2async_get_result(query_form)
                if response is not None:
                    if response['code'] == 10000:
                        status = response['data']['status']
                        if status == 'done':
                            write_jiment_log(
                                f"视频生成结果查询完成\n{json.dumps(response, indent=4)}")
                            return response.get('data').get('video_url')
                        elif status == 'in_queue':
                            max_retry_count -= 1
                            time.sleep(0.5)
                        else:
                            write_jiment_log(f"视频生成结果查询异常 status: {status}")
                            logging.error(f"视频生成结果查询异常 status: {status}")
                            raise AltaApiError("视频生成失败")
                    else:
                        write_jiment_log(
                            f"视频生成结果查询异常 response:\n{json.dumps(response, indent=4)}")
        else:
            write_jiment_log(
                f"视频生成异步任务创建失败 response:\n{json.dumps(response, indent=4)}")
            logging.error("提交任务失败")
            raise AltaApiError(f"提交任务失败.")


# class AltaI2VTaskQueryNode(AltaNodeBase):
#     """Alta Image to Video Task Query Node"""

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "api_key": (
#                     IO.STRING, {
#                         "tooltip": "Alta API Key"
#                     }
#                 ),
#                 "api_secret": (
#                     IO.STRING, {
#                         "tooltip": "Alta API Secret"
#                     }
#                 ),
#                 "task_id": (
#                     IO.STRING, {
#                         "tooltip": "Alta Task ID"
#                     }
#                 ),
#                 "req_type": model_field_to_node_input(
#                     IO.COMBO,
#                     AltaImage2VideoRequest,
#                     "req_type",
#                     enum_type=AltaI2VReqType,
#                 )
#             },
#         }

#     RETURN_TYPES = ("VIDEO", )
#     RETURN_NAMES = ("VIDEO", )
#     DESCRIPTION = "Alta Image to Video Task Query Node"

#     async def api_call(
#         self,
#         api_key: str,
#         api_secret: str,
#         task_id: str,
#         req_type: AltaI2VReqType,
#         **kwargs,
#     ) -> str:
#         visual_service = VisualService()
#         visual_service.set_ak(api_key)
#         visual_service.set_sk(api_secret)
#         form = {
#             "req_key": I2VReqMap.get(req_type),
#             "task_id": task_id
#         }
#         video_url = await self.async_api_call(visual_service, form)

#         return comfy_io.NodeOutput(await download_url_to_video_output(video_url))

#     async def async_api_call(self, visual_service, form) -> str:
#         max_retry_count = 1000
#         while max_retry_count > 0:
#             response = visual_service.cv_sync2async_get_result(form)
#             if response is not None:
#                 if response['code'] == 10000:
#                     status = response['data']['status']
#                     if status == 'done':
#                         write_jiment_log(
#                             f"视频生成结果查询完成\n{json.dumps(response, indent=4)}")
#                         return response.get('data').get('video_url')
#                     elif status == 'in_queue':
#                         max_retry_count -= 1
#                         time.sleep(0.5)
#                     else:
#                         write_jiment_log(f"视频生成结果查询异常 status: {status}")
#                         logging.error(f"视频生成结果查询异常 status: {status}")
#                         raise AltaApiError("视频生成失败")
#                 else:
#                     write_jiment_log(
#                         f"视频生成结果查询异常 response:\n{json.dumps(response, indent=4)}")


NODE_CLASS_MAPPINGS = {
    "Alta:AltaImage2Video": AltaImage2VideoNode,
    # "Alta:AltaImage2VideoQuery": AltaI2VTaskQueryNode
}