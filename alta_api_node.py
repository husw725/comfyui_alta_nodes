"""
Alta API Nodes
API Doc
http://44.243.33.213:8000/docs#/
"""

from __future__ import annotations
import os
import requests
import mimetypes
import json
import time
from typing import Optional
from comfy_api.latest import io as comfy_io
from comfy.comfy_types.node_typing import IO, ComfyNodeABC


class AltaApiError(Exception):
    """Base exception for Alta API errors."""
    pass


HOST = 'http://44.254.93.105:8000'


def seg_to_text(seg, text_key):
    return f'[{round(seg['start'], 1)} - {round(seg['end'], 1)}] {seg[text_key]}'


class AltaAPINode(ComfyNodeABC):
    """Alta api base node."""
    FUNCTION = "api_call"
    OUTPUT_NODE = True
    CATEGORY = "Alta"

    def print_result(self, r):
        try:
            print(json.dumps(r.json(), indent='    '))
        except Exception as e:
            print(str(e))
            AltaApiError(str(e))


    def get(self, api, params):
        st = time.time()
        r = requests.get(f"{HOST}/{api}", params=params)
        print(f"API {api} cost {int((time.time() - st)*1000)}ms")
        self.print_result(r)
        return r

    def post(self, api, data=None, json=None, files=None):
        st = time.time()
        r = requests.post(f"{HOST}/{api}", data=data, json=json, files=files)
        print(f"API {api} cost {int((time.time() - st)*1000)}ms")
        self.print_result(r)
        return r


class AltaAPIProcessVideoNode(AltaAPINode):
    RETURN_TYPES = ("STRING", "STRING", "STRING" )
    RETURN_NAMES = ("VOCALS_URL", "MUSIC_URL", "VIDEO_NOAUDIO_URL" )
    DESCRIPTION = "Accepts a video (up to 1GB) and processes it to separate video, vocals and music."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": (
                    IO.STRING, {
                        "tooltip": "The file path of the video"
                    }
                ),
                "video_name": (
                    IO.STRING, {
                        "tooltip": "The unique name for the Series used in Step 1 (e.g., 'project_alpha')."
                    }
                ),
                "video_number": (
                    IO.STRING, {
                        "tooltip": "The number or identifier for the video used in Step 1 (e.g., '001')."
                    }
                )
            }
        }

    async def api_call(
        self,
        video_path: str,
        video_name: str,
        video_number: str,
        **kwargs,
    ) -> str:
        try:
            with open(video_path, mode='rb') as f:
                mime_type, _ = mimetypes.guess_type(video_path)
                r = self.post("step-1", data={"name": video_name, "video_number": video_number}, files={
                    'video': (os.path.basename(video_path), f, mime_type)})
                data = r.json()
                links = data['download_links']
                return comfy_io.NodeOutput(links['vocals_enhanced'], links['music_enhanced'], links['video_no_audio'])
        except Exception as e:
            raise AltaApiError(str(e))


class AltaAPITranscribeAudioNode(AltaAPINode):
    RETURN_TYPES = ("STRING", "STRING", )
    RETURN_NAMES = ("RESULT_JSON_URL", "VOCALS_RESULT", )
    DESCRIPTION = "Transcribes and diarizes the vocals from Step 1 using OpenAI's Whisper and advanced diarization."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_name": (
                    IO.STRING, {
                        "tooltip": "The unique name for the Series used in Step 1 (e.g., 'project_alpha')."
                    }
                ),
                "video_number": (
                    IO.STRING, {
                        "tooltip": "The number or identifier for the video used in Step 1 (e.g., '001')."
                    }
                )
            },
            "hidden": {
                "seed": (
                    IO.INT,
                    {
                        "default": -1
                    }
                ),
            }
        }

    async def api_call(
        self,
        video_name: str,
        video_number: str,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            r = self.post(
                "step-2", data={"name": video_name, "video_number": video_number})
            data = r.json()
            data_file_url = data.get('transcript_link')
            r = requests.get(data_file_url)
            result_data = json.loads(r.text)
            result_for_display = "\n".join([seg_to_text(seg, 'text') for seg in result_data.get('segments')])
            return comfy_io.NodeOutput(data_file_url, result_for_display)
        except Exception as e:
            raise AltaApiError(str(e))


class AltaAPITranslateNode(AltaAPINode):
    RETURN_TYPES = ("STRING", "STRING", "STRING", )
    RETURN_NAMES = ("TRANSLATION_JSON_URL",
                    "TTS_SEGMENTS_JSON_URL", "TRANSLATION",)
    DESCRIPTION = "Process a translation request with prosody-aware translation and TTS enhancement."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_name": (
                    IO.STRING, {
                        "tooltip": "The unique name for the Series used in Step 1 (e.g., 'project_alpha')."
                    }
                ),
                "video_number": (
                    IO.STRING, {
                        "tooltip": "The number or identifier for the video used in Step 1 (e.g., '001')."
                    }
                ),
                # "target_language": (
                #     IO.STRING, {
                #         "tooltip": "The target language to use."
                #     }
                # ),
            },
            "hidden": {
                "seed": (
                    IO.INT,
                    {
                        "default": -1
                    }
                ),
            }
        }

    async def api_call(
        self,
        video_name: str,
        video_number: str,
        # target_language: str,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            data = {"drama_name": video_name, "episode_number": video_number,
                    "target_language": "en"}
            r = self.post("translate", json=data)
            data = r.json()
            links = data['download_links']
            trans_link = links['translation']
            tts_link = links['tts_segments']
            r = requests.get(trans_link)
            trans_data = json.loads(r.text)
            result_for_display = "\n".join(
                [seg_to_text(seg, 'translation_for_tts') for seg in trans_data.get('segments')])
            return comfy_io.NodeOutput(trans_link, tts_link, result_for_display)
        except Exception as e:
            raise AltaApiError(str(e))


class AltaAPITTSNode(AltaAPINode):
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("AUDIO_URL", )
    DESCRIPTION = "Process a translation request with prosody-aware translation and TTS enhancement."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_name": (
                    IO.STRING, {
                        "tooltip": "The unique name for the Series used in Step 1 (e.g., 'project_alpha')."
                    }
                ),
                "video_number": (
                    IO.STRING, {
                        "tooltip": "The number or identifier for the video used in Step 1 (e.g., '001')."
                    }
                ),
            },
            "hidden": {
                "seed": (
                    IO.INT,
                    {
                        "default": -1
                    }
                ),
            }
        }

    async def api_call(
        self,
        video_name: str,
        video_number: str,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            data = {"drama_name": video_name, "episode_number": video_number, "engine": "coqui"}
            r = self.post("tts", json=data)
            data = r.json()
            links = data['final_audio']
            return comfy_io.NodeOutput(links['download_link'])
        except Exception as e:
            raise AltaApiError(str(e))


class AltaAPILipSyncNode(AltaAPINode):
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("VIDEO_URL", )
    DESCRIPTION = "Perform lip-syncing on a video using the specified strategy."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_name": (
                    IO.STRING, {
                        "tooltip": "The unique name for the Series used in Step 1 (e.g., 'project_alpha')."
                    }
                ),
                "video_number": (
                    IO.STRING, {
                        "tooltip": "The number or identifier for the video used in Step 1 (e.g., '001')."
                    }
                ),
            },
            "hidden": {
                "seed": (
                    IO.INT,
                    {
                        "default": -1
                    }
                ),
            }
        }

    async def api_call(
        self,
        video_name: str,
        video_number: str,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            data = {"drama_name": video_name,
                    "episode_number": video_number, "strategy": "A"}
            r = self.post("lipsync", json=data)
            data = r.json()
            links = data['final_audio']
            return comfy_io.NodeOutput(links['download_link'])
        except Exception as e:
            raise AltaApiError(str(e))



NODE_CLASS_MAPPINGS = {
    "Alta:API_ProcessVideo": AltaAPIProcessVideoNode,
    "Alta:API_TranscribeAudio": AltaAPITranscribeAudioNode,
    "Alta:API_Translate": AltaAPITranslateNode,
    "Alta:API_TTS": AltaAPITTSNode,
    "Alta:API_LipSync": AltaAPILipSyncNode,
}