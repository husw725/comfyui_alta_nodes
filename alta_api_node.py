"""
Alta API Nodes
API Doc
http://44.243.33.213:8000/docs#/
"""

from __future__ import annotations
import os
import requests
import mimetypes
from typing import Optional
from comfy_api.latest import io as comfy_io
from comfy.comfy_types.node_typing import IO, ComfyNodeABC


class AltaApiError(Exception):
    """Base exception for Alta API errors."""
    pass


class AltaAPIProcessVideoNode(ComfyNodeABC):
    """Accepts a video (up to 1GB) and processes it to separate video, vocals and music."""
    FUNCTION = "api_call"
    OUTPUT_NODE = True
    CATEGORY = "Alta"
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
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    async def api_call(
        self,
        video_path: str,
        video_name: str,
        video_number: str,
        node_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        api = "http://44.243.33.213:8000/step-1"
        try:
            with open(video_path, mode='rb') as f:
                mime_type, _ = mimetypes.guess_type(video_path)
                r = requests.post(api, data={"name": video_name, "video_number": video_number}, files={
                    'video': (os.path.basename(video_path), f, mime_type)})
                data = r.json()
                links = data['download_links']

                return comfy_io.NodeOutput(links['vocals_enhanced'], links['music_enhanced'], links['video_no_audio'])
        except Exception as e:
            raise AltaApiError(str(e))


class AltaAPITranscribeAudioNode(ComfyNodeABC):
    """Transcribes and diarizes the vocals from Step 1 using OpenAI's Whisper and advanced diarization."""
    FUNCTION = "api_call"
    OUTPUT_NODE = True
    CATEGORY = "Alta"
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("RESULT_JSON_URL", )
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
                "unique_id": "UNIQUE_ID",
            }
        }

    async def api_call(
        self,
        video_name: str,
        video_number: str,
        node_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        api = "http://44.243.33.213:8000/step-2"
        r = requests.post(api, data={"name": video_name, "video_number": video_number})
        data = r.json()
        return comfy_io.NodeOutput(data.get('transcript_link', ''))


NODE_CLASS_MAPPINGS = {
    "Alta:API_ProcessVideo": AltaAPIProcessVideoNode,
    "Alta:API_TranscribeAudio": AltaAPITranscribeAudioNode,
}