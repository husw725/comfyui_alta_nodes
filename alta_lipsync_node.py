import os
import time
import tempfile
from dotenv import load_dotenv
from sync import Sync
from sync.common import Audio, GenerationOptions, Video
from sync.core.api_error import ApiError
from sync.core.file import File

# Load .env automatically
load_dotenv()

class SyncLipsyncNode:
    """
    ComfyUI node for Sync Lipsync-2 generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "input_video.mp4"}),
                "audio_path": ("STRING", {"default": "input_audio.wav"}),
                # "mode": (["loop", "cut","stretch"], {"default": "cut"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": os.getenv("SYNC_API_KEY", "")}),
                "poll_interval": ("INT", {"default": 10, "min": 2, "max": 60}),
                "download_result": (["false", "true"], {"default": "false"}),
            }
        }

    RETURN_TYPES = ("STRING",)   # Output is the URL or local file path
    RETURN_NAMES = ("output_video",)
    FUNCTION = "run"
    CATEGORY = "alta/Lipsync"

    def run(self, video_path, audio_path, api_key, poll_interval, download_result):
        if not api_key:
            raise ValueError("❌ Missing SYNC_API_KEY. Set in .env or pass via api_key input.")

        # Initialize Sync client
        sync = Sync(api_key=api_key)
        try:
            generation = sync.generations.create_with_files(
                audio=audio_path,
                video=video_path,
                model="lipsync-2",
                options=GenerationOptions(
                    sync_mode="cut",
                ),
            )
        except ApiError as e:
            raise RuntimeError(
                f"❌ Sync API request failed (status={e.status_code}): {e.body}"
            )

        job_id = generation.id
        print(f"🎬 Generation submitted successfully, job id: {job_id}")

        # Polling until completed
        while True:
            generation = sync.generations.get(job_id)
            status = generation.status
            print(f"⏳ Polling job {job_id} ... status={status}")
            if status in ["COMPLETED", "FAILED", "REJECTED"]:
                break
            time.sleep(poll_interval)

        if status != "COMPLETED":
            raise RuntimeError(f"❌ Generation failed, status={status}")

        output_url = generation.output_url
        print(f"✅ Generation completed: {output_url}")

        if download_result == "true":
            import requests
            response = requests.get(output_url)
            output_path = os.path.join(tempfile.gettempdir(), f"sync_result_{job_id}.mp4")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"📥 Result saved to {output_path}")
            return (output_path,)

        return (output_url,)
    

class SyncLipsyncByUrlInputNode:
    """
    ComfyUI node for Sync Lipsync-2 generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default": "https://example.com/input_video.mp4"}),
                "audio_url": ("STRING", {"default": "https://example.com/input_audio.wav"}),
                # "mode": (["loop", "cut","stretch"], {"default": "cut"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": os.getenv("SYNC_API_KEY", "")}),
                "poll_interval": ("INT", {"default": 10, "min": 2, "max": 60}),
                "download_result": (["false", "true"], {"default": "false"}),
            }
        }

    RETURN_TYPES = ("STRING",)   # Output is the URL or local file path
    RETURN_NAMES = ("output_video",)
    FUNCTION = "run"
    CATEGORY = "alta/Lipsync"

    def run(self, video_url, audio_url, api_key, poll_interval, download_result):
        if not api_key:
            raise ValueError("❌ Missing SYNC_API_KEY. Set in .env or pass via api_key input.")

        # Initialize Sync client
        sync = Sync(api_key=api_key)

        try:
            generation = sync.generations.create(
                input=[
                    Video(url=video_url),
                    Audio(url=audio_url),
                ],
                model="lipsync-2",
                options=GenerationOptions(
                    sync_mode="cut",
                ),
            )
        except ApiError as e:
            raise RuntimeError(
                f"❌ Sync API request failed (status={e.status_code}): {e.body}"
            )

        job_id = generation.id
        print(f"🎬 Generation submitted successfully, job id: {job_id}")

        # Polling until completed
        while True:
            generation = sync.generations.get(job_id)
            status = generation.status
            print(f"⏳ Polling job {job_id} ... status={status}")
            if status in ["COMPLETED", "FAILED", "REJECTED"]:
                break
            time.sleep(poll_interval)

        if status != "COMPLETED":
            raise RuntimeError(f"❌ Generation failed, status={status}")

        output_url = generation.output_url
        print(f"✅ Generation completed: {output_url}")

        if download_result == "true":
            import requests
            response = requests.get(output_url)
            output_path = os.path.join(tempfile.gettempdir(), f"sync_result_{job_id}.mp4")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"📥 Result saved to {output_path}")
            return (output_path,)

        return (output_url,)


# Register node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "Alta:SyncLipsyncNode(path)": SyncLipsyncNode,
    "Alta:SyncLipsyncNode(url)": SyncLipsyncByUrlInputNode,
}

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Alta:SyncLipsyncNode": "🎤 Sync Lipsync Generator",
#     "Alta:SyncLipsyncByUrlInputNode": "🎤 Sync Lipsync Generator (by URL)",
# }