import os
import tempfile
from typing import List, Tuple
from pathlib import Path
from pyannote.audio import Pipeline

import numpy as np
import soundfile as sf


class PyannoteSpeakerDiarizationNode:
    """ComfyUI 节点：输入音频文件或上游 audio 对象，输出说话人分离结果"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_token": ("STRING", {"default": "your_hf_token_here"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_path": ("STRING", {"default": ""}),
                "cache_dir": ("STRING", {"default": "./models/pyannote"}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("speaker_segments",)
    FUNCTION = "run_diarization"
    CATEGORY = "alta/Audio/Analysis"

    def run_diarization(self, hf_token: str, audio=None, audio_path: str = "", cache_dir: str = "./models/pyannote"):
        import numpy as np
        import soundfile as sf
        import tempfile
        import os
        from pyannote.audio import Pipeline

        # --------------------------
        # 1️⃣ 解析音频输入
        # --------------------------
        audio_file = None

        if audio is not None:
            # case 1: dict {"waveform": tensor, "sample_rate": int}
            if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]
                if hasattr(waveform, "cpu"):  # torch.Tensor -> numpy
                    waveform = waveform.cpu().numpy()
                tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp_file.name, np.array(waveform).T if np.ndim(waveform) > 1 else np.array(waveform), sample_rate)
                tmp_file.flush()
                audio_file = tmp_file.name
                print(f"[Pyannote] Using dict audio input -> temp file: {audio_file}")

            # case 2: tuple or list (waveform, sample_rate)
            elif isinstance(audio, (tuple, list)) and len(audio) == 2:
                waveform, sample_rate = audio
                tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp_file.name, np.array(waveform).T if np.ndim(waveform) > 1 else np.array(waveform), sample_rate)
                tmp_file.flush()
                audio_file = tmp_file.name
                print(f"[Pyannote] Using tuple audio input -> temp file: {audio_file}")

            # case 3: str path
            elif isinstance(audio, str) and os.path.exists(audio):
                audio_file = audio
                print(f"[Pyannote] Using string audio input: {audio_file}")

            else:
                raise ValueError(f"Invalid 'audio' input type: {type(audio)}")

        elif audio_path and os.path.exists(audio_path):
            audio_file = audio_path
            print(f"[Pyannote] Using audio_path: {audio_path}")
        else:
            raise ValueError("Please provide either 'audio' input or a valid 'audio_path'.")

        # --------------------------
        # 2️⃣ 加载 pyannote pipeline
        # --------------------------
        print(f"[Pyannote] Loading model from cache_dir={cache_dir}")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token,
            cache_dir=cache_dir
        )

        # --------------------------
        # 3️⃣ 执行分离
        # --------------------------
        print(f"[Pyannote] Running diarization on {audio_file}")
        diarization = pipeline(audio_file)

        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker
            })

        print(f"[Pyannote] Found {len(result)} segments")

        # --------------------------
        # 4️⃣ 清理临时文件
        # --------------------------
        if audio_file and audio is not None and not isinstance(audio, str):
            os.remove(audio_file)

        return (result,)
# 注册节点
NODE_CLASS_MAPPINGS = {
    "Alta:SpeakerDiarization": PyannoteSpeakerDiarizationNode
}

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Alta:SpeakerDiarization": "🎙️ Speaker Diarization (Pyannote)"
# }
