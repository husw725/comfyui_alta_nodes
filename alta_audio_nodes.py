
import numpy as np
import soundfile as sf

class PyannoteSpeakerDiarizationNode:
    """ComfyUI 节点：基于 pyannote/speaker-diarization-community-1 的说话人分离"""

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
                "use_gpu": ("BOOL", {"default": True}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("speaker_segments",)
    FUNCTION = "run_diarization"
    CATEGORY = "alta/Audio/Analysis"

    def run_diarization(
        self,
        hf_token: str,
        audio=None,
        audio_path: str = "",
        cache_dir: str = "./models/pyannote",
        use_gpu: bool = True,
    ):
        import os
        import tempfile
        import torch
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook

        audio_file = None

        # --------------------------
        # Prepare input audio
        # --------------------------
        if audio is not None:
            if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]

                if hasattr(waveform, "cpu"):
                    waveform = waveform.cpu().numpy()

                waveform = np.squeeze(waveform)
                if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
                    waveform = waveform.T

                tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp_file.name, waveform, sample_rate)
                audio_file = tmp_file.name
                print(f"[Pyannote 4.1] Using dict audio input -> temp file: {audio_file}")

            elif isinstance(audio, (tuple, list)) and len(audio) == 2:
                waveform, sample_rate = audio
                if hasattr(waveform, "cpu"):
                    waveform = waveform.cpu().numpy()
                waveform = np.squeeze(waveform)
                if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
                    waveform = waveform.T

                tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp_file.name, waveform, sample_rate)
                audio_file = tmp_file.name
                print(f"[Pyannote 4.1] Using tuple audio input -> temp file: {audio_file}")

            elif isinstance(audio, str) and os.path.exists(audio):
                audio_file = audio
                print(f"[Pyannote 4.1] Using string audio input: {audio_file}")
            else:
                raise ValueError(f"Invalid 'audio' input type: {type(audio)}")

        elif audio_path and os.path.exists(audio_path):
            audio_file = audio_path
            print(f"[Pyannote 4.1] Using audio_path: {audio_path}")
        else:
            raise ValueError("Please provide either 'audio' input or a valid 'audio_path'.")

        # --------------------------
        # Load pipeline
        # --------------------------
        print(f"[Pyannote 4.1] Loading community pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
            cache_dir=cache_dir
        )

        # Move to GPU if available
        if use_gpu and torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            print("[Pyannote 4.1] Using GPU")
        else:
            print("[Pyannote 4.1] Using CPU")

        # --------------------------
        # Run diarization
        # --------------------------
        print(f"[Pyannote 4.1] Running diarization on {audio_file}")
        with ProgressHook() as hook:
            output = pipeline(audio_file, hook=hook)

        result = []
        for turn, speaker in output.speaker_diarization:
            result.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": f"speaker_{speaker}"
            })

        print(f"[Pyannote 4.1] Found {len(result)} segments")

        # Cleanup
        if audio_file and audio is not None and not isinstance(audio, str):
            os.remove(audio_file)

        return (result,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "Alta:SpeakerDiarization": PyannoteSpeakerDiarizationNode
}

# 可选显示名映射
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Alta:SpeakerDiarization": "🎙️ Speaker Diarization (Pyannote 4.1 Community)"
# }