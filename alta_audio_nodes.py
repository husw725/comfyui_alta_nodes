
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

        audio_file = None

        if audio is not None:
            if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]

                # torch.Tensor -> numpy
                if hasattr(waveform, "cpu"):
                    waveform = waveform.cpu().numpy()

                # squeeze 掉多余维度
                waveform = np.squeeze(waveform)

                # 确保 (samples, channels) 格式
                if waveform.ndim == 1:
                    pass  # mono, ok
                elif waveform.ndim == 2:
                    # check if (channels, samples)
                    if waveform.shape[0] < waveform.shape[1]:
                        waveform = waveform.T
                else:
                    raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

                tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp_file.name, waveform, sample_rate)
                tmp_file.flush()
                audio_file = tmp_file.name
                print(f"[Pyannote] Using dict audio input -> temp file: {audio_file}")

            elif isinstance(audio, (tuple, list)) and len(audio) == 2:
                waveform, sample_rate = audio
                if hasattr(waveform, "cpu"):
                    waveform = waveform.cpu().numpy()
                waveform = np.squeeze(waveform)
                if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
                    waveform = waveform.T

                tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp_file.name, waveform, sample_rate)
                tmp_file.flush()
                audio_file = tmp_file.name
                print(f"[Pyannote] Using tuple audio input -> temp file: {audio_file}")

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
        # Run diarization
        # --------------------------
        print(f"[Pyannote] Loading model from cache_dir={cache_dir}")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token,
            cache_dir=cache_dir
        )

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
