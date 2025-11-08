
import numpy as np
import soundfile as sf
import torchaudio
import os,io,av
from typing import Tuple

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
                # "cache_dir": ("STRING", {"default": "./models/pyannote"}),
                "num_speakers": ("INT", {"default": 0,"tooltip":"if set to 0, use min_speakers and max_speakers, otherwise use num_speakers"}),
                "min_speakers": ("INT", {"default": 1}),
                "max_speakers": ("INT", {"default": 10}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "min_duration_off_label": ("FLOAT", {"default": 0.8,
                                                     "tooltip": "Minimum silence duration to consider a speaker change. Shorter pauses are merged with previous speech."}),
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
        # cache_dir: str = "./models/pyannote",
        num_speakers: int = 0,
        min_speakers: int = 1,
        max_speakers: int = 10,
        use_gpu: bool = True,
        min_duration_off_label: float = 0.8,
    ):
        import os
        import tempfile
        import torch
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook

        # --------------------------
        # Prepare input audio
        # --------------------------
        audio_file = None
        waveform = None

        if audio is not None:
            if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]
            elif isinstance(audio, (tuple, list)) and len(audio) == 2:
                waveform, sample_rate = audio
            elif isinstance(audio, str) and os.path.exists(audio):
                audio_file = audio
            else:
                raise ValueError(f"Invalid 'audio' input type: {type(audio)}")

            if waveform is not None:
                if hasattr(waveform, "cpu"):
                    waveform = waveform.cpu().numpy()
                waveform = np.squeeze(waveform)
                if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
                    waveform = waveform.T

                tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp_file.name, waveform, sample_rate)
                audio_file = tmp_file.name
                print(f"[Pyannote] Using waveform input -> temp file: {audio_file}")

        elif audio_path and os.path.exists(audio_path):
            audio_file = audio_path
            print(f"[Pyannote] Using audio_path: {audio_file}")
        else:
            raise ValueError("Please provide either 'audio' input or a valid 'audio_path'.")

        # --------------------------
        # Load pipeline
        # --------------------------
        print("[Pyannote] Loading speaker diarization pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )
        pipeline.clustering.method = "average"
        pipeline.clustering.threshold = 0.6  # 默认可能是0.5，调大可以合并相似声音

        if use_gpu and torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            print("[Pyannote] Using GPU")
        else:
            print("[Pyannote] Using CPU")

        pipeline.segmentation.min_duration_off = min_duration_off_label

        # --------------------------
        # Run diarization
        # --------------------------
        print(f"[Pyannote] Running diarization on {audio_file}")
        with ProgressHook() as hook:
            if num_speakers > 0:
                output = pipeline(
                    audio_file,
                    hook=hook,
                    num_speakers=num_speakers,
                )
            else:
                output = pipeline(audio_file, hook=hook,min_speakers=min_speakers, max_speakers=max_speakers)

        # --------------------------
        # Convert output to list
        # --------------------------
        result = []
        for turn, speaker in output.speaker_diarization:
            result.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker
            })

        print(f"[Pyannote] Found {len(result)} segments")

        # --------------------------
        # Cleanup temp file
        # --------------------------
        if audio_file and audio is not None and not isinstance(audio, str):
            os.remove(audio_file)

        return (result,)

from typing import Any, Dict, List


class SaveAudioToPath:
    """
    ComfyUI node to save AUDIO to disk.
    Supports mono/stereo/multi-channel audio.
    Supports formats: flac, mp3, opus, wav
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "output_path": ("STRING", {"multiline": False}),
                "format": (["flac", "mp3", "opus", "wav"], {"default": "flac"}),
                "quality": (["64k", "96k", "128k", "192k", "320k", "V0"], {"default": "128k"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_audio"
    CATEGORY = "Audio"

    def save_audio(
        self,
        audio: Dict[str, Any],
        output_path: str,
        format: str = "flac",
        quality: str = "128k",
    ) -> Tuple[str]:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            waveform: torch.Tensor = audio["waveform"].cpu()
            sample_rate: int = audio["sample_rate"]

            # Opus requires specific sample rates
            OPUS_RATES = [8000, 12000, 16000, 24000, 48000]
            if format == "opus":
                if sample_rate > 48000 or sample_rate not in OPUS_RATES:
                    sample_rate = min([r for r in OPUS_RATES if r >= min(sample_rate, 48000)])
                    waveform = torch.nn.functional.interpolate(
                        waveform.unsqueeze(0), size=int(sample_rate*waveform.shape[-1]/audio["sample_rate"]), mode='linear'
                    ).squeeze(0)

            results = []
            for batch_number, wave in enumerate(waveform):
                # Handle batch filename
                path = output_path
                if "%batch_num%" in output_path:
                    path = output_path.replace("%batch_num%", str(batch_number))
                elif waveform.shape[0] > 1:
                    name, ext = os.path.splitext(output_path)
                    path = f"{name}_{batch_number}{ext}"

                # Interleave channels for PyAV
                channels, samples = wave.shape
                wave_np = wave.numpy().astype(np.float32)

                if channels == 1:
                    interleaved = wave_np[0]
                    layout = "mono"
                else:
                    interleaved = np.empty((samples * channels,), dtype=np.float32)
                    for c in range(channels):
                        interleaved[c::channels] = wave_np[c]
                    layout = "stereo" if channels == 2 else f"{channels}.0"

                # Write using PyAV
                output_buffer = io.BytesIO()
                container = av.open(output_buffer, mode="w", format=format)

                # Codec selection
                if format == "opus":
                    stream = container.add_stream("libopus", rate=sample_rate, layout=layout)
                    bitrate_map = {"64k": 64000, "96k": 96000, "128k": 128000, "192k": 192000, "320k": 320000}
                    stream.bit_rate = bitrate_map.get(quality, 128000)
                elif format == "mp3":
                    stream = container.add_stream("libmp3lame", rate=sample_rate, layout=layout)
                    if quality == "V0":
                        stream.codec_context.qscale = 1
                    else:
                        bitrate_map = {"128k": 128000, "320k": 320000}
                        stream.bit_rate = bitrate_map.get(quality, 128000)
                elif format == "wav":
                    stream = container.add_stream("pcm_s16le", rate=sample_rate, layout=layout)
                else:
                    stream = container.add_stream("flac", rate=sample_rate, layout=layout)

                # Create frame
                frame = av.AudioFrame.from_ndarray(interleaved.reshape(1, -1), format="flt", layout=layout)
                frame.sample_rate = sample_rate
                frame.pts = 0

                # Encode and mux
                container.mux(stream.encode(frame))
                container.mux(stream.encode(None))
                container.close()

                # Save to disk
                output_buffer.seek(0)
                with open(path, "wb") as f:
                    f.write(output_buffer.getbuffer())

                results.append(path)

            return (results[0] if len(results) == 1 else json.dumps(results),)

        except Exception as e:
            return (f"Error saving audio: {e}",)


import os
import torch
import torchaudio

class LoadAudioByPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT",)
    RETURN_NAMES = ("audio_tensor", "sample_rate",)
    FUNCTION = "load_audio"
    CATEGORY = "Alta/Audio"
    DESCRIPTION = "Load an audio file from a given path."

    def load_audio(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        # 加载音频
        waveform, sample_rate = torchaudio.load(path)

        # 如果是立体声，可选择转换为单声道（可按需注释掉）
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        print(f"Loaded audio: {path} (sr={sample_rate}, len={waveform.shape[-1]})")

        return (waveform, float(sample_rate))

import torch

# 全局缓存字典
_AUDIO_CACHE = {}

# -------------------------
# Store Node
# -------------------------
class StoreAudioInMemory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "default_audio"}),
                "audio": ("AUDIO",),
                "sample_rate": ("FLOAT", {"default": 44100}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "store"
    CATEGORY = "Alta/Audio/Memory"
    DESCRIPTION = "Store audio in memory for later retrieval."

    def store(self, name, audio, sample_rate):
        _AUDIO_CACHE[name] = (audio.clone(), sample_rate)
        print(f"[AudioCache] Stored '{name}' (len={audio.shape[-1]}, sr={sample_rate})")
        return ()


# -------------------------
# Load Node
# -------------------------
class LoadAudioFromMemory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "default_audio"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "BOOLEAN",)
    RETURN_NAMES = ("audio", "sample_rate", "exist",)
    FUNCTION = "load"
    CATEGORY = "Alta/Audio/Memory"
    DESCRIPTION = "Load audio from memory by name. Returns exist=False if not found."

    def load(self, name):
        if name not in _AUDIO_CACHE:
            print(f"[AudioCache] '{name}' not found.")
            # 返回空的 tensor + 默认采样率 + exist=False
            empty_audio = torch.zeros((1, 0))
            return (empty_audio, 0.0, False)

        audio, sr = _AUDIO_CACHE[name]
        print(f"[AudioCache] Loaded '{name}' (len={audio.shape[-1]}, sr={sr})")
        return (audio.clone(), float(sr), True)


# -------------------------
# Delete Node
# -------------------------
class DeleteAudioFromMemory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "default_audio"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("deleted",)
    FUNCTION = "delete"
    CATEGORY = "Alta/Audio/Memory"
    DESCRIPTION = "Delete audio from memory cache and return whether it existed."

    def delete(self, name):
        if name in _AUDIO_CACHE:
            del _AUDIO_CACHE[name]
            print(f"[AudioCache] Deleted '{name}'")
            return (True,)
        else:
            print(f"[AudioCache] '{name}' not found.")
            return (False,)



# RNNoiseDenoise_Enhanced.py
# 放入 ComfyUI/custom_nodes/

# import os
# import tempfile
# import numpy as np
# try:
#     import pyrnnoise
# except ImportError:
#     raise ImportError("请先安装 pyrnnoise: pip install pyrnnoise")
# import soundfile as sf

# class RNNoiseDenoiseEnhancedNode:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {},
#             "optional": {
#                 "audio": ("AUDIO",),
#                 "audio_path": ("STRING", {"default": ""}),
#                 "highpass_cutoff": ("FLOAT", {"default": 80.0}),
#                 "lowpass_cutoff": ("FLOAT", {"default": 9000.0}),
#                 "noise_gate_threshold": ("FLOAT", {"default": -40.0})
#             }
#         }

#     RETURN_TYPES = ("AUDIO",)
#     RETURN_NAMES = ("clean_audio",)
#     FUNCTION = "denoise_enhanced"
#     CATEGORY = "Audio/Processing"
#     DESCRIPTION = "Use RNNoise + high/low pass filter + noise gate to clean up vocals."

#     def denoise_enhanced(
#         self,
#         audio=None,
#         audio_path: str = "",
#         highpass_cutoff: float = 80.0,
#         lowpass_cutoff: float = 9000.0,
#         noise_gate_threshold: float = -40.0
#     ):
#         # --------------------------
#         # Prepare waveform
#         # --------------------------
#         waveform = None
#         sample_rate = 44100
#         if audio is not None:
#             if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
#                 waveform = audio["waveform"]
#                 sample_rate = audio["sample_rate"]
#                 if hasattr(waveform, "cpu"):
#                     waveform = waveform.cpu().numpy()
#             elif isinstance(audio, (tuple, list)) and len(audio) == 2:
#                 waveform, sample_rate = audio
#                 if hasattr(waveform, "cpu"):
#                     waveform = waveform.cpu().numpy()
#             elif isinstance(audio, str) and os.path.exists(audio):
#                 audio_path = audio

#         if waveform is None:
#             if audio_path and os.path.exists(audio_path):
#                 waveform, sample_rate = sf.read(audio_path, dtype="float32")
#             else:
#                 raise ValueError("Please provide 'audio' or a valid 'audio_path'")

#         # Ensure 2D array [channels, samples]
#         if waveform.ndim == 1:
#             chunk = waveform[np.newaxis, :]
#         elif waveform.shape[0] < waveform.shape[1]:
#             chunk = waveform
#         else:
#             chunk = waveform.T  # [samples, channels] -> [channels, samples]

#         # --------------------------
#         # RNNoise denoise
#         # --------------------------
#         denoiser = pyrnnoise.RNNoise(sample_rate)
#         clean_frames = []

#         for _, frame in denoiser.denoise_chunk(chunk, partial=True):
#             clean_frames.append(frame)

#         clean_waveform = np.concatenate(clean_frames, axis=-1)
#         if clean_waveform.shape[0] == 1:
#             clean_waveform = clean_waveform[0]  # mono flatten

#         # --------------------------
#         # Highpass / Lowpass FFT filter
#         # --------------------------
#         fft = np.fft.rfft(clean_waveform)
#         freqs = np.fft.rfftfreq(len(clean_waveform), d=1/sample_rate)
#         fft[(freqs < highpass_cutoff) | (freqs > lowpass_cutoff)] = 0
#         clean_waveform = np.fft.irfft(fft)

#         # --------------------------
#         # Noise gate
#         # --------------------------
#         threshold_amp = 10 ** (noise_gate_threshold / 20.0)
#         clean_waveform[np.abs(clean_waveform) < threshold_amp] = 0.0

#         # return ({"waveform": clean_waveform, "sample_rate": sample_rate},)
    
#     # ComfyUI expects [1, channels, samples]
#         if clean_waveform.ndim == 1:
#             clean_waveform = clean_waveform[np.newaxis, :]  # shape (1, samples)
#         else:
#             clean_waveform = clean_waveform  # shape (channels, samples)

#         # ComfyUI expects [1, channels, samples]
#         clean_tensor = torch.tensor(clean_waveform, dtype=torch.float32).unsqueeze(0)

#         return ({"waveform": clean_tensor, "sample_rate": sample_rate},)



# 注册节点
NODE_CLASS_MAPPINGS = {
    "Alta:SpeakerDiarization": PyannoteSpeakerDiarizationNode,
    "Alta:SaveAudioToPath": SaveAudioToPath,
    "Alta:LoadAudioByPath": LoadAudioByPath,
    "Alta:StoreAudioInMemory": StoreAudioInMemory,
    "Alta:LoadAudioFromMemory": LoadAudioFromMemory,
    "Alta:DeleteAudioFromMemory": DeleteAudioFromMemory,
    # "Alta:AudioDenoise": RNNoiseDenoiseEnhancedNode,
}

# 可选显示名映射
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Alta:SpeakerDiarization": "🎙️ Speaker Diarization (Pyannote 4.1 Community)"
# }