
import numpy as np
import soundfile as sf
import torchaudio
import os,io,av
from typing import Tuple

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





import folder_paths
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



# 注册节点
NODE_CLASS_MAPPINGS = {
    "Alta:SpeakerDiarization": PyannoteSpeakerDiarizationNode,
    "Alta:SaveAudioToPath": SaveAudioToPath,
}

# 可选显示名映射
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Alta:SpeakerDiarization": "🎙️ Speaker Diarization (Pyannote 4.1 Community)"
# }