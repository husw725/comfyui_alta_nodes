# Alta ComfyUI Nodes

This repository contains a collection of custom nodes for ComfyUI, designed to create a comprehensive video dubbing pipeline.

## Overview

These nodes leverage the Alta API and other services to perform a series of operations for automatic video translation and dubbing. The pipeline is as follows:

1.  **Video Processing**: Separates the input video into video, vocals, and music streams.
2.  **Transcription**: Transcribes the separated vocals into text using OpenAI's Whisper.
3.  **Translation**: Translates the transcribed text into the target language.
4.  **Text-to-Speech (TTS)**: Converts the translated text into speech.
5.  **Lip-Sync**: Synchronizes the generated audio with the original video for accurate lip movement.

## Features

-   End-to-end video dubbing workflow.
-   Integration with the Alta API for streamlined processing.
-   High-quality transcription, translation, and TTS.
-   Advanced lip-syncing using the `syncsdk`.
-   Nodes for loading and saving various media types.

## Installation

1.  Clone this repository into your `ComfyUI/custom_nodes` directory:
    ```bash
    git clone <repository_url>
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Nodes

### Alta API Nodes

-   **Alta:API\_ProcessVideo**: Accepts a video and separates it into video, vocals, and music.
-   **Alta:API\_TranscribeAudio**: Transcribes the vocals from the previous step.
-   **Alta:API\_Translate**: Translates the transcription.
-   **Alta:API\_TTS**: Generates audio from the translated text.
-   **Alta:API\_LipSync**: Performs lip-syncing on the video with the generated audio.

### Lipsync Nodes

-   **Alta:SyncLipsyncNode(path)**: Performs lipsync on a video and audio from local file paths.
-   **Alta:SyncLipsyncNode(url)**: Performs lipsync on a video and audio from URLs.

### Media Loading Nodes

-   **Alta:LoadVideoPath**: Loads a video from a file path.
-   **Alta:LoadVideosFromFolder**: Loads all videos from a specified folder.
-   **Alta:LoadFilesFromFolder**: Loads all files from a specified folder.

## Dependencies

The following Python packages are required:

-   `openai`
-   `ultralytics`
-   `syncsdk`
-   `python-dotenv`
-   `pyannote.audio`
-   `volcengine`

## API Keys

This workflow requires API keys for the Alta API and the Sync API. Make sure to set them as environment variables, for example in a `.env` file:

```
SYNC_API_KEY="your_sync_api_key"
```
