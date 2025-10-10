import os
import requests
import shutil
from tqdm import tqdm
from server import PromptServer


class AltaDownloader:
    # OUTPUT_NODE = True
    CATEGORY = "Alta"
    FUNCTION = "download"
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("FILE_PATH", )

    def __init__(self):
        self.status = "Idle"
        self.progress = 0.0
        self.node_id = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False}),
                "dest_path": ("STRING", {"multiline": False}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    def download(self, url, dest_path, node_id):
        if not url or not dest_path:
            print(
                f"AltaDownloader: Missing required values: url='{url}', dest_path='{dest_path}'")
            return ()
            
        if os.path.exists(dest_path):
            print(f"AltaDownloader: File already exists: {dest_path}")
            return ()

        print(f'AltaDownloader: Downloading {url} to {dest_path}')
        self.node_id = node_id

        print(f"Downloading {url} to {dest_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        temp_path = dest_path + '.tmp'

        downloaded = 0
        last_progress_update = 0
        file_name = os.path.basename(dest_path)
        try:
            with open(temp_path, 'wb') as file:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=dest_path) as pbar:
                    for data in response.iter_content(chunk_size=4*1024*1024):
                        size = file.write(data)
                        downloaded += size
                        pbar.update(size)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100.0
                            if (progress - last_progress_update) > 0.2:
                                print(
                                    f"Downloading {file_name}... {progress:.1f}%")
                                last_progress_update = progress
                            if progress is not None and hasattr(self, 'node_id'):
                                PromptServer.instance.send_sync("progress", {
                                    "node": self.node_id,
                                    "value": progress,
                                    "max": 100
                                })

            shutil.move(temp_path, dest_path)
            print(f"Complete! {file_name} saved to {dest_path}")
            return dest_path

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

        return ()


NODE_CLASS_MAPPINGS = {
    "Alta:FileDownloader": AltaDownloader,
}
