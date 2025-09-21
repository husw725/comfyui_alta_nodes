NODE_CLASS_MAPPINGS = {}

try:
    from .alta_save_image import SaveImagePlus, NODE_CLASS_MAPPINGS as SAVE_MAPPING
    NODE_CLASS_MAPPINGS.update(SAVE_MAPPING)
except ImportError as e:
    print(f"Error importing alta_save_image: {e}")


try:
    from .alta_load_images import LoadImagesFromDirectoryPath, NODE_CLASS_MAPPINGS  as LOAD_MAPPING
    NODE_CLASS_MAPPINGS.update(LOAD_MAPPING)
except ImportError as e:
    print(f"Error importing alta_load_images: {e}")


try:
    from .nodes_jimeng import NODE_CLASS_MAPPINGS as JIMENG_MAPPING
    NODE_CLASS_MAPPINGS.update(JIMENG_MAPPING)
except ImportError as e:
    print(f"Error importing nodes_jimeng: {e}")


try:
    from .alta_load_video import NODE_CLASS_MAPPINGS as VIDEO_MAPPING
    NODE_CLASS_MAPPINGS.update(VIDEO_MAPPING)
except ImportError as e:
    print(f"Error importing alta_load_video: {e}")


try:
    from .alta_save_video import  NODE_CLASS_MAPPINGS as VIDEO_SAVE_MAPPING
    NODE_CLASS_MAPPINGS.update(VIDEO_SAVE_MAPPING)
except ImportError as e:
    print(f"Error importing alta_save_video: {e}")
