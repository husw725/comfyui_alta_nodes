from .alta_save_image import SaveImagePlus, NODE_CLASS_MAPPINGS as SAVE_MAPPING
from .alta_load_images import LoadImagesFromDirectoryPath, NODE_CLASS_MAPPINGS  as LOAD_MAPPING
from .GptImageMerge import GPTImageMerge, NODE_CLASS_MAPPINGS as GPTIMAGE_MAPPING


# 合并成一个字典
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(SAVE_MAPPING)
NODE_CLASS_MAPPINGS.update(LOAD_MAPPING)
NODE_CLASS_MAPPINGS.update(GPTIMAGE_MAPPING)