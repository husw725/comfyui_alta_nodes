from .logger import logger

NODE_CLASS_MAPPINGS = {}
logger.warning("================= ALTA NODES LOAD START ================")
try:
    from .alta_save_image import SaveImagePlus, NODE_CLASS_MAPPINGS as SAVE_MAPPING
    NODE_CLASS_MAPPINGS.update(SAVE_MAPPING)
    logger.info("alta_save_image loaded")
except ImportError as e:
    logger.error("Error importing alta_save_image",e)  # 会打印完整 traceback

try:
    from .alta_load_images import LoadImagesFromDirectoryPath, NODE_CLASS_MAPPINGS as LOAD_MAPPING
    NODE_CLASS_MAPPINGS.update(LOAD_MAPPING)
    logger.info("alta_load_images loaded")
except ImportError as e:
    logger.error("Error importing alta_load_images",e)

try:
    from .nodes_jimeng import NODE_CLASS_MAPPINGS as JIMENG_MAPPING
    NODE_CLASS_MAPPINGS.update(JIMENG_MAPPING)
    logger.info("nodes_jimeng loaded")
except ImportError as e:
    logger.error("Error importing nodes_jimeng",e)

try:
    from .alta_load_video import NODE_CLASS_MAPPINGS as VIDEO_MAPPING
    NODE_CLASS_MAPPINGS.update(VIDEO_MAPPING)
    logger.info("alta_load_video loaded")
except ImportError as e:
    logger.error("Error importing alta_load_video",e)

try:
    from .alta_save_video import NODE_CLASS_MAPPINGS as VIDEO_SAVE_MAPPING
    NODE_CLASS_MAPPINGS.update(VIDEO_SAVE_MAPPING)
    logger.info("alta_save_video loaded")
except ImportError as e:
    logger.error("Error importing alta_save_video",e)

try:
    from .alta_save_str import NODE_CLASS_MAPPINGS as STR_SAVE_MAPPING
    NODE_CLASS_MAPPINGS.update(STR_SAVE_MAPPING)
    logger.info("alta_save_str loaded")
except ImportError as e:
    logger.error("Error importing alta_save_str",e)

logger.warning("================= ALTA NODES LOADED ================")