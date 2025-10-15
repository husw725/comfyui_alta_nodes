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

try:
    from .alta_utils_nodes import NODE_CLASS_MAPPINGS as UTILS_MAPPING
    NODE_CLASS_MAPPINGS.update(UTILS_MAPPING)
    logger.info("alta_utils_nodes loaded")
except ImportError as e:
    logger.error("Error importing alta_utils_nodes",e)

try:
    from .alta_merge_images import NODE_CLASS_MAPPINGS as __MERGE_MAPPING
    NODE_CLASS_MAPPINGS.update(__MERGE_MAPPING)
    logger.info("alta_merge_images loaded")
except ImportError as e:
    logger.error("Error importing alta_merge_images", e)

try:
    from .alta_api_node import NODE_CLASS_MAPPINGS as __MERGE_MAPPING
    NODE_CLASS_MAPPINGS.update(__MERGE_MAPPING)
    logger.info("alta_api_node loaded")
except ImportError as e:
    logger.error("Error importing alta_api_node", e)

try:
    from .alta_downloader import NODE_CLASS_MAPPINGS as __MERGE_MAPPING
    NODE_CLASS_MAPPINGS.update(__MERGE_MAPPING)
    logger.info("alta_downloader loaded")
except ImportError as e:
    logger.error("Error importing alta_downloader", e)

try:
    from .alta_cartoon_face_mask import NODE_CLASS_MAPPINGS as cfm
    NODE_CLASS_MAPPINGS.update(cfm)
    logger.info("cartoon_face_mask loaded")
except ImportError as e:
    logger.error("Error importing cartoon_face_mask",e)
except Exception as e:
    logger.error("Error importing cartoon_face_mask", e)


try:
    from .alta_Wan_optimizer import NODE_CLASS_MAPPINGS as WAN_MAPPING
    NODE_CLASS_MAPPINGS.update(WAN_MAPPING)
    logger.info("alta_Wan_optimizer loaded")
except ImportError as e:
    logger.error("Error importing alta_Wan_optimizer",e)
except Exception as e:
    logger.error("Error importing alta_Wan_optimizer", e)

logger.warning("================= ALTA NODES LOADED ================")