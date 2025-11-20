import os

# Encoding for subprocess I/O
FILE_ENCODING = "utf-8"

# Blender executable (adjust this path for your installation)
BLENDER_EXECUTABLE = "C:\\Users\\letiz\\Documents\\Huvant\\Tac2AR\\Blender\\blender.exe"

# Project/session identifiers
CLIENT_ID = "DemoClient"
PROJECT_SESSION_ID = "Case001"

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "blender_out")
TEXTURES_DIR = os.path.join(OUTPUT_DIR, "textures")
INPUT_MESH_DIR =  "C:\\Users\\letiz\\Documents\\Huvant\\nnUNet_update\\Blender_pipeline\\meshes_in"# folder for converted meshes
NIFTI_INPUT_DIR =  "C:\\Users\\letiz\\Documents\\Huvant\\inference_new_case_fold0\\fullsize\\99997.nii.gz"     # folder containing .nii/.nii.gz files

# Shader registry and manifest
BLENDER_SHADER_REGISTRY_FILE = "C:\\Users\\letiz\\Documents\\Huvant\\nnUNet_update\\Blender_pipeline\\blender_shader_registry.yaml"  # YAML input
BLENDER_SHADER_REGISTRY_TMP = os.path.join(OUTPUT_DIR, "shader_registry.json")  # JSON generated
SEGMENTS_DATA_MANIFEST_FILE = "C:\\Users\\letiz\\Documents\\Huvant\\nnUNet_update\\Blender_pipeline\\segments_manifest.json"  # segments manifest (JSON)
SEGMENT_MAPPINGS_FILE = "C:\\Users\\letiz\\Documents\\Huvant\\nnUNet_update\\Blender_pipeline\\segmentMappings.yaml"  # segment mappings (YAML)

# Export filenames (filenames only; code joins with OUTPUT_DIR)
PBR_FILENAME = f"{PROJECT_SESSION_ID}_pbr.glb"
URP_FILENAME = f"{PROJECT_SESSION_ID}_urp.fbx"

# Scene/root naming
ROOT_NAME_BASE = f"{PROJECT_SESSION_ID}_Root"

# Geometry & bake settings
WORLD_SCALE_FACTOR = 0.001        # e.g., mm -> m
#MERGE_DISTANCE = 0.0005
MERGE_DISTANCE = 0.001          # in scene units after scaling
MAX_FACES_PER_MESH = 150_000
TEXTURE_SIZE = 1024               # 1024/2048/4096
BLENDER_DEVICE = "GPU"            # "GPU" or "CPU"
NORMAL_SMOOTHING_METHOD = "AUTO_SMOOTH"  # or "FACE"/"ANGLE"

#DISSOLVE_DEGENERATE_THRESHOLD
ROOT_WORLD_POSITION = (0.0, 0.0, 0.0)
SHADERS_DIR= "C:\\Users\\letiz\\Documents\\Huvant\\nnUNet_update\\Blender_pipeline\\shaders"
DISSOLVE_DEGENERATE_THRESHOLD= 0.00015