from dotenv import load_dotenv

load_dotenv()

import os
import base64
import gc
import sys
from multiprocessing import Process, Queue
import multiprocessing
from ExtendedClasses import ExtendedController, DEFAULT_CKPT, ConceptKBFeaturePipeline
import logging
import PIL
from PIL import Image
import torch
from typing import List, Dict, Union, Any
import concurrent.futures
import pickle
from fastapi.responses import StreamingResponse, FileResponse
import os
import io
from StreamingMethods import (
    streaming_hierachical_predict_result,
    streaming_diff_images,
    streaming_heatmap,
    streaming_heatmap_class_difference,
    streaming_is_concept_in_image,
)

logger = logging.getLogger("uvicorn.error")

LOC_SEG_CONCEPT_DIR = os.environ.get("LOC_SEG_CONCEPT_DIR")
CACHE_DIR = os.environ.get(
    "CACHE_DIR", "/shared/nas2/knguye71/ecole-june-demo/cache/ckpt_dir"
)
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/shared/nas2/knguye71/ecole-june-demo/image_dir")
TENSOR_DIR = os.environ.get("TENSOR_DIR", "/shared/nas2/knguye71/ecole-june-demo/tensor_dir")
CONCEPT_KB_CKPT = os.environ.get("CONCEPT_KB_CKPT", "/shared/nas2/knguye71/ecole-june-demo/conceptKB_ckpt")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    AGENT_GPU_LIST = list(range(num_gpus))
else:
    RuntimeError("No GPU available. Please check your CUDA installation.")


# Set the multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

logger = logging.getLogger("uvicorn.error")
logger.info(str("Starting server..."))

# List available GPUs
logger.info(str(f"GPUs available: {torch.cuda.device_count()}, current device: {torch.cuda.current_device()}"))

sys.path.append("/shared/nas2/knguye71/ecole-backend/ods/src")
from feature_extraction import (
    build_sam,
    build_feature_extractor,
)
from image_processing import build_localizer_and_segmenter
from kb_ops.retrieve import CLIPConceptRetriever
from model.concept import ConceptKB, ConceptExample, ConceptKBConfig
from feature_extraction.trained_attrs import N_ATTRS_DINO
import uuid
import time
import json
from io import BytesIO
def rle(x):
    """
    Perform run-length encoding on a boolean PyTorch tensor.
    
    Args:
        tensor (torch.Tensor): A 1D boolean tensor to be encoded.
        
    Returns:
        values (torch.Tensor): The unique values (0 or 1).
        lengths (torch.Tensor): The lengths of the runs.
    """
    assert x.dim() == 1, "Input tensor must be a 1D tensor"
    n = x.shape[0]
    if n == 0:
        return (
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.bool),
        )

    # Find where the value changes
    change_indices = (
        torch.diff(
            x,
            prepend=torch.tensor([0], dtype=x.dtype),
            append=torch.tensor([0], dtype=x.dtype),
        )
        .nonzero()
        .flatten()
    )

    # Lengths of the runs
    lengths = torch.diff(change_indices, prepend=torch.tensor([0])).to(
        dtype=torch.int64
    )

    # Values of the runs
    values = x[change_indices[:-1]]

    return change_indices[:-1], lengths, values


def convert_bool_tensor_to_byte_string(obj: torch.BoolTensor) -> str:
    # Convert a torch.BoolTensor to a byte string
    # Flatten the N-D tensor to 1D
    if (obj.shape == torch.Size([1])):
        return True if obj[0] == 1 else False

    shape = obj.shape
    flattened_tensor = torch.flatten(obj)
    indices, lengths, values = rle(flattened_tensor)
    return {
        "type": "bool_tensor_rle",
        "shape": list(shape),
        "data": lengths.numpy().tobytes().decode("utf-8", errors="ignore"),
        "values": values.numpy().tobytes().decode("utf-8", errors="ignore"),
    }
def convert_PIL_Image_to_byte_string(obj: PIL.Image.Image) -> str:
    # Convert a PIL.Image.Image to a byte string
    with BytesIO() as output:
        obj.save(output, format="JPEG")
        return {
            "type": "PIL_image",
            "size": obj.size,
            "data": output.getvalue().decode("utf-8", errors="ignore"),
        }
def convert_PIL_Image_to_base64_string(obj: PIL.Image.Image) -> str:
    # Convert a PIL.Image.Image to a base64 string
    with BytesIO() as output:
        obj.save(output, format="JPEG")
        return base64.b64encode(output.getvalue()).decode("utf-8")

async def yield_nested_objects(obj: Any, level: int = 1) -> Any:
    """
    Helper function to recursively yield nested objects in markdown format.
    Each yield string starts with "result: " to indicate the start of a new result.
    Lists and nested dictionaries are handled appropriately with hierarchical formatting.
    """

    if isinstance(obj, torch.Tensor):
        if obj.dtype == torch.bool:
            yield f"result: ```python\n{obj.shape}\n{convert_bool_tensor_to_byte_string(obj)}\n\n``"
        else:
            yield f"result: ```python\n{obj.shape}\n{obj.tolist()}\n\n```"
    elif isinstance(obj, PIL.Image.Image):
        new_id = str(uuid.uuid4())
        image_path = os.path.join(IMAGE_DIR, f"{new_id}.jpg")
        obj.save(image_path)
        yield f"result: Image_id: {new_id}\n\n"
        base64_image = convert_PIL_Image_to_base64_string(obj)
        yield f"result: ![](data:image/jpg;base64,{base64_image})\n\n"
    elif hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            if key == "part_masks":
                continue
            yield f"result: {'#' * level} {key}\n\n"
            async for msg in yield_nested_objects(value, level + 1):
                yield msg
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key == "part_masks":
                continue
            yield f"result: {'#' * level} {key}\n\n"
            async for msg in yield_nested_objects(value, level + 1):
                yield msg
    elif isinstance(obj, list):
        for idx, element in enumerate(obj):
            yield f"result: {'#' * level} Node {idx}\n\n"
            async for msg in yield_nested_objects(element, level + 1):
                yield msg
    else:
        try:
            obj_str = json.dumps(obj, indent=4)
            yield f"result: ```json\n{obj_str}\n\n```"
        except (TypeError, OverflowError):
            yield f"result: {str(obj)}\n\n"


# Function to recursively convert nested objects
def convert_to_serializable(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        if "part_crops" in obj.__dict__:
            # save part crops to Image Dir
            list_part_crops_ids = []
            for idx, part_crop in enumerate(obj.__dict__["part_crops"]):
                new_id = str(uuid.uuid4())
                part_crop.save(os.path.join(IMAGE_DIR, f"{new_id}.jpg"))
                list_part_crops_ids.append(new_id)
            obj.__dict__["part_crops"] = list_part_crops_ids
        if "part_masks" in obj.__dict__:
            obj.__dict__["part_masks"] = []
        return {
            key: convert_to_serializable(value) for key, value in obj.__dict__.items()
        }
    elif isinstance(obj, PIL.Image.Image):
        # return convert_PIL_Image_to_byte_string(obj)
        new_id = str(uuid.uuid4())
        obj.save(os.path.join(IMAGE_DIR, f"{new_id}.jpg"))
        return new_id
    elif isinstance(obj, torch.Tensor):
        if isinstance(obj, torch.BoolTensor):
            return convert_bool_tensor_to_byte_string(obj)
        return obj.tolist()

    elif isinstance(obj, dict):
        if "part_crops" in obj:
            # save part crops to Image Dir
            list_part_crops_ids = []
            for idx, part_crop in enumerate(obj["part_crops"]):
                new_id = str(uuid.uuid4())
                part_crop.save(os.path.join(IMAGE_DIR, f"{new_id}.jpg"))
                list_part_crops_ids.append(new_id)
            obj["part_crops"] = list_part_crops_ids

        if "part_masks" in obj:
            obj["part_masks"] = []
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    else:
        return obj


def initialize_models(gpuid=0):
    """Initialize and return a dictionary of models."""
    sam = build_sam(device=f"cuda:{gpuid}")
    loc_and_seg = build_localizer_and_segmenter(sam, None)
    feature_extractor = build_feature_extractor(device=f"cuda:{gpuid}")
    default_kb = ConceptKB()
    default_kb.initialize(
        ConceptKBConfig(
            n_trained_attrs=N_ATTRS_DINO,
        )
    )
    default_kb.to(f"cuda:{gpuid}")

    feature_pipeline = ConceptKBFeaturePipeline(loc_and_seg, feature_extractor)
    controller = ExtendedController(default_kb, feature_pipeline)
    logger.info(str("Models initialized"))

    models = {
        "sam": sam,
        "loc_and_seg": loc_and_seg,
        "feature_extractor": feature_extractor,
        "controller": controller,
    }
    return models


def model_process(input_queue, output_queue, gpuid=0):
    models = initialize_models(
        gpuid
    )  # Ensure initialize_models is defined and correctly initializes models based on the gpuid.

    while True:
        try:
            task_id, status, model_key, func_name, args, kwargs = input_queue.get()

            if func_name == "shutdown":
                break

            model = models.get(model_key, None)
            if model is None:
                output_queue.put(
                    (task_id, "error", f"Error: Model '{model_key}' not found")
                )
                continue

            if not hasattr(model, func_name):
                output_queue.put(
                    (
                        task_id,
                        "error",
                        f"Error: Model '{model_key}' has no function '{func_name}'",
                    )
                )
                continue
            # Perform the function call on the model
            result = getattr(model, func_name)(*args, **kwargs)
            output_queue.put((task_id, "done", result))
        except Exception as e:
            # General exception handling for any unexpected errors
            output_queue.put((task_id, "error", str(e)))


class Agent:
    def __init__(self, gpuid=0):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.process = Process(
            target=model_process, args=(self.input_queue, self.output_queue, gpuid)
        )
        self.gpuid = gpuid
        self.process.start()

    def call(self, model_key, func_name, *args, **kwargs):
        task_id = str(uuid.uuid4())
        status = "processing"
        self.input_queue.put((task_id, status, model_key, func_name, args, kwargs))

        while True:
            result = (
                self.output_queue.get()
            )  # This will block until a result is available
            if result[0] == task_id:
                if result[1] == "done":
                    return result[2]  # Return the actual result
                elif result[1] == "error":
                    import traceback
                    import sys
                    logger.error(traceback.format_exc())
                    logger.error(sys.exc_info())
                    raise Exception(f"Error processing task {task_id}: {result[2]}")
                else:
                    raise Exception(
                        f"Unexpected status for task {task_id}: {result[1]}"
                    )

    def shutdown(self):
        if self.process.is_alive():
            self.input_queue.put((None, None, None, "shutdown", None, None))
            self.process.join()
        if not self.input_queue.empty():
            self.input_queue.close()
            self.input_queue.join_thread()
        if not self.output_queue.empty():
            self.output_queue.close()
            self.output_queue.join_thread()
        logger.info(str(f"Agent on GPU {self.gpuid} shut down"))
        
    def __del__(self):
        self.shutdown()


import os
from collections import deque
import json
from collections import OrderedDict


class AgentManager:
    def __init__(self):
        self.agents = {
            f"agent{idx}": Agent(gpuid) for idx, gpuid in enumerate(AGENT_GPU_LIST)
            # "agent2": Agent(2),  # Example agent for model B
            # Add more agents as required
        }
        # Initialize a round-robin queue for distributing tasks
        self.round_robin_queue = deque(self.agents.keys())
        self.users_concept_kb = OrderedDict()
        self.save_path_concept_kb = CONCEPT_KB_CKPT
        self.checkpoint_path_dict = self.__init_checkpoint_path_dict()

    def __init_checkpoint_path_dict(self):
        new_dict = {}
        for root, dirs, files in os.walk(self.save_path_concept_kb):
            if root == self.save_path_concept_kb:
                continue
            if len(files) > 0:
                user_id = os.path.basename(root)
                new_dict[user_id] = [os.path.join(root, file) for file in files]
                new_dict[user_id].sort()
        return new_dict

    def shutdown(self):
        for agent in self.agents.values():
            agent.shutdown()
        # for user_id in self.users_concept_kb:
        #     self.save_concept_kb(user_id)
        # with open(f"{self.save_path_concept_kb}/checkpoint_path_dict.json", "w") as f:
        #     json.dump(self.checkpoint_path_dict, f)

        # Release the tensor from memory (if it's no longer needed)
        gc.collect()
        # Additionally, to clear unused memory from the GPU cache
        torch.cuda.empty_cache()

        logger.info(str("All agents shut down and concept KBs saved"))

    def get_next_agent_key(self):
        # Rotate the queue and return the next agent key
        self.round_robin_queue.rotate(-1)
        return self.round_robin_queue[0]

    def get_kb(self, user_id:str):
        if user_id in self.users_concept_kb:
            return self.users_concept_kb[user_id]
        elif user_id in self.checkpoint_path_dict:
            return ConceptKB.load(self.checkpoint_path_dict[user_id][-1])
        else:
            return None

    def get_concept_kb(self, user_id: str, agent):

        if agent == None:
            RuntimeError("Agent is None")
        else:
            gpu_id = agent.gpuid
        logger.info(str(f"Getting concept KB for user {user_id}"))
        if (
            user_id not in self.users_concept_kb
            and user_id not in self.checkpoint_path_dict
        ):
            concept_kb = ConceptKB.load(DEFAULT_CKPT)
            self.users_concept_kb[user_id] = concept_kb
            ckpt_path = self.save_concept_kb(user_id)
            self.checkpoint_path_dict[user_id] = [ckpt_path]
        elif user_id in self.users_concept_kb:
            concept_kb = self.users_concept_kb[user_id]
        elif user_id in self.checkpoint_path_dict:
            concept_kb = ConceptKB.load(self.checkpoint_path_dict[user_id][-1])
        else:
            raise Exception("Error in get_concept_kb")

        # Drop the concept KB if number of kb exceeds 5
        if user_id in self.users_concept_kb and len(self.users_concept_kb) > 4:
            pop_id, user_kb = self.users_concept_kb.popitem(last=False)
            logger.info(str(f"user {pop_id} concept KB dropped"))
            user_kb.to("cpu")  
            # Release the tensor from memory (if it's no longer needed)
            del user_kb
            gc.collect()  # Explicitly call garbage collector
            # Additionally, to clear unused memory from the GPU cache
            torch.cuda.empty_cache()
        concept_kb.to(f"cuda:{gpu_id}")
        logger.info(str(f"Concept KB loaded for user {user_id} to CUDA {gpu_id}"))
        self.users_concept_kb[user_id] = concept_kb
        return concept_kb

    def save_concept_kb(self, user_id):
        if user_id in self.users_concept_kb:

            concept_kb = self.users_concept_kb[user_id]
            checkpoint_path = f"{self.save_path_concept_kb}/{user_id}/concept_kb_epoch_{time.time()}.pt"
            if not os.path.exists(f"{self.save_path_concept_kb}/{user_id}"):
                os.makedirs(f"{self.save_path_concept_kb}/{user_id}")
            concept_kb.save(checkpoint_path)

            if user_id not in self.checkpoint_path_dict:
                self.checkpoint_path_dict[user_id] = [checkpoint_path]
            else:
                self.checkpoint_path_dict[user_id].append(checkpoint_path)
            return checkpoint_path
        else:
            return None

    def wrapper_user_id_no_save(self, user_id: str, func, *args, **kwargs):

        agent_key = self.get_next_agent_key()
        concept_kb = None
        try:
            # load concept kb by user_id
            concept_kb = self.get_concept_kb(user_id, self.agents[agent_key])

            # Load user KB first
            self.agents[agent_key].call("controller", "load_kb", concept_kb)

            # run and return result
            return self.agents[agent_key].call("controller", func, *args, **kwargs)
        except Exception as e:
            import traceback
            import sys
            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise Exception(f"Error in wrapper_user_id_no_save: {str(e)}")
        finally:
            if concept_kb is not None:
                concept_kb.to("cpu")
                gc.collect()  # Explicitly call garbage collector
                # Optionally, you can clear the unused memory from the GPU cache
                torch.cuda.empty_cache()

    def wrapper_user_id_save(self, user_id: str, func, *args, **kwargs):

        agent_key = self.get_next_agent_key()
        concept_kb = None
        try:
            # load concept kb by user_id
            concept_kb = self.get_concept_kb(user_id, self.agents[agent_key])
            if concept_kb is None:
                raise Exception(f"Error: Concept KB not found for user {user_id}")
            # Load user KB first
            self.agents[agent_key].call("controller", "load_kb", concept_kb)

            # run and return result
            concept_kb = self.agents[agent_key].call("controller", func, *args, **kwargs)
            self.users_concept_kb[user_id] = concept_kb
            self.save_concept_kb(user_id)
            del self.users_concept_kb[user_id]
            gc.collect()  # Explicitly call garbage collector

            return concept_kb
        except Exception as e:
            raise Exception(f"Error in wrapper_user_id_save: {str(e)}")
        finally:
            # run this block to release memory
            if concept_kb is not None:
                concept_kb.to("cpu")           
                gc.collect()  # Explicitly call garbage collector
                # Optionally, you can clear the unused memory from the GPU cache
                torch.cuda.empty_cache()

    def predict_concept(self, user_id: str, image, threshold):
        result = self.wrapper_user_id_no_save(
            user_id, "predict_concept", image, threshold
        )
        # res = convert_to_serializable(result)
        return result

    def predict_from_subtree(
        self,
        user_id: str,
        image: Image.Image,
        root_concept_name: str,
        unk_threshold: float = 0.1,
    ) -> list[dict]:
        result = self.wrapper_user_id_no_save(
            user_id, "predict_from_subtree", image, root_concept_name, unk_threshold
        )
        return result
        # return convert_to_serializable(result)
        
    
    def train_concepts(self, user_id: str, concept_names: list[str], **train_concept_kwargs):
        result = self.wrapper_user_id_save(
            user_id, "train_concepts", concept_names, **train_concept_kwargs
        )
        return result
    
    

    def predict_hierarchical(
        self,
        user_id: str,
        image: Image.Image,
        unk_threshold: float = 0.1,
        include_component_concepts: bool = False,
    ) -> list[dict]:
        result = self.wrapper_user_id_no_save(
            user_id,
            "predict_hierarchical",
            image,
            unk_threshold,
            include_component_concepts,
        )
        return result
        # return convert_to_serializable(result)

    def predict_root_concept(
        self, user_id: str, image: Image.Image, unk_threshold: float = 0.1
    ) -> dict:
        result = self.wrapper_user_id_no_save(
            user_id, "predict_root_concept", image, unk_threshold
        )
        return result
        # return convert_to_serializable(result)

    def is_concept_in_image(
        self,
        user_id: str,
        image: Image.Image,
        concept_name: str,
        unk_threshold: float = 0.1,
    ) -> bool:
        result = self.wrapper_user_id_no_save(
            user_id, "is_concept_in_image", image, concept_name, unk_threshold
        )
        return result
        # return convert_to_serializable(result)

    def add_hyponym(self, user_id: str, child_name: str, parent_name: str, child_max_retrieval_distance: float = 0.,streaming: bool = False):
        result = self.wrapper_user_id_save(
            user_id, "add_hyponym", child_name, parent_name, child_max_retrieval_distance
        )
        if streaming:
            return f"Hyponym {child_name} added to {parent_name}\n\n"
        else:
            return {"status": "success"}
        # return convert_to_serializable(result)

    def add_component_concept(self, user_id: str, component_concept_name: str, concept_name: str, component_max_retrieval_distance: float = 0., streaming: bool = False):
        result = self.wrapper_user_id_save(
            user_id, "add_component_concept", component_concept_name, concept_name, component_max_retrieval_distance
        )
        if streaming:
            return f"Component {component_concept_name} added to {concept_name}\n\n"
        else:
            return {"status": "success"}
        # return convert_to_serializable(result)

    def add_concept_negatives(self, user_id: str, concept_name: str, negatives: list[PIL.Image.Image], streaming: bool = False):
        time_start = time.time()
        concept_examples = self._loc_and_seg_multiple_images(negatives, concept_name)
        result = self.wrapper_user_id_save(
            user_id, "add_concept_negatives", concept_name, concept_examples, 
        )
        if streaming:
            yield f"status: Add negative examples successfully time: {time.time() - time_start}\n\n"
            
        else:
            return result

    def add_component_concept(
        self,
        user_id: str,
        component_concept_name: str,
        concept_name: str,
        component_max_retrieval_distance: float = 0.0,
        streaming: bool = False,
    ):
        result = self.wrapper_user_id_save(
            user_id, "add_component_concept", component_concept_name, concept_name, component_max_retrieval_distance
        )
        if streaming:
            return f"Component {component_concept_name} added to {concept_name}\n\n"
        else:
            return {"status": "success"}
        # return convert_to_serializable(result)

    def reset_kb(self, user_id: str, clear_all: bool = False, streaming: bool = False):
        logger.info(str(f"clear all: {clear_all}" ))
        if user_id == "":
            user_id = "default_user"
        if user_id  in self.users_concept_kb:
            self.users_concept_kb.pop(user_id, None)
        if clear_all:
            self.users_concept_kb[user_id] = ConceptKB()
            # %% Initialize ConceptKB
            self.users_concept_kb[user_id].initialize(
                ConceptKBConfig(
                    n_trained_attrs=N_ATTRS_DINO,
                )
            )
        else:
            self.users_concept_kb[user_id] = ConceptKB.load(DEFAULT_CKPT)
        self.save_concept_kb(user_id)
        if streaming:
            return f"Knowledge base reset {'from scratch' if clear_all else 'default checkpoint'}\n\n"
        else:
            return {"status": "success"}

    def _loc_and_seg_single_image(self, image: PIL.Image.Image, concept_name: str):
        try:
            time_start = time.time()
            cache_dir = CACHE_DIR
            save_log_and_seg_concept_dir = LOC_SEG_CONCEPT_DIR
            agent_key = self.get_next_agent_key()
            loc_seg_output = self.agents[agent_key].call(
                "loc_and_seg", "localize_and_segment", image
            )

            # save log_seg_output to cache
            if not os.path.exists(save_log_and_seg_concept_dir):
                os.makedirs(save_log_and_seg_concept_dir)

            # save image to cache
            new_id = str(uuid.uuid4())
            image_path = os.path.join(cache_dir, f"{new_id}.jpg")
            # save image to cache
            image.save(image_path)

            path = os.path.join(
                save_log_and_seg_concept_dir,
                new_id + ".pkl",
            )
            with open(path, "wb") as f:
                pickle.dump(loc_seg_output, f)
            logger.info(str("Loc and seg single image time: " + str(time.time() - time_start)))
            return ConceptExample(
                concept_name=concept_name,
                image_path=image_path,
                image_segmentations_path=path,
                image_features_path=None,
            )
        finally:
            image.close()
            torch.cuda.empty_cache()

    def _loc_and_seg_multiple_images(self, images: list[PIL.Image.Image], concept_name: str):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self._loc_and_seg_single_image, images, [concept_name] * len(images)))

        return results

    def _train_concept(
        self, user_id: str, concept_name: str, concept_examples: list[ConceptExample], previous_concept=None, streaming=False
    ):
        result = self.wrapper_user_id_save(
            user_id, "teach_concept", concept_name, concept_examples, previous_concept
        )
        if streaming:
            return  f"status: \nConcept {concept_name} trained with {len(concept_examples)} examples\n\n"
        else:
            return {"status": "success"}

    async def train_concept(
        self, user_id: str, concept_name: str, images: list[PIL.Image.Image], previous_concept=None, streaming=False
    ):
        time_start = time.time()
        concept_examples = self._loc_and_seg_multiple_images(images, concept_name)
        logger.info(str("Loc and seg time: " + str(time.time() - time_start)))
        if streaming:
            yield f"status: Total time for localization and segmentation for {len(images)} images: {time.time() - time_start}"
            time_start = time.time()
            yield self._train_concept(user_id, concept_name, concept_examples, previous_concept, streaming=streaming)
            logger.info(str("Teach concept time: " + str(time.time() - time_start)))
            yield f"status: Teach concept time: {time.time() - time_start}"
        else:
            self._train_concept(user_id, concept_name, concept_examples, previous_concept, streaming=streaming)


    def add_examples(self, user_id:str, images: list[PIL.Image.Image], concept_name: str = None, streaming=False):
        time_start = time.time()
        concept_examples = self._loc_and_seg_multiple_images(images, concept_name)
        result = self.wrapper_user_id_save(
            user_id, "add_examples", concept_examples, concept_name
        )
        if streaming:
            yield f"status: Add examples successfully time: {time.time() - time_start}"
        else:
            return {"status": "success"}

    def train_concepts(self, user_id:str, concept_names: list[str], streaming: bool = False, **train_concept_kwargs):
        time_start = time.time()
        if streaming:
            yield f"status: Training {len(concept_names)} concepts\n\n"
            result = self.wrapper_user_id_save(
                user_id, "train_concepts", concept_names, **train_concept_kwargs
            )
            yield f"status: Training completed\n\n"
            yield f"status: Total time for training {len(concept_names)} concepts: {time.time() - time_start}"
        else:
            result = self.wrapper_user_id_save(
                user_id, "train_concepts", concept_names, **train_concept_kwargs
            )
            return {"status": "success"}
        
    def get_zs_attributes(self, user_id: str, concept_name: str):
        return self.wrapper_user_id_no_save(user_id, "get_zs_attributes", concept_name)

    def heatmap_image_comparison(self, user_id: str, image1: Image, image2: Image):
        result = self.wrapper_user_id_no_save(
            user_id, "heatmap_image_comparison", image1, image2
        )
        return result

    def heatmap_class_difference(self, user_id: str, concept1_name: str, concept2_name: str, image: Image = None):
        logger.info(str(f"heatmap_class_difference: {concept1_name} - {concept2_name}"))
        logger.info(str(f"heatmap_class_difference: {image}"))
        result = self.wrapper_user_id_no_save(
            user_id, "heatmap_class_difference", concept1_name, concept2_name, image
        )
        return result

    def heatmap(self, user_id: str, image: Image, concept_name: str, only_score_increasing_regions: bool = False, only_score_decreasing_regions: bool = False):
        result = self.wrapper_user_id_no_save(
            user_id, "heatmap", image, concept_name, only_score_increasing_regions, only_score_decreasing_regions
        )
        return result

    ##################
    # Utils          #
    ##################
    # def _get_differences(
    #     self,
    #     attr_scores1: torch.Tensor,
    #     attr_scores2: torch.Tensor,
    #     attr_names: List[str],
    #     weights1: torch.Tensor = None,
    #     weights2: torch.Tensor = None,
    #     top_k=5,
    # ) -> Dict[str, List[Union[str, float]]]:
    #     # Compute top attribute probability differences
    #     if weights1 is not None and weights2 is not None:
    #         probs1 = attr_scores1.squeeze().sigmoid()
    #         probs2 = attr_scores2.squeeze().sigmoid()
    #     else:
    #         probs1 = attr_scores1.squeeze()
    #         probs2 = attr_scores2.squeeze()
    #     diffs = (probs1 - probs2).abs()

    #     top_k = min(top_k, len(diffs))

    #     if weights1 is not None and weights2 is not None:
    #         attr_weights1 = weights1
    #         attr_weights2 = weights2
    #         # Weight the differences by the attribute weights
    #         weighted_diffs = diffs * (attr_weights1.abs() + attr_weights2.abs())

    #         top_diffs, top_inds = weighted_diffs.topk(top_k)
    #     else:
    #         top_diffs, top_inds = diffs.topk(top_k)

    #     top_attr_names = [attr_names[i] for i in top_inds]

    #     # Get the top k attribute probabilities and convert to list
    #     probs1 = probs1[top_inds].tolist()
    #     probs2 = probs2[top_inds].tolist()

    #     return {
    #         "name": top_attr_names,
    #         "probs1": probs1,
    #         "probs2": probs2,
    #     }

    # def get_predictor_heatmap_list(self, user_id: str, list_concept_names, image):
    #     result = self.wrapper_user_id_no_save(
    #         user_id, "get_predictor_heatmap_list", list_concept_names, image
    #     )
    #     return result

################
# FastAPI      #
################
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image
from contextlib import asynccontextmanager
tags_metadata = [
    {
        "name": "predict",
        "description": "Predict methods from an image.",
    },
    {
        "name": "train",
        "description": "Train methods from images."
    },
    {
        "name": "compare",
        "description": "Compare methods"
    },
    {
        "name": "kb_ops",
        "description": "Knowledge base operations"
    }
]


app = FastAPI(openapi_tags=tags_metadata, title="ECOLE API", description="API for ECOLE Image Demo", version="0.1.0")


@asynccontextmanager
async def agent_lifespan(agent):
    # Initialization code here. For example, agent.start() if your agent needs explicit starting.
    yield agent
    # Teardown code here
    agent.shutdown()


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent_manager = AgentManager()
    app.state.agentmanager = agent_manager
    logger.info(str("AgentManager initialized"))
    yield
    agent_manager.shutdown()
    logger.info(str("AgentManager shut down"))

app.router.lifespan_context = lifespan

# Initialize your Agent here within the context manager to ensure proper management
# @app.on_event("startup")
# async def startup_event():
#     app.state.agentmanager = AgentManager()

############################
# Predict methods          #
############################


@app.post("/predict_concept", tags=["predict"])
async def predict_concept(
    user_id: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    threshold: str = "0.7",
    streaming: str = "false",
):
    if streaming == "true":
        img = Image.open(image.file).convert("RGB")
        async def streamer(img, threshold):
            time_start = time.time()
            # Convert to PIL Image
            yield "status: Predicting concept..."
            try:
                result = app.state.agentmanager.predict_concept(user_id, img, float(threshold))
                logger.info(str("Predict concept time: " + str(time.time() - time_start)))
                # yield f"result: {json.dumps(result)}"
                async for msg in yield_nested_objects(result):
                    yield msg
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img, threshold),
            media_type="text/event-stream",
        )
    else:
        time_start = time.time()
        # Convert to PIL Image
        img = Image.open(image.file)
        img = img.convert("RGB")
        try:
            result = app.state.agentmanager.predict_concept(user_id, img, float(threshold))
            logger.info(str("Predict concept time: " + str(time.time() - time_start)))
            return JSONResponse(content=result)
        except Exception as e:
            import traceback
            import sys
            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict_from_subtree", tags=["predict"])
async def predict_from_subtree(
    user_id: str,  # Required field
    root_concept_name: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.1",
    streaming: str = "false",
):  
    if streaming == "true":
        img = Image.open(image.file).convert("RGB")
        async def streamer(img, root_concept_name, unk_threshold):
            time_start = time.time()
            # Convert to PIL Image
            yield "status: Predicting from subtree..."
            try:
                result = app.state.agentmanager.predict_from_subtree(
                    user_id, img, root_concept_name, float(unk_threshold)
                )
                logger.info(str("Predict from subtree time: " + str(time.time() - time_start)))
                # yield f"result: {json.dumps(result)}"
                async for msg in streaming_hierachical_predict_result(result):
                    yield msg
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img, root_concept_name, unk_threshold),
            media_type="text/event-stream",
        )
    else:
        time_start = time.time()
        # Convert to PIL Image
        img = Image.open(image.file)
        img = img.convert("RGB")
        try:
            result = app.state.agentmanager.predict_from_subtree(
                user_id, img, root_concept_name, float(unk_threshold)
            )
            logger.info(str("Predict from subtree time: " + str(time.time() - time_start)))
            return JSONResponse(content=result)
        except Exception as e:
            import traceback
            import sys
            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict_hierarchical", tags=["predict"])
async def predict_hierarchical(
    user_id: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.1",
    include_component_concepts: str = "False",
    streaming: str = "false",
):
    logger.info(f"streaming: {streaming}")
    if streaming == "true":
        img = Image.open(image.file).convert("RGB")
        async def streamer(img, unk_threshold, include_component_concepts):
            time_start = time.time()
            # Convert to PIL Image

            yield "status: Predicting hierarchical...\n\n"

            try:
                result = app.state.agentmanager.predict_hierarchical(
                    user_id, img, float(unk_threshold), include_component_concepts if include_component_concepts == "True" else False
                )

                logger.info(str("Predict hierarchical time: " + str(time.time() - time_start)))
                # async for msg in yield_nested_objects(result):
                #     yield msg
                async for msg in streaming_hierachical_predict_result(result):
                    yield msg
                # yield f"result: {json.dumps(result)}"
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img, unk_threshold, include_component_concepts),
            media_type="text/event-stream",
        )
    else:
        time_start = time.time()
        # Convert to PIL Image
        img = Image.open(image.file)
        img = img.convert("RGB")
        try:
            result = app.state.agentmanager.predict_hierarchical(
                user_id, img, float(unk_threshold), include_component_concepts
            )
            logger.info(str("Predict hierarchical time: " + str(time.time() - time_start)))
            return JSONResponse(content=result)
        except Exception as e:
            import traceback
            import sys
            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict_root_concept", tags=["predict"])
async def predict_root_concept(
    user_id: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.1",
    streaming: str = "false",
):
    if streaming == "true":
        img = Image.open(image.file).convert("RGB")
        async def streamer(img, unk_threshold):
            time_start = time.time()
            # Convert to PIL Image
            yield "status: Predicting root concept..."
            try:
                result = app.state.agentmanager.predict_root_concept(
                    user_id, img, float(unk_threshold)
                )
                logger.info(str("Predict root concept time: " + str(time.time() - time_start)))
                logger.info(str("result", result))
                async for msg in yield_nested_objects(result):
                    yield msg
                # yield f"result: {json.dumps(result)}"
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img, unk_threshold),
            media_type="text/event-stream",
        )
    else:
        time_start = time.time()
        # Convert to PIL Image
        img = Image.open(image.file)
        img = img.convert("RGB")
        try:
            result = app.state.agentmanager.predict_root_concept(
                user_id, img, float(unk_threshold)
            )
            logger.info(str("Predict root concept time: " + str(time.time() - time_start)))
            return JSONResponse(content=result)
        except Exception as e:
            import traceback
            import sys
            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))

@app.post("/is_concept_in_image", tags=["predict"])
async def is_concept_in_image(
    user_id: str,  # Required field
    concept_name: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.1",
    streaming: str = "false",
):  
    if streaming == "true":
        img = Image.open(image.file).convert("RGB")
        async def streamer(img, concept_name, unk_threshold):
            time_start = time.time()
            # Convert to PIL Image
            yield "status: Checking if concept in image..."
            try:
                result = app.state.agentmanager.is_concept_in_image(
                    user_id, img, concept_name, float(unk_threshold)
                )
                if concept_name in result.concept_names:
                    heatmap = app.state.agentmanager.heatmap(user_id, img, concept_name)
                logger.info(str("Is concept in image time: " + str(time.time() - time_start)))
                async for msg in streaming_is_concept_in_image(
                    result, concept_name, heatmap
                ):
                    yield msg
                # yield f"result: {json.dumps(result)}"
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img, concept_name, unk_threshold),
            media_type="text/event-stream",
        )
    else:
        time_start = time.time()
        # Convert to PIL Image
        img = Image.open(image.file)
        img = img.convert("RGB")
        try:
            result = app.state.agentmanager.is_concept_in_image(
                user_id, img, concept_name, float(unk_threshold)
            )
            logger.info(str("Is concept in image time: " + str(time.time() - time_start)))
            return JSONResponse(content=result)
        except Exception as e:
            import traceback
            import sys
            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))
###############
# Compare Ops #
###############
@app.post("/heatmap_image_comparison", tags=["compare"])
async def heatmap_image_comparison(
    user_id: str,  # Required field
    image_1: UploadFile = File(...),  # Required file upload,
    image_2: UploadFile = File(...),  # Required file upload,
    streaming: str = "false",
):
    if streaming == "true":
        img1 = Image.open(image_1.file).convert("RGB")
        img2 = Image.open(image_2.file).convert("RGB")
        async def streamer(img1, img2):
            time_start = time.time()
            yield "status: Generating heatmap image comparison..."
            try:
                result = app.state.agentmanager.heatmap_image_comparison(
                    user_id, img1, img2
                )
                for msg in streaming_diff_images(result):
                    yield msg
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img1, img2),
            media_type="text/event-stream",
        )
    else:
        img1 = Image.open(image_1.file).convert("RGB")
        img2 = Image.open(image_2.file).convert("RGB")
        try:
            result = app.state.agentmanager.heatmap_image_comparison(
                user_id, img1, img2
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/heatmap_class_difference", tags=["compare"])
def heatmap_class_difference(
    user_id: str,  # Required field
    concept1_name: str,  # Required field
    concept2_name: str,
    image: UploadFile = None,  # file upload,
    streaming: str = "false",
):
    """
    If image is provided, implements:
        "Why is this <class x> and not <class y>"

    Otherwise, implements:
        "What is the difference between <class x> and <class y>"
        
    Args:
        user_id: str: Required field
        concept1_name: str: Required field
        concept2_name: str: Required field
        image: UploadFile: file upload
        streaming: str: "false"
    
    Returns:
        JSONResponse: content=result
        
    Raises:
        HTTPException: status_code=404, detail=str(e)
        
    Yields:
        StreamingResponse: streamer(concept1_name, concept2_name, img)
    """

    if streaming == "true":
        img = Image.open(image.file).convert("RGB") if image is not None else None
        async def streamer( concept1_name, concept2_name, img):
            time_start = time.time()
            yield "status: Generating heatmap class difference..."
            try:
                result = app.state.agentmanager.heatmap_class_difference(
                    user_id,
                    concept1_name,
                    concept2_name,
                    image=img,
                )
                logger.info(str("Heatmap class difference time: " + str(time.time() - time_start)))

                for msg in streaming_heatmap_class_difference(result, concept1_name, concept2_name, img):
                    yield msg
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(concept1_name, concept2_name, img),
            media_type="text/event-stream",
        )
    else:
        img = Image.open(image.file).convert("RGB") if image is not None else None
        try:
            result = app.state.agentmanager.heatmap_class_difference(
                user_id,
                concept1_name,
                concept2_name,
                img,
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

@app.post("/heatmap", tags=["compare"])
def heatmap(
    user_id: str,  # Required field
    concept_name: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    only_score_increasing_regions: str = "false",
    only_score_decreasing_regions: str = "false",
    streaming: str = "false",
):
    """
    If only_score_increasing_regions is True, implements:
        "Why is this a <class x>"

    If only_score_decreasing_regions is True, implements:
        "What are the differences between this and <class x>"

    If neither is true, shows the full heatmap (both increasing and decreasing regions).

    """
    only_score_increasing_regions = (
        True
        if only_score_increasing_regions == "true"
        else False
    )
    only_score_decreasing_regions = (
        True
        if only_score_decreasing_regions == "true"
        else False
    )
    logger.info(f"streaming: {streaming}")
    if streaming == "true":
        img = Image.open(image.file).convert("RGB")
        async def streamer(img, concept_name):
            time_start = time.time()
            yield "status: Generating heatmap..."
            try:
                result = app.state.agentmanager.heatmap(
                    user_id, img, concept_name, only_score_increasing_regions, only_score_decreasing_regions
                )
                logger.info(str("Heatmap time: " + str(time.time() - time_start)))
                for msg in streaming_heatmap(
                    result,
                    concept_name,
                    only_score_increasing_regions,
                    only_score_decreasing_regions,
                ): 
                    yield msg  
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img, concept_name),
            media_type="text/event-stream",
        )
    else:
        img = Image.open(image.file).convert("RGB")
        try:
            result = app.state.agentmanager.heatmap(
                user_id, img, concept_name, only_score_increasing_regions, only_score_decreasing_regions
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))


###############
# Kb Ops      #
###############

@app.post("/reset_kb", tags=["kb_ops"])
async def reset_kb(
    user_id: str,  # Required field
    clear_all: str = "false",
    streaming: str = "false",
):
    if streaming == "true":
        async def streamer(user_id, clear_all):
            time_start = time.time()
            yield "status: Resetting concept KB..."
            try:
                if clear_all == "True":
                    app.state.agentmanager.reset_kb(user_id, True)
                else:
                    app.state.agentmanager.reset_kb(user_id, False)
                logger.info(str("Reset concept KB time: " + str(time.time() - time_start)))
                yield "status: Reset successful"
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(user_id, clear_all),
            media_type="text/event-stream",
        )
    else:
        try:
            if clear_all == "True":
                app.state.agentmanager.reset_conceptkb(user_id, True)
            else:
                app.state.agentmanager.reset_conceptkb(user_id, False)

            return Response(status_code=200)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

@app.post("/add_hyponym", tags=["kb_ops"])
async def add_hyponym(
    user_id: str,  # Required field
    child_name: str ,  # Required field
    parent_name: str ,  # Required field
    child_max_retrieval_distance: str = "0",
    streaming: str = "alse",
):
    if streaming == "true":
        async def streamer(user_id, child_name, parent_name, child_max_retrieval_distance):
            time_start = time.time()
            yield "status: Adding hyponyms..."
            try:
                result = app.state.agentmanager.add_hyponym(
                    user_id,
                    child_name,
                    parent_name,
                    float(child_max_retrieval_distance),
                    streaming=streaming,
                )
                logger.info(str("Add hyponym time: " + str(time.time() - time_start)))
                yield f"result: {result}"
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}\n\n"

        return StreamingResponse(
            streamer(user_id, child_name, parent_name, child_max_retrieval_distance),
            media_type="text/event-stream",
        )
    else:
        try:
            result = app.state.agentmanager.add_hyponym(
                user_id, child_name, parent_name, float(child_max_retrieval_distance)
            )
            return JSONResponse(content=result)
        except Exception as e:
            import traceback
            import sys

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))

@app.post("/add_component_concept", tags=["kb_ops"])
async def add_component_concept(
    user_id: str,  # Required field
    component_concept_name: str ,  # Required field
    concept_name: str ,  # Required field
    component_max_retrieval_distance: str = "0",
    streaming: str = "false",
):
    if streaming == "true":
        time_start = time.time()
        async def streamer(user_id, component_concept_name, concept_name, component_max_retrieval_distance):
            yield "status: Adding component concept..."
            try:
                result = app.state.agentmanager.add_component_concept(
                    user_id,
                    component_concept_name,
                    concept_name,
                    float(component_max_retrieval_distance),
                    streaming=streaming,
                )
                logger.info(str("Add component concept time: " + str(time.time() - time_start)))
                yield f"result: {result}"
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"
        return StreamingResponse(
            streamer(user_id, component_concept_name, concept_name, component_max_retrieval_distance),
            media_type="text/event-stream",
        )
    else:
        try:
            result = app.state.agentmanager.add_component_concept(
                user_id, component_concept_name, concept_name, float(component_max_retrieval_distance)
            )
            return JSONResponse(content=result)
        except Exception as e:
            import traceback
            import sys

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))

@app.post("/add_examples", tags=["kb_ops"])
async def add_examples(
    user_id: str,  # Required field
    concept_name: str ,  # Required field
    images: List[UploadFile] = File(...),  # Required field
    streaming: str = "false",
):
    if streaming == "true":
        async def streamer(user_id, concept_name, images):
            yield "status: Adding examples..."
            try:
                images = [Image.open(image.file).convert("RGB") for image in images]
                result = app.state.agentmanager.add_examples(
                    user_id, concept_name, images, streaming=streaming
                )
                yield f"result: {result}"
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(user_id, concept_name, images),
            media_type="text/event-stream",
        )
    else:
        try:
            images = [Image.open(image.file).convert("RGB") for image in images]
            result = app.state.agentmanager.add_examples(
                user_id, concept_name, images
            )
            return JSONResponse(content=result)
        except Exception as e:
            import traceback
            import sys

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))
        

@app.post("/add_concept_negatives", tags=["kb_ops"])
async def add_concept_negatives(
    user_id: str,  # Required field
    concept_name: str ,  # Required field
    negatives: List[UploadFile] = File(...),  # Required field
    streaming: str = "false",
):
    if streaming == "true":
        async def streamer(user_id, concept_name, negatives):
            yield "status: Adding concept negatives..."
            try:
                negatives = [Image.open(negative.file).convert("RGB") for negative in negatives]
                result = app.state.agentmanager.add_concept_negatives(
                    user_id, concept_name, negatives, 
                    streaming=streaming
                )
                yield f"result: {result}"
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(user_id, concept_name, negatives),
            media_type="text/event-stream",
        )
    else:
        try:
            negatives = [Image.open(negative.file).convert("RGB") for negative in negatives]
            result = app.state.agentmanager.add_concept_negatives(
                user_id, concept_name, negatives
            )
            return JSONResponse(content=result)
        except Exception as e:
            import traceback
            import sys

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))

@app.post("/train_concept", tags=["train"])
async def train_concept(
    user_id: str,  # Required field
    concept_name: str,  # Required field
    images: List[UploadFile] = File(...),  # Required field
    # previous_concept: str = "None",
    streaming: str = "false",
):
    image_files = [Image.open(image.file).convert("RGB") for image in images]
    if streaming == "true":
        async def streamer(user_id, concept_name, image_files, streaming):
            yield "status: \nTraining concept...\n"
            try:
                time_start = time.time()
                async for res in app.state.agentmanager.train_concept(
                    user_id, concept_name, image_files, streaming=streaming
                ):
                    yield f"status: {res}"
                logger.info(str("Train concept time: " + str(time.time() - time_start)))
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(user_id, concept_name, image_files, streaming),
            media_type="text/event-stream",
        )
    else:
        time_start = time.time()
        try:
            await app.state.agentmanager.train_concept(
                user_id, concept_name, image_files
            )
            logger.info(str("Train concept time: " + str(time.time() - time_start)))
            return JSONResponse(content={"status": "success"})
        except Exception as e:
            import traceback
            import sys

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))
        
@app.post("/train_concepts", tags=["train"])
async def train_concepts(
    user_id: str,  # Required field
    concepts: List[str] = Form(...),  # Required field
    streaming: str = "false",
):
    if streaming == "true":
        async def streamer(user_id, concepts):
            yield "status: \nTraining concepts...\n"
            try:
                time_start = time.time()
                async for res in app.state.agentmanager.train_concepts(
                    user_id, concepts
                ):
                    yield f"status: {res}"
                logger.info(str("Train concepts time: " + str(time.time() - time_start)))
            except Exception as e:
                import traceback
                import sys
                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(user_id, concepts),
            media_type="text/event-stream",
        )
    else:
        time_start = time.time()
        try:
            await app.state.agentmanager.train_concepts(
                user_id, concepts
            )
            logger.info(str("Train concepts time: " + str(time.time() - time_start)))
            return JSONResponse(content={"status": "success"})
        except Exception as e:
            import traceback
            import sys

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))

# @app.post("/diff_between_predictions")
# async def diff_between_predictions(
#     user_id: str,  # Required field
#     image1: UploadFile = File(...),  # Required file upload,
#     image2: UploadFile = File(...),  # Required file upload,
#     threshold: str = "0",
# ):
#     time_start = time.time()
#     if user_id == "":
#         user_id = "default_user"
#     try:
#         img1 = Image.open(image1.file).convert("RGB")
#         img2 = Image.open(image2.file).convert("RGB")
#         # Call your segmentation function (assuming `server.run_segmentation` is defined elsewhere)
#         dict_result = app.state.agentmanager.diff_between_predictions(
#             user_id, img1, img2, float(threshold)
#         )
#         print("Predict concept time: " + str(time.time() - time_start))
#         return dict_result
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=str(e))


# @app.get("/diff_concepts")
# async def diff_concepts(
#     user_id: str = "",  # Required field
#     concept1: str = "",  # Required field
#     concept2: str = "",
#     threshold: str = "0",
# ):
#     time_start = time.time()
#     if user_id == "":
#         user_id = "default_user"
#     try:
#         differences = app.state.agentmanager.diff_concepts(user_id, concept1, concept2)
#         print("Predict concept time: " + str(time.time() - time_start))
#         return differences
#     except Exception as e:
#         import traceback
#         import sys
#         print(traceback.format_exc())
#         print(sys.exc_info())
#         raise HTTPException(status_code=404, detail=str(e))


# @app.post("/train_concept")
# async def add_examples(user_id: str, concept_name: str, examples: List[UploadFile], previous_concept: str = "None"):
#     time_start = time.time()
#     img_list = [Image.open(example.file).convert("RGB") for example in examples]
#     try:
#         app.state.agentmanager.train_concept(user_id, concept_name, img_list, None if previous_concept == "None" else previous_concept)
#         print("Teach concept time: " + str(time.time() - time_start))
#         return {"status": "success"}
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=str(e))


# @app.get("/reset-server")
# async def reset_server(user_id: str, status: str = ""):
#     try:
#         if status == "all":
#             app.state.agentmanager.reset_conceptkb(user_id, True)
#         else:
#             app.state.agentmanager.reset_conceptkb(user_id, False)

#         return Response(status_code=200)
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=str(e))


# @app.get("/get_zs_attributes")
# async def get_zs_attributes(user_id: str, concept_name: str):
#     if user_id == "":
#         user_id = "default_user"
#     try:
#         return app.state.agentmanager.get_zs_attributes(user_id, concept_name)
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=str(e))

# @app.post("/get-predictor-heatmap-list")
# async def get_predictor_heatmap_list(user_id: str, list_concept_names: str, image: UploadFile = File(...)):
#     img = Image.open(image.file)
#     img = img.convert("RGB")
#     list_names = list_concept_names.split(", ")
#     try:
#         return app.state.agentmanager.get_predictor_heatmap_list(
#             user_id, list_names, img
#         )
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=str(e))


# # @app.on_event("shutdown")
# # async def shutdown_event():
# #     app.state.agentmanager.shutdown()  # Ensure the agent is cleanly shut down
@app.post("/images")
def upload_image(image: UploadFile = File(...)):
    img = Image.open(image.file)
    img.save(os.path.join(IMAGE_DIR, image.filename))
    return {"filename": image.filename}


@app.get("/images/{uid}")
def get_image(uid):
    filename = f"{uid}.jpg"
    return FileResponse(os.path.join(IMAGE_DIR, filename), media_type="image/jpg")

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=16004)

@app.post("/tensor")
def upload_tensor(tensor: UploadFile = File(...)):
    tensor.save(os.path.join(TENSOR_DIR, tensor.filename))
    return {"filename": tensor.filename}

@app.get("/tensor/{uid}")
def get_tensor(uid):
    filename = f"{uid}.pt"
    return FileResponse(
        os.path.join(TENSOR_DIR, filename), media_type="application/octet-stream"
    )
