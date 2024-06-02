from dotenv import load_dotenv

load_dotenv()

import os
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

LOC_SEG_CONCEPT_DIR = os.environ.get("LOC_SEG_CONCEPT_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    AGENT_GPU_LIST = list(range(num_gpus))
else:
    RuntimeError("No GPU available. Please check your CUDA installation.")


# Set the multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

logger = logging.getLogger("uvicorn.error")
logger.info("Starting server...")

# List available GPUs
logger.info(f"GPUs available: {torch.cuda.device_count()}, current device: {torch.cuda.current_device()}")

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

def convert_bool_tensor_to_byte_string(obj: torch.BoolTensor) -> str:
    # Convert a torch.BoolTensor to a byte string
    return {
        "type": "bool_tensor",
        "shape": list(obj.shape),
        "data": obj.numpy().tobytes().decode("utf-8", errors="ignore"),
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

# Function to recursively convert nested objects
def convert_to_serializable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if isinstance(obj, torch.BoolTensor):
            return convert_bool_tensor_to_byte_string(obj)
        return obj.tolist()
    elif isinstance(obj, PIL.Image.Image):
        return convert_PIL_Image_to_byte_string(obj)
    elif isinstance(obj, dict):
        # if "part_crops" in obj:
        #     del obj["part_crops"]
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    elif hasattr(obj, "__dict__"):
        # if hasattr(obj, "part_crops"):
        #     del obj.part_crops
        return {
            key: convert_to_serializable(value) for key, value in obj.__dict__.items()
        }
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
    # retriever = CLIPConceptRetriever(
    #     default_kb.concepts,
    #     feature_extractor.clip,
    #     feature_extractor.processor,
    # )

    feature_pipeline = ConceptKBFeaturePipeline(loc_and_seg, feature_extractor)
    controller = ExtendedController(default_kb, feature_pipeline)

    # controller = ExtendedController(
    #     loc_and_seg, default_kb, feature_extractor, retriever
    # )

    logger.info("Models initialized")

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
                    print(traceback.format_exc())
                    print(sys.exc_info())
                    raise Exception(f"Error processing task {task_id}: {result[2]}")
                else:
                    raise Exception(
                        f"Unexpected status for task {task_id}: {result[1]}"
                    )

    def shutdown(self):
        self.input_queue.put((None, None, None, "shutdown", None, None))
        self.process.join()


import os
from collections import deque
import json
from collections import OrderedDict


CONCEPT_KB_CKPT = "/shared/nas2/knguye71/ecole-backend/conceptKB_ckpt"


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
        with open(f"{self.save_path_concept_kb}/checkpoint_path_dict.json", "w") as f:
            json.dump(self.checkpoint_path_dict, f)
        print("All agents shut down and concept KBs saved")

    def get_next_agent_key(self):
        # Rotate the queue and return the next agent key
        self.round_robin_queue.rotate(-1)
        return self.round_robin_queue[0]

    def get_concept_kb(self, user_id: str, agent):
        if agent == None:
            gpu_id = 0
        else:
            gpu_id = agent.gpuid
        print(f"Getting concept KB for user {user_id}")
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
            print(f"user {pop_id} concept KB dropped")
            user_kb.to("cpu")  
            # Release the tensor from memory (if it's no longer needed)
            del user_kb
            gc.collect()  # Explicitly call garbage collector
            # Additionally, to clear unused memory from the GPU cache
            torch.cuda.empty_cache()
        concept_kb.to(f"cuda:{gpu_id}")
        print(f"Concept KB loaded for user {user_id} to CUDA {gpu_id}")
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
            print(traceback.format_exc())
            print(sys.exc_info())
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
        res = convert_to_serializable(result)
        return res

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
        return convert_to_serializable(result)

    def predict_hierarchical(
        self,
        user_id: str,
        image: Image.Image,
        unk_threshold: float = 0.1,
        include_component_concepts: bool = False,
    ) -> list[dict]:
        result = self.wrapper_user_id_no_save(
            user_id, "predict_hierarchical", image, unk_threshold, include_component_concepts
        )
        return convert_to_serializable(result)

    def predict_root_concept(
        self, user_id: str, image: Image.Image, unk_threshold: float = 0.1
    ) -> dict:
        result = self.wrapper_user_id_no_save(
            user_id, "predict_root_concept", image, unk_threshold
        )
        return convert_to_serializable(result)

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
        return convert_to_serializable(result)

    def add_hyponyms(self, user_id: str, child_name: str, parent_name: str, child_max_retrieval_distance: float = 0.):
        result = self.wrapper_user_id_save(
            user_id, "add_hyponyms", child_name, parent_name, child_max_retrieval_distance
        )
        return convert_to_serializable(result)

    def add_component_concept(self, user_id: str, component_concept_name: str, concept_name: str, component_max_retrieval_distance: float = 0.):
        result = self.wrapper_user_id_save(
            user_id, "add_component_concept", component_concept_name, concept_name, component_max_retrieval_distance
        )
        return convert_to_serializable(result)

    def add_concept_negatives(self, user_id: str, concept_name: str, negatives: list[ConceptExample]):
        result = self.wrapper_user_id_save(
            user_id, "add_concept_negatives", concept_name, negatives
        )
        return convert_to_serializable(result)

    def add_component_concept(
        self,
        user_id: str,
        component_concept_name: str,
        concept_name: str,
        component_max_retrieval_distance: float = 0.0,
    ):
        result = self.wrapper_user_id_save(
            user_id, "add_component_concept", component_concept_name, concept_name, component_max_retrieval_distance
        )
        return convert_to_serializable(result)

    # def diff_between_predictions(
    #     self,
    #     user_id: str,
    #     image1: PIL.Image.Image,
    #     image2: PIL.Image.Image,
    #     threshold=0,
    #     top_k=5,
    # ):
    #     result = self.wrapper_user_id_no_save(
    #         user_id, "diff_between_predictions", image1, image2, threshold, top_k
    #     )
    #     return result

    # def diff_concepts(self, user_id: str, concept1: str, concept2: str):
    #     """
    #     Returns the differences in the top k attributes between the two concepts.
    #     """
    #     concept1 = concept1.lower()
    #     concept2 = concept2.lower()

    #     result = self.wrapper_user_id_no_save(
    #         user_id, "diff_concepts", concept1, concept2, top_k=5
    #     )

    #     return result

    def reset_kb(self, user_id: str, clear_all: bool = False):
        print("clear all: ", clear_all)
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
        return {"status": "success"}

    def _loc_and_seg_single_image(self, image: PIL.Image.Image):
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
            print("Loc and seg single image time: ", time.time() - time_start)
            return ConceptExample(
                image_path=image_path,
                image_segmentations_path=path,
                image_features_path=None,
            )
        finally:
            image.close()
            torch.cuda.empty_cache()

    def _loc_and_seg_multiple_images(self, images: list[PIL.Image.Image]):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self._loc_and_seg_single_image, images))

        return results

    def _train_concept(
        self, user_id: str, concept_name: str, concept_examples: list[ConceptExample], previous_concept=None
    ):
        result = self.wrapper_user_id_save(
            user_id, "teach_concept", concept_name, concept_examples, previous_concept
        )
        return {"status": "success"}

    def train_concept(
        self, user_id: str, concept_name: str, images: list[PIL.Image.Image], previous_concept=None
    ):
        time_start = time.time()
        concept_examples = self._loc_and_seg_multiple_images(images)
        print("Loc and seg time: ", time.time() - time_start)
        time_start = time.time()
        self._train_concept(user_id, concept_name, concept_examples, previous_concept)
        print("Teach concept time: ", time.time() - time_start)

    def get_zs_attributes(self, user_id: str, concept_name: str):
        return self.wrapper_user_id_no_save(user_id, "get_zs_attributes", concept_name)

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
    print("AgentManager initialized")
    yield
    agent_manager.shutdown()
    print("AgentManager shut down")

app.router.lifespan_context = lifespan

# Initialize your Agent here within the context manager to ensure proper management
# @app.on_event("startup")
# async def startup_event():
#     app.state.agentmanager = AgentManager()

############################
# Predict methods          #
############################

@app.post("/predict-concept", tags=["predict"])
async def predict_concept(
    user_id: str = Form(...),  # Required field
    image: UploadFile = File(...),  # Required file upload,
    threshold: str = "0.7",
):
    time_start = time.time()
    # Convert to PIL Image
    img = Image.open(image.file)
    img = img.convert("RGB")
    # try:
    if user_id == "":
        user_id = "default_user"
    try:
        # Call your segmentation function (assuming `server.run_segmentation` is defined elsewhere)
        result = app.state.agentmanager.predict_concept(
            user_id, img, float(threshold)
        )
        print("Predict concept time: ", time.time() - time_start)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        import sys
        print(traceback.format_exc())
        print(sys.exc_info())
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict-from-subtree", tags=["predict"])
async def predict_from_subtree(
    user_id: str = Form(...),  # Required field
    image: UploadFile = File(...),  # Required file upload,
    root_concept_name: str = Form(...),  # Required field
    unk_threshold: str = "0.1",
):
    time_start = time.time()
    # Convert to PIL Image
    img = Image.open(image.file)
    img = img.convert("RGB")
    try:
        result = app.state.agentmanager.predict_from_subtree(
            user_id, img, root_concept_name, float(unk_threshold)
        )
        print("Predict from subtree time: ", time.time() - time_start)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        import sys
        print(traceback.format_exc())
        print(sys.exc_info())
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict-hierarchical", tags=["predict"])
async def predict_hierarchical(
    user_id: str = Form(...),  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.1",
    include_component_concepts: str = "False",
):
    time_start = time.time()
    # Convert to PIL Image
    img = Image.open(image.file)
    img = img.convert("RGB")
    try:
        result = app.state.agentmanager.predict_hierarchical(
            user_id, img, float(unk_threshold), include_component_concepts
        )
        print("Predict hierarchical time: ", time.time() - time_start)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        import sys
        print(traceback.format_exc())
        print(sys.exc_info())
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict-root-concept", tags=["predict"])
async def predict_root_concept(
    user_id: str = Form(...),  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.1",
):
    time_start = time.time()
    # Convert to PIL Image
    img = Image.open(image.file)
    img = img.convert("RGB")
    try:
        result = app.state.agentmanager.predict_root_concept(
            user_id, img, float(unk_threshold)
        )
        print("Predict root concept time: ", time.time() - time_start)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        import sys
        print(traceback.format_exc())
        print(sys.exc_info())
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/is-concept-in-image", tags=["predict"])
async def is_concept_in_image(
    user_id: str = Form(...),  # Required field
    image: UploadFile = File(...),  # Required file upload,
    concept_name: str = Form(...),  # Required field
    unk_threshold: str = "0.1",
):
    time_start = time.time()
    # Convert to PIL Image
    img = Image.open(image.file)
    img = img.convert("RGB")
    try:
        result = app.state.agentmanager.is_concept_in_image(
            user_id, img, concept_name, float(unk_threshold)
        )
        print("Is concept in image time: ", time.time() - time_start)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        import sys
        print(traceback.format_exc())
        print(sys.exc_info())
        raise HTTPException(status_code=404, detail=str(e))


###############
# Kb Ops      #
###############

@app.post("/reset-kb", tags=["kb_ops"])
async def reset_kb(
    user_id: str = Form(...),  # Required field
    clear_all: str = "False",
):
    try:
        if clear_all == "True":
            app.state.agentmanager.reset_conceptkb(user_id, True)
        else:
            app.state.agentmanager.reset_conceptkb(user_id, False)

        return Response(status_code=200)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/add-hyponyms", tags=["kb_ops"])
async def add_hyponyms(
    user_id: str = Form(...),  # Required field
    child_name: str = Form(...),  # Required field
    parent_name: str = Form(...),  # Required field
    child_max_retrieval_distance: str = "0",
):
    if user_id == "":
        user_id = "default_user"
    try:
        result = app.state.agentmanager.add_hyponyms(
            user_id, child_name, parent_name, float(child_max_retrieval_distance)
        )
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        import sys

        print(traceback.format_exc())
        print(sys.exc_info())
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/add-component-concept", tags=["kb_ops"])
async def add_component_concept(
    user_id: str = Form(...),  # Required field
    component_concept_name: str = Form(...),  # Required field
    concept_name: str = Form(...),  # Required field
    component_max_retrieval_distance: str = "0",
):
    if user_id == "":
        user_id = "default_user"
    try:
        result = app.state.agentmanager.add_component_concept(
            user_id, component_concept_name, concept_name, float(component_max_retrieval_distance)
        )
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        import sys

        print(traceback.format_exc())
        print(sys.exc_info())
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/add-concept-negatives", tags=["kb_ops"])
async def add_concept_negatives(
    user_id: str = Form(...),  # Required field
    concept_name: str = Form(...),  # Required field
    negatives: list[UploadFile] = File(...),  # Required field
):
    if user_id == "":
        user_id = "default_user"
    try:
        negatives = [Image.open(negative.file).convert("RGB") for negative in negatives]
        result = app.state.agentmanager.add_concept_negatives(
            user_id, concept_name, negatives
        )
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        import sys

        print(traceback.format_exc())
        print(sys.exc_info())
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/train-concept", tags=["train"])
async def train_concept(
    user_id: str = Form(...),  # Required field
    concept_name: str = Form(...),  # Required field
    examples: List[UploadFile] = File(...),  # Required field
    previous_concept: str = "None",
):
    time_start = time.time()
    img_list = [Image.open(example.file).convert("RGB") for example in examples]
    try:
        app.state.agentmanager.train_concept(user_id, concept_name, img_list, None if previous_concept == "None" else previous_concept)
        print("Teach concept time: ", time.time() - time_start)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# @app.post("/diff_between_predictions")
# async def diff_between_predictions(
#     user_id: str = Form(...),  # Required field
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
#         print("Predict concept time: ", time.time() - time_start)
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
#         print("Predict concept time: ", time.time() - time_start)
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
#         print("Teach concept time: ", time.time() - time_start)
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


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=16008)
