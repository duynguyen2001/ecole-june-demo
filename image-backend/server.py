# %%

from dotenv import load_dotenv

load_dotenv()
# %%
import base64
import concurrent.futures
import gc
import logging
import multiprocessing
import os
import pickle
import sys
from multiprocessing import Process, Queue
from typing import Any, List

import PIL.Image
import torch
from ExtendedClasses import (DEFAULT_CKPT, ConceptExample,
                             ConceptKBFeaturePipeline, ExtendedController)
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from streaming_methods import (streaming_diff_images, streaming_heatmap,
                               streaming_heatmap_class_difference,
                               streaming_hierachical_predict_result,
                               yield_nested_objects)

logger = logging.getLogger("uvicorn.error")

LOC_SEG_CONCEPT_DIR = os.environ.get(
    "LOC_SEG_CONCEPT_DIR", "/shared/nas2/knguye71/ecole-june-demo/cache/log_and_seg"
)
CACHE_DIR = os.environ.get(
    "CACHE_DIR", "/shared/nas2/knguye71/ecole-june-demo/cache/ckpt_dir"
)
IMAGE_DIR = os.environ.get(
    "IMAGE_DIR", "/shared/nas2/knguye71/ecole-june-demo/image_dir"
)
TENSOR_DIR = os.environ.get(
    "TENSOR_DIR", "/shared/nas2/knguye71/ecole-june-demo/tensor_dir"
)
CONCEPT_KB_CKPT = os.environ.get(
    "CONCEPT_KB_CKPT", "/shared/nas2/knguye71/ecole-june-demo/conceptKB_ckpt"
)

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    AGENT_GPU_LIST = list(range(num_gpus))
else:
    RuntimeError("No GPU available. Please check your CUDA installation.")
    AGENT_GPU_LIST = [0]


# Set the multiprocessing start method to 'spawn'
multiprocessing.set_start_method("spawn", force=True)

logger = logging.getLogger("uvicorn.error")
logger.info(str("Starting server..."))

# List available GPUs
logger.info(
    str(
        f"GPUs available: {torch.cuda.device_count()}, current device: {torch.cuda.current_device()}"
    )
)

import time
import uuid
from io import BytesIO

from feature_extraction import build_feature_extractor, build_sam
from feature_extraction.trained_attrs import N_ATTRS_DINO
from image_processing import build_localizer_and_segmenter
from model.concept import ConceptExample, ConceptKB, ConceptKBConfig


class Agent:
    def initialize_models(self, device: str = "cpu") -> dict[str, Any]:
        """Initialize and return a dictionary of models."""
        sam = build_sam(device=device)
        loc_and_seg = build_localizer_and_segmenter(sam, None)
        feature_extractor = build_feature_extractor(device=device)
        default_kb = ConceptKB()
        default_kb.initialize(
            ConceptKBConfig(
                n_trained_attrs=N_ATTRS_DINO,
            )
        )
        default_kb.to(device)

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

    def model_process(
        self, input_queue: Queue, output_queue: Queue, device="cpu"
    ) -> None:

        models = self.initialize_models(
            device
        )  # Ensure initialize_models is defined and correctly initializes models based on the cuda device

        while True:
            task_id, status, model_key, func_name, args, kwargs = input_queue.get()
            try:
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

    def __init__(self, device) -> None:
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.process = Process(
            target=self.model_process, args=(self.input_queue, self.output_queue, device)
        )
        self.device = device
        self.process.start()

    def call(self, model_key, func_name, *args, **kwargs) -> Any:
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
                    import sys
                    import traceback

                    logger.error(traceback.format_exc())
                    logger.error(sys.exc_info())
                    raise Exception(f"Error processing task {task_id}: {result[2]}")
                else:
                    raise Exception(
                        f"Unexpected status for task {task_id}: {result[1]}"
                    )

    def shutdown(self) -> None:
        if self.process.is_alive():
            self.input_queue.put((None, None, None, "shutdown", None, None))
            self.process.join()
        if not self.input_queue.empty():
            self.input_queue.close()
            self.input_queue.join_thread()
        if not self.output_queue.empty():
            self.output_queue.close()
            self.output_queue.join_thread()
        logger.info(str(f"Agent on GPU {self.device} shut down"))

    def __del__(self):
        self.shutdown()


import os
from collections import deque


class AgentManager:
    """
    The AgentManager class manages the agents and the concept knowledge bases (KBs) for each user.
    """

    def __init__(
        self,
        agent_gpu_list: list[int] = AGENT_GPU_LIST,
        concept_kb_dir: str = CONCEPT_KB_CKPT,
        default_ckpt: str = DEFAULT_CKPT,
    ) -> None:
        """
        The AgentManager class manages the agents and the concept knowledge bases (KBs) for each user.


        Args:
            agent_gpu_list (list[int], optional): List of GPU IDs to assign to each agent. Defaults to AGENT_GPU_LIST.
        """
        self.agents = {
            f"agent{idx}": Agent(f"cuda:{gpuid}") for idx, gpuid in enumerate(agent_gpu_list)
        }
        # Initialize a round-robin queue for distributing tasks
        self.round_robin_queue = deque(self.agents.keys())
        self.concept_kb_dir = concept_kb_dir
        self.default_ckpt = default_ckpt
        self.checkpoint_path_dict = self.__init_checkpoint_path_dict()

    def __init_checkpoint_path_dict(self) -> dict[str, list[str]]:
        """
        Get the checkpoint path dictionary for each user. The dictionary contains the user IDs and their checkpoint paths order by time.

        Returns:
            dict[str, list[str]]: Dictionary of user IDs and their checkpoint paths.
        """
        new_dict = {}
        for root, dirs, files in os.walk(self.concept_kb_dir):
            if root == self.concept_kb_dir:
                continue
            if len(files) > 0:
                user_id = os.path.basename(root)
                new_dict[user_id] = [os.path.join(root, file) for file in files]
                new_dict[user_id].sort()
        return new_dict

    def shutdown(self):
        for agent in self.agents.values():
            agent.shutdown()

        # Release the tensor from memory (if it's no longer needed)
        gc.collect()
        # Additionally, to clear unused memory from the GPU cache
        torch.cuda.empty_cache()

        logger.info(str("All agents shut down and concept KBs saved"))

    def get_next_agent_key(self) -> str:
        # Rotate the queue and return the next agent key
        self.round_robin_queue.rotate(-1)
        return self.round_robin_queue[0]

    ####################
    # ConceptKB ops    #
    ####################
    def get_concept_kb(self, user_id: str, device: str = "cpu") -> ConceptKB:
        """
        This method gets the concept knowledge base (KB) for a user.

        Args:
            user_id (str): User ID
            device (str, optional): Device to load the concept KB. Defaults to "cpu".

        Returns:
            ConceptKB: Concept knowledge base
        """
        logger.info(str(f"Getting concept KB for user {user_id}"))
        if user_id in self.checkpoint_path_dict:
            concept_kb = ConceptKB.load(self.checkpoint_path_dict[user_id][-1])
        else:
            # load default concept kb
            concept_kb = ConceptKB.load(self.default_ckpt)
            # save default concept kb
            os.makedirs(f"{self.concept_kb_dir}/{user_id}", exist_ok=True)
            checkpoint_path = f"{self.concept_kb_dir}/{user_id}/concept_kb_epoch_{time.time()}.pt"
            concept_kb.save(checkpoint_path)
            self.checkpoint_path_dict[user_id] = [checkpoint_path]

        # Move concept kb to device
        concept_kb.to(device=device)
        logger.info(str(f"Concept KB loaded for user {user_id} to device {device}"))
        return concept_kb

    def save_concept_kb(self, user_id: str, concept_kb: ConceptKB) -> str:
        """
        Save the concept knowledge base (KB) for a user.

        Args:
            user_id (str): User ID
            concept_kb (ConceptKB): Concept knowledge base

        Returns:
            str: Checkpoint path
        """
        os.makedirs(f"{self.concept_kb_dir}/{user_id}", exist_ok=True)
        checkpoint_path = f"{self.concept_kb_dir}/{user_id}/concept_kb_epoch_{time.time()}.pt"
        concept_kb.to("cpu")
        concept_kb.save(checkpoint_path)
        if user_id not in self.checkpoint_path_dict:
            self.checkpoint_path_dict[user_id] = [checkpoint_path]
        self.checkpoint_path_dict[user_id].append(checkpoint_path)
        return checkpoint_path
    
    def undo_concept_kb(self, user_id: str) -> str:
        """
        Undo the last checkpoint of the concept knowledge base (KB) for a user.

        Args:
            user_id (str): User ID

        Returns:
            str: Checkpoint path
        """
        if user_id in self.checkpoint_path_dict and len(self.checkpoint_path_dict[user_id]) > 1:
            checkpoint_path = self.checkpoint_path_dict[user_id][-2]
            if os.path.exists(checkpoint_path):
                os.remove(self.checkpoint_path_dict[user_id][-1])
            self.checkpoint_path_dict[user_id].pop()
            return checkpoint_path
        else:
            return "No checkpoint to undo"

    def executeControllerFunctionNoSave(self, user_id: str, func, *args, **kwargs) -> Any:
        """
        Execute a function in the controller without saving the concept knowledge base.

        Args:
            user_id (str): User ID
            func (_type_): Function to execute
            *args (_type_): Function arguments
            **kwargs (_type_): Function keyword arguments

        Raises:
            Exception: Error in executeControllerFunctionNoSave

        Returns:
            Any: Function result
        """
        agent_key = self.get_next_agent_key()
        concept_kb = None
        try:
            # load concept kb by user_id
            concept_kb = self.get_concept_kb(user_id, self.agents[agent_key].device)

            # Load user KB first
            self.agents[agent_key].call("controller", "load_kb", concept_kb)

            # run and return result
            return self.agents[agent_key].call("controller", func, *args, **kwargs)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise Exception(f"Error in executeControllerFunctionNoSave: {str(e)}")
        finally:
            if concept_kb is not None:
                concept_kb.to("cpu")
                gc.collect()  # Explicitly call garbage collector
                # Optionally, you can clear the unused memory from the GPU cache
                torch.cuda.empty_cache()

    def executeControllerFunctionWithSave(self, user_id: str, func, *args, **kwargs) -> str:
        """
        If the function is successful, save the concept knowledge base.
        

        Args:
            user_id (str): User ID
            func (_type_): Function to execute

        Raises:
            Exception: _description_
            Exception: _description_

        Returns:
            str: Checkpoint path
        """
        agent_key = self.get_next_agent_key()
        concept_kb = None
        try:
            # load concept kb by user_id
            concept_kb = self.get_concept_kb(user_id, self.agents[agent_key].device)
            # Load user KB first
            self.agents[agent_key].call("controller", "load_kb", concept_kb)

            # run and return result
            concept_kb = self.agents[agent_key].call(
                "controller", func, *args, **kwargs
            )

            # move concept kb to cpu
            concept_kb.to("cpu")
            ckpt_path = self.save_concept_kb(user_id, concept_kb)
            return ckpt_path
        except Exception as e:
            # General exception handling for any unexpected errors
            import sys
            import traceback

            logger.error(sys.exc_info())
            logger.error(traceback.format_exc())
            raise Exception(f"Error in executeControllerFunctionWithSave: {str(e)}")
        finally:
            gc.collect()  # Explicitly call garbage collector
            # Optionally, you can clear the unused memory from the GPU cache
            torch.cuda.empty_cache()

    def predict_from_subtree(
        self,
        user_id: str,
        image: Image.Image,
        root_concept_name: str,
        unk_threshold: float = 0.1,
    ) -> list[dict]:
        return self.executeControllerFunctionNoSave(
            user_id, "predict_from_subtree", image, root_concept_name, unk_threshold
        )

    def predict_hierarchical(
        self,
        user_id: str,
        image: Image.Image,
        unk_threshold: float = 0.1,
        include_component_concepts: bool = False,
    ) -> list[dict]:
        return self.executeControllerFunctionNoSave(
            user_id,
            "predict_hierarchical",
            image,
            unk_threshold,
            include_component_concepts,
        )

    def predict_root_concept(
        self, user_id: str, image: Image.Image, unk_threshold: float = 0.1
    ) -> dict:
        return self.executeControllerFunctionNoSave(
            user_id, "predict_root_concept", image, unk_threshold
        )

    def is_concept_in_image(
        self,
        user_id: str,
        image: Image.Image,
        concept_name: str,
        unk_threshold: float = 0.1,
    ) -> bool:
        return self.executeControllerFunctionNoSave(
            user_id, "is_concept_in_image", image, concept_name, unk_threshold
        )

    def add_hyponym(
        self,
        user_id: str,
        child_name: str,
        parent_name: str,
        child_max_retrieval_distance: float = 0.0,
        streaming: bool = False,
    ):
        result = self.executeControllerFunctionWithSave(
            user_id,
            "add_hyponym",
            child_name,
            parent_name,
            child_max_retrieval_distance,
        )
        if streaming:
            return f"Hyponym {child_name} added to {parent_name}\n\n"
        else:
            return {"status": "success"}
        # return convert_to_serializable(result)

    def add_concept_negatives(
        self,
        user_id: str,
        images: list[PIL.Image.Image],
        concept_name: str,
        streaming: bool = False,
    ):
        time_start = time.time()
        concept_examples = self._loc_and_seg_multiple_images(images, concept_name)
        for example in concept_examples:
            example.is_negative = True
        result = self.executeControllerFunctionWithSave(
            user_id,
            "add_concept_negatives",
            concept_name,
            concept_examples,
        )
        if streaming:
            yield f"status: Add negative examples successfully time: {time.time() - time_start}\n\n"
            yield f"result: {len(concept_examples)} negative examples added to concept {concept_name}\n\n"

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
        result = self.executeControllerFunctionWithSave(
            user_id,
            "add_component_concept",
            component_concept_name,
            concept_name,
            component_max_retrieval_distance,
        )
        if streaming:
            return f"Component {component_concept_name} added to {concept_name}\n\n"
        else:
            return {"status": "success"}

    ####################################
    # ConceptKB methods for streaming  #
    ####################################

    def reset_kb(self, user_id: str, clear_all: bool = False, streaming: bool = False):
        """
        Reset the knowledge base for a user.

        """
        logger.info(f"clear all: {clear_all}")

        if user_id == "":
            user_id = "default_user"
        if clear_all:
            concept_kb = ConceptKB()
            concept_kb.initialize(
                ConceptKBConfig(
                    n_trained_attrs=N_ATTRS_DINO,
                )
            )
            concept_kb.to("cpu")
        else:
            concept_kb = ConceptKB.load(self.default_ckpt)
        
        self.save_concept_kb(user_id, concept_kb)
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
                new_id + ".pkl"
            )
            with open(path, "wb") as f:
                pickle.dump(loc_seg_output.cpu(detach=True), f)
            logger.info(
                str("Loc and seg single image time: " + str(time.time() - time_start))
            )
            return ConceptExample(
                concept_name=concept_name,
                image_path=image_path,
                image_segmentations_path=path,
                image_features_path=None,
            )
        finally:
            image.close()
            torch.cuda.empty_cache()

    def _loc_and_seg_multiple_images(
        self, images: list[PIL.Image.Image], concept_name: str
    ):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    self._loc_and_seg_single_image, images, [concept_name] * len(images)
                )
            )
        return results

    def _train_concept(
        self,
        user_id: str,
        concept_name: str,
        concept_examples: list[ConceptExample],
        previous_concept=None,
        streaming=False,
    ):
        result = self.executeControllerFunctionWithSave(
            user_id, "teach_concept", concept_name, concept_examples, previous_concept
        )
        if streaming:
            return f"status: \nConcept {concept_name} trained with {len(concept_examples)} examples\n\n"
        else:
            return {"status": "success"}

    async def train_concept(
        self,
        user_id: str,
        concept_name: str,
        images: list[PIL.Image.Image],
        previous_concept=None,
        streaming=False,
    ):
        time_start = time.time()
        concept_examples = self._loc_and_seg_multiple_images(images, concept_name)
        logger.info(str("Loc and seg time: " + str(time.time() - time_start)))
        if streaming:
            yield f"status: Total time for localization and segmentation for {len(images)} images: {time.time() - time_start}"
            time_start = time.time()
            yield self._train_concept(
                user_id,
                concept_name,
                concept_examples,
                previous_concept,
                streaming=streaming,
            )
            logger.info(str("Teach concept time: " + str(time.time() - time_start)))
            yield f"status: Teach concept time: {time.time() - time_start}"
        else:
            self._train_concept(
                user_id,
                concept_name,
                concept_examples,
                previous_concept,
                streaming=streaming,
            )

    def add_examples(
        self,
        user_id: str,
        images: list[PIL.Image.Image],
        concept_name: str = None,
        streaming=False,
    ):
        time_start = time.time()
        concept_examples = self._loc_and_seg_multiple_images(images, concept_name)

        print(f"concept_examples: {concept_examples}")
        print(f"concept_name: {concept_name}")

        self.executeControllerFunctionWithSave(
            user_id,
            "add_examples",
            examples=concept_examples,
            concept_name=concept_name,
        )

        if streaming:
            yield f"status: Add examples successfully time: {time.time() - time_start}"
            yield f"result: {len(concept_examples)} examples added to concept {concept_name}\n\n"
        else:
            return {"status": "success"}

    def train_concepts(
        self,
        user_id: str,
        concept_names: list[str],
        streaming: bool = False,
        **train_concept_kwargs,
    ):
        time_start = time.time()
        logger.info(f"\n\nconcept_names: {concept_names}\n\n")
        if streaming:
            yield f"status: Training {len(concept_names)} concepts\n\n"
            self.executeControllerFunctionWithSave(
                user_id, "train_concepts", concept_names, **train_concept_kwargs
            )
            yield f"status: Training completed\n\n"
            yield f"status: Total time for training {len(concept_names)} concepts: {time.time() - time_start}"
            yield f"result: Training succcessfully with {len(concept_names)} concepts\n\n"
        else:
            result = self.executeControllerFunctionWithSave(
                user_id, "train_concepts", concept_names, **train_concept_kwargs
            )
            return {"status": "success"}

    def get_zs_attributes(self, user_id: str, concept_name: str):
        return self.executeControllerFunctionNoSave(user_id, "get_zs_attributes", concept_name)

    def heatmap_image_comparison(self, user_id: str, image1: PIL.Image.Image, image2: PIL.Image.Image):
        result = self.executeControllerFunctionNoSave(
            user_id, "heatmap_image_comparison", image1, image2
        )
        return result

    def heatmap_class_difference(
        self, user_id: str, concept1_name: str, concept2_name: str, image: PIL.Image.Image = None
    ):
        logger.info(str(f"heatmap_class_difference: {concept1_name} - {concept2_name}"))
        logger.info(str(f"heatmap_class_difference: {image}"))
        result = self.executeControllerFunctionNoSave(
            user_id, "heatmap_class_difference", concept1_name, concept2_name, image
        )
        return result

    def heatmap(
        self,
        user_id: str,
        image: PIL.Image.Image,
        concept_name: str,
        only_score_increasing_regions: bool = False,
        only_score_decreasing_regions: bool = False,
    ):
        result = self.executeControllerFunctionNoSave(
            user_id,
            "heatmap",
            image,
            concept_name,
            only_score_increasing_regions,
            only_score_decreasing_regions,
        )
        return result


from contextlib import asynccontextmanager

import PIL.Image
################
# FastAPI      #
################
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

tags_metadata = [
    {
        "name": "predict",
        "description": "Predict methods from an image.",
    },
    {"name": "train", "description": "Train methods from images."},
    {"name": "compare", "description": "Compare methods"},
    {"name": "kb_ops", "description": "Knowledge base operations"},
]


app = FastAPI(
    openapi_tags=tags_metadata,
    title="ECOLE API",
    description="API for ECOLE Image Demo",
    version="0.1.0",
)


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


############################
# Predict methods          #
############################
@app.post("/predict_from_subtree", tags=["predict"])
async def predict_from_subtree(
    user_id: str,  # Required field
    root_concept_name: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.7",
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
                logger.info(
                    str("Predict from subtree time: " + str(time.time() - time_start))
                )
                # yield f"result: {json.dumps(result)}"
                async for msg in streaming_hierachical_predict_result(result):
                    yield msg
            except Exception as e:
                import sys
                import traceback

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
            logger.info(
                str("Predict from subtree time: " + str(time.time() - time_start))
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict_hierarchical", tags=["predict"])
async def predict_hierarchical(
    user_id: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.7",
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
                    user_id,
                    img,
                    float(unk_threshold),
                    (
                        include_component_concepts
                        if include_component_concepts == "True"
                        else False
                    ),
                )

                logger.info(
                    str("Predict hierarchical time: " + str(time.time() - time_start))
                )
                # async for msg in yield_nested_objects(result):
                #     yield msg
                async for msg in streaming_hierachical_predict_result(result):
                    yield msg
                # yield f"result: {json.dumps(result)}"
            except Exception as e:
                import sys
                import traceback

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
            logger.info(
                str("Predict hierarchical time: " + str(time.time() - time_start))
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict_root_concept", tags=["predict"])
async def predict_root_concept(
    user_id: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.7",
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
                logger.info(
                    str("Predict root concept time: " + str(time.time() - time_start))
                )
                logger.info(str("result", result))
                async for msg in yield_nested_objects(result):
                    yield msg
                # yield f"result: {json.dumps(result)}"
            except Exception as e:
                import sys
                import traceback

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
            logger.info(
                str("Predict root concept time: " + str(time.time() - time_start))
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

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
                heatmap = app.state.agentmanager.heatmap(user_id, img, concept_name)
                logger.info(
                    str("Is concept in image time: " + str(time.time() - time_start))
                )
                for msg in streaming_heatmap(heatmap, concept_name):
                    yield msg
            except Exception as e:
                import sys
                import traceback

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
            logger.info(
                str("Is concept in image time: " + str(time.time() - time_start))
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

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
                import sys
                import traceback

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

        async def streamer(concept1_name, concept2_name, img):
            time_start = time.time()
            yield "status: Generating heatmap class difference..."
            try:
                result = app.state.agentmanager.heatmap_class_difference(
                    user_id,
                    concept1_name,
                    concept2_name,
                    image=img,
                )
                logger.info(
                    str(
                        "Heatmap class difference time: "
                        + str(time.time() - time_start)
                    )
                )

                for msg in streaming_heatmap_class_difference(
                    result, concept1_name, concept2_name, img
                ):
                    yield msg
            except Exception as e:
                import sys
                import traceback

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
        True if only_score_increasing_regions == "true" else False
    )
    only_score_decreasing_regions = (
        True if only_score_decreasing_regions == "true" else False
    )
    logger.info(f"streaming: {streaming}")
    if streaming == "true":
        img = Image.open(image.file).convert("RGB")

        async def streamer(img, concept_name):
            time_start = time.time()
            yield "status: Generating heatmap..."
            try:
                result = app.state.agentmanager.heatmap(
                    user_id,
                    img,
                    concept_name,
                    only_score_increasing_regions,
                    only_score_decreasing_regions,
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
                import sys
                import traceback

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
                user_id,
                img,
                concept_name,
                only_score_increasing_regions,
                only_score_decreasing_regions,
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
                logger.info(
                    str("Reset concept KB time: " + str(time.time() - time_start))
                )
                yield f"result: Knowledge base successfully reset {'from scratch' if clear_all == "true" else 'default checkpoint'}\n\n"
            except Exception as e:
                import sys
                import traceback

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
    child_name: str,  # Required field
    parent_name: str,  # Required field
    child_max_retrieval_distance: str = "0",
    streaming: str = "alse",
):
    if streaming == "true":

        async def streamer(
            user_id, child_name, parent_name, child_max_retrieval_distance
        ):
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
                import sys
                import traceback

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
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/add_component_concept", tags=["kb_ops"])
async def add_component_concept(
    user_id: str,  # Required field
    component_concept_name: str,  # Required field
    concept_name: str,  # Required field
    component_max_retrieval_distance: str = "0",
    streaming: str = "false",
):
    if streaming == "true":
        time_start = time.time()

        async def streamer(
            user_id,
            component_concept_name,
            concept_name,
            component_max_retrieval_distance,
        ):
            yield "status: Adding component concept..."
            try:
                result = app.state.agentmanager.add_component_concept(
                    user_id,
                    component_concept_name,
                    concept_name,
                    float(component_max_retrieval_distance),
                    streaming=streaming,
                )
                logger.info(
                    str("Add component concept time: " + str(time.time() - time_start))
                )
                yield f"result: {result}"
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(
                user_id,
                component_concept_name,
                concept_name,
                component_max_retrieval_distance,
            ),
            media_type="text/event-stream",
        )
    else:
        try:
            result = app.state.agentmanager.add_component_concept(
                user_id,
                component_concept_name,
                concept_name,
                float(component_max_retrieval_distance),
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/add_examples", tags=["kb_ops"])
async def add_examples(
    user_id: str,  # Required field
    concept_name: str,  # Required field
    images: List[UploadFile] = File(...),  # Required field
    streaming: str = "false",
):

    images = [Image.open(image.file).convert("RGB") for image in images]
    if streaming == "true":

        async def streamer(user_id, concept_name, imgs):
            yield "status: Adding examples..."
            try:
                result = app.state.agentmanager.add_examples(
                    user_id, imgs, concept_name, streaming=streaming
                )
                for msg in result:
                    yield msg
            except Exception as e:
                import sys
                import traceback

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
            result = app.state.agentmanager.add_examples(user_id, concept_name, images)
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/add_concept_negatives", tags=["kb_ops"])
async def add_concept_negatives(
    user_id: str,  # Required field
    concept_name: str,  # Required field
    images: List[UploadFile] = File(...),  # Required field
    streaming: str = "false",
):

    negatives = [Image.open(negative.file).convert("RGB") for negative in images]
    if streaming == "true":

        async def streamer(user_id, concept_name, imgs):
            yield "status: Adding concept negatives..."
            try:
                result = app.state.agentmanager.add_concept_negatives(
                    user_id, imgs, concept_name, streaming=streaming
                )
                for msg in result:
                    yield msg
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(user_id, concept_name, negatives),
            media_type="text/event-stream",
        )
    else:
        try:
            negatives = [
                Image.open(negative.file).convert("RGB") for negative in negatives
            ]
            result = app.state.agentmanager.add_concept_negatives(
                user_id,
                negatives,
                concept_name,
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

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
                import sys
                import traceback

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
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/train_concepts", tags=["train"])
async def train_concepts(
    user_id: str,  # Required field
    concepts: str,
    streaming: str = "false",
):
    if streaming == "true":

        async def streamer(user_id, cncpts):
            yield "status: \nTraining concepts...\n"
            try:
                time_start = time.time()
                for res in app.state.agentmanager.train_concepts(user_id, cncpts):
                    yield res
                logger.info(
                    str("Train concepts time: " + str(time.time() - time_start))
                )
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(user_id, concepts.split(", ")),
            media_type="text/event-stream",
        )
    else:
        time_start = time.time()
        try:
            await app.state.agentmanager.train_concepts(user_id, concepts)
            logger.info(str("Train concepts time: " + str(time.time() - time_start)))
            return JSONResponse(content={"status": "success"})
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


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
    filename = f"{uid}.json"
    return FileResponse(
        os.path.join(TENSOR_DIR, filename), media_type="application/octet-stream"
    )
