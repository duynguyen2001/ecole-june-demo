import concurrent.futures
import gc
import logging
import multiprocessing
import os
import pickle
import sys
import traceback
from multiprocessing import Process, Queue
from typing import Any, List

import PIL.Image
import torch
from ExtendedClasses import (DEFAULT_CKPT, Concept, ConceptExample,
                             ConceptKBFeaturePipeline,
                             ExtendedCLIPConceptRetriever, ExtendedController)
from PIL import Image

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
JSON_DIR = os.environ.get(
    "JSON_DIR", "/shared/nas2/knguye71/ecole-june-demo/json_dir"
)
VIDEO_DIR = os.environ.get(
    "VIDEO_DIR", "/shared/nas2/knguye71/ecole-june-demo/video_dir"
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

import asyncio
import threading
import time
import uuid
from io import BytesIO

from feature_extraction import build_feature_extractor, build_sam
from feature_extraction.trained_attrs import N_ATTRS_DINO
from image_processing import build_localizer_and_segmenter
from model.concept import (ConceptExample, ConceptKB, ConceptKBConfig,
                           concept_kb)
from streaming_methods import streaming_concept_kb


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
        retriever = ExtendedCLIPConceptRetriever(
            default_kb.concepts, feature_extractor.clip, feature_extractor.processor
        )
        logger.info(str("Models initialized"))

        models = {
            "sam": sam,
            "loc_and_seg": loc_and_seg,
            "retriever": retriever,
            "feature_extractor": feature_extractor,
            "controller": controller,
        }
        return models

    def model_process(
        self, input_queue: Queue, output_queue: Queue, device="cpu"
    ) -> None:

        models = self.initialize_models(device)

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
                logger.error(sys.exc_info())
                logger.error(traceback.format_exc())
                output_queue.put((task_id, "error", str(e)))
                raise e

    def __init__(self, device) -> None:
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.process = Process(
            target=self.model_process,
            args=(self.input_queue, self.output_queue, device),
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
                    logger.error(sys.exc_info())
                    logger.error(traceback.format_exc())
                    raise Exception(f"Error processing task {task_id}: {result[2]}")
                else:
                    logger.error(sys.exc_info())
                    logger.error(traceback.format_exc())
                    raise Exception(
                        f"Unexpected status for task {task_id}: {result[1]}"
                    )
            else:
                # Put the result back into the queue
                self.output_queue.put(result)

    async def call_async(self, model_key, func_name, *args, **kwargs) -> Any:
        task_id = str(uuid.uuid4())
        status = "processing"

        self.input_queue.put((task_id, status, model_key, func_name, args, kwargs))

        while True:
            result = (
                self.output_queue.get()
            )
            if result[0] == task_id:
                if result[1] == "done":
                    yield "result: " + str(result[2]) + "\n\n"
                elif result[1] == "error":
                    logger.error(sys.exc_info())
                    logger.error(traceback.format_exc())
                    raise Exception(f"Error processing task {task_id}: {result[2]}")
                elif result[1] == "status":
                    yield "status: " + str(result[2]) + "\n\n"
                else:
                    logger.error(sys.exc_info())
                    logger.error(traceback.format_exc())
                    raise Exception(
                        f"Unexpected status for task {task_id}: {result[1]}"
                    )
            else:
                # Put the result back into the queue
                self.output_queue.put(result)

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
            f"agent{idx}": Agent(f"cuda:{gpuid}")
            for idx, gpuid in enumerate(agent_gpu_list)
        }
        # Initialize a round-robin queue for distributing tasks
        self.round_robin_queue = deque(self.agents.keys())
        self.concept_kb_dir = concept_kb_dir
        self.default_ckpt = default_ckpt
        self.cache_kb = {}
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

    ###################################
    # Tasks related to the controller #
    ###################################

    def executeControllerFunctionNoSave(
        self, user_id: str, func, *args, **kwargs
    ) -> Any:
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
            concept_kb = self.get_concept_kb(user_id, self.agents[agent_key].device, get_temp=False)

            # Load user KB first
            self.agents[agent_key].call("controller", "load_kb", concept_kb)

            # remove keyword temp if it is
            if "temp" in kwargs:
                del kwargs['temp']

            # run and return result
            return self.agents[agent_key].call("controller", func, *args, **kwargs)
        except Exception as e:

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise e
        finally:
            if concept_kb is not None:
                for concept in concept_kb:
                    concept.predictor.to("cpu")
                gc.collect()  # Explicitly call garbage collector
                # Optionally, you can clear the unused memory from the GPU cache
                torch.cuda.empty_cache()

    def executeControllerFunctionWithSave(
        self, user_id: str, func, *args, **kwargs
    ) -> str:
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
        temp = kwargs.get("temp", False)
        try:
            # load concept kb by user_id
            concept_kb = self.get_concept_kb(user_id, self.agents[agent_key].device)
            # Load user KB first
            self.agents[agent_key].call("controller", "load_kb", concept_kb)

            # remove keyword temp if it is
            if "temp" in kwargs:
                del kwargs["temp"]

            # run and return result
            concept_kb = self.agents[agent_key].call(
                "controller", func, *args, **kwargs
            )

            # move concept kb to cpu
            for concept in concept_kb:
                concept.predictor.to("cpu")
            ckpt_path = self.save_concept_kb(user_id, concept_kb, temp)
            return ckpt_path
        except Exception as e:
            # General exception handling for any unexpected errors
            logger.error(sys.exc_info())
            logger.error(traceback.format_exc())
            # raise Exception(f"Error in executeControllerFunctionWithSave: {str(e)}")
            raise e
        finally:
            gc.collect()  # Explicitly call garbage collector
            # Optionally, you can clear the unused memory from the GPU cache
            torch.cuda.empty_cache()

    def executeRetrieverFunction(self, user_id: str, func, *args, **kwargs) -> Any:
        """
        Execute a function in the retriever.

        Args:
            func (): Function to execute

        Raises:
            Exception: Error in executeRetrieverFunction

        Returns:
            Any: Function result
        """
        agent_key = self.get_next_agent_key()
        concept_kb = self.get_concept_kb(user_id, self.agents[agent_key].device)
        try:
            self.agents[agent_key].call("retriever", "load_kb", concept_kb)
            result = self.agents[agent_key].call("retriever", func, *args, **kwargs)

            for concept in concept_kb:
                concept.predictor.to("cpu")
            return result
        except Exception as e:

            logger.error(sys.exc_info())
            logger.error(traceback.format_exc())
            # raise Exception(f"Error in executeRetrieverFunction: {str(e)}")
            raise e

    ##################
    # ConceptKB ops  #
    ##################
    def list_concept_dir(self, user_id: str, get_temp: bool = True) -> list[str]:
        """
        List the concept knowledge base (KB) checkpoint files for a user.

        Args:
            user_id (str): User ID

        Returns:
            list[str]: List of checkpoint paths
        """
        # Move concept kb to device
        user_dir = f"{self.concept_kb_dir}/{user_id}"
        os.makedirs(user_dir, exist_ok=True)

        # get all checkpoint in user_id and get the last checkpoint is the lastname in the list
        if get_temp:
            files = [path for path in os.listdir(user_dir) if path.endswith(".pt")]
        else:
            files = [
                path for path in os.listdir(user_dir) if path.endswith(".pt") and "_temp" not in path
            ]
        files.sort(reverse= True)
        return files

    def get_concept_kb(self, user_id: str, device: str = "cpu", get_temp = True) -> ConceptKB:
        """
        This method gets the concept knowledge base (KB) for a user.

        Args:
            user_id (str): User ID
            device (str, optional): Device to load the concept KB. Defaults to "cpu".

        Returns:
            ConceptKB: Concept knowledge base
        """
        logger.info(str(f"Getting concept KB for user {user_id}"))

        # if user_id in self.checkpoint_path_dict:
        #     concept_kb_path = self.checkpoint_path_dict[user_id][-1]
        #     if concept_kb_path in self.cache_kb:
        #         concept_kb = self.cache_kb[concept_kb_path]
        #     else:
        #         concept_kb = ConceptKB.load(concept_kb_path)
        # else:
        #     # load default concept kb
        #     concept_kb = ConceptKB.load(self.default_ckpt)
        #     # save default concept kb
        #     os.makedirs(f"{self.concept_kb_dir}/{user_id}", exist_ok=True)
        #     checkpoint_path = (
        #         f"{self.concept_kb_dir}/{user_id}/concept_kb_epoch_{time.time()}.pt"
        #     )
        #     concept_kb.save(checkpoint_path)
        #     self.checkpoint_path_dict[user_id] = [checkpoint_path]
        #     self.cache_kb[checkpoint_path] = concept_kb

        # if len(self.cache_kb) > 10:
        #     self.cache_kb.pop(list(self.cache_kb.keys())[0])

        files = self.list_concept_dir(user_id, get_temp)
        logger.info(f"{files}")
        if len(files) == 0:
            concept_kb = ConceptKB.load(self.default_ckpt)
            # save default concept kb
            self.save_concept_kb(user_id, concept_kb)
        else:
            full_path = os.path.join(self.concept_kb_dir, user_id, files[0])
            concept_kb = ConceptKB.load(full_path)

        for concept in concept_kb:
            concept.predictor.to(device)
        return concept_kb

    def save_concept_kb(self, user_id: str, concept_kb: ConceptKB, temp = False) -> str:
        """
        Save the concept knowledge base (KB) for a user.

        Args:
            user_id (str): User ID
            concept_kb (ConceptKB): Concept knowledge base

        Returns:
            str: Checkpoint path
        """
        os.makedirs(f"{self.concept_kb_dir}/{user_id}", exist_ok=True)
        if temp:
            checkpoint_path = (
                f"{self.concept_kb_dir}/{user_id}/concept_kb_epoch_{time.time()}_temp.pt"
            )
        else:
            checkpoint_path = (
                f"{self.concept_kb_dir}/{user_id}/concept_kb_epoch_{time.time()}.pt"
            )

        for concept in concept_kb:
            concept.predictor.to("cpu")
        concept_kb.save(checkpoint_path)
        if user_id not in self.checkpoint_path_dict:
            self.checkpoint_path_dict[user_id] = [checkpoint_path]
        self.checkpoint_path_dict[user_id].append(checkpoint_path)
        self.cache_kb[checkpoint_path] = concept_kb
        return checkpoint_path

    def undo_kb(self, user_id: str) -> str:
        """
        Undo the last checkpoint of the concept knowledge base (KB) for a user.

        Args:
            user_id (str): User ID

        Returns:
            str: Checkpoint path
        """
        # if (
        #     user_id in self.checkpoint_path_dict
        #     and len(self.checkpoint_path_dict[user_id]) > 1
        # ):
        #     checkpoint_path = self.checkpoint_path_dict[user_id][-2]
        #     if os.path.exists(checkpoint_path):
        #         os.remove(self.checkpoint_path_dict[user_id][-1])
        #     self.checkpoint_path_dict[user_id].pop()
        #     self.cache_kb[user_id] = ConceptKB.load(checkpoint_path)
        #     return checkpoint_path
        # else:
        #     return "No checkpoint to undo"

        files = self.list_concept_dir(user_id, get_temp=True)
        # find the second last checkpoint but not temp checkpoint, then remove all checkpoint before it
        count, index = 0, 0
        if len(files)  == 0:
            return "No checkpoint to undo"
        else:
            for file in files:
                if "_temp" not in file:
                    count += 1
                    if count == 2:
                        break
                index += 1
            for i in range(index):
                if os.path.exists(f"{self.concept_kb_dir}/{user_id}/{files[i]}"):
                    os.remove(f"{self.concept_kb_dir}/{user_id}/{files[i]}")

            return f"{self.concept_kb_dir}/{user_id}/{files[index]}"

    def retrieve_concept(
        self, user_id: str, concept_name: str, max_retrieval_distance: float = 0.5
    ):
        """ """
        concept_name = concept_name.strip()
        concept_kb = self.get_concept_kb(user_id)
        if concept_name in concept_kb:
            # move concept kb to cpu
            for concept in concept_kb:
                concept.predictor.to("cpu")
            return concept_kb[concept_name]

        elif concept_name.lower() in concept_kb:
            return concept_kb[concept_name.lower()]

        else:
            try:

                retrieved_concept = self.executeRetrieverFunction(
                    user_id, "retrieve", concept_name, 1
                )[0]
                logger.info(
                    f'Retrieved concept "{retrieved_concept.concept.name}" with distance: {retrieved_concept.distance}'
                )
                if retrieved_concept.distance > max_retrieval_distance:
                    logger.info(
                        f"Retrieved concept distance {retrieved_concept.distance} is greater than max retrieval distance {max_retrieval_distance}. Add new concept to KB."
                    )
                return retrieved_concept.concept
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                # raise Exception(f"Error in retrieve_concept: {str(e)}")
                raise e

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
            temp = True
        )
        if streaming:
            return f"Hyponym {child_name} added to {parent_name}\n\n"
        else:
            return {"status": "success"}
        # return convert_to_serializable(result)

    async def add_concept_negatives(
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
        self.executeControllerFunctionWithSave(
            user_id,
            "add_concept_negatives",
            concept_name,
            concept_examples,
            temp = True
        )
        if streaming:
            yield f"status: Add negative examples successfully time: {time.time() - time_start}\n\n"
            yield f"result: {len(concept_examples)} negative examples added to concept {concept_name}\n\n"

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
            temp = True
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
            concept_kb = ConceptKB.load(self.default_ckpt)
            concept_kb._concepts = {}
        else:
            concept_kb = ConceptKB.load(self.default_ckpt)

        self.save_concept_kb(user_id, concept_kb)
        if streaming:
            return f"Knowledge base reset {'from scratch' if clear_all else 'default checkpoint'}\n\n"
        else:
            return {"status": "success"}

    def load_checkpoint(
        self, user_id: str, checkpoint_id: str, streaming: bool = False
    ):
        """
        Load a checkpoint for a user.

        """
        checkpoint_path_of_user = None
        for checkpoint_path in self.checkpoint_path_dict.get(user_id, []):
            if checkpoint_id in checkpoint_path:
                concept_kb = ConceptKB.load(checkpoint_path)
                checkpoint_path_of_user = self.save_concept_kb(user_id, concept_kb)

        if checkpoint_path_of_user is None:
            raise Exception(f"Checkpoint {checkpoint_id} not found")

        if streaming:
            yield f"Checkpoint {checkpoint_path_of_user} loaded\n\n"

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

            path = os.path.join(save_log_and_seg_concept_dir, new_id + ".pkl")
            with open(path, "wb") as f:
                pickle.dump(loc_seg_output.cpu(), f)
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

    async def add_examples(
        self,
        user_id: str,
        images: list[PIL.Image.Image],
        concept_name: str,
        streaming=False,
    ):

        if concept_name is None:
            RuntimeError("Concept name is required")
        time_start = time.time()
        concept_examples = self._loc_and_seg_multiple_images(images, concept_name)

        print(f"concept_examples: {concept_examples}")
        print(f"concept_name: {concept_name}")

        self.executeControllerFunctionWithSave(
            user_id,
            "add_concept_examples",
            examples=concept_examples,
            concept_name=concept_name,
            temp = True
        )

        if streaming:
            yield f"status: Added examples in {(time.time() - time_start): .2f}s\n\n"
            yield f"result: {len(concept_examples)} examples added to concept {concept_name}\n\n"

    async def train_concepts(
        self,
        user_id: str,
        concept_names: list[str],
        streaming: bool = False,
        **train_concept_kwargs,
    ):
        time_start = time.time()
        logger.info(f"\n\nconcept_names: {concept_names}\n\n")
        try:
            if streaming:
                # for msg in streaming_concept_kb(self.get_concept_kb(user_id, get_temp=True)):
                #     yield msg
                yield f"status: Training concepts...\n\n"

                # Start the status thread
                running_thread = threading.Thread(target=self.executeControllerFunctionWithSave, args=(user_id, "train_concepts", concept_names,), kwargs=train_concept_kwargs)

                try:
                    running_thread.start()
                    while running_thread.is_alive():
                        yield "status: "
                        await asyncio.sleep(1)
                finally:
                    running_thread.join()

                yield f"status: Training completed\n\n"
                yield f"status: Total time for training  concepts: {time.time() - time_start}"
                yield f"result: Trained concept(s) successfully\n\n"
        except Exception as e:

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            # raise Exception(f"Error in train_concepts: {str(e)}")
            raise e

    def get_zs_attributes(self, user_id: str, concept_name: str):
        return self.executeControllerFunctionNoSave(
            user_id, "get_zs_attributes", concept_name
        )

    def heatmap_image_comparison(
        self, user_id: str, image1: PIL.Image.Image, image2: PIL.Image.Image
    ):
        result = self.executeControllerFunctionNoSave(
            user_id, "heatmap_image_comparison", image1, image2
        )
        return result


    def heatmap_class_difference(
        self,
        user_id: str,
        concept1_name: str,
        concept2_name: str,
        image: PIL.Image.Image = None,
    ) -> dict:
        logger.info(f"heatmap_class_difference: {concept1_name} - {concept2_name}")
        logger.info(f"heatmap_class_difference: {image}")

        rst = {}
        concept1_part_names, concept2_part_names = self.executeControllerFunctionNoSave(
            user_id, "compare_component_concepts", concept1_name, concept2_name
        )
        rst["concept1_part_names"] = concept1_part_names
        rst["concept2_part_names"] = concept2_part_names

        if concept1_name and concept2_name:
            rst["concept1"] = self.retrieve_concept(user_id, concept1_name)
            rst["concept2"] = self.retrieve_concept(user_id, concept2_name)

            if rst["concept1"].component_concepts and rst["concept2"].component_concepts:
                return rst

        result = self.executeControllerFunctionNoSave(
            user_id, "heatmap_class_difference", concept1_name, concept2_name, image
        )
        result.update(rst)

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
