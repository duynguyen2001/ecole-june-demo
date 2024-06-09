# %%
import sys

sys.path.append("/shared/nas2/knguye71/ecole-june-demo/ecole_mo9_demo/src")
import gc
import logging
import os
import sys
# import base64
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

# from feature_extraction.trained_attrs import N_ATTRS_SUBSET
import PIL.Image
import torch
from controller import Controller
from feature_extraction.trained_attrs import N_ATTRS_DINO
from kb_ops import ConceptKBPredictor, ConceptKBTrainer
from kb_ops.caching.cacher import ConceptKBFeatureCacher
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
# from typing import Union, Literal
# from llm import LLMClient
# from score import AttributeScorer
from kb_ops.retrieve import CLIPConceptRetriever
from model.concept import Concept, ConceptExample, ConceptKB, ConceptKBConfig
from PIL import Image

# import torch.nn as nn
# import numpy as np
# from rembg import remove
# from typing import List, Dict, Union
# from feature_extraction.dino_features import get_rescaled_features, rescale_features
# import concurrent.futures

# logger = logging.getLogger("uvicorn.error")
# DEFAULT_CKPT = os.environ.get(
#     "DEFAULT_CKPT",
#     "/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_06_05-20:23:53-yd491eo3-all_planes_and_guns-infer_localize/concept_kb_epoch_50.pt",
# )
DEFAULT_CKPT = "/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_06_06-23:31:12-8ckp59v8-all_planes_and_guns/concept_kb_epoch_50.pt"
FEATURE_CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "./feature_cache/")
logger = logging.getLogger("uvicorn.error")
class ExtendedController(Controller):
    def move_to_cpu_and_return_concept_kb(self) -> ConceptKB:
        currentkb = self.concept_kb
        currentkb.to("cpu")
        return currentkb

    def add_hyponym(
        self,
        child_name: str,
        parent_name: str,
        child_max_retrieval_distance: float = 0.0,
    ) -> ConceptKB:
        super().add_hyponym(child_name, parent_name, child_max_retrieval_distance)
        return self.move_to_cpu_and_return_concept_kb()

    def add_component_concept(
        self,
        component_concept_name: str,
        concept_name: str,
        component_max_retrieval_distance: float = 0.0,
    ):
        super().add_component_concept(
            component_concept_name, concept_name, component_max_retrieval_distance
        )
        return self.move_to_cpu_and_return_concept_kb()

    def train_concepts(self, concept_names: list[str], **train_concept_kwargs) -> ConceptKB:
        concepts = self.get_markov_blanket(concept_names)
        # Initialize a pool of workers
        with ThreadPoolExecutor() as executor:
            for concept in concepts:
                executor.submit(self._train_concept_wrapper, concept.name, train_concept_kwargs)
        
        
        logger.info("Training complete.")
        logger.info(
            "current_cuda_device",
            torch.cuda.current_device(),
            torch.cuda.get_device_name(),
        )

        return self.move_to_cpu_and_return_concept_kb()

    def _train_concept_wrapper(self, concept_name, train_concept_kwargs):
        logger.info(f"Training concept: {concept_name}")
        self.train_concept(concept_name, **train_concept_kwargs)

    def add_examples(self, examples: list[ConceptExample], concept_name: str = None, concept: Concept = None):
        super().add_examples(examples, concept_name, concept)
        return self.move_to_cpu_and_return_concept_kb()

    def add_concept_negatives(self, concept_name: str, negatives: list[ConceptExample]) -> ConceptKB:
        super().add_concept_negatives(concept_name, negatives)
        return self.move_to_cpu_and_return_concept_kb()

    #     ############################
    #     # Functionality for KB Ops #
    #     ###########################

    def load_kb(self, concept_kb):
        cache_dir = FEATURE_CACHE_DIR  + str(
                    time.time()
                )
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cacher = ConceptKBFeatureCacher(
            concept_kb, self.feature_pipeline, cache_dir=cache_dir
        )
        model, processor = (
            self.feature_pipeline.feature_extractor.clip,
            self.feature_pipeline.feature_extractor.processor,
        )
        retriever = CLIPConceptRetriever(concept_kb.concepts, model, processor)
        self.__init__(
            concept_kb,
            self.feature_pipeline,
            retriever = retriever,
            cacher = cacher,
        )
        # Free up memory
        gc.collect()
        torch.cuda.empty_cache()


# %%


if __name__ == "__main__":

    import controller as ctrl
    from feature_extraction import build_feature_extractor, build_sam
    from image_processing import build_localizer_and_segmenter

    img_path = "/shared/nas2/knguye71/ecole-june-demo/samples/DemoJune2024-2/cargo_jet/000001.jpg"
    ckpt_path = "/shared/nas2/knguye71/ecole-june-demo/conceptKB_ckpt/05b09a5e2bdbc93d04059ecf2741478f0e01df72c8f9f88c2dd1ef0084bfc90f/concept_kb_epoch_1717755420.4449189.pt"

    # %%
    kb = ConceptKB.load(ckpt_path)
    loc_and_seg = build_localizer_and_segmenter(
        build_sam(), None
    )
    fe = build_feature_extractor()
    feature_pipeline = ctrl.ConceptKBFeaturePipeline(loc_and_seg, fe)

    controller = ExtendedController(kb, feature_pipeline)
    # %%

    img = PIL.Image.open(img_path).convert("RGB")
    result = controller.predict_concept(img, unk_threshold=0.1)


    # %%

    result_2 = controller.predict_hierarchical(img, unk_threshold=0.1)

    result_2
