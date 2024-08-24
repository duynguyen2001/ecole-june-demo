# %%
import sys
from unittest.mock import DEFAULT

import torch.multiprocessing as mp

sys.path.append("/shared/nas2/knguye71/ecole-june-demo/ecole_mo9_demo/src")
import copy
import cProfile
import gc
import logging
import os
import sys
# import base64
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional

# from feature_extraction.trained_attrs import N_ATTRS_SUBSET
import PIL.Image
import torch
from controller import Controller
from controller.train.train_parallel import ConcurrentTrainingConceptSelector
from kb_ops import ConceptKBTrainer
from kb_ops.caching.cacher import ConceptKBFeatureCacher
from kb_ops.concurrency.locking.lock_generators import LockType
from kb_ops.dataset import FeatureDataset, extend_with_global_negatives
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
from kb_ops.retrieve import CLIPConceptRetriever
from model.concept import Concept, ConceptExample, ConceptKB
from PIL import Image

torch.autograd.set_detect_anomaly(True)
# if mp.get_start_method(allow_none=True) != "spawn":
#     mp.set_start_method("spawn", force=True)
# DEFAULT_CKPT = os.environ.get(
#     "DEFAULT_CKPT",
#     "/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_06_05-20:23:53-yd491eo3-all_planes_and_guns-infer_localize/concept_kb_epoch_50.pt",
# )
# DEFAULT_CKPT = "/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_06_06-23:31:12-8ckp59v8-all_planes_and_guns/concept_kb_epoch_50.pt"
# DEFAULT_CKPT = "/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_23-02:35:55-all_planes_and_guns_v3-rm_bg_with_component_rm_bg_containing_positives/concept_kb_epoch_487.pt"
DEFAULT_CKPT = "/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_07_31-13:35:28-0aeepf7x-all_planes_and_guns_v4/concept_kb_epoch_496.pt"
CACHE_DIR = "/shared/nas2/knguye71/ecole-june-demo/cache"
logger = logging.getLogger("uvicorn.access")

class ExtendedController(Controller):

    def predict_hierarchical(
        self,
        image: Image,
        unk_threshold: float = 0.1,
        include_component_concepts: bool = False,
    ) -> list[dict]:
        prediction_path: list[PredictOutput] = self.predictor.hierarchical_predict(
            image_data=image,
            unk_threshold=unk_threshold,
            include_component_concepts=include_component_concepts,
        )

        # Get heatmap visualizations for each concept components in the prediction path
        for pred in prediction_path:
            component_concepts = pred.predicted_concept_components_to_scores.keys()
            pred.predicted_concept_components_heatmaps = {
                concept: self.heatmap(image, concept, return_detection_score=True) for concept in component_concepts
            }
            pred.concept_heatmap = self.heatmap(image, pred.predicted_label, return_detection_score=True)

        if prediction_path[-1]["is_below_unk_threshold"]:
            predicted_label = (
                "unknown"
                if len(prediction_path) == 1
                else prediction_path[-2]["predicted_label"]
            )
            concept_path = [pred["predicted_label"] for pred in prediction_path[:-1]]
        else:
            predicted_label = prediction_path[-1]["predicted_label"]
            concept_path = [pred["predicted_label"] for pred in prediction_path]

        return {
            "prediction_path": prediction_path,  # list[PredictOutput]
            "concept_path": concept_path,  # list[str]
            "predicted_label": predicted_label,  # str
        }

    def predict_from_subtree(
        self, image: Image, root_concept_name: str, unk_threshold: float = 0.1
    ) -> list[dict]:
        root_concept = self.retrieve_concept(root_concept_name)
        prediction_path: list[PredictOutput] = self.predictor.hierarchical_predict(
            image_data=image, root_concepts=[root_concept], unk_threshold=unk_threshold
        )
        # Get heatmap visualizations for each concept components in the prediction path
        for pred in prediction_path:
            component_concepts = pred.predicted_concept_components_to_scores.keys()
            pred.predicted_concept_components_heatmaps = {
                concept: self.heatmap(image, concept) for concept in component_concepts
            }

        if prediction_path[-1]["is_below_unk_threshold"]:
            predicted_label = (
                "unknown"
                if len(prediction_path) == 1
                else prediction_path[-2]["predicted_label"]
            )
            concept_path = [pred["predicted_label"] for pred in prediction_path[:-1]]
        else:
            predicted_label = prediction_path[-1]["predicted_label"]
            concept_path = [pred["predicted_label"] for pred in prediction_path]

        return {
            "prediction_path": prediction_path,  # list[PredictOutput]
            "concept_path": concept_path,  # list[str]
            "predicted_label": predicted_label,  # str
        }

    def add_hyponym(
        self,
        child_name: str,
        parent_name: str,
        child_max_retrieval_distance: float = 0.0,
    ) -> ConceptKB:
        print(f"Adding hyponym: {child_name} to parent: {parent_name}")
        super().add_hyponym(child_name, parent_name, child_max_retrieval_distance)

        return self.concept_kb.cpu()

    def add_component_concept(
        self,
        component_concept_name: str,
        concept_name: str,
        component_max_retrieval_distance: float = 0.0,
    ):
        super().add_component_concept(
            component_concept_name, concept_name, component_max_retrieval_distance
        )
        return self.concept_kb.cpu()

    def train_concepts(
        self,
        concept_names: list[str],
        n_epochs: int = 100,
        use_concepts_as_negatives: bool = True,
        max_retrieval_distance=0.01,
        concepts_to_train_kwargs: dict = {},
        **train_concept_kwargs,
    ) -> Optional[ConceptKB]:
        time_start = time.time()
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            AGENT_GPU_LIST = list(range(num_gpus))
            self.train_concepts_parallel(
                concept_names=concept_names,
                n_epochs=n_epochs,
                use_concepts_as_negatives=use_concepts_as_negatives,
                max_retrieval_distance=max_retrieval_distance,
                concepts_to_train_kwargs=concepts_to_train_kwargs,
                devices=AGENT_GPU_LIST,
                lock_type=LockType.FILE_LOCK,
                **train_concept_kwargs,
            )

        print("Training complete.")

        print(f"Time to train concepts: {time.time() - time_start}")
        return self.concept_kb.cpu()

    def is_concept_in_image(
        self, image: Image, concept_name: str, unk_threshold: float = 0.1
    ) -> bool:
        return self.heatmap(image, concept_name, only_score_increasing_regions=True)

    def add_concept_examples(
        self,
        examples: list[ConceptExample],
        concept_name: str = None,
    ):
        try:
            concept = self.retrieve_concept(concept_name, max_retrieval_distance=0.1)
        except Exception as e:
            logger.info("Concept not found in KB. Adding new concept.")
            concept = self.add_concept(concept_name, concept=None)

        for example in examples:
            example.concept_name = concept.name

        self.add_examples(examples, concept=concept, concept_name=None)
        return self.concept_kb.cpu()

    def add_concept_negatives(
        self, concept_name: str, negatives: list[ConceptExample]
    ) -> ConceptKB:
        concept = None
        if concept is None:
            try:
                concept = self.retrieve_concept(concept_name)
            except Exception as e:
                logger.info("Concept not found in KB. Adding new concept.")
                concept = self.add_concept(concept_name)
        for example in negatives:
            example.concept_name = concept.name
        super().add_concept_negatives(concept.name, negatives)

        return self.concept_kb.cpu()

    #     ############################
    #     # Functionality for KB Ops #
    #     ###########################

    def load_kb(self, concept_kb):
        current_time = str(time.time())
        cache_dir = CACHE_DIR + "/feature_cache/" + current_time
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(cache_dir + "/segmentations", exist_ok=True)
        os.makedirs(cache_dir + "/features", exist_ok=True)
        cacher = ConceptKBFeatureCacher(
            concept_kb,
            self.feature_pipeline,
            cache_dir=cache_dir,
        )
        model, processor = (
            self.feature_pipeline.feature_extractor.clip,
            self.feature_pipeline.feature_extractor.processor,
        )
        retriever = CLIPConceptRetriever(concept_kb.concepts, model, processor)
        trainer = ConceptKBTrainer(concept_kb, self.feature_pipeline)
        self.__init__(
            concept_kb,
            self.feature_pipeline,
            retriever=retriever,
            cacher=cacher,
            trainer=trainer,
        )
        # Free up memory
        gc.collect()
        torch.cuda.empty_cache()


class ExtendedCLIPConceptRetriever(CLIPConceptRetriever):
    def load_kb(self, concept_kb: ConceptKB):
        super().__init__(
            concepts=concept_kb.concepts,
            clip_model=self.clip_model,
            clip_processor=self.clip_processor,
        )


# %%


if __name__ == "__main__":

    import controller as ctrl
    from feature_extraction import build_feature_extractor, build_sam
    from image_processing import build_localizer_and_segmenter

    img_path = "/shared/nas2/knguye71/ecole-june-demo/samples/DemoJune2024-2/cargo_jet/000001.jpg"
    ckpt_path = "/shared/nas2/knguye71/ecole-june-demo/conceptKB_ckpt/05b09a5e2bdbc93d04059ecf2741478f0e01df72c8f9f88c2dd1ef0084bfc90f/concept_kb_epoch_1717755420.4449189.pt"

    # %%
    kb = ConceptKB.load(ckpt_path)
    loc_and_seg = build_localizer_and_segmenter(build_sam(), None)
    fe = build_feature_extractor()
    feature_pipeline = ctrl.ConceptKBFeaturePipeline(loc_and_seg, fe)

    controller = ExtendedController(kb, feature_pipeline)
    # %%

    img = PIL.Image.open(img_path).convert("RGB")
    result = controller.predict_concept(img, unk_threshold=0.1)

    # %%

    result_2 = controller.predict_hierarchical(img, unk_threshold=0.1)

    result_2
