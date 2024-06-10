# %%
import sys
from asyncio import futures
from calendar import c

sys.path.append("/shared/nas2/knguye71/ecole-june-demo/ecole_mo9_demo/src")
import gc
import logging
import os
import sys
# import base64
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

# from feature_extraction.trained_attrs import N_ATTRS_SUBSET
import PIL.Image
import torch
from controller import Controller
from kb_ops import ConceptKBTrainer
from kb_ops.caching.cacher import ConceptKBFeatureCacher
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
from kb_ops.retrieve import CLIPConceptRetriever
from model.concept import Concept, ConceptExample, ConceptKB
from PIL import Image

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

    def train_concepts(
        self, concept_names: list[str], **train_concept_kwargs
    ) -> ConceptKB:

        # Ensure features are prepared, only generating those which don't already exist or are dirty
        # Cache all concepts, since we might sample from concepts whose examples haven't been cached yet
        logger.info(f"Training concepts over here: {concept_names}")
        print(f"Training concepts over here: {concept_names}")
        concepts = self.get_markov_blanket(concept_names)

        for concept in concepts:
            try:
                logger.info(f"Training concept: {concept.name}\n\n")
                print(f"Training concept: {concept.name}\n\n")
                self.train_concept(concept.name, **train_concept_kwargs)
                # self._train_concept_wrapper(concept.name, **train_concept_kwargs)
            except Exception as e:
                print("+++++++++++++++++++++++++++++++++++++++++++++++")
                traceback.print_exc()
                print(sys.exc_info())
                print(f"Error training concepts: {e}")
                print("concept device", concept.predictor.device)
                print(
                    "controller device",
                    self.feature_pipeline.feature_extractor.clip.device,
                )
                print("concept_kb device", {c.name: c.predictor.device for c in self.concept_kb.concepts})
                print("cuda device", torch.cuda.current_device())
                print("cuda device name", torch.cuda.get_device_name())
                print("cuda device count", torch.cuda.device_count())
                print("+++++++++++++++++++++++++++++++++++++++++++++++")

                raise e
        logger.info("Training complete.")
        logger.info(
            "current_cuda_device",
            torch.cuda.current_device(),
            torch.cuda.get_device_name(),
        )

        return self.move_to_cpu_and_return_concept_kb()

    def _train_concept_wrapper(
        self,
        concept_name: str,
        stopping_condition: Literal["n_epochs"] = "n_epochs",
        n_epochs: int = 5,
        max_retrieval_distance=0.01,
        use_concepts_as_negatives: bool = True,
    ):
        logger.info(f"Training concept: {concept_name}")

        # Try to retrieve concept
        concept = self.retrieve_concept(
            concept_name,
            max_retrieval_distance=max_retrieval_distance,
        )  # Low retrieval distance to force exact match

        logger.info(f'Retrieved concept with name: "{concept.name}"')
        print("concept cuda", concept.predictor.device)
        print("controller cuda", self.feature_pipeline.feature_extractor.clip.device)
        # Hook to recache zs_attr_features after negative examples have been sampled
        # This is faster than calling recache_zs_attr_features on all examples in the concept_kb
        def cache_hook(examples):
            self.cacher.recache_zs_attr_features(concept, examples=examples)

            if (
                not self.use_concept_predictors_for_concept_components
            ):  # Using fixed scores for concept-image pairs
                self.cacher.recache_component_concept_scores(concept, examples=examples)

        if stopping_condition == "n_epochs" or len(self.concept_kb) <= 1:
            if len(self.concept_kb) == 1:
                logger.info(
                    f"No other concepts in the ConceptKB; training concept in isolation for {n_epochs} epochs."
                )

            self.trainer.train_concept(
                concept,
                concepts = [concept],
                stopping_condition="n_epochs",
                n_epochs=n_epochs,
                post_sampling_hook=cache_hook,
                lr=1e-2,
                use_concepts_as_negatives=use_concepts_as_negatives,
            )

    def is_concept_in_image(
        self, image: Image, concept_name: str, unk_threshold: float = 0.1
    ) -> bool:
        return self.heatmap(image, concept_name, only_score_increasing_regions=True)

    def add_examples(self, examples: list[ConceptExample], concept_name: str | None = None, concept: Concept = None) -> ConceptKB:
        super().add_examples(examples, concept_name, concept)
        return self.move_to_cpu_and_return_concept_kb()

    def add_concept_examples(
        self,
        examples: list[ConceptExample],
        concept_name: str = None,
    ):
        try:
            concept = self.retrieve_concept(concept_name, max_retrieval_distance=0.1)
        except Exception as e:
            logger.info("Concept not found in KB. Adding new concept.")
            concept = self.add_concept(concept_name, concept = None)

        for example in examples:
            example.concept_name = concept.name

        self.add_examples(examples, concept=concept, concept_name=None)
        return self.move_to_cpu_and_return_concept_kb()

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

        return self.move_to_cpu_and_return_concept_kb()

    #     ############################
    #     # Functionality for KB Ops #
    #     ###########################

    def load_kb(self, concept_kb):
        cache_dir = FEATURE_CACHE_DIR + str(time.time())
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
        super().__init__(concepts=concept_kb.concepts, clip_model=self.clip_model, clip_processor=self.clip_processor)


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
