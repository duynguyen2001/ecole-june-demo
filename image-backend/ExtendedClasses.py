# %%
import sys

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
from controller.train import ConcurrentTrainingConceptSelector
from kb_ops import ConceptKBTrainer
from kb_ops.caching.cacher import ConceptKBFeatureCacher
from kb_ops.dataset import FeatureDataset, extend_with_global_negatives
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline
from kb_ops.retrieve import CLIPConceptRetriever
from model.concept import Concept, ConceptExample, ConceptKB
from PIL import Image

torch.autograd.set_detect_anomaly(True)
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)
# DEFAULT_CKPT = os.environ.get(
#     "DEFAULT_CKPT",
#     "/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_06_05-20:23:53-yd491eo3-all_planes_and_guns-infer_localize/concept_kb_epoch_50.pt",
# )
DEFAULT_CKPT = "/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_06_06-23:31:12-8ckp59v8-all_planes_and_guns/concept_kb_epoch_50.pt"
CACHE_DIR = "/shared/nas2/knguye71/ecole-june-demo/cache"
logger = logging.getLogger("uvicorn.access")


def train_wrapper(
    trainer: ConceptKBTrainer,
    concept,
    examples,
    dataset,
    ret_concept_queue: mp.Queue,
    n_epochs=5,
    use_concepts_as_negatives=True,
    device="cpu",
    **train_concept_kwargs,
):
    try:
        logger.info(f"Training concept: {concept.name}\n\n")
        print(f"Training concept: {concept.name}\n\n")
        # Hook to recache zs_attr_features after negative examples have been sampled
        # This is faster than calling recache_zs_attr_features on all examples in the concept_kb
        train_concept = copy.deepcopy(concept)
        train_concept.predictor.to(device)
        # Train for fixed number of epochs
        train_kwargs = train_concept_kwargs.copy()
        trainer.train_concept(
            train_concept,
            samples_and_dataset=(examples, dataset),
            n_epochs=n_epochs,
            **train_kwargs,
        )
        train_concept.predictor.to("cpu")
        ret_concept_queue.put((concept, train_concept))
        print(f"Training complete for concept: {concept.name}")
    except Exception as e:
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
        traceback.print_exc()
        print(sys.exc_info())
        print(f"Error training concepts: {e}")
        print("concept device", concept.predictor.device)
        print("cuda device", torch.cuda.current_device())
        print("cuda device name", torch.cuda.get_device_name())
        print("cuda device count", torch.cuda.device_count())
        print("+++++++++++++++++++++++++++++++++++++++++++++++")

        raise e
    finally:
        torch.cuda.empty_cache()


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

    def move_to_cpu_and_return_concept_kb(self) -> ConceptKB:
        for concept in self.concept_kb.concepts:
            concept.predictor.to("cpu")
        return self.concept_kb

    def add_hyponym(
        self,
        child_name: str,
        parent_name: str,
        child_max_retrieval_distance: float = 0.0,
    ) -> ConceptKB:
        print(f"Adding hyponym: {child_name} to parent: {parent_name}")
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

    # def train_concept(
    #     self,
    #     concept_name: str,
    #     stopping_condition: Literal["n_epochs"] = "n_epochs",
    #     new_examples: list[ConceptExample] = [],
    #     n_epochs: int = 5,
    #     max_retrieval_distance=0.01,
    #     use_concepts_as_negatives: bool = True,
    # ):
    #     """
    #     Trains the specified concept with name concept_name for the specified number of epochs.

    #     Args:
    #         concept_name: The concept to train. If it does not exist, it will be created.
    #         stopping_condition: The condition to stop training. Must be 'n_epochs'.
    #         new_examples: If provided, these examples will be added to the concept's examples list.
    #     """
    #     # Try to retrieve concept
    #     concept = self.retrieve_concept(
    #         concept_name, max_retrieval_distance=max_retrieval_distance
    #     )  # Low retrieval distance to force exact match
    #     logger.info(f'Retrieved concept with name: "{concept.name}"')

    #     # Hook to recache zs_attr_features after negative examples have been sampled
    #     # This is faster than calling recache_zs_attr_features on all examples in the concept_kb
    #     def cache_hook(examples):
    #         self.cacher.recache_zs_attr_features(concept, examples=examples)

    #         # Handle component concepts
    #         if self.use_concept_predictors_for_concept_components:
    #             for component in concept.component_concepts.values():
    #                 self.cacher.recache_zs_attr_features(
    #                     component, examples=examples
    #                 )  # Needed to predict the componnt concept

    #         else:  # Using fixed scores for concept-image pairs
    #             self.cacher.recache_component_concept_scores(concept, examples=examples)

    #     if stopping_condition == "n_epochs" or len(self.concept_kb) <= 1:
    #         if len(self.concept_kb) == 1:
    #             logger.info(
    #                 f"No other concepts in the ConceptKB; training concept in isolation for {n_epochs} epochs."
    #             )

    #         self.trainer.train_concept(
    #             concept,
    #             stopping_condition="n_epochs",
    #             n_epochs=n_epochs,
    #             post_sampling_hook=cache_hook,
    #             lr=1e-2,
    #             use_concepts_as_negatives=use_concepts_as_negatives,
    #         )

    #     else:
    #         raise ValueError("Unrecognized stopping condition")

    # def train_concepts(
    #     self, concept_names: list[str], **train_concept_kwargs
    # ) -> ConceptKB:

    #     # Ensure features are prepared, only generating those which don't already exist or are dirty
    #     # Cache all concepts, since we might sample from concepts whose examples haven't been cached yet
    #     logger.info(f"Training concepts over here: {concept_names}")
    #     print(f"Training concepts over here: {concept_names}")
    #     concepts = self.get_markov_blanket(concept_names)
    #     if len(concepts) == 0:
    #         raise ValueError("No concepts found in the ConceptKB.")

    #     # Ensure features are prepared, only generating those which don't already exist or are dirty
    #     # Cache all concepts, since we might sample from concepts whose examples haven't been cached yet
    #     self.cacher.cache_segmentations(only_uncached_or_dirty=True)
    #     self.cacher.cache_features(only_uncached_or_dirty=True)

    #     # Initialize multiprocessing
    #     processes = []
    #     train_concept_kwargs['device'] = concepts[0].predictor.device
    #     concept_kb = self.move_to_cpu_and_return_concept_kb()

    #     def cache_hook(concept, examples):
    #         self.cacher.recache_zs_attr_features(concept, examples=examples)

    #         # Handle component concepts
    #         if self.use_concept_predictors_for_concept_components:
    #             for component in concept.component_concepts.values():
    #                 self.cacher.recache_zs_attr_features(
    #                     component, examples=examples
    #                 )

    #         else:  # Using fixed scores for concept-image pairs
    #             self.cacher.recache_component_concept_scores(concept, examples=examples)
    #     trainer = self.trainer
    #     def get_train_dataset(concept):
    #         logger.info(f"Creating train dataset for concept: {concept.name}")
    #         return concept.name, create_train_ds(concept, concept_kb, **train_concept_kwargs)

    #     time_start = time.time()
    #     with ProcessPoolExecutor(max_workers=8) as executor:
    #         dict_train_ds = dict(executor.map(get_train_dataset, concepts))

    #     logger.info(f"Time to create train datasets: {time.time() - time_start}")

    #     for concept in concepts:
    #         logger.info(f"Training concept: {concept.name}, device: {concept.predictor.device}")
    #         cache_hook(concept, dict_train_ds[concept.name]['tot_samples'])

    #         p = mp.Process(
    #             target=train_wrapper,
    #             args=(concept_kb, concept, dict_train_ds[concept.name]['train_ds']),
    #             kwargs=train_concept_kwargs,
    #         )
    #         p.start()
    #         processes.append(p)

    #     for p in processes:
    #         p.join()

    #     for concept in concepts:
    #         self.concept_kb._concepts[concept.name] = concept

    #     logger.info("Training complete.")
    #     logger.info(f"Trained with the following concepts: {[concept.name for concept in concepts]}" )
    #     logger.info(
    #         "current_cuda_device",
    #         torch.cuda.current_device(),
    #         torch.cuda.get_device_name(),
    #     )

    #     return self.move_to_cpu_and_return_concept_kb()

    # def train_concepts(
    #     self,
    #     concept_names: list[str],
    #     n_epochs: int = 50,
    #     use_concepts_as_negatives: bool = True,
    #     max_retrieval_distance=0.01,
    #     **train_concept_kwargs,
    # ):
    #     time_start = time.time()
    #     super().train_concepts(
    #             concept_names,
    #             n_epochs=n_epochs,
    #             use_concepts_as_negatives=use_concepts_as_negatives,
    #             max_retrieval_distance=max_retrieval_distance,
    #             **train_concept_kwargs,
    #         )
    #     print(f"Time to train concepts: {time.time() - time_start}")
    #     return self.move_to_cpu_and_return_concept_kb()

    def train_concepts(
        self,
        concept_names: list[str],
        n_epochs: int = 50,
        use_concepts_as_negatives: bool = True,
        max_retrieval_distance=0.01,
        concepts_to_train_kwargs: dict = {},
        **train_concept_kwargs,
    ) -> Optional[ConceptKB]:
        time_start = time.time()
        # Ensure features are prepared, only generating those which don't already exist or are dirty
        # Cache all concepts, since we might sample from concepts whose examples haven't been cached yet
        self.cacher.cache_segmentations(only_uncached_or_dirty=True)
        self.cacher.cache_features(only_uncached_or_dirty=True)
        self.move_to_cpu_and_return_concept_kb()
        # TODO Add a variant that merges all of the datasets (merging duplicate examples using datasets' concepts_to_train field) and runs trainer.train()

        concepts = [
            self.retrieve_concept(c, max_retrieval_distance=max_retrieval_distance)
            for c in concept_names
        ]

        concepts_to_train = {}
        for concept in concepts:
            concepts_to_train.update(
                dict.fromkeys(
                    self._get_concepts_to_train_to_update_concept(
                        concept, **concepts_to_train_kwargs
                    )
                )
            )
        concept_selector = ConcurrentTrainingConceptSelector(list(concepts_to_train))
        p_list = []

        feature_pipeline = ConceptKBFeaturePipeline(None, None)
        trainer = ConceptKBTrainer(self.concept_kb, feature_pipeline)

        # Initialize multiprocessing
        i = 0
        manager = mp.Manager()
        ret_concept_queue = manager.Queue()

        # Train concepts in parallel
        while concept_selector.num_concepts_remaining > 0:

            # checking if there are any concepts that finished training
            while not ret_concept_queue.empty():
                old_concept, new_concept = ret_concept_queue.get()
                self.concept_kb._concepts[new_concept.name] = new_concept
                print(
                    f"Marking concept as completed: {old_concept.name}, before: {concept_selector.num_concepts_remaining}"
                )
                concept_selector.mark_concept_completed(old_concept)
            if not concept_selector.has_concept_available():

                print(
                    "No current concept to train : ",
                    concept_selector.num_concepts_remaining,
                    i,
                )
                i += 1
                time.sleep(5)
                continue
            concept = concept_selector.get_next_concept()

            examples, dataset = self.trainer.construct_dataset_for_concept_training(
                concept, use_concepts_as_negatives=use_concepts_as_negatives
            )

            # Recache zero-shot attributes for sampled examples
            self.cacher.recache_zs_attr_features(concept, examples=examples)

            if self.use_concept_predictors_for_concept_components:
                for component in concept.component_concepts.values():
                    self.cacher.recache_zs_attr_features(
                        component, examples=examples
                    )  # Needed to predict the componnt concept

            else:  # Using fixed scores for concept-image pairs
                self.cacher.recache_component_concept_scores(concept, examples=examples)

            p = mp.Process(
                target=train_wrapper,
                args=(
                    trainer,
                    concept,
                    examples,
                    dataset,
                    ret_concept_queue,
                    n_epochs,
                    use_concepts_as_negatives,
                    "cuda",
                ),
                kwargs=train_concept_kwargs,
            )
            p.start()
            p_list.append(p)

        for p in p_list:
            p.join()

        while not ret_concept_queue.empty():
            old_concept, new_concept = ret_concept_queue.get()
            print("len of queue", ret_concept_queue.qsize())
            self.concept_kb._concepts[new_concept.name] = new_concept
            concept_selector.mark_concept_completed(old_concept)

        print("Training complete.")

        print(f"Time to train concepts: {time.time() - time_start}")
        return self.concept_kb

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
