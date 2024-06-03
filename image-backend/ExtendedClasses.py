# %%
import sys

sys.path.append("/shared/nas2/knguye71/ecole-june-demo/ecole_mo9_demo/src")
from controller import Controller
from feature_extraction.trained_attrs import N_ATTRS_DINO
from kb_ops import ConceptKBTrainer, ConceptKBPredictor
from model.concept import ConceptKB, ConceptKBConfig, ConceptExample

# from typing import Union, Literal
# from llm import LLMClient
# from score import AttributeScorer
from kb_ops.retrieve import CLIPConceptRetriever
from kb_ops.caching import ConceptKBFeatureCacher
from kb_ops.feature_pipeline import ConceptKBFeaturePipeline

# from feature_extraction.trained_attrs import N_ATTRS_SUBSET
import PIL
import torch
# import base64
import time
import traceback
import os
import sys
import gc
from PIL import Image
# import torch.nn as nn
# import numpy as np
# from rembg import remove
# from typing import List, Dict, Union
# from feature_extraction.dino_features import get_rescaled_features, rescale_features
# import concurrent.futures

# logger = logging.getLogger("uvicorn.error")
DEFAULT_CKPT = os.environ.get("DEFAULT_CKPT", "/shared/nas2/knguye71/ecole-backend/ckpts/concept_kb_epoch_2021_06_29-15:00:00.pt")
FEATURE_CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "./feature_cache/")

class ExtendedController(Controller):

    def predict_concept(
        self,
        image= None,
        loc_and_seg_output = None,
        unk_threshold: float = 0.1,
        leaf_nodes_only: bool = True,
        restrict_to_concepts: list[str] = [],
    ) -> dict:
        """
        Predicts the concept of an image and returns the predicted label and a plot of the predicted classes.

        Returns: dict with keys 'predicted_label' and 'plot' of types str and PIL.Image, respectively.
        """
        self.cached_images.append(image)

        try:
            if restrict_to_concepts:
                assert not leaf_nodes_only, 'Specifying concepts to restrict prediction to is only supported when leaf_nodes_only=False.'
                concepts = [self.retrieve_concept(concept_name) for concept_name in restrict_to_concepts]
            else:
                concepts = None

            prediction = self.predictor.predict(
                image_data=image,
                unk_threshold=unk_threshold,
                return_segmentations=True,
                leaf_nodes_only=leaf_nodes_only,
                concepts=concepts
            )
        except Exception as e:
            print(traceback.format_exc())
            # or
            print(sys.exc_info()[2])
            if "argmax" in str(e.with_traceback(sys.exc_info()[2])):
                return {"predicted_label": "unknown", "is_below_unk_threshold": True}
            return {"error": str(e.with_traceback(sys.exc_info()[2]))}

        return prediction

    def compare_concepts(self, concept_name1, concept_name2, top_k=5):
        try:
            concept1 = self.retrieve_concept(concept_name1, max_retrieval_distance=0.3)
        except Exception as e:
            raise RuntimeError(f'No concept found for "{concept_name1}".')
        try:
            concept2 = self.retrieve_concept(concept_name2, max_retrieval_distance=0.3)
        except Exception as e:
            raise RuntimeError(f'No concept found for "{concept_name2}".')

        print("concept1", concept1.name)
        print("concept2", concept2.name)

        # weights1 = concept1.predictor.img_trained_attr_weights.weights.data.cpu()
        # weights2 = concept2.predictor.img_trained_attr_weights.weights.data.cpu()

        # attr_names = (
        #     self.feature_pipeline.feature_extractor.trained_attr_predictor.attr_names
        # )

        # return self._get_differences(
        #     torch.Tensor(weights1), torch.Tensor(weights2), attr_names, top_k=top_k
        # )

    #     def diff_between_predictions(
    #         self,
    #         image1: PIL.Image.Image,
    #         image2: PIL.Image.Image,
    #         threshold=0,
    #         top_k=5,
    #     ):
    #         """
    #         Predicts the concept for two images and returns the differences in the top k attributes
    #         between the two predictions.
    #         """

    #         def pred_concept(image) -> dict:
    #             return self.predict_concept_processed(image, threshold, with_heatmap=False)

    #         [pred1, pred2] = list(map(pred_concept, [image1, image2]))

    #         def compare_zs_attr(concept_name_1, concept_name_2):
    #             try:
    #                 concept1 = self.retrieve_concept(
    #                     concept_name_1, max_retrieval_distance=0.2
    #                 )
    #                 concept1_zs_attr_names = [attr.name for attr in concept1.zs_attributes]
    #             except Exception as e:
    #                 concept1_zs_attr_names = []

    #             try:
    #                 concept2 = self.retrieve_concept(
    #                     concept_name_2, max_retrieval_distance=0.2
    #                 )
    #                 concept2_zs_attr_names = [attr.name for attr in concept2.zs_attributes]
    #             except Exception as e:
    #                 concept2_zs_attr_names = []

    #             all_queries = list(set([
    #                 a for a in concept1_zs_attr_names + concept2_zs_attr_names
    #             ]))
    #             scores_1 = (
    #                 self.feature_pipeline.feature_extractor.zs_attr_predictor.predict(
    #                     [image1], all_queries, apply_sigmoid=False
    #                 )
    #             )  # (1, n_zs_attrs)

    #             scores_2 = (
    #                 self.feature_pipeline.feature_extractor.zs_attr_predictor.predict(
    #                     [image2], all_queries, apply_sigmoid=False
    #                 )
    #             )

    #             # concept1_weights = (
    #             #     concept1.predictor.img_zs_attr_weights.weights.cpu()
    #             #     if concept1
    #             #     else [0] * len(concept1_zs_attr_names)
    #             # )

    #             # concept2_weights = (
    #             #     concept2.predictor.img_zs_attr_weights.weights.cpu()
    #             #     if concept2
    #             #     else torch.Tensor([0] * len(all_queries))
    #             # )

    #             # predictor_weights = concept1_weights, concept2_weights
    #             print("score1", scores_1)
    #             print("score2", scores_2)

    #             return {
    #                 "name": all_queries,
    #                 "probs1": scores_1.squeeze().tolist(),
    #                 "probs2": scores_2.squeeze().tolist(),
    #             }

    #         return {
    #             "pred1": pred1,
    #             "pred2": pred2,
    #             "diff": self._get_differences(
    #                 torch.Tensor(
    #                     pred1["prediction"]["predicted_concept_outputs"][
    #                         "trained_attr_img_scores"
    #                     ]
    #                 ),
    #                 torch.Tensor(
    #                     pred2["prediction"]["predicted_concept_outputs"][
    #                         "trained_attr_img_scores"
    #                     ]
    #                 ),
    #                 pred1["prediction"]["trained_attributes"],
    #                 weights1=torch.Tensor(pred1["predictor_weights"]),
    #                 weights2=torch.Tensor(pred2["predictor_weights"]),
    #                 top_k=top_k,
    #             ),
    #             "diff_zs_attr": compare_zs_attr(pred1["predicted_label"], pred2["predicted_label"]),
    #         }

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

    def save_kb(self, save_dir: str):
        current_time = time.strftime("%Y_%m_%d-%H:%M:%S")
        checkpoint_path = save_dir + f"/concept_kb_epoch_{current_time}.pt"
        self.concept_kb.save(checkpoint_path)
        # Loop through all the concepts and delete the segmentations and the cached features
        for concept in self.concept_kb._concepts.values():
            for example in concept.examples:
                example.image_segmentations = None
                example.image_features = None
        return checkpoint_path, self.concept_kb

    def reset_KB(self, clear_all=False):
        if clear_all:
            current_kb = ConceptKB()
            # Initialize ConceptKB
            current_kb.initialize(
                ConceptKBConfig(
                    img_feature_dim=N_ATTRS_DINO,
                )
            )
        else:
            current_kb = ConceptKB.load(DEFAULT_CKPT)
        self.load_kb(current_kb)
        current_kb.to("cpu")
        return current_kb

        # def predict_concept_processed(self, img: PIL.Image.Image, threshold=0.7, with_heatmap=False):
        #     """
        #     Processes the prediction of an image and returns the predicted label and a json file of the predicted classes.

        #     Args:
        #         img (Image.Image): the image to be processed
        #         threshold (float, optional): Threshold to . Defaults to 0.1.
        #         retry (int, optional): _description_. Defaults to 0.

        #     Returns:
        #         _type_: _description_
        #     """
        #     result = self.predict_concept(img, threshold)

        #     if "plot" in result:
        #         import base64
        #         # Convert the plot to base64 encoding
        #         plot = result["plot"]
        #         plot_bytes = plot.tobytes()
        #         encoded_plot = base64.b64encode(plot_bytes).decode("utf-8")
        #         result["plot"] = encoded_plot

        #         if "prediction" in result:
        #             if "predictors_scores" in result["prediction"]:
        #                 names, values = self.compute_top_k(
        #                     result["prediction"]["predictors_scores"],
        #                     result["prediction"]["concept_names"],
        #                     5,
        #                 )
        #                 result["predicted_top_k"] = {
        #                     "names": names,
        #                     "values": values,
        #                 }
        #                 result["prediction"]["predictors_scores"] = result["prediction"][
        #                     "predictors_scores"
        #                 ].tolist()

        #             if "predicted_concept_outputs" in result["prediction"]:
        #                 attributes = {}
        #                 for k, v in result["prediction"][
        #                     "predicted_concept_outputs"
        #                 ].__dict__.items():
        #                     if isinstance(v, torch.Tensor):
        #                         attributes[k] = v.tolist()
        #                         if not isinstance(attributes[k], list):
        #                             attributes[k] = [attributes[k]]
        #                 print("is_below_unk_threshold", result["prediction"]["is_below_unk_threshold"])
        #                 attributes["is_below_unk_threshold"] = result["prediction"][
        #                     "is_below_unk_threshold"
        #                 ]
        #                 del result["prediction"]["predicted_concept_outputs"]
        #                 result["prediction"]["predicted_concept_outputs"] = attributes
        #             result["prediction"][
        #                 "trained_attributes"
        #             ] = (
        #                 self.feature_pipeline.feature_extractor.trained_attr_predictor.attr_names
        #             )

        #             if "segmentations" in result["prediction"]:
        #                 result["prediction"]["segmentations"] = self.process_segmentations(
        #                     result["prediction"]["segmentations"]
        #                 )
        #             if result["predicted_label"] != "unknown":
        #                 concept = self.concepts.get_concept(result["predicted_label"])
        #                 result["concept_attributes"] = concept.zs_attributes
        #                 result["predictor_weights"] = (
        #                     concept.predictor.img_trained_attr_weights.weights.data.cpu()
        #                     .numpy()
        #                     .tolist()
        #                 )

        #             result["predicted_label"] = (
        #                 result["prediction"]["predicted_label"]
        #                 if not attributes["is_below_unk_threshold"]
        #                 else "unknown"
        #             )

        #             if with_heatmap and result["prediction"]["predicted_label"] != "unknown":
        #                 heatmap = self.get_predictor_heatmap(
        #                     result["prediction"]["predicted_label"], img
        #                 )
        #                 result["heatmap"] = self.process_heatmap(heatmap)
        # return result

    #     ########################
    #     # Utils Functionality  #
    #     ########################

    #     def compute_top_k(self, scores: torch.Tensor, names: list, top_k: int):

    #         scores: torch.Tensor = scores.sigmoid()  # (n,)
    #         names: list = names

    #         top_k = min(top_k, len(scores))
    #         values, indices = scores.topk(top_k)
    #         names = [names[i] for i in indices]

    #         # Reverse for plot so highest score is at top
    #         values = list(reversed(values.tolist()))
    #         names = list(reversed(names))
    #         return names, values

    #     def process_segmentations(self, segmentations: dict):
    #         # Convert the tensor to bytes
    #         tensor_bytes = segmentations.part_masks.numpy().tobytes()
    #         # Encode the bytes using base64
    #         encoded_str = base64.b64encode(tensor_bytes).decode("utf-8")

    #         # Store the shape and data type for later reconstruction
    #         tensor_shape = segmentations.part_masks.shape
    #         tensor_dtype = segmentations.part_masks.dtype

    #         # Encode the binary tensor to a compressed string using base64
    #         return {
    #             "localized_bbox": segmentations.localized_bbox.tolist(),
    #             "part_masks": {
    #                 "shape": tensor_shape,
    #                 "dtype": str(tensor_dtype),
    #                 "data": encoded_str,
    #             },
    #         }

    #     def process_heatmap(self, heatmap: torch.Tensor):
    #         # Convert the tensor to bytes
    #         tensor_bytes = heatmap.numpy().tobytes()
    #         # Encode the bytes using base64
    #         encoded_str = base64.b64encode(tensor_bytes).decode("utf-8")

    #         # Store the shape and data type for later reconstruction
    #         tensor_shape = heatmap.shape
    #         tensor_dtype = heatmap.dtype

    #         # Encode the binary tensor to a compressed string using base64
    #         return {
    #             "shape": tensor_shape,
    #             "dtype": str(tensor_dtype),
    #             "data": encoded_str,
    #         }

    #     def _get_differences(
    #         self,
    #         attr_scores1: torch.Tensor,
    #         attr_scores2: torch.Tensor,
    #         attr_names: List[str],
    #         weights1: torch.Tensor = None,
    #         weights2: torch.Tensor = None,
    #         top_k=5,
    #     ) -> Dict[str, List[Union[str, float]]]:
    #         # Compute top attribute probability differences
    #         if weights1 is not None and weights2 is not None:
    #             probs1 = attr_scores1.squeeze().sigmoid()
    #             probs2 = attr_scores2.squeeze().sigmoid()
    #         else:
    #             probs1 = attr_scores1.squeeze()
    #             probs2 = attr_scores2.squeeze()
    #         diffs = (probs1 - probs2).abs()

    #         top_k = min(top_k, len(diffs))

    #         if weights1 is not None and weights2 is not None:
    #             attr_weights1 = weights1
    #             attr_weights2 = weights2
    #             # Weight the differences by the attribute weights
    #             weighted_diffs = diffs * (attr_weights1.abs() + attr_weights2.abs())

    #             top_diffs, top_inds = weighted_diffs.topk(top_k)
    #         else:
    #             top_diffs, top_inds = diffs.topk(top_k)

    #         top_attr_names = [attr_names[i] for i in top_inds]

    #         # Get the top k attribute probabilities and convert to list
    #         probs1 = probs1[top_inds].tolist()
    #         probs2 = probs2[top_inds].tolist()

    #         return {
    #             "name": top_attr_names,
    #             "probs1": probs1,
    #             "probs2": probs2,
    #         }

    def teach_concept(self, concept_name, concept_examples: ConceptExample, previous_concept_name= None):
        print(
            "current_cuda_device",
            torch.cuda.current_device(),
            torch.cuda.get_device_name(),
        )
        try:
            # Train concept in isolation
            self.train_concept(concept_name, new_examples=concept_examples, n_epochs=10)

            if previous_concept_name:
                self.train_concept(
                    previous_concept_name, max_retrieval_distance=0.0, n_epochs=10
                )

            currentkb = self.concept_kb
            currentkb.to("cpu")
            return currentkb
        except Exception as e:
            print(traceback.format_exc())
            print(sys.exc_info()[2])
            raise RuntimeError(f"Failed to train concept {concept_name}.")
        finally:
            self.reset_KB(clear_all=True)
            gc.collect()
            torch.cuda.empty_cache()

#     def get_zs_attributes(self, concept_name: str):
#         concept_name = concept_name.strip().lower()

#         try:
#             concept = self.retrieve_concept(concept_name, max_retrieval_distance=0.2)
#         except Exception:
#             return []

#         ret_list = [attr.name for attr in concept.zs_attributes]
#         return ret_list

#     def get_predictor_heatmap_list(self, list_of_concepts_name: List[str], img: PIL.Image.Image):
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             def get_heatmap_with_concept_name (concept_name):
#                 return {
#                     "name": concept_name,
#                     "heatmap": self.process_heatmap(
#                         self.get_predictor_heatmap(concept_name, img)
#                     ),
#                 }
#             results = executor.map(
#                 lambda concept_name: get_heatmap_with_concept_name(concept_name),
#                 list_of_concepts_name,
#             )
#         return list(results)

#     def get_predictor_heatmap(self, concept_name: str, img: PIL.Image.Image, strategy: Literal["clamping", "normalize"] = "clamping"):
#         concept_name = concept_name.strip().lower()

#         try:
#             concept = self.retrieve_concept(concept_name, max_retrieval_distance=0.2)

#         except Exception:
#             raise RuntimeError(f"No concept found for {concept_name}.")

#         try:
#             # Get feature extractor
#             feature_predictor = (
#                 nn.Sequential(
#                     concept.predictor.img_features_predictor,
#                     concept.predictor.img_features_weight,
#                 )
#                 .eval()
#                 .cpu()
#             )

#             # remove background
#             bw_img = remove(img, post_process_mask=True, session=self.feature_pipeline.loc_and_seg.localizer.rembg_session, only_mask=True)
#             img_mask = np.array(bw_img) > 0 # (h, w)

#             # Patch features
#             cls_feats, patch_feats = get_rescaled_features(self.feature_pipeline.feature_extractor.dino_feature_extractor, [img], interpolate_on_cpu=True)
#             patch_feats = patch_feats[0]  # (h, w, d)
#             patch_feats = rescale_features(patch_feats, img)  # (h, w, d)

#             # Get heatmap
#             with torch.no_grad():
#                 # Need to move to CPU otherwise runs out of GPU mem on big images
#                 heatmap = feature_predictor(patch_feats.cpu()).squeeze()  # (h, w)

#             # Move img_features_predictor back to correct device (train is called by train method)
#             feature_predictor.cuda()


#             if strategy == "clamping":
#                 heatmap = self.clamping(heatmap)
#             elif strategy == "normalize":
#                 heatmap = self.normalize(heatmap)
#             print("heatmap", heatmap.shape)
#             # Mask image background
#             heatmap = heatmap * img_mask  # (h, w, 3)
#             print("heatmap after", heatmap.shape)
#             return heatmap.detach().cpu()
#         except Exception as e:
#             print(traceback.format_exc())
#             print(sys.exc_info()[2])
#             raise RuntimeError(f"Failed to get heatmap for {concept_name}.")

#     def normalize(self, x: torch.Tensor):
#         x = x.squeeze()
#         return (x - x.min()) / (x.max() - x.min())

#     def clamping(self, heatmap: torch.Tensor, radius: int = 5, center: int = 0):
#         heatmap = heatmap - center # Center at zero
#         heatmap = heatmap.clamp(-radius, radius) # Restrict to [-radius, radius]
#         heatmap = (heatmap + radius) / (2 * radius) # Normalize to [0, 1] with zero at .5
#         return heatmap


# %%


if __name__ == "__main__":

    import controller as ctrl
    from feature_extraction import build_feature_extractor, build_sam
    from image_processing import build_localizer_and_segmenter

    img_path = "/shared/nas2/knguye71/ecole-june-demo/samples/DemoJune2024-2/cargo_jet/000001.jpg"
    ckpt_path = "/shared/nas2/blume5/fa23/ecole/checkpoints/concept_kb/2024_05_30-11:30:28-rafd2xjd-no_biplane_no_cargo_jet/concept_kb_epoch_50.pt"

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

    result

    # %%

    result_2 = controller.predict_hierarchical(img, unk_threshold=0.1)

    result_2

# %%
