import base64
import math
import os
import uuid
from typing import Any, Generator

import numpy as np
import PIL
import PIL.Image
import torch
from kb_ops.predict import PredictOutput

IMAGE_DIR = os.environ.get(
    "IMAGE_DIR", "/shared/nas2/knguye71/ecole-june-demo/image_dir"
)
TENSOR_DIR = os.environ.get(
    "TENSOR_DIR", "/shared/nas2/knguye71/ecole-june-demo/tensor_dir"
)
import logging

logger = logging.getLogger("uvicorn.error")

import json

DINO_SUBSET = "/shared/nas2/knguye71/ecole-june-demo/ecole_mo9_demo/src/feature_extraction/trained_attrs/dino_class_id_to_index.json"
with open(DINO_SUBSET) as f:
    DINO_INDEX_TO_ATTR = json.load(f)
LIST_DINO_ATTR = list(DINO_INDEX_TO_ATTR.values())


def sigmoid(x) -> float:
    return 1 / (1 + math.exp(-x))


def correct_grammar(sentence: str)  -> str:
    return sentence
    # blob = TextBlob(sentence)
    # return str(blob.correct())


def barchart_md_template(
    x_value: list[float],
    y_value: list[str],
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    threshold: float,
    rev_list: list[str],
    sort: bool = False,
    sigmoided: bool = False,
) -> str:
    if sort:
        x_value, y_value = zip(
            *sorted(zip(x_value, y_value), key=lambda x: x[0], reverse=True)
        )

    if sigmoided:
        x_value = [sigmoid(x) for x in x_value]

    return (
        "```barchart\n{"
        + f"""
  "title": "{title}",
  "x_label": "{x_axis_label}",
  "y_label": "{y_axis_label}",
  "threshold": "{threshold}",
  "x": {x_value},
  "y": {list(y_value)}
"""
        + "}\n```"
    )


def decision_tree_md_template(prediction_path: list[dict]):
    return f"""
```decision-tree
{prediction_path}
```
"""


def upload_image_and_return_id(image: PIL.Image.Image):
    # store image in local storage
    image_id = str(uuid.uuid4())
    os.makedirs(IMAGE_DIR, exist_ok=True)
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    image.save(image_path)
    return image_id


def upload_binary_tensor_and_return_id(tensor: torch.Tensor) -> str:
    """
    Uploads a binary tensor to the local storage and returns the tensor id and shape.

    Args:
        tensor (torch.Tensor): The tensor to upload.

    Returns:
        str: The tensor id.
    """
    # store image in local storage
    tensor_id = str(uuid.uuid4())
    # convert tensor to base64 string
    # Flatten the boolean tensor to a 1D array
    flattened_data = np.ravel(tensor).astype(np.uint8)

    # Convert the flattened data to bytes
    byte_data = flattened_data.tobytes()

    # Encode the byte data to a base64 string
    base64_data = base64.b64encode(byte_data).decode("utf-8")

    # Get the shape of the tensor
    shape = list(tensor.shape)

    # Create the EncodedTensor object
    encoded_tensor = {"dtype": "torch.bool", "data": base64_data, "shape": shape}
    tensor_path = os.path.join(TENSOR_DIR, f"{tensor_id}.json")
    os.makedirs(TENSOR_DIR, exist_ok=True)
    with open(tensor_path, "w") as f:
        json.dump(encoded_tensor, f)
    return tensor_id


def image_with_mask_md_template(
    image: PIL.Image.Image,
    mask: torch.Tensor,
    general_attributes: list[dict] = [],
    general_attributes_image: list[PIL.Image.Image] = [],
):
    image_id = upload_image_and_return_id(image)
    mask_id = upload_binary_tensor_and_return_id(mask)
    return f"""
```image-with-mask
{{
  "image": "{image_id}",
  "mask": "{mask_id}",
  "general_attributes": {general_attributes},
  "general_attributes_for_image": {general_attributes_image}
}}
```
"""


def format_prediction_result(
    output: PredictOutput, rev_dict: dict = None, top_k: int = 5
):
    nodes = []
    rev_list = (
        [rev_dict[name] for name in output["concept_names"]] if rev_dict else None
    )
    predicted_label = output.predicted_label
    if predicted_label == "unknown":
        return "I do not know what object is in the image\n\n"

    predicted_concept_outputs = output.predicted_concept_outputs
    if predicted_concept_outputs and predicted_label != "unknown":
        mask_scores = predicted_concept_outputs.trained_attr_region_scores.tolist()
        img_trained_attr_scores = (
            predicted_concept_outputs.trained_attr_img_scores.tolist()
        )

    predicted_concept_components_to_scores = (
        output.predicted_concept_components_to_scores
    )
    component_concept_scores = None
    component_concept_names = None
    if predicted_concept_components_to_scores:
        component_concept_scores = predicted_concept_components_to_scores.values()
        component_concept_names = predicted_concept_components_to_scores.keys()
    predicted_concept_components_heatmaps = output.predicted_concept_compoents_heatmaps
    component_concept_heatmaps = None
    if predicted_concept_components_heatmaps:
        component_concept_heatmaps = list(
            predicted_concept_components_heatmaps.values()
        )

    if segmentations := output.segmentations:
        masks = segmentations.part_masks
        attr_names = LIST_DINO_ATTR
        region_general_attributes = []
        for index in range(len(masks)):
            region_general_attribute_mask = {
                attr: mask_scores[index][j] for j, attr in enumerate(attr_names)
            }
            region_general_attribute_mask = dict(
                sorted(
                    region_general_attribute_mask.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:top_k]
            )

            region_general_attributes.append(region_general_attribute_mask)

    if img_trained_attr_scores:
        img_trained_attr_scores = dict(zip(attr_names, img_trained_attr_scores))
        img_trained_attr_scores = dict(
            sorted(img_trained_attr_scores.items(), key=lambda x: x[1], reverse=True)[
                :top_k
            ]
        )
    nodes.append(
        image_with_mask_md_template(
            segmentations["input_image"],
            masks,
            general_attributes=region_general_attributes,
            general_attributes_image=img_trained_attr_scores,
        )
        if segmentations
        and segmentations["input_image"]
        and region_general_attributes
        and len(masks)
        and img_trained_attr_scores
        else ""
    )
    nodes.append(
        correct_grammar(f'There is a  "{predicted_label}" in the image\n\n')
        if predicted_label and predicted_label != "unknown"
        else "I do not know what object is in the image\n\n"
    )
    nodes.append("### Concept Scores")
    nodes.append(
        barchart_md_template(
            output["predictors_scores"].tolist(),
            output["concept_names"],
            "Concept Scores",
            "Scores",
            "Concepts",
            0.1,
            rev_list,
            sort=True,
            sigmoided=True,
        )
    )
    nodes.append("### Component Concepts")
    if component_concept_heatmaps:
        # nodes.append(barchart_md_template(component_concept_scores, component_concept_names, 'Component Concepts Scores', 'Scores', 'Concepts', 0.1, None, sort=True, sigmoided=True) )
        nodes.append(
            image_block(component_concept_heatmaps, list(component_concept_names))
        )
    else:
        nodes.append("No component concepts found")
    return nodes


async def streaming_hierachical_predict_result(
    output: dict, sigmoided: bool = True, streaming_subclass: bool = False
):
    """
    Implements:
        hierachical_predict_result: What is in the image?

    Args:
        output (dict): The output from the model.
        sigmoided (bool): Whether to apply sigmoid to the scores.
        streaming_subclass (bool): Whether to stream subclass predictions

    Returns:
        str: The generated text.
    """
    # stream output first
    if predicted_label := output.get("predicted_label"):
        if predicted_label == "unknown":
            yield "result: I do not know what object is in the image\n\n"
        else:
            yield correct_grammar(
                f'result: The object in the image is a "{predicted_label}"\n\n'
            )

    # stream prediction path
    if prediction_path := output.get("prediction_path"):
        # build a de
        rev_dict = {}
        decision_tree = []
        nodes = []
        concept_path = output.get("concept_path")

        for i, pred in enumerate(prediction_path):
            sorted_data = sorted(
                zip(pred["concept_names"], pred["predictors_scores"].tolist()),
                key=lambda x: x[1],
                reverse=True,
            )

            for concept_name, concept_score in sorted_data:
                rev_dict[concept_name] = str(uuid.uuid4())
                decision_tree.append(
                    {
                        "concept_name": concept_name,
                        "score": sigmoid(concept_score) if sigmoided else concept_score,
                        "parent": concept_path[i - 1] if i > 0 else "root",
                        "id": rev_dict[concept_name] if i > 0 else "root",
                    }
                )
            best_node = sorted_data[0]
            nodes.append(f"""result: ### {best_node[0]}\n\n""")
            nodes.extend(
                [
                    f"result: {res}"
                    for res in format_prediction_result(pred, rev_dict)
                    if res
                ]
            )

        yield "result: " + decision_tree_md_template(decision_tree) + "\n\n"
        # stream nodes
        for node in nodes:
            yield node


def image_block(images: PIL.Image.Image, names: list[str] = []):
    """
    Implements:
        images: Show the images.

    Args:
        images (list[PIL.Image.Image]): The list of images.
        names (list[str]): The list of names for the images.

    Returns:
        str: The generated text.
    """
    image_id_list = []
    for image in images:
        image_id = upload_image_and_return_id(image)
        image_id_list.append(image_id)
    return f"""result: ```images
{{
  "images": {image_id_list},
  "names": {names}
}}
```
"""


def streaming_heatmap_class_difference(
    output: dict, concept_1: str, concept_2: str, img: PIL.Image.Image
) -> Generator[str, None, None]:
    """
    If image is provided, implements:
        "Why is this <class x> and not <class y>"

    Otherwise, implements:
        "What is the difference between <class x> and <class y>"


    """
    if img is None:
        yield f"result: The difference between the two concepts of {concept_1} and {concept_2}  is highlighted in the two example images.\n\n"

        c1_minus_c2_image1 = output["concept1_minus_concept2_on_concept1_image"]
        c2_minus_c1_image1 = output["concept2_minus_concept1_on_concept1_image"]
        c1_minus_c2_image2 = output["concept1_minus_concept2_on_concept2_image"]
        c2_minus_c1_image2 = output["concept2_minus_concept1_on_concept2_image"]

        yield f"""result: The highlighted regions in the first image are the regions that are more likely to be a {concept_1} than a {concept_2} and the highlighted regions in the second image are the regions that are more likely to be a {concept_2} than a {concept_1}.\n\n"""
        yield image_block(
            [c1_minus_c2_image1, c2_minus_c1_image1],
            names=[concept_1 + " predictor", concept_2 + " predictor"],
        )

        yield f"""result: The highlighted regions in the third image are the regions that are more likely to be a {concept_2} than a {concept_1} and the highlighted regions in the fourth image are the regions that are more likely to be a {concept_1} than a {concept_2}.\n\n"""
        yield image_block(
            [c1_minus_c2_image2, c2_minus_c1_image2],
            names=[concept_1 + " predictor", concept_2 + " predictor"],
        )
    else:
        concept1_minus_concept2 = output["concept1_minus_concept2"]
        concept2_minus_concept1 = output["concept2_minus_concept1"]
        yield f"result: The difference between the two concepts of {concept_1} and {concept_2}  is highlighted in the image.\n\n"
        yield f"""result: The highlighted regions in the first image are the regions that are more likely to be a {concept_1} than a {concept_2} and vice versa.\n\n"""

        yield image_block(
            [concept1_minus_concept2, concept2_minus_concept1],
            names=[concept_1 + " predictor", concept_2 + " predictor"],
        )


def streaming_heatmap(
    heatmap: PIL.Image.Image,
    concept: str,
    only_score_increasing_regions: bool = False,
    only_score_decreasing_regions: bool = False,
) -> Generator[str, None, None]:
    """
    Streams the heatmap image with the concept name.
    If only_score_increasing_regions is True, implements:
        "What regions are more likely to be a <concept> in the image?"
    If only_score_decreasing_regions is True, implements:
        "What regions are less likely to be a <concept> in the image?"
    Otherwise, implements:
        "Is there a <concept> in the image?"

    Args:
        heatmap (PIL.Image.Image): The heatmap image.
        concept (str): The concept name.
        only_score_increasing_regions (bool, optional): display only regions with increasing scores. Defaults to False.
        only_score_decreasing_regions (bool, optional): display only regions with decreasing scores. Defaults to False.

    Yields:
        Generator[str, None, None]: The generated text.
    """
    if only_score_increasing_regions:
        yield f"result: The regions that are more likely to be a {concept} are highlighted in the image.\n\n"
    elif only_score_decreasing_regions:
        yield f"result: The regions that are less likely to be a {concept} are highlighted in the image.\n\n"
    else:  # Visualize all regions
        yield f"result: The regions that are more likely to be a {concept} are highlighted in the image with a yellow color. Whereas, the regions that are less likely to be a {concept} are highlighted in the image with a blue color.\n\n"

    yield image_block([heatmap], names=[concept + " predictor"])


def streaming_diff_images(output: dict) -> Generator[str, None, None]:
    """
    Streams the difference between two images.
    Implements:
        "What is the difference between the first image and the second image?"
    If the two images are the same, implements:
        "What is the most distinctive region of the object in the image?"

    Args:
        output (dict): The output from the model.

    Yields:
        Generator[str, None, None]: The generated text.
    """
    concept1_prediction = output["concept1_prediction"]
    concept2_prediction = output["concept2_prediction"]
    c1_minus_c2_image1 = output["concept1_minus_concept2_on_image1"]
    c2_minus_c1_image1 = output["concept2_minus_concept1_on_image1"]
    c1_minus_c2_image2 = output["concept1_minus_concept2_on_image2"]
    c2_minus_c1_image2 = output["concept2_minus_concept1_on_image2"]

    predicted_label1 = concept1_prediction.predicted_label
    predicted_label2 = concept2_prediction.predicted_label

    if predicted_label1 == "unknown" or predicted_label2 == "unknown":
        if predicted_label1 == "unknown" and predicted_label2 == "unknown":
            yield "result: I do not know what object is in the first and second image.\n\n"
        elif predicted_label1 == "unknown":
            yield "result: I do not know what object is in the first image.\n\n"
        else:
            yield "result: I do not know what object is in the second image.\n\n"
    else:
        if predicted_label1 == predicted_label2:
            yield f"""result: I predict that the object in the first image and the object in the second image are both a {predicted_label1}.\n\n"""
            yield f"""result: # Image Comparison\n\n"""
            yield f"""result: These are the most distinctive regions of the object in the first image and the second image.\n\n"""
            yield image_block(
                [c1_minus_c2_image1, c2_minus_c1_image2],
                names=[
                    predicted_label1 + " predictor",
                    predicted_label2 + " predictor",
                ],
            )
        else:
            yield f"""result: I predict that the object in the first image is a {predicted_label1} and the object in the second image is a {predicted_label2}.\n\n"""
            yield f"""result: # Image Comparison\n\n"""
            yield f"""result: The highlighted regions in the first image are the regions that are more likely to be a {predicted_label1} than a {predicted_label2} and the highlighted regions in the second image are the regions that are more likely to be a {predicted_label2} than a {predicted_label1}.
            """
            yield image_block(
                [c1_minus_c2_image1, c2_minus_c1_image1],
                names=[
                    predicted_label1 + " predictor",
                    predicted_label2 + " predictor",
                ],
            )

            yield f"""result: The highlighted regions in the third image are the regions that are more likely to be a {predicted_label2} than a {predicted_label1} and the highlighted regions in the fourth image are the regions that are more likely to be a {predicted_label1} than a {predicted_label2}.
            """
            yield image_block(
                [c1_minus_c2_image2, c2_minus_c1_image2],
                names=[
                    predicted_label1 + " predictor",
                    predicted_label2 + " predictor",
                ],
            )

            yield f"""result: # Concept Prediction\n\n"""
            yield f"""result: ## First image\n\n"""
            yield (
                f"""result: The predicted object in the first image is a {predicted_label1}\n\n"""
                if predicted_label1 != "unknown"
                else "result: I do not know what object is in the first image.\n\n"
            )
            for msg in format_prediction_result(concept1_prediction):
                yield f"""result: {msg}\n\n"""

            yield f"""result: ## Second image\n\n"""
            yield (
                f"""result: The predicted object in the second image is a {predicted_label2}\n\n"""
                if predicted_label2 != "unknown"
                else "result: I do not know what object is in the second image.\n\n"
            )
            for msg in format_prediction_result(concept2_prediction):
                yield f"""result: {msg}\n\n"""


def streaming_is_concept_in_image(
    result, concept_name: str, heatmap: PIL.Image.Image
) -> Generator[str, None, None]:
    """
    Streams whether a concept is in the image.
    Implements:
        is_concept_in_image: Is there a <concept> in the image?

    Args:
        result (PredictOutput): The output from the model.
        concept_name (str): The concept name.
        heatmap (PIL.Image.Image): The heatmap image.

    Yields:
        Generator[str, None, None]: The generated text.
    """

    concept_names = result.concept_names
    concept_scores = result.predictors_scores
    if concept_name in concept_names:
        concept_index = concept_names.index(concept_name)
        concept_score = concept_scores[concept_index]
        if concept_score > 0.0:
            yield f"result: Yes, the concept of {concept_name} is in the image with a score of {sigmoid(concept_score):.2%}.\n\n"
            if heatmap:
                yield f"""result: The regions that are more likely to be a {concept_name} are highlighted in the image.\n\n"""
                yield image_block([heatmap], names=[concept_name])

        else:
            yield f"result: No, the concept of {concept_name} is not in the image.\n\n"
    else:
        yield f"result: No, the concept of {concept_name} is not in the image.\n\n"

    yield f"""result: # Concept Prediction\n\n"""
    yield f"""result: {barchart_md_template(concept_scores.tolist(), concept_names, 'Concept Scores', 'Scores', 'Concepts', 0.1, None, sort=True, sigmoided=True)}\n\n"""


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
