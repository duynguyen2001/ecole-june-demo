import base64
import os
import uuid
from io import BytesIO
from re import I
from typing import Any

import PIL.Image
import torch

IMAGE_DIR = os.environ.get(
    "IMAGE_DIR", "/shared/nas2/knguye71/ecole-june-demo/image_dir"
)

def rle(x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def convert_bool_tensor_to_byte_string(obj: torch.BoolTensor) -> dict[str, Any]:
    # Convert a torch.BoolTensor to a byte string
    # Flatten the N-D tensor to 1D
    if obj.shape == torch.Size([1]):
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


def convert_PIL_Image_to_byte_string(obj: PIL.Image.Image) -> dict[str, Any]:
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
            return str(obj.shape)
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
