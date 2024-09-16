import asyncio
import code
import json
import logging
import os
import time
from typing import AsyncGenerator, List

import httpx
from model.Counter import Counter
from server_utils import (convert_base64_to_upload_file,
                          streaming_response_end, streaming_response_yield,
                          substitute_brackets)

logger = logging.getLogger("uvicorn.error")

# Load the environment variables
IMAGE_BACKEND_API = os.environ.get(
    "IMAGE_BACKEND_API", "http://blender12.cs.illinois.edu:16004"
)

#


async def _make_request(function_name: str, args: dict[str, str], files: object, user_id: str= "default_user", extra_args: dict[str, str] = {}, callback = None, counter = Counter()) -> AsyncGenerator[str, None]:
    """
    Make a request to the image backend API.

    Args:
        function_name (str): The name of the function to execute on the image.
        args (dict): The arguments for the function.
        files (object): The image files to upload.
        user_id (str): The user ID.
        extra_args (dict): The extra arguments for the function.
        
    Returns:
        dict: The response from the image backend API.
    """
    url = f"{IMAGE_BACKEND_API}/{function_name}"

    # Make a request to the image backend API
    # Here you would make the actual async HTTP request, for example using httpx:
    args["user_id"] = user_id
    # Enable streaming
    args["streaming"] = "true"
    if extra_args:
        for arg in extra_args:
            args[arg] = extra_args[arg]
    # yield streaming_response_yield(f"Making a request to the image backend API with function {url}, args: {args}\n\n", counter)
    logger.info(f"Making a request to the image backend API with function {url}, args: {args}, images: {files}")
    async with httpx.AsyncClient(timeout=2000) as client:
        async with client.stream("POST", url, params=args, files=files) as response:
            try:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    if chunk.startswith("status: "):
                        yield streaming_response_yield(chunk.replace("status: ", ""), counter)
                    elif chunk.startswith("error: "):
                        yield streaming_response_end(chunk.replace("error: ", ""), counter)
                    elif chunk.startswith("result: "):
                        yield streaming_response_yield(
                            chunk.replace("result: ", ""), counter
                        )
                        callback(chunk.replace("result: ", "\n\n"))
                    else:
                        yield streaming_response_yield(chunk, counter)
                    yield streaming_response_yield(f"\n\n", counter)
            except httpx.HTTPStatusError as e:
                yield streaming_response_end(f"HTTP Error: {e}", counter)

async def make_requests(functions, params, images, code_blocks, user_id = "default_user", callback = None, counter: Counter = Counter()) -> AsyncGenerator[str, None]:
    """
    Make a request to the image backend API.

    Parameters:
    functions (list): The list of functions to execute on the image.
    params (dict): The parameters extracted from the sentence.

    Returns:
    dict: The response from the image backend API.
    """
    i = 0
    if code_blocks and  isinstance(code_blocks, list):
        for code_block_pair in code_blocks:
            if code_block_pair[0] == "video":
                json_obj = json.loads(code_block_pair[1])
                print("json_obj", json_obj)
                params["id"] = json_obj["id"]
                    
    for function in functions:
        function_name = function["name"]
        args = function["args"]
        extra_args = function.get("extra_args", {})

        # Make a request to the image backend API
        arg_dict = {}
        image_upload_list = []
        for arg, arg_api_name in args.items():
            if arg_api_name.startswith("images"):
                image_upload_list.extend([convert_base64_to_upload_file("images", image) for image in images])
            elif arg_api_name == "concepts":
                if arg_api_name not in arg_dict:
                    arg_dict[arg_api_name] = []
                arg_dict[arg_api_name].append(params[arg])
            elif arg_api_name.startswith("image_"):
                index = int(arg.split("_")[-1]) - 1
                image_upload_list.append(
                    convert_base64_to_upload_file(arg_api_name, images[index])
                )
            elif arg_api_name.startswith("image"):
                image_upload_list.append(convert_base64_to_upload_file("image", images[0]))
            else:
                arg_dict[arg_api_name] = params[arg]

        if "concepts" in arg_dict:
            arg_dict["concepts"] = ", ".join(arg_dict["concepts"])

        description = function["description"]
        if description:
            yield streaming_response_yield(
                substitute_brackets(description, params) + "\n\n", counter
            )
        
        async for msg in _make_request(function_name, arg_dict, image_upload_list, user_id, extra_args, callback, counter= counter):
            yield msg
    yield streaming_response_yield("success\n\n", counter)
