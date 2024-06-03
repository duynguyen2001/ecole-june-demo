from typing import List
import json
import httpx
from utils import streaming_response_yield, streaming_response_end
import os
import base64
import io

import base64
import io
import os
import tempfile
import mimetypes
from uuid import uuid4


IMAGE_BACKEND_API = os.environ.get("IMAGE_BACKEND_API", "http://localhost:16004")
def convert_base64_to_upload_file(arg, base64_string):
    """
    convert base64 image to upload file
    For example, given the input string:
    arg = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/4QBYRXhpZgAATU0AKgAAAAgAA1IBAAABAAEAAK4BAgAABAAEAAYIBAgAABAAEAAYUBAgAABAAEAAYYB"



    Args:
        arg (str): string to identify the image
        base64_string (str): base64 image

    Returns:
        _type_:
    """

    # Extract the MIME type and the base64 part
    if base64_string.startswith("data:"):
        header, base64_data = base64_string.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
    else:
        mime_type = "application/octet-stream"
        base64_data = base64_string

    # Decode the base64 string
    file_data = base64.b64decode(base64_data)

    # Prepare a file-like object from the decoded data
    file_like = io.BytesIO(file_data)

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=mimetypes.guess_extension(mime_type)
    )
    try:
        # Write the decoded data to the temporary file
        temp_file.write(file_data)
        temp_file.close()

        return (arg, (os.path.basename(temp_file.name), file_like, mime_type))

    finally:
        # Clean up the temporary file
        os.remove(temp_file.name)


async def _make_request(function_name, args, files, user_id= "default_user"):
    """
    Make a request to the image backend API.

    Parameters:
    function_name (str): The name of the function to execute on the image.
    args (dict): The parameters extracted from the sentence.
    files (list): The list of images to upload.
    """
    url = f"{IMAGE_BACKEND_API}/{function_name}"

    # Make a request to the image backend API
    # Here you would make the actual async HTTP request, for example using httpx:
    args["user_id"] = user_id
    args["streaming"] = "true"
    print("args", args)
    yield streaming_response_yield(f"Making a request to the image backend API with function {url}, args: {args}\n\n", 0)

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, data=args, files=files) as response:
            async for chunk in response.aiter_text():
                if chunk.startswith("status: "):
                    yield streaming_response_yield(chunk.replace("status: ", ""), 0)
                elif chunk.startswith("error: "):
                    yield streaming_response_end(chunk.replace("error: ", ""), 0)
                elif chunk.startswith("result: "):
                    yield streaming_response_yield(chunk.replace("result: ", "\n\n"), 0)
                else:
                    yield streaming_response_yield(chunk, 0)


async def make_requests(functions, params, images, user_id = "default_user"):
    """
    Make a request to the image backend API.

    Parameters:
    functions (list): The list of functions to execute on the image.
    params (dict): The parameters extracted from the sentence.

    Returns:
    dict: The response from the image backend API.
    """
    response = {}

    for function in functions:
        function_name = function["name"]
        args = function["args"]

        # Make a request to the image backend API
        arg_dict = {}
        image_upload_list = []
        for arg, arg_api_name in args.items():
            if arg_api_name.startswith("images"):
                image_upload_list.append([convert_base64_to_upload_file("images", image) for image in images])
            elif arg_api_name.startswith("image_"):
                index = int(arg.split("_")[-1]) - 1
                image_upload_list.append(
                    convert_base64_to_upload_file(arg_api_name, images[index])
                )
            elif arg_api_name.startswith("image"):
                image_upload_list.append(convert_base64_to_upload_file("image", images[0]))
            else:
                arg_dict[arg_api_name] = params[arg]

        async for msg in _make_request(function_name, arg_dict, image_upload_list, user_id):
            yield msg
