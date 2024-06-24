import base64
import io
import json
import mimetypes
import os
import re
import tempfile
from difflib import SequenceMatcher
from typing import Any

from model.Counter import Counter

START_TOKEN = "<|im_start|>"
END_TOKEN = "<|im_end|>"


def clean_string(input_string: str) -> str:
    """
    Clean the input string by removing articles ('a', 'an', 'the'), double and single quotes, and extra spaces.

    Args:
        input_string (str): The input string to clean.

    Returns:
        str: The cleaned string.
    """
    # Step 1: Remove articles ('a', 'an', 'the')
    articles_pattern = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
    cleaned_string = articles_pattern.sub("", input_string)

    # Step 2: Remove double and single quotes
    cleaned_string = cleaned_string.replace('"', "").replace("'", "")

    # Step 3: Clean up extra spaces
    cleaned_string = " ".join(cleaned_string.split())

    return cleaned_string


def streaming_response_yield(output_string, counter: Counter):
    counter.increment()
    return (
        "data: "
        + json.dumps(
            {
                "token": {
                    "id": counter.count,
                    "text": output_string + "\n\n",
                    "logprob": 0,
                    "special": False,
                },
                "generated_text": None,
                "details": None,
            }
        )
        + "\n\n"
    )


def streaming_response_end(output_string: str, counter: Counter) -> str:
    """
    Generate the end of the streaming response.

    Args:
        output_string (str): The output string.
        counter (Counter): The counter object.

    Returns:
        str: The end of the streaming response.
    """
    # auto increment index
    counter.increment()

    # return the end of the streaming response
    return (
        "data: "
        + json.dumps(
            {
                "token": {
                    "id": counter.count,
                    "text": "",
                    "special": True,
                    "logprob": 0,
                },
                "generated_text": f"""{output_string}\n\n""",
                "details": {
                    "finish_reason": "eos_token",
                    "num_tokens": len(output_string.split()),
                },
            }
        )
        + "\n\n"
    )


def strip_tokens(
    input_string: str, start_token: str = START_TOKEN, end_token: str = END_TOKEN
):
    """
    Strips the conversation inputs to list of strings.
    For instance, given the input string:
    start_token = "<|start|>"
    end_token = "<|end|>"
    input_string = "<|start|>user Hi <|end|><|start|>bot Hello <|end|><|start|>user How are you? <|end|><|start|>bot"
    return [
        {"role": "user", "prompt": "Hi"},
        {"role": "bot", "prompt": "Hello"},
        {"role": "user", "prompt": "How are you?"}
    ]

    Args:
        input_string (str): The input string.
        start_token (str, optional): The start token. Defaults to "<|start|>".
        end_token (str, optional): The end token. Defaults to "<|end|>".

    Returns:
        list[dict]: The list of conversation inputs.
    """

    # split by start token
    split_by_start = input_string.split(start_token)
    # remove the first element
    split_by_start = split_by_start[1:]

    # remove end token
    split_by_end = [x.split(end_token)[0] for x in split_by_start]

    split_by_role = []
    for sentence in split_by_end:
        # get markdown images if any
        sentence_list = sentence.split("\n")
        role = sentence_list[0]
        sentence = "\n".join(sentence_list[1:])
        images = []
        if "![" in sentence:
            sub_sentences = sentence.split("![")
            for sub_sentence in sub_sentences:
                if "](" in sub_sentence:
                    images.append(sub_sentence.split("](")[1].split(")")[0])
        # Remove images from sentence
        prompt = re.sub(r"!\[.*\]\(.*\)", "", sentence)

        split_by_role.append({"role": role, "prompt": prompt, "images": images})

    # remove empty strings
    return split_by_role


def get_last_user_input(conversations, skip =0):
    """
    Get the last user input from the conversation.
    For example, given the conversation:
    [
        {"role": "user", "prompt": "Hi"},
        {"role": "bot", "prompt": "Hello"},
        {"role": "user", "prompt": "How are you?"}
    ]
    return "How are you?"
    """
    skipped = 0
    for conv in reversed(conversations):
        if conv["role"] == "user":
            if skipped == skip:
                return conv
            skipped += 1
    return None


with open("/shared/nas2/knguye71/ecole-june-demo/parser/functions.json", "r") as f:
    templates = json.load(f)


def similarity(a: str, b: str) -> float:
    """
    Find the similarity between two strings.

    Args:
        a (str): The first string.
        b (str): The second string.

    Returns:
        _type_: _description_
    """
    return SequenceMatcher(None, a, b).ratio()


def match_template(generator: Any, sentence: str, threshold: float = 0.3) -> dict:
    """
    Match a sentence to a template based on similarity.

    Args:
        generator (Any): The LLM generator object.
        sentence (str): The input sentence to match.
        threshold (float, optional): Threshold for the similarity score. Defaults to 0.3.

    Returns:
        dict: The matched template.
    """
    best_match = None
    highest_similarity = 0

    for template in templates:
        template_title = template["title"]
        sim = similarity(sentence, template_title)
        print("template", template_title, "sim", sim)
        if sim > highest_similarity:
            highest_similarity = sim
            best_match = template

    if highest_similarity >= threshold:
        matched_template = best_match.copy()

        params = extract_params_from_sentence_with_prompt(
            generator, sentence, matched_template["params"], matched_template["title"]
        )

        return {
            "template_title": matched_template["title"],
            "params": params,
            "functions": matched_template["functions"],
            "image_required": matched_template["image_required"],
        }

    return {
        "template_title": None,
        "params": None,
        "functions": None,
        "image_required": False,
    }


def extract_params_from_sentence_with_prompt(
    generator: Any, sentence: str, template_params: str, template: str
) -> dict:
    """
    Extract parameters from a given sentence using GPT-2 with a prompt.

    Parameters:
    generator (Any): The LLM generator object.
    sentence (str): The input sentence from which to extract parameters.
    template_params (str): The list of placeholders to extract from the sentence.
    template (str): The template sentence with placeholders for parameters.

    Returns:
    dict: A dictionary of extracted parameters.
    """
    params = {}

    # Create and process prompt for each placeholder
    for placeholder in template_params:
        prompt = f"GPT4 Correct User: The known template is '{template}'. The sentence to analyze: '{sentence}'. What is the '{placeholder}' in the sentence?<|end_of_turn|>GPT4 Correct Assistant: In the sentence, the '{placeholder}' is "
        generated_text = generator(prompt, max_length=200, num_return_sequences=1)[0][
            "generated_text"
        ]

        # Extract the value for the placeholder
        extracted_value = (
            generated_text.replace(prompt, "")
            .strip()
            .split(".")[0]
            .split("\n")[0]
            .strip()
        )
        params[placeholder] = clean_string(extracted_value)

    return params


def convert_base64_to_upload_file(
    arg: str, base64_string: str
) -> tuple[str, tuple[str, io.BytesIO, str]]:
    """
    convert base64 image to upload file, then output the file name, file-like object, and MIME type, along with the argument name.
    For example, given the input string:

    Args:
        arg (str): The argument name.
        base64_string (str): The base64 string.

    Returns:
        [tuple[str, tuple[str, io.BytesIO, str]]]: The argument name and the tuple of file name, file-like object, and MIME type.
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


def substitute_brackets(template: str, values: dict) -> str:
    """
    Substitute the placeholders in a template with the given values.

    Args:
        template (str): The template string.
        values (dict): The dictionary of values to substitute.

    Returns:
        str: The template with the placeholders substituted with the values.
    """
    # Substitute the placeholders in the template with the values
    return template.format(**values)
