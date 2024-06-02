from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import pipeline
from typing import Any
import json
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Initialize the text generation pipeline
generator = pipeline(
    "text-generation", device="cuda", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
START_TOKEN = "<|im_start|>"
END_TOKEN = "<|im_end|>"
IMAGE_BACKEND_API = os.environ.get("IMAGE_BACKEND_API", "http://localhost:8000")


@app.get("/")
async def read_root():
    return {"Hello": "World"}

def streaming_response_yield(output_string, index):
    return "data: " + json.dumps(
        {
            "token": {
                "id": str(index),
                "text": output_string + " ",
                "logprob": 0,
                "special": False,
            },
            "generated_text": None,
            "details": None,
        }
    ) + "\n\n"

def streaming_response_end(output_string, index):
    return "data: " + json.dumps(
        {
            "token": {
                "id": index,
                "text": "",
                "special": True,
                "logprob": 0,
            },
            "generated_text": output_string,
            "details": {
                "finish_reason": "eos_token",
                "num_tokens": len(output_string.split()),
            },
        }
    ) + "\n\n"

def strip_tokens(input_string, start_token = START_TOKEN, end_token = END_TOKEN):
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
    print("split_by_role", split_by_role)

    # remove empty strings
    return split_by_role

def get_last_user_input(conversations):
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
    for conv in reversed(conversations):
        if conv["role"] == "user":
            return conv
    return None

import re
from difflib import SequenceMatcher
with open("/shared/nas2/knguye71/ecole-june-demo/parser/functions.json", "r") as f:
    templates = json.load(f)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def match_template(sentence, threshold=0.3):
    print ("sentence", sentence)
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

        params = extract_params_from_sentence_with_prompt(sentence, matched_template['params'], matched_template["title"])
        
        return {
            "template_title": matched_template["title"],
            "params": params,
            "functions": matched_template["functions"],
            "image_required": matched_template["image_required"]
        }

    return {
        "template_title": None,
        "params": None,
        "functions": None,
        "image_required": False
    }

def extract_params_from_sentence_with_prompt(sentence, template_params, template):
    """
    Extract parameters from a given sentence using GPT-2 with a prompt.

    Parameters:
    sentence (str): The input sentence from which to extract parameters.
    template_params (str): The list of placeholders to extract from the sentence.
    template (str): The template sentence with placeholders for parameters.

    Returns:
    dict: A dictionary of extracted parameters.
    """
    params = {}

    # Create and process prompt for each placeholder
    for placeholder in template_params:
        prompt = f"The known template sentence is {template}. Extract the '{placeholder}' in curly brackets from this sentence: '{sentence}'.  The {placeholder} is:"
        generated_text = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']

        # Extract the value for the placeholder
        extracted_value = generated_text.replace(prompt, "").strip().split('.')[0].split('\n')[0].strip()
        params[placeholder] = extracted_value

    return params


def make_request
@app.post("/generate")
async def generate_text(request: Request):

    # Read the input data
    data = await request.json()
    prompt = data["inputs"]

    conversations = strip_tokens(prompt)

    last_input = get_last_user_input(conversations)

    # Generate text

    async def text_streamer():
        prompt = last_input["prompt"]
        print("prompt", prompt)
        obj = match_template(prompt)
        template_title = obj["template_title"]
        params = obj["params"]
        functions = obj["functions"]
        image_required = obj["image_required"]
        
        # Check if the template requires an image and if there is an image in the last input
        if image_required and len(last_input["images"]) == 0:
            yield streaming_response_end("Sorry, but you forgot to input an image", 0)
            return
        if template_title is not None:
            res_str = f"Template matched: {template_title}\nParameters extracted: {params}\nFunctions to be executed: {functions}\n image_required: {image_required}"
            
        else:
            res_str = "No template matched." 
            llm_res = generator(prompt, max_length=50, num_return_sequences=1)
            res_str += llm_res[0]["generated_text"]
        for i, token in enumerate(res_str.split()):
            yield streaming_response_yield(token, i)
        yield streaming_response_end(res_str, len(res_str.split()))

        # token_id = 0
        # for text in result[0]["generated_text"].split():

        #     yield streaming_response_yield(text, token_id)
        #     token_id += 1

        # yield streaming_response_end(result[0]["generated_text"], token_id)

    async def event_generator():
        event_gen = text_streamer()
        async for event in event_gen:
            # If the client has disconnected, stop the generator
            if await request.is_disconnected():
                break
            yield event

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
