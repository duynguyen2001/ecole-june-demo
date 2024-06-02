from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import pipeline
from typing import Any
import json
import time

app = FastAPI()

# Initialize the text generation pipeline
generator = pipeline("text-generation", device="cuda")
START_TOKEN = "<|im_start|>"
END_TOKEN = "<|im_end|>"


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
    Strips the specified start and end tokens from the input string.

    Args:
    input_string (str): The string from which to strip the tokens.
    start_token (str): The token to strip from the start of the string.
    end_token (str): The token to strip from the end of the string.

    Returns:
    str: The input string with the start and end tokens removed.
    """
    if input_string.startswith(start_token):
        input_string = input_string[len(start_token) :]
    if input_string.endswith(end_token):
        input_string = input_string[: -len(end_token)]
    return input_string

import re
from difflib import SequenceMatcher
with open("/shared/nas2/knguye71/ecole-june-demo/parser/functions.json", "r") as f:
    templates = json.load(f)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def match_template(sentence, threshold=0.5):
    print ("sentence", sentence)
    best_match = None
    highest_similarity = 0

    for template in templates:
        template_title = template["title"]
        sim = similarity(sentence, template_title)
        if sim > highest_similarity:
            highest_similarity = sim
            best_match = template

    if highest_similarity >= threshold:
        matched_template = best_match.copy()
        
        params = extract_params_from_sentence_with_prompt(sentence, matched_template["prompt"])
        print("params", params)
        return {"matched_template": match_template, "params": params, "function": matched_template["function"]}

    return None

def extract_params_from_sentence_with_prompt(sentence, template):
    """
    Extract parameters from a given sentence using GPT-2 with a prompt.

    Parameters:
    sentence (str): The input sentence from which to extract parameters.
    template (str): The template sentence with placeholders for parameters.

    Returns:
    dict: A dictionary of extracted parameters.
    """
    params = {}
    
    # Identify placeholders in the template
    placeholders = [word.strip('{}') for word in template.split() if word.startswith('{') and word.endswith('}')]
    
    # Create and process prompt for each placeholder
    for placeholder in placeholders:
        prompt = f"Extract the {placeholder} from this sentence: '{sentence}'. The {placeholder} is:"
        generated_text = generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        
        # Extract the value for the placeholder
        extracted_value = generated_text.replace(prompt, "").strip().split('.')[0]
        params[placeholder] = extracted_value
    
    return params

@app.post("/generate")
async def generate_text(request: Request):
    print("Request received")

    # Read the input data
    data = await request.json()
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 50)

    prompt = strip_tokens(prompt)
    print("prompt", prompt)

    # Generate text
    # result = generator(prompt, max_length=max_length)
    # print("result", result)

    async def text_streamer():
        obj = match_template(prompt)
        match_template = obj["matched_template"]
        params = obj["params"]
        functions = obj["function"]
        res_str = f"Matched template: {match_template['title']}\n + Parameters: {params} Function: {functions}"
        print(res_str)
        yield streaming_response_end(res_str, 0)
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
