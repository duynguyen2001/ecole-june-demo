from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import pipeline
from typing import Any
import json
import os
from dotenv import load_dotenv
import httpx
from utils import streaming_response_yield, streaming_response_end, match_template, strip_tokens, get_last_user_input, Counter
from api_call import make_requests
load_dotenv("/shared/nas2/knguye71/ecole-june-demo/parser/.env")
app = FastAPI()

# Initialize the text generation pipeline
generator = pipeline(
    "text-generation",
    model="openchat/openchat-3.5-0106",
    device_map="auto",
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/generate")
async def generate_text(request: Request) -> None:
    """
    Generate text based on the input prompt.

    Args:
        request (Request): The request object.

    Returns:
        streaming_response: The generated text.

    Yields:
        str: The generated text. In format "data: {json}\n\n".
    """

    # Read the input data
    data = await request.json()
    prompt = data["inputs"]
    userId = data["userId"] if "userId" in data else "default_user"
    print(f"\n\n\n\nUser {userId}\n\n\n\n")

    conversations = strip_tokens(prompt)

    last_input = get_last_user_input(conversations)

    # Generate text

    async def text_streamer():
        counter = Counter()
        prompt = last_input["prompt"]
        obj = match_template(generator, prompt)
        template_title = obj["template_title"]
        params = obj["params"]
        functions = obj["functions"]
        image_required = obj["image_required"]

        # Check if the template requires an image and if there is an image in the last input
        if image_required and len(last_input["images"]) == 0:
            yield streaming_response_end(
                "Sorry, but you forgot to input an image", counter
            )
            return
        elif template_title is not None:
            res_str = f"Template matched: {template_title}\nParameters extracted: {params}\n\n"
        else:
            res_str = "No template matched." 
            llm_res = generator(prompt, max_length=50, num_return_sequences=1)
            res_str += llm_res[0]["generated_text"]

        yield streaming_response_yield(res_str, counter)

        last_res_str = ""
        def update_last_stream(chunk):
            nonlocal last_res_str
            last_res_str += chunk

        if template_title is not None:
            async for response in make_requests(
                functions, params, last_input["images"], userId, callback = update_last_stream
            ):
                yield response
                
        yield streaming_response_end(last_res_str, counter)
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

    uvicorn.run(app, host="127.0.0.1", port=8003)
