from api_call import make_requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from model.Counter import Counter
from transformers import pipeline
from utils import (get_last_user_input, match_template, streaming_response_end,
                   streaming_response_yield, strip_tokens)

load_dotenv()
app = FastAPI()

# Initialize the text generation pipeline
generator = pipeline(
    "text-generation",
    model="openchat/openchat-3.5-0106",
    device_map="auto",
)

@app.get("/")
async def read_root():
    """
    Return message you got to the correct endpoint. By default, the endpoint is /generate.

    Returns:
        str: The message.
    """
    return {"message": "You got to the correct endpoint. Please use the /generate endpoint to generate text."}


@app.post("/generate")
async def generate_text(request: Request) -> None:
    """
    Generate text based on the input prompt. The input prompt should be a list of conversation turns.
    The function will match the input prompt with a template and extract the parameters from the prompt.
    The function will then call the required APIs to generate the text.

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

    conversations = strip_tokens(prompt)

    last_input = get_last_user_input(conversations)

    # Generate text based on the input prompt
    async def text_streamer():
        """
        The main function that generates the text based on the input prompt. The function will match the input prompt with a template and extract the parameters from the prompt.

        Yields:
            str: The generated text. In format "data: {json}\n\n".
        """
        
        # Initialize the counter
        counter = Counter()
        
        # Matching the last input prompt with a template
        prompt = last_input["prompt"]
        obj = match_template(generator, prompt)
        
        # Get the template title, parameters, functions, and image_required flag
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
            # Show status message
            res_str = "Processing...\n\n"
        else:
            # if there is no template, just generate text
            llm_res = generator(prompt, max_length=50, num_return_sequences=1)
            res_str += llm_res[0]["generated_text"]

        # yield the status message
        yield streaming_response_yield(res_str, counter)

        # Initialize the last response string
        last_res_str = ""
        
        # Update the last response string with the new chunk function
        def update_last_stream(chunk):
            nonlocal last_res_str
            last_res_str += chunk

        # Call the APIs to generate the text
        if template_title is not None:
            async for response in make_requests(
                functions, params, last_input["images"], userId, callback = update_last_stream
            ):
                yield response
        
        # yield the final response string, this will be the final response, only happens when all the APIs have been called successfully
        yield streaming_response_end(last_res_str, counter)

    async def event_generator():
        """
        Wrapper function for the text_streamer function. This function will return a StreamingResponse object.

        Yields:
            str: The generated text. In format "data: {json}\n\n".
        """
        event_gen = text_streamer()
        async for event in event_gen:
            # If the client has disconnected, stop the generator
            if await request.is_disconnected():
                break
            yield event

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
def health():
    """
    Health check endpoint.

    Returns:
        dict: The health check response.
    """
    return {"status": "healthy"}
import json


@app.get("/commands")
def commands():
    """
    Return the list of available commands.

    Returns:
        dict: The list of available commands.
    """
    with open("/shared/nas2/knguye71/ecole-june-demo/parser/functions.json") as f:
        commands = json.load(f)
    return JSONResponse(content=commands)
# Run the server using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
