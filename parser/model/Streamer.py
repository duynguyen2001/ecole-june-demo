import asyncio
import inspect
import json
from typing import AsyncGenerator, Generator, Optional, Union

from Counter import Counter


class TextStreamer:
    """
    The TextStreamer class is used to stream text responses to the client. The class is used to generate the streaming response for the yield statement.
    """
    def __init__(self):
        self.request_count = Counter()

    def format_string(self, request: str, finalized: bool = False):
        """
        Generate the streaming response. This function is used to generate the streaming response for the yield statement.

        Args:
            output_string (str): The output string.
            counter (Counter): The counter object.

        """
        if finalized:
            return (
                "data: "
                + json.dumps(
                    {
                        "token": {
                            "id": self.request_count.count,
                            "text": "",
                            "special": True,
                            "logprob": 0,
                        },
                        "generated_text": f"""{request}\n\n""",
                        "details": {
                            "finish_reason": "eos_token",
                            "num_tokens": len(request.split()),
                        },
                    }
                )
                + "\n\n"
            )
        else:
            return (
                "data: "
                + json.dumps(
                    {
                        "token": {
                            "id": request.count,
                            "text": request + "\n\n",
                            "logprob": 0,
                            "special": False,
                        },
                        "generated_text": None,
                        "details": None,
                    }
                )
                + "\n\n"
            )

    def stream(
        self, requests: Union[Generator, list], finalized=False
    ) -> Generator[str, str, None]:
        if inspect.isgenerator(requests):
            for request in requests:
                self.request_count.increment()
                formatted_request = self.format_string(request, finalized)
                yield formatted_request
                response = self.process_request(formatted_request)
                yield response
        else:
            for request in requests:
                self.request_count.increment()
                formatted_request = self.format_string(request, finalized)
                yield formatted_request
                response = self.process_request(formatted_request)
                yield response

    async def stream_async(
        self, async_requests: Union[AsyncGenerator, Generator, list], finalized=False
    ) -> AsyncGenerator[str, str]:
        """
        Stream a list of requests asynchronously.

        Args:

        """
        if inspect.isasyncgen(async_requests):
            async for request in async_requests:
                self.request_count.increment()
                formatted_request = self.format_string(request, finalized=finalized)
                yield formatted_request
                response = await self.process_request_async(formatted_request)
                yield response
        elif inspect.isgenerator(async_requests):
            for request in async_requests:
                self.request_count.increment()
                formatted_request = self.format_string(request, finalized)
                yield formatted_request
                response = self.process_request(formatted_request)
                yield response
        else:
            for request in async_requests:
                self.request_count.increment()
                formatted_request = self.format_string(request, finalized)
                yield formatted_request
                response = await self.process_request_async(formatted_request)
                yield response


# Example usage with a list of requests:
requests = ["First message", "Second message", "Third message"]

text_streamer = TextStreamer()
for message in text_streamer.stream(requests):
    print(message)


# Example usage with a synchronous generator:
def request_generator():
    requests = [
        "First sync gen message",
        "Second sync gen message",
        "Third sync gen message",
    ]
    for request in requests:
        yield request


for message in text_streamer.stream(request_generator()):
    print(message)


# Example usage with an asynchronous generator:
async def async_request_generator():
    requests = ["First async message", "Second async message", "Third async message"]
    for request in requests:
        await asyncio.sleep(0.5)  # Simulate async request generation
        yield request


async def main():
    async for message in text_streamer.stream_async(async_request_generator()):
        print(message)


# Run the async example
asyncio.run(main())
