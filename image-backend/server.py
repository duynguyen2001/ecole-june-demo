# %%

from dotenv import load_dotenv

load_dotenv()
import logging
# %%
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import List

import torch

torch.autograd.set_detect_anomaly(True)

import PIL.Image
from agent_manager import IMAGE_DIR, TENSOR_DIR, AgentManager
################
# FastAPI      #
################
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import (FileResponse, JSONResponse, Response,
                               StreamingResponse)
from streaming_methods import (streaming_concept_kb, streaming_diff_images,
                               streaming_heatmap,
                               streaming_heatmap_class_difference,
                               streaming_hierachical_predict_result,
                               yield_nested_objects)

logger = logging.getLogger("uvicorn.error")

tags_metadata = [
    {
        "name": "predict",
        "description": "Predict methods from an image.",
    },
    {"name": "train", "description": "Train methods from images."},
    {"name": "compare", "description": "Compare methods"},
    {"name": "kb_ops", "description": "Knowledge base operations"},
    {"name": "data_ops", "description": "Data operations"},
]


app = FastAPI(
    openapi_tags=tags_metadata,
    title="ECOLE API",
    description="API for ECOLE Image Demo",
    version="0.1.0",
)


@asynccontextmanager
async def agent_lifespan(agent):
    # Initialization code here. For example, agent.start() if your agent needs explicit starting.
    yield agent
    # Teardown code here
    agent.shutdown()


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent_manager = AgentManager()
    app.state.agentmanager = agent_manager
    logger.info(str("AgentManager initialized"))
    yield
    agent_manager.shutdown()
    logger.info(str("AgentManager shut down"))


app.router.lifespan_context = lifespan


############################
# Predict methods          #
############################
@app.post("/predict_from_subtree", tags=["predict"])
async def predict_from_subtree(
    user_id: str,  # Required field
    root_concept_name: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.7",
    streaming: str = "false",
):
    if streaming == "true":
        img = PIL.Image.open(image.file).convert("RGB")

        async def streamer(img, root_concept_name, unk_threshold):
            time_start = time.time()
            # Convert to PIL Image
            yield "status: Predicting from subtree..."
            try:
                result = app.state.agentmanager.predict_from_subtree(
                    user_id, img, root_concept_name, float(unk_threshold)
                )
                logger.info(
                    str("Predict from subtree time: " + str(time.time() - time_start))
                )
                # yield f"result: {json.dumps(result)}"
                async for msg in streaming_hierachical_predict_result(result):
                    yield msg
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img, root_concept_name, unk_threshold),
            media_type="text/event-stream",
        )
    else:
        time_start = time.time()
        # Convert to PIL Image
        img = PIL.Image.open(image.file)
        img = img.convert("RGB")
        try:
            result = app.state.agentmanager.predict_from_subtree(
                user_id, img, root_concept_name, float(unk_threshold)
            )
            logger.info(
                str("Predict from subtree time: " + str(time.time() - time_start))
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict_hierarchical", tags=["predict"])
async def predict_hierarchical(
    user_id: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.7",
    include_component_concepts: str = "False",
    streaming: str = "false",
    show_explanation: str = "false"
):
    logger.info(f"streaming: {streaming}")
    if streaming == "true":
        img = PIL.Image.open(image.file).convert("RGB")

        async def streamer(img, unk_threshold, include_component_concepts, show_explain):
            time_start = time.time()
            # Convert to PIL Image

            yield "status: Predicting hierarchical...\n\n"

            try:
                result = app.state.agentmanager.predict_hierarchical(
                    user_id,
                    img,
                    float(unk_threshold),
                    (
                        include_component_concepts
                        if include_component_concepts == "True"
                        else False
                    ),
                )

                logger.info(
                    str("Predict hierarchical time: " + str(time.time() - time_start))
                )
                # async for msg in yield_nested_objects(result):
                #     yield msg
                show_exp = (show_explain == "true")
                async for msg in streaming_hierachical_predict_result(result, show_explanation=show_exp):
                    yield msg
                # for msg in yield_nested_objects(result):
                #     yield msg
                # yield f"result: {json.dumps(result)}"
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img, unk_threshold, include_component_concepts, show_explanation),
            media_type="text/event-stream",
        )
    else:
        time_start = time.time()
        # Convert to PIL Image
        img = PIL.Image.open(image.file)
        img = img.convert("RGB")
        try:
            result = app.state.agentmanager.predict_hierarchical(
                user_id, img, float(unk_threshold), include_component_concepts
            )
            logger.info(
                str("Predict hierarchical time: " + str(time.time() - time_start))
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict_root_concept", tags=["predict"])
async def predict_root_concept(
    user_id: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.7",
    streaming: str = "false",
):
    if streaming == "true":
        img = PIL.Image.open(image.file).convert("RGB")

        async def streamer(img, unk_threshold):
            time_start = time.time()
            # Convert to PIL Image
            yield "status: Predicting root concept..."
            try:
                result = app.state.agentmanager.predict_root_concept(
                    user_id, img, float(unk_threshold)
                )
                logger.info(
                    str("Predict root concept time: " + str(time.time() - time_start))
                )
                for msg in yield_nested_objects(result):
                    yield msg
                # yield f"result: {json.dumps(result)}"
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img, unk_threshold),
            media_type="text/event-stream",
        )
    else:
        time_start = time.time()
        # Convert to PIL Image
        img = PIL.Image.open(image.file)
        img = img.convert("RGB")
        try:
            result = app.state.agentmanager.predict_root_concept(
                user_id, img, float(unk_threshold)
            )
            logger.info(
                str("Predict root concept time: " + str(time.time() - time_start))
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/is_concept_in_image", tags=["predict"])
async def is_concept_in_image(
    user_id: str,  # Required field
    concept_name: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    unk_threshold: str = "0.1",
    streaming: str = "false",
):
    if streaming == "true":
        img = PIL.Image.open(image.file).convert("RGB")

        async def streamer(img, concept_name, unk_threshold):
            time_start = time.time()
            # Convert to PIL Image
            yield "status: Checking if concept in image..."
            try:
                heatmap = app.state.agentmanager.heatmap(user_id, img, concept_name)
                logger.info(
                    str("Is concept in image time: " + str(time.time() - time_start))
                )
                for msg in streaming_heatmap(heatmap, concept_name):
                    yield msg
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img, concept_name, unk_threshold),
            media_type="text/event-stream",
        )
    else:
        time_start = time.time()
        # Convert to PIL Image
        img = PIL.Image.open(image.file)
        img = img.convert("RGB")
        try:
            result = app.state.agentmanager.is_concept_in_image(
                user_id, img, concept_name, float(unk_threshold)
            )
            logger.info(
                str("Is concept in image time: " + str(time.time() - time_start))
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


###############
# Compare Ops #
###############
@app.post("/heatmap_image_comparison", tags=["compare"])
async def heatmap_image_comparison(
    user_id: str,  # Required field
    image_1: UploadFile = File(...),  # Required file upload,
    image_2: UploadFile = File(...),  # Required file upload,
    streaming: str = "false",
):
    if streaming == "true":
        img1 = PIL.Image.open(image_1.file).convert("RGB")
        img2 = PIL.Image.open(image_2.file).convert("RGB")

        async def streamer(img1, img2):
            time_start = time.time()
            yield "status: Generating heatmap image comparison..."
            try:
                result = app.state.agentmanager.heatmap_image_comparison(
                    user_id, img1, img2
                )
                for msg in streaming_diff_images(result):
                    yield msg
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img1, img2),
            media_type="text/event-stream",
        )
    else:
        img1 = PIL.Image.open(image_1.file).convert("RGB")
        img2 = PIL.Image.open(image_2.file).convert("RGB")
        try:
            result = app.state.agentmanager.heatmap_image_comparison(
                user_id, img1, img2
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/heatmap_class_difference", tags=["compare"])
def heatmap_class_difference(
    user_id: str,  # Required field
    concept1_name: str,  # Required field
    concept2_name: str,
    image: UploadFile | None = None,  # file upload,
    streaming: str = "false",
):
    """
    If image is provided, implements:
        "Why is this <class x> and not <class y>"

    Otherwise, implements:
        "What is the difference between <class x> and <class y>"

    Args:
        user_id: str: Required field
        concept1_name: str: Required field
        concept2_name: str: Required field
        image: UploadFile: file upload
        streaming: str: "false"

    Returns:
        JSONResponse: content=result

    Raises:
        HTTPException: status_code=404, detail=str(e)

    Yields:
        StreamingResponse: streamer(concept1_name, concept2_name, img)
    """

    if streaming == "true":
        img = PIL.Image.open(image.file).convert("RGB") if image is not None else None

        async def streamer(concept1_name, concept2_name, img):
            time_start = time.time()
            yield "status: Generating heatmap class difference..."
            try:
                result = app.state.agentmanager.heatmap_class_difference(
                    user_id,
                    concept1_name,
                    concept2_name,
                    image=img,
                )
                logger.info(
                    str(
                        "Heatmap class difference time: "
                        + str(time.time() - time_start)
                    )
                )

                for msg in streaming_heatmap_class_difference(
                    result, concept1_name, concept2_name, img
                ):
                    yield msg
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(concept1_name, concept2_name, img),
            media_type="text/event-stream",
        )
    else:
        img = PIL.Image.open(image.file).convert("RGB") if image is not None else None
        try:
            result = app.state.agentmanager.heatmap_class_difference(
                user_id,
                concept1_name,
                concept2_name,
                img,
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/heatmap", tags=["compare"])
def heatmap(
    user_id: str,  # Required field
    concept_name: str,  # Required field
    image: UploadFile = File(...),  # Required file upload,
    only_score_increasing_regions: str = "false",
    only_score_decreasing_regions: str = "false",
    streaming: str = "false",
):
    """
    If only_score_increasing_regions is True, implements:
        "Why is this a <class x>"

    If only_score_decreasing_regions is True, implements:
        "What are the differences between this and <class x>"

    If neither is true, shows the full heatmap (both increasing and decreasing regions).

    """
    only_score_increasing_regions_param = (
        True if only_score_increasing_regions == "true" else False
    )
    only_score_decreasing_regions_param = (
        True if only_score_decreasing_regions == "true" else False
    )
    logger.info(f"streaming: {streaming}")
    if streaming == "true":
        img = PIL.Image.open(image.file).convert("RGB")

        async def streamer(img, concept_name):
            time_start = time.time()
            yield "status: Generating heatmap..."
            try:
                result = app.state.agentmanager.heatmap(
                    user_id,
                    img,
                    concept_name,
                    only_score_increasing_regions_param,
                    only_score_decreasing_regions_param,
                )
                logger.info(str("Heatmap time: " + str(time.time() - time_start)))
                for msg in streaming_heatmap(
                    result,
                    concept_name,
                    only_score_increasing_regions_param,
                    only_score_decreasing_regions_param,
                ):
                    yield msg
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(img, concept_name),
            media_type="text/event-stream",
        )
    else:
        img = PIL.Image.open(image.file).convert("RGB")
        try:
            result = app.state.agentmanager.heatmap(
                user_id,
                img,
                concept_name,
                only_score_increasing_regions,
                only_score_decreasing_regions,
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))


###############
# Kb Ops      #
###############


@app.post("/reset_kb", tags=["kb_ops"])
async def reset_kb(
    user_id: str,  # Required field
    clear_all: str = "false",
    streaming: str = "false",
):
    if streaming == "true":

        async def streamer(user_id, clear_all):
            time_start = time.time()
            yield "status: Resetting concept KB..."
            try:
                if clear_all == "true":
                    app.state.agentmanager.reset_kb(user_id, True)
                else:
                    app.state.agentmanager.reset_kb(user_id, False)
                logger.info(
                    str("Reset concept KB time: " + str(time.time() - time_start))
                )
                yield f"result: Knowledge base successfully reset {'from scratch' if clear_all == "true" else 'default checkpoint'}\n\n"
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(user_id, clear_all),
            media_type="text/event-stream",
        )
    else:
        try:
            if clear_all == "True":
                app.state.agentmanager.reset_conceptkb(user_id, True)
            else:
                app.state.agentmanager.reset_conceptkb(user_id, False)

            return Response(status_code=200)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

@app.post("/undo_kb", tags=["kb_ops"])
def undo_kb(user_id: str, streaming: str = "false"):
    if streaming == "true":
        async def streamer(user_id):
            time_start = time.time()
            yield "status: Undoing last operation..."
            try:
                app.state.agentmanager.undo_kb(user_id)
                logger.info(str("Undo KB time: " + str(time.time() - time_start)))
                yield f"result: Last operation undone\n\n"
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(user_id),
            media_type="text/event-stream",
        )
    else:
        try:
            app.state.agentmanager.undo_kb(user_id)
            return Response(status_code=200)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

@app.post("/add_hyponym", tags=["kb_ops"])
async def add_hyponym(
    user_id: str,  # Required field
    child_name: str,  # Required field
    parent_name: str,  # Required field
    child_max_retrieval_distance: str = "0",
    streaming: str = "alse",
):
    if streaming == "true":

        async def streamer(
            user_id, child_name, parent_name, child_max_retrieval_distance
        ):
            time_start = time.time()
            yield "status: Adding hyponyms..."
            try:
                result = app.state.agentmanager.add_hyponym(
                    user_id,
                    child_name,
                    parent_name,
                    float(child_max_retrieval_distance),
                    streaming=streaming,
                )
                logger.info(str("Add hyponym time: " + str(time.time() - time_start)))
                yield f"result: {result}"
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}\n\n"

        return StreamingResponse(
            streamer(user_id, child_name, parent_name, child_max_retrieval_distance),
            media_type="text/event-stream",
        )
    else:
        try:
            result = app.state.agentmanager.add_hyponym(
                user_id, child_name, parent_name, float(child_max_retrieval_distance)
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/add_component_concept", tags=["kb_ops"])
async def add_component_concept(
    user_id: str,  # Required field
    component_concept_name: str,  # Required field
    concept_name: str,  # Required field
    component_max_retrieval_distance: str = "0.7",
    streaming: str = "false",
):
    if streaming == "true":
        time_start = time.time()

        async def streamer(
            user_id,
            component_concept_name,
            concept_name,
            component_max_retrieval_distance,
        ):
            yield "status: Adding component concept..."
            try:
                result = app.state.agentmanager.add_component_concept(
                    user_id,
                    component_concept_name,
                    concept_name,
                    float(component_max_retrieval_distance),
                    streaming=streaming,
                )
                logger.info(
                    str("Add component concept time: " + str(time.time() - time_start))
                )
                yield f"result: {result}\n\n"
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(
                user_id,
                component_concept_name,
                concept_name,
                component_max_retrieval_distance,
            ),
            media_type="text/event-stream",
        )
    else:
        try:
            result = app.state.agentmanager.add_component_concept(
                user_id,
                component_concept_name,
                concept_name,
                float(component_max_retrieval_distance),
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/add_examples", tags=["kb_ops"])
async def add_examples(
    user_id: str,  # Required field
    concept_name: str,  # Required field
    images: List[UploadFile] = File(...),  # Required field
    streaming: str = "false",
):

    processed_images = [PIL.Image.open(image.file).convert("RGB") for image in images]
    if streaming == "true":

        async def streamer(user_id, concept_name, imgs):
            yield "status: Adding examples...\n\n"
            try:
                async for msg in app.state.agentmanager.add_examples(
                    user_id, imgs, concept_name, streaming=streaming
                ):
                    yield msg
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"
            yield "result: Examples added\n\n"

        return StreamingResponse(
            streamer(user_id, concept_name, processed_images),
            media_type="text/event-stream",
        )
    else:
        try:
            processed_images = [PIL.Image.open(image.file).convert("RGB") for image in images]
            result = app.state.agentmanager.add_examples(user_id, concept_name, processed_images)
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/add_concept_negatives", tags=["kb_ops"])
async def add_concept_negatives(
    user_id: str,  # Required field
    concept_name: str,  # Required field
    images: List[UploadFile] = File(...),  # Required field
    streaming: str = "false",
):
    
    processed_images = [
        PIL.Image.open(negative.file).convert("RGB") for negative in images
    ]
    if streaming == "true":

        async def streamer(user_id, concept_name, imgs):
            yield "status: Adding concept negatives..."
            try:
                result = app.state.agentmanager.add_concept_negatives(
                    user_id, imgs, concept_name, streaming=streaming
                )
                async for msg in result:
                    yield msg
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(user_id, concept_name, processed_images),
            media_type="text/event-stream",
        )
    else:
        try:
            result = app.state.agentmanager.add_concept_negatives(
                user_id,
                processed_images,
                concept_name,
            )
            return JSONResponse(content=result)
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/get_concept_kb", tags=["kb_ops"])
def get_concept_kb(user_id: str, streaming: str = "false"):
    if streaming == "true":
            async def streamer(user_id):
                yield "status: Getting concept KB..."
                try:
                    result = app.state.agentmanager.get_concept_kb(user_id)
                    for msg in streaming_concept_kb(result):
                        yield msg
                except Exception as e:
                    logger.error(traceback.format_exc())
                    logger.error(sys.exc_info())
                    yield f"error: {str(e)}"
    
            return StreamingResponse(
                streamer(user_id),
                media_type="text/event-stream",
            )
    try:
        result = app.state.agentmanager.get_concept_kb(user_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    
@app.post("/get_checkpoint_list", tags=["kb_ops"])
def get_checkpoint_list(user_id: str, streaming: str = "false"):
    if streaming == "true":
            
            async def streamer(user_id):
                yield "status: Getting checkpoint list..."
                try:
                    result = app.state.agentmanager.get_checkpoint_list(user_id)
                    yield "result:```checkpoint_list: " + str(result) + "```\n\n"
                except Exception as e:
                    import sys
                    import traceback
    
                    logger.error(traceback.format_exc())
                    logger.error(sys.exc_info())
                    yield f"error: {str(e)}"
    
            return StreamingResponse(
                streamer(user_id),
                media_type="text/event-stream",
            )
    try:
        result = app.state.agentmanager.get_checkpoint_list(user_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    

    
@app.post("/load_checkpoint", tags=["kb_ops"])
def load_checkpoint(user_id: str, checkpoint_name: str, streaming: str = "false"):
    if streaming == "true":
            
            async def streamer(user_id, checkpoint_name):
                yield "status: Loading checkpoint..."
                try:
                    result = app.state.agentmanager.load_checkpoint(user_id, checkpoint_name)
                    yield f"result: {result}"
                except Exception as e:
                    import sys
                    import traceback
    
                    logger.error(traceback.format_exc())
                    logger.error(sys.exc_info())
                    yield f"error: {str(e)}"
    
            return StreamingResponse(
                streamer(user_id, checkpoint_name),
                media_type="text/event-stream",
            )
    try:
        result = app.state.agentmanager.load_checkpoint(user_id, checkpoint_name)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/train_concepts", tags=["train"])
async def train_concepts(
    user_id: str,  # Required field
    concepts: str,
    streaming: str = "false",
):

    async def streamer(user_id, cncpts):
        yield "status: \nTraining concepts...\n"
        try:
            time_start = time.time()
            async for res in app.state.agentmanager.train_concepts(user_id, cncpts, streaming=streaming):
                yield res
            logger.info(
                str("Train concepts time: " + str(time.time() - time_start))
            )
            
        except Exception as e:
            import sys
            import traceback

            logger.error(traceback.format_exc())
            logger.error(sys.exc_info())
            yield f"error: {str(e)}"

    return StreamingResponse(
        streamer(user_id, concepts.split(", ")),
        media_type="text/event-stream",
    )

        
@app.post("/undo_concept_kb")
def undo_concept_kb(user_id: str, streaming: str = "false"):
    if streaming == "true":
        async def streamer(user_id):
            time_start = time.time()
            yield "status: Undoing last operation..."
            try:
                app.state.agentmanager.undo_kb(user_id)
                logger.info(str("Undo KB time: " + str(time.time() - time_start)))
                yield f"result: Last operation undone\n\n"
            except Exception as e:
                import sys
                import traceback

                logger.error(traceback.format_exc())
                logger.error(sys.exc_info())
                yield f"error: {str(e)}"

        return StreamingResponse(
            streamer(user_id),
            media_type="text/event-stream",
        )
    else:
        try:
            app.state.agentmanager.undo_concept_kb(user_id)
            return Response(status_code=200)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))


@app.post("/images")
def upload_image(image: UploadFile = File(...)):
    img = PIL.Image.open(image.file)
    img.save(os.path.join(IMAGE_DIR, image.filename))
    return {"filename": image.filename}


@app.get("/images/{uid}", tags=["data_ops"])
def get_image(uid):
    filename = f"{uid}.jpg"
    return FileResponse(os.path.join(IMAGE_DIR, filename), media_type="image/jpg")

@app.get("/tensor/{uid}", tags=["data_ops"])
def get_tensor(uid):
    filename = f"{uid}.json"
    return FileResponse(
        os.path.join(TENSOR_DIR, filename), media_type="application/octet-stream"
    )

import asyncio


@app.post("/healthcheck")
async def healthcheck():
    async def streamer():
        for _ in range(5):
            await asyncio.sleep(1)
            yield "status: Healthy\n\n"
        
        await asyncio.sleep(1)
        yield "result: All systems operational\n\n"
    return StreamingResponse(
        streamer(),
        media_type="text/event-stream",
    )
    

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=16004)

