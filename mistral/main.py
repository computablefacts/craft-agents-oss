import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
from dotenv import load_dotenv
import logging
import json
import asyncio

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    try:
        payload = await request.json()
        if "store" in payload:
            del payload["store"]
        if "stream_options" in payload:
            del payload["stream_options"]
        if "max_completion_tokens" in payload:
            del payload["max_completion_tokens"]
        stream = payload.get("stream", False)
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        if stream:
            # logger.info(f"Mode streaming activé pour la requête : {payload}")
            # --- Mode STREAMING ---
            async def generate():
                try:
                    timeout = httpx.Timeout(120.0, connect=60.0)
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        async def fetch_response(retry=True):
                            async with client.stream(
                                    "POST",
                                    f"{MISTRAL_BASE_URL}/chat/completions",
                                    json=payload,
                                    headers=headers
                            ) as mistral_response:
                                if mistral_response.status_code == 429 and retry:
                                    logger.warning(f"Rate Limit atteint. Attente de 1 seconde...")
                                    await asyncio.sleep(1)
                                    async for chunk in fetch_response(retry=False):
                                        yield chunk
                                    return
                                if mistral_response.status_code != 200:
                                    error_detail = await mistral_response.aread()
                                    try:
                                        error_json = json.loads(error_detail)
                                        logger.error(f"Erreur Mistral (streaming): {error_json}")
                                        yield f"data: {json.dumps({'error': error_json})}\n\n"
                                    except:
                                        error_text = error_detail.decode("utf-8", errors="replace")
                                        logger.error(f"Erreur Mistral (streaming): {error_text}")
                                        yield f"data: {json.dumps({'error': error_text})}\n\n"
                                    return
                                async for line in mistral_response.aiter_lines():
                                    line = line.strip()
                                    if not line:
                                        continue
                                    if line.startswith("data:"):
                                        yield f"{line}\n\n"
                                    elif line == "[DONE]":
                                        yield f"data: [DONE]\n\n"
                        async for chunk in fetch_response():
                            yield chunk
                except Exception as e:
                    logger.exception(f"Erreur dans le flux streaming")
                    yield f"data: {json.dumps({'error': str(e) or 'Internal Stream Error'})}\n\n"
            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            # logger.info(f"Mode non-streaming activé pour la requête : {payload}")
            # --- Mode NON-STREAMING ---
            timeout = httpx.Timeout(120.0, connect=60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{MISTRAL_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers
                )
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Mistral API error: {response.text}"
                    )
                return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def proxy_models():
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{MISTRAL_BASE_URL}/models",
            headers=headers
        )
        return response.json()