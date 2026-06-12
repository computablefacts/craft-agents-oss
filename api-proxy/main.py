import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx
from dotenv import load_dotenv
import logging
import json
import asyncio

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API Proxy Fusionné (Mistral + DeepInfra)")

# ── Configuration des providers ──

PROVIDERS = {
    "deepinfra": {
        "api_key": os.getenv("DEEPINFRA_API_KEY"),
        "base_url": "https://api.deepinfra.com/v1/openai",
        "label": "DeepInfra",
    },
    "mistral": {
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "base_url": "https://api.mistral.ai/v1",
        "label": "Mistral",
    },
}


# ── Helpers ──

def get_provider_config(provider: str):
    config = PROVIDERS.get(provider)
    if not config:
        raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found. Supported: {list(PROVIDERS.keys())}")
    if not config["api_key"]:
        raise HTTPException(status_code=500, detail=f"{config['label']} API key not configured. Set {provider.upper()}_API_KEY in .env")
    return config


def parse_model(payload: dict) -> tuple[str, str]:
    """Extract (model_name, provider_slug) from the model field.

    Formats accceptés :
    - "deepseek-ai/DeepSeek-V4-Flash@deepinfra"  → ("deepseek-ai/DeepSeek-V4-Flash", "deepinfra")
    - "mistral-medium-3-5@mistral"               → ("mistral-medium-3-5", "mistral")
    """
    original = payload.get("model", "")
    if "@" not in original:
        raise HTTPException(
            status_code=400,
            detail=f"Model must include provider suffix (e.g., 'model@deepinfra'). Got: '{original}'"
        )
    model_name, provider = original.rsplit("@", 1)
    if provider not in PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider '{provider}'. Supported: {list(PROVIDERS.keys())}"
        )
    return model_name, provider


def clean_payload(payload: dict):
    """Remove Craft Agents specific fields that external APIs don't support."""
    for field in ("store", "stream_options", "max_completion_tokens"):
        payload.pop(field, None)


# ── Génération du flux streaming (factorisée) ──

async def stream_from_provider(provider_cfg: dict, payload: dict):
    headers = {
        "Authorization": f"Bearer {provider_cfg['api_key']}",
        "Content-Type": "application/json",
    }
    url = f"{provider_cfg['base_url']}/chat/completions"
    label = provider_cfg["label"]

    try:
        timeout = httpx.Timeout(120.0, connect=60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async def fetch_response(retry=True):
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code == 429 and retry:
                        logger.warning(f"[{label}] Rate limit atteint. Attente de 1 seconde...")
                        await asyncio.sleep(1)
                        async for chunk in fetch_response(retry=False):
                            yield chunk
                        return
                    if resp.status_code != 200:
                        error_detail = await resp.aread()
                        try:
                            error_json = json.loads(error_detail)
                            logger.error(f"[{label}] Erreur (streaming): {error_json}")
                            yield f"data: {json.dumps({'error': error_json})}\n\n"
                        except Exception:
                            error_text = error_detail.decode("utf-8", errors="replace")
                            logger.error(f"[{label}] Erreur (streaming): {error_text}")
                            yield f"data: {json.dumps({'error': error_text})}\n\n"
                        return
                    async for line in resp.aiter_lines():
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
        logger.exception(f"[{label}] Erreur dans le flux streaming")
        yield f"data: {json.dumps({'error': str(e) or 'Internal Stream Error'})}\n\n"


# ── Routes ──

@app.post("/chat/completions")
async def proxy_chat(request: Request):
    try:
        payload = await request.json()
        clean_payload(payload)

        # Extract model name & provider from the "model@provider" format
        model_name, provider = parse_model(payload)
        payload["model"] = model_name
        provider_cfg = get_provider_config(provider)
        stream = payload.get("stream", False)
        headers = {
            "Authorization": f"Bearer {provider_cfg['api_key']}",
            "Content-Type": "application/json",
        }
        url = f"{provider_cfg['base_url']}/chat/completions"
        label = provider_cfg["label"]

        if stream:
            return StreamingResponse(
                stream_from_provider(provider_cfg, payload),
                media_type="text/event-stream",
            )
        else:
            timeout = httpx.Timeout(120.0, connect=60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"[{label}] API error: {response.text}",
                    )
                return response.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def proxy_models(provider: str):
    provider_cfg = get_provider_config(provider)
    headers = {"Authorization": f"Bearer {provider_cfg['api_key']}"}
    url = f"{provider_cfg['base_url']}/models"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()


@app.get("/")
async def root():
    return {
        "service": "API Proxy Fusionné",
        "providers": list(PROVIDERS.keys()),
        "usage": {
            "chat": 'POST /chat/completions  with {"model": "model_name@provider"}',
            "models": 'GET /models?provider=deepinfra',
        },
        "examples": {
            "deepinfra": '{"model": "deepseek-ai/DeepSeek-V4-Flash@deepinfra", ...}',
            "mistral": '{"model": "mistral-medium-3-5@mistral", ...}',
        },
    }
