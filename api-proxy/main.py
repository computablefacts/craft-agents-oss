import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import httpx
from dotenv import load_dotenv
import logging
import json
import asyncio

# Module email (IMAP/SMTP)
from email_handler import (
    load_accounts, get_account, list_accounts,
    fetch_emails, list_imap_folders, send_email, list_local_emails,
    EmailAccount,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Répertoire workspace pour les emails ──
WORKSPACE_EMAILS_DIR = os.path.expanduser(
    os.getenv("WORKSPACE_EMAILS_DIR", "~/.craft-agent/workspaces/my-workspace")
)

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


# ═══════════════════════════════════════════════════════════════════
#  Endpoints Email (IMAP/SMTP)
# ═══════════════════════════════════════════════════════════════════


# ── Modèles Pydantic pour les requêtes ──

class FetchRequest(BaseModel):
    account_id: Optional[str] = None
    max_age_minutes: int = 60
    folders: Optional[list[str]] = None
    start_date: Optional[str] = None


class SendRequest(BaseModel):
    account_id: str
    to: list[str]
    subject: str
    body: str
    cc: Optional[list[str]] = None
    bcc: Optional[list[str]] = None
    body_html: Optional[str] = None
    attachment_paths: Optional[list[str]] = None


class ListRequest(BaseModel):
    query: Optional[str] = None
    from_addr: Optional[str] = None
    limit: int = 50
    offset: int = 0


# ── GET /email/accounts — Liste les comptes configurés ──

@app.get("/email/accounts")
async def get_accounts():
    """Retourne la liste des comptes email configurés (sans mots de passe)."""
    try:
        accounts = list_accounts(WORKSPACE_EMAILS_DIR)
        return {"accounts": accounts, "total": len(accounts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement comptes : {e}")


# ── POST /email/fetch — Récupère les emails récents ──

@app.post("/email/fetch")
async def fetch(req: FetchRequest):
    """
    Récupère les emails récents depuis un ou tous les comptes configurés.

    - account_id : si non spécifié, récupère depuis tous les comptes
    - max_age_minutes : horizon temporel (défaut: 60 min) — ignoré si start_date est fourni
    - folders : liste des dossiers IMAP (défaut: ["INBOX"])
    - start_date : date de début ISO ou "YYYY-MM-DD" (optionnel, prime sur max_age_minutes)
      Exemples : "2026-01-01", "2026-01-15T10:00:00+01:00"
    """
    try:
        data = load_accounts(WORKSPACE_EMAILS_DIR)
        all_accounts = [EmailAccount(**a) for a in data.get("accounts", [])]

        if not all_accounts:
            raise HTTPException(
                status_code=400,
                detail="Aucun compte configuré. Créez d'abord le fichier "
                       f"{os.path.join(WORKSPACE_EMAILS_DIR, 'emails', 'accounts.json')}",
            )

        # Filtrer par account_id si spécifié
        if req.account_id:
            targets = [a for a in all_accounts if a.id == req.account_id]
            if not targets:
                raise HTTPException(
                    status_code=404,
                    detail=f"Compte '{req.account_id}' introuvable",
                )
        else:
            targets = all_accounts

        # Calcul de la date de début : start_date prime sur max_age_minutes
        if req.start_date:
            from datetime import datetime, timezone
            try:
                start_dt = datetime.fromisoformat(req.start_date)
            except ValueError:
                try:
                    start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail="Format start_date invalide. Utilisez ISO (ex: 2026-01-15 ou 2026-01-15T10:00:00+01:00)",
                    )
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            since_dt = start_dt
        else:
            since_dt = None  # fetch_emails utilisera max_age_minutes

        results = []
        total_fetched = 0
        total_attachments = 0
        total_errors = []

        emails_dir = os.path.join(WORKSPACE_EMAILS_DIR, "emails")
        os.makedirs(emails_dir, exist_ok=True)

        for account in targets:
            result = fetch_emails(
                account=account,
                max_age_minutes=req.max_age_minutes,
                folders=req.folders,
                since_dt=since_dt,
                workspace_dir=emails_dir,
            )
            results.append({
                "account_id": result.account_id,
                "account_label": result.account_label,
                "fetched": result.fetched,
                "skipped_duplicates": result.skipped_duplicates,
                "attachments_saved": result.attachments_saved,
                "errors": result.errors,
            })
            total_fetched += result.fetched
            total_attachments += result.attachments_saved
            total_errors.extend(result.errors)

        return {
            "status": "ok",
            "total_fetched": total_fetched,
            "total_attachments": total_attachments,
            "total_errors": len(total_errors),
            "accounts": results,
            "emails_dir": emails_dir,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur fetch emails")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /email/folders — Liste les dossiers IMAP ──

@app.get("/email/folders")
async def get_folders(
    account_id: Optional[str] = None,
):
    """
    Liste les dossiers/dossiers disponibles sur le(s) serveur(s) IMAP.

    - account_id : si non spécifié, liste ceux de tous les comptes
    """
    try:
        data = load_accounts(WORKSPACE_EMAILS_DIR)
        all_accounts = [EmailAccount(**a) for a in data.get("accounts", [])]

        if not all_accounts:
            raise HTTPException(
                status_code=400,
                detail="Aucun compte configuré. Créez d'abord le fichier "
                       f"{os.path.join(WORKSPACE_EMAILS_DIR, 'emails', 'accounts.json')}",
            )

        if account_id:
            targets = [a for a in all_accounts if a.id == account_id]
            if not targets:
                raise HTTPException(
                    status_code=404,
                    detail=f"Compte '{account_id}' introuvable",
                )
        else:
            targets = all_accounts

        results = []
        for account in targets:
            folders_data = list_imap_folders(account)
            results.append(folders_data)

        return {"accounts": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur list folders")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /email/send — Envoie un email ──

@app.post("/email/send")
async def send(req: SendRequest):
    """
    Envoie un email via SMTP.

    - account_id : compte à utiliser (requis)
    - to : liste des destinataires
    - subject : sujet
    - body : corps texte
    - cc / bcc : copies (optionnel)
    - body_html : corps HTML (optionnel)
    - attachment_paths : chemins fichiers à joindre (optionnel)
    """
    try:
        account = get_account(req.account_id, WORKSPACE_EMAILS_DIR)
        if not account:
            raise HTTPException(
                status_code=404,
                detail=f"Compte '{req.account_id}' introuvable",
            )

        emails_dir = os.path.join(WORKSPACE_EMAILS_DIR, "emails")
        os.makedirs(emails_dir, exist_ok=True)

        result = send_email(
            account=account,
            to=req.to,
            subject=req.subject,
            body=req.body,
            cc=req.cc,
            bcc=req.bcc,
            body_html=req.body_html,
            attachment_paths=req.attachment_paths,
            workspace_dir=emails_dir,
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("detail"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur send email")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /email/list — Liste les emails téléchargés localement ──

@app.get("/email/list")
async def list_emails(
    query: Optional[str] = None,
    from_addr: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Liste et recherche les emails téléchargés localement.

    - query : recherche textuelle (sujet, expéditeur)
    - from_addr : filtre par expéditeur
    - limit : nombre max de résultats (défaut: 50)
    - offset : pagination (défaut: 0)
    """
    try:
        emails_dir = os.path.join(WORKSPACE_EMAILS_DIR, "emails")
        result = list_local_emails(
            query=query,
            from_addr=from_addr,
            limit=limit,
            offset=offset,
            workspace_dir=emails_dir,
        )
        result["emails_dir"] = emails_dir
        result["attachments_dir"] = os.path.join(emails_dir, "attachments")
        return result
    except Exception as e:
        logger.exception("Erreur list emails")
        raise HTTPException(status_code=500, detail=str(e))
