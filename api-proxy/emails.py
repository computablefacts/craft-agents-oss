import asyncio
import email
import hashlib
import imaplib
import json
import logging
import mimetypes
import os
import re
import smtplib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from email.header import decode_header
from email.message import EmailMessage
from email.policy import default as email_policy
from email.utils import formataddr
from fastapi import APIRouter, HTTPException
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Répertoire workspace pour les emails ──
WORKSPACE_EMAILS_DIR = os.path.expanduser(
    os.getenv("WORKSPACE_EMAILS_DIR", "~/.craft-agent/workspaces/my-workspace")
)


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


# ── GET /emails/accounts — Liste les comptes configurés ──

@router.get("/emails/accounts")
async def get_accounts():
    """Retourne la liste des comptes email configurés (sans mots de passe)."""
    try:
        accounts = list_accounts(WORKSPACE_EMAILS_DIR)
        return {"accounts": accounts, "total": len(accounts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement comptes : {e}")


# ── POST /emails/fetch — Récupère les emails récents ──

@router.post("/emails/fetch")
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
                "errors": result.errors
            })
            total_fetched += result.fetched
            total_attachments += result.attachments_saved
            total_errors.extend(result.errors)

        return {
            "status": "success" if not total_errors else "partial_success",
            "summary": {
                "total_fetched": total_fetched,
                "total_attachments": total_attachments,
                "accounts_processed": len(targets),
                "errors_count": len(total_errors)
            },
            "accounts": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur fetch emails")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /emails/folders — Liste les dossiers IMAP ──

@router.get("/emails/folders")
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


# ── POST /emails/send — Envoie un email ──

@router.post("/emails/send")
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


# ── Répertoire de base des emails ──
DEFAULT_EMAILS_DIR = os.path.expanduser(
    "~/.craft-agent/workspaces/my-workspace/emails"
)
ACCOUNTS_FILE = "accounts.json"
ATTACHMENTS_DIR = "attachments"

# Taille max d'une pièce-jointe (50 Mo)
MAX_ATTACHMENT_SIZE = 50 * 1024 * 1024


# ═══════════════════════════════════════════════════════════════════
#  Modèles de données
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EmailAccount:
    """Configuration d'un compte email."""
    id: str
    label: str
    imap_server: str
    imap_port: int = 993
    smtp_server: str = ""
    smtp_port: int = 465
    smtp_use_tls: bool = True
    username: str = ""
    password: str = ""

    def __post_init__(self):
        if not self.smtp_server:
            self.smtp_server = self.imap_server


@dataclass
class FetchResult:
    """Résultat d'une récupération d'emails."""
    account_id: str
    account_label: str
    fetched: int = 0
    skipped_duplicates: int = 0
    attachments_saved: int = 0
    errors: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
#  Utilitaires
# ═══════════════════════════════════════════════════════════════════

def _safe_filename(text: str, max_len: int = 80) -> str:
    """Nettoie une chaîne pour en faire un nom de fichier sûr."""
    text = str(text).strip().lower()
    # Remplacer les caractères problématiques
    text = re.sub(r'[<>:"/\\|?*@]', '_', text)
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'_+', '_', text)
    text = text.strip('_.')
    if len(text) > max_len:
        text = text[:max_len]
    return text if text else "unknown"


def _short_hash(msg_id: str) -> str:
    """Génère un hash court (8 caractères) à partir d'un Message-ID."""
    return hashlib.md5(msg_id.encode('utf-8')).hexdigest()[:8]


def _datetime_from_email(msg: email.message.Message) -> Optional[datetime]:
    """Extrait la date d'un email, en timezone-aware."""
    date_str = msg.get("Date", "")
    if not date_str:
        return None
    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _decode_header_value(value) -> str:
    """Décode une valeur d'en-tête email (gère les encodages)."""
    if value is None:
        return ""
    try:
        decoded = decode_header(value)
        return str(make_header(decoded))
    except Exception:
        return str(value)


def _parse_addresses(addr_str: str) -> list:
    """Parse une chaîne d'adresses email en liste simple."""
    if not addr_str:
        return []
    try:
        # decode_header peut être nécessaire ici aussi si l'en-tête est encodé
        decoded_addr = _decode_header_value(addr_str)
        return [a.strip() for a in decoded_addr.split(",") if a.strip()]
    except Exception:
        return [addr_str]


def _format_sender_for_filename(sender: str) -> str:
    """Extrait une version courte de l'expéditeur pour le nom de fichier."""
    match = re.search(r'<([^>]+)>', sender)
    email_addr = match.group(1) if match else sender
    return _safe_filename(email_addr.split('@')[0])


def _get_email_dir(workspace_dir: str = None) -> str:
    base = workspace_dir or DEFAULT_EMAILS_DIR
    return base


def _get_accounts_path(workspace_dir: str = None) -> str:
    return os.path.join(_get_email_dir(workspace_dir), ACCOUNTS_FILE)


def _get_attachments_dir(workspace_dir: str = None) -> str:
    d = os.path.join(_get_email_dir(workspace_dir), ATTACHMENTS_DIR)
    os.makedirs(d, exist_ok=True)
    return d


# ═══════════════════════════════════════════════════════════════════
#  Gestion des comptes
# ═══════════════════════════════════════════════════════════════════

def load_accounts(workspace_dir: str = None) -> dict:
    """Charge les comptes depuis accounts.json."""
    path = _get_accounts_path(workspace_dir)
    if not os.path.exists(path):
        return {"accounts": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erreur lecture accounts.json: {e}")
        return {"accounts": []}


def get_account(account_id: str, workspace_dir: str = None) -> Optional[EmailAccount]:
    """Récupère un compte spécifique par son ID."""
    data = load_accounts(workspace_dir)
    for a in data.get("accounts", []):
        if a.get("id") == account_id:
            return EmailAccount(**a)
    return None


def list_accounts(workspace_dir: str = None) -> list:
    """Liste les comptes (sans les mots de passe)."""
    data = load_accounts(workspace_dir)
    accounts = []
    for a in data.get("accounts", []):
        acc_copy = a.copy()
        acc_copy.pop("password", None)
        accounts.append(acc_copy)
    return accounts


# ═══════════════════════════════════════════════════════════════════
#  Logique IMAP (Fetch)
# ═══════════════════════════════════════════════════════════════════

def _search_between_dates(imap: imaplib.IMAP4_SSL, folder: str, since_dt: datetime, before_dt: datetime):
    """Recherche des messages entre deux dates (format IMAP: 11-Jun-2026)."""
    imap.select(folder, readonly=True)

    # Format attendu par IMAP : DD-Mon-YYYY (ex: 01-Jan-2024)
    # Note: IMAP SEARCH SINCE est inclusif, BEFORE est exclusif.
    since_str = since_dt.strftime("%d-%b-%Y")
    before_str = before_dt.strftime("%d-%b-%Y")

    search_criterion = f'(SINCE "{since_str}" BEFORE "{before_str}")'
    status, messages = imap.search(None, search_criterion)

    if status != "OK":
        return []
    return messages[0].split()


def _fetch_email_data(imap: imaplib.IMAP4_SSL, msg_id: int):
    """Récupère le contenu brut RFC822 d'un email."""
    status, data = imap.fetch(str(msg_id), "(RFC822)")
    if status != "OK":
        return None
    for response_part in data:
        if isinstance(response_part, tuple):
            return response_part[1]
    return None


def _parse_and_save_email(raw_email: bytes, account: EmailAccount, folder: str, workspace_dir: str) -> Optional[dict]:
    """Parse l'email, sauvegarde le fichier .eml."""
    msg = email.message_from_bytes(raw_email, policy=email_policy)

    msg_id = msg.get("Message-ID", f"no-id-{time.time()}")
    msg_id_hash = _short_hash(msg_id)

    # Vérifier si déjà sauvegardé
    filename_prefix = f"*_{msg_id_hash}.eml"
    existing_files = list(Path(workspace_dir).glob(f"*_{msg_id_hash}.eml"))
    if existing_files:
        return None

    dt = _datetime_from_email(msg) or datetime.now(timezone.utc)
    subject = _decode_header_value(msg.get("Subject", "(No Subject)"))
    sender = _decode_header_value(msg.get("From", "unknown"))

    # Nom de fichier : YYYYMMDD_HHMMSS_sender_hash.eml
    timestamp = dt.strftime("%Y%m%d_%H%M%S")
    sender_slug = _format_sender_for_filename(sender)
    filename = f"{timestamp}_{sender_slug}_{msg_id_hash}.eml"

    # Sauvegarde du fichier EML
    file_path = os.path.join(workspace_dir, filename)
    with open(file_path, "wb") as f:
        f.write(raw_email)

    # Extraction des pièces jointes
    attachments_dir = _get_attachments_dir(workspace_dir)
    attachment_files = _extract_attachments(msg, msg_id_hash, attachments_dir)

    return {
        "filename": filename,
        "message_id_hash": msg_id_hash,
        "attachment_files": attachment_files
    }


def _extract_attachments(msg: email.message.Message, msg_id_hash: str, attachments_dir: str) -> list:
    """Parcourt les parties de l'email pour extraire les fichiers."""
    attachment_files = []
    for part in msg.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        if part.get('Content-Disposition') is None:
            continue

        filename = part.get_filename()
        if not filename:
            continue

        filename = _decode_header_value(filename)
        # Préfixer par le hash du message pour éviter les collisions entre emails
        safe_name = f"{msg_id_hash}_{_safe_filename(filename)}"
        save_path = os.path.join(attachments_dir, safe_name)

        payload = part.get_payload(decode=True)
        if payload:
            if len(payload) > MAX_ATTACHMENT_SIZE:
                logger.warning(f"Pièce jointe trop grande ignorée: {filename} ({len(payload)} octets)")
                continue

            with open(save_path, "wb") as f:
                f.write(payload)

            attachment_files.append({
                "original_name": filename,
                "stored_name": safe_name,
                "size": len(payload),
                "content_type": part.get_content_type()
            })

    return attachment_files


def _fetch_folder(imap: imaplib.IMAP4_SSL, account: EmailAccount, folder: str, since_dt: datetime, workspace_dir: str,
                  result: FetchResult):
    """Récupère les emails d'un dossier spécifique depuis since_dt."""
    try:
        # On cherche jusqu'à demain pour être sûr de tout prendre aujourd'hui
        before_dt = datetime.now(timezone.utc) + timedelta(days=1)
        msg_ids = _search_between_dates(imap, folder, since_dt, before_dt)

        logger.info(f"[{account.label}] {folder} : {len(msg_ids)} messages trouvés depuis {since_dt.date()}")

        for m_id in msg_ids:
            try:
                raw_data = _fetch_email_data(imap, int(m_id))
                if not raw_data:
                    continue

                info = _parse_and_save_email(raw_data, account, folder, workspace_dir)
                if info:
                    result.fetched += 1
                    result.attachments_saved += len(info.get("attachment_files", []))
                else:
                    result.skipped_duplicates += 1

            except Exception as e:
                err_msg = f"Erreur fetch message {m_id} dans {folder}: {e}"
                logger.error(err_msg)
                result.errors.append(err_msg)

    except Exception as e:
        err_msg = f"Erreur accès dossier {folder}: {e}"
        logger.error(err_msg)
        result.errors.append(err_msg)


def fetch_emails(account: EmailAccount, max_age_minutes: int = 60, folders: list = None,
                 since_dt: Optional[datetime] = None, workspace_dir: str = None) -> FetchResult:
    """Se connecte en IMAP et récupère les nouveaux emails."""
    result = FetchResult(account.id, account.label)
    if not folders:
        folders = ["INBOX"]

    # Si since_dt n'est pas fourni, on calcule à partir de max_age_minutes
    if not since_dt:
        since_dt = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)

    try:
        imap = imaplib.IMAP4_SSL(account.imap_server, account.imap_port)
        imap.login(account.username, account.password)

        for folder in folders:
            _fetch_folder(imap, account, folder, since_dt, workspace_dir, result)

        imap.logout()
    except Exception as e:
        err_msg = f"Erreur connexion IMAP {account.label}: {e}"
        logger.error(err_msg)
        result.errors.append(err_msg)

    return result


def list_imap_folders(account: EmailAccount, workspace_dir: str = None) -> dict:
    """Récupère la liste des dossiers IMAP disponibles."""
    try:
        imap = imaplib.IMAP4_SSL(account.imap_server, account.imap_port)
        imap.login(account.username, account.password)
        status, folder_list = imap.list()
        folders = []
        if status == "OK":
            for f in folder_list:
                # Format typique: '(\\\\HasNoChildren) "/" "INBOX"'
                parts = f.decode().split(' "/" ')
                if len(parts) == 2:
                    folders.append(parts[1].strip('"'))
                else:
                    folders.append(f.decode())

        imap.logout()
        return {
            "account_id": account.id,
            "account_label": account.label,
            "folders": folders
        }
    except Exception as e:
        return {
            "account_id": account.id,
            "account_label": account.label,
            "error": str(e)
        }


# ═══════════════════════════════════════════════════════════════════
#  Logique SMTP (Send)
# ═══════════════════════════════════════════════════════════════════

def send_email(account: EmailAccount, to: list, subject: str, body: str, cc: list = None, bcc: list = None,
               body_html: str = None, attachment_paths: list = None, workspace_dir: str = None) -> dict:
    """Envoie un email via SMTP."""
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = formataddr((account.label, account.username))
        msg["To"] = ", ".join(to)
        if cc:
            msg["Cc"] = ", ".join(cc)
        if bcc:
            msg["Bcc"] = ", ".join(bcc)

        msg.set_content(body)

        if body_html:
            msg.add_alternative(body_html, subtype='html')

        # Ajout des pièces jointes
        if attachment_paths:
            for path_str in attachment_paths:
                p = Path(path_str)
                if not p.exists():
                    # Essayer relativement au workspace
                    p = Path(workspace_dir) / path_str
                    if not p.exists():
                        logger.warning(f"Pièce jointe introuvable : {path_str}")
                        continue

                ctype, encoding = mimetypes.guess_type(str(p))
                if ctype is None or encoding is not None:
                    ctype = 'application/octet-stream'
                maintype, subtype = ctype.split('/', 1)

                with open(p, 'rb') as f:
                    msg.add_attachment(
                        f.read(),
                        maintype=maintype,
                        subtype=subtype,
                        filename=p.name
                    )

        # Envoi
        if account.smtp_use_tls:
            with smtplib.SMTP_SSL(account.smtp_server, account.smtp_port) as server:
                server.login(account.username, account.password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(account.smtp_server, account.smtp_port) as server:
                server.starttls()
                server.login(account.username, account.password)
                server.send_message(msg)

        return {
            "status": "success",
            "message_id": msg["Message-ID"],
            "to": to
        }

    except Exception as e:
        logger.error(f"Erreur envoi email: {e}")
        return {"status": "error", "detail": str(e)}
