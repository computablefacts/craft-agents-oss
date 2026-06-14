"""
email_handler.py — Module de gestion des emails IMAP/SMTP pour l'API Proxy.

Fournit les fonctions pour :
- Lire et écrire la configuration des comptes email
- Récupérer les emails via IMAP et les stocker au format EML
- Extraire les pièces-jointes vers un répertoire commun
- Envoyer des emails via SMTP
- Lister et rechercher les emails téléchargés localement
"""

import imaplib
import smtplib
import email
from email.message import EmailMessage
from email.policy import default as email_policy
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime, formataddr
import os
import json
import hashlib
import re
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# ── Répertoire de base des emails ──
DEFAULT_EMAILS_DIR = os.path.expanduser(
    "~/.craft-agent/workspaces/my-workspace/emails"
)
ACCOUNTS_FILE = "accounts.json"
ATTACHMENTS_DIR = "attachments"
INDEX_FILE = "index.json"

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
class EmailInfo:
    """Métadonnées d'un email téléchargé."""
    filename: str
    message_id_hash: str
    date: str
    subject: str
    sender: str
    recipients: list
    account_id: str
    has_attachments: bool
    attachment_files: list = field(default_factory=list)
    size: int = 0
    folder: str = "INBOX"


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
    """Parse une chaîne d'adresse email en liste de (name, email)."""
    if not addr_str:
        return []
    from email.utils import getaddresses
    try:
        return getaddresses([addr_str])
    except Exception:
        return [(addr_str, addr_str)]


def _format_sender_for_filename(sender: str) -> str:
    """Extrait une partie sûre pour le nom de fichier depuis l'expéditeur."""
    # Prendre l'adresse email entre <>
    match = re.search(r'<([^>]+)>', sender)
    if match:
        sender = match.group(1)
    # Nettoyer
    return _safe_filename(sender, max_len=50)


def _get_email_dir(workspace_dir: str = None) -> str:
    """Retourne le chemin du répertoire des emails."""
    if workspace_dir:
        return workspace_dir
    return DEFAULT_EMAILS_DIR


def _get_accounts_path(workspace_dir: str = None) -> str:
    return os.path.join(_get_email_dir(workspace_dir), ACCOUNTS_FILE)


def _get_attachments_dir(workspace_dir: str = None) -> str:
    return os.path.join(_get_email_dir(workspace_dir), ATTACHMENTS_DIR)


def _get_index_path(workspace_dir: str = None) -> str:
    return os.path.join(_get_email_dir(workspace_dir), INDEX_FILE)


# ═══════════════════════════════════════════════════════════════════
#  Gestion des comptes
# ═══════════════════════════════════════════════════════════════════

def load_accounts(workspace_dir: str = None) -> dict:
    """Charge la configuration des comptes depuis accounts.json."""
    path = _get_accounts_path(workspace_dir)
    if not os.path.exists(path):
        logger.warning(f"Fichier comptes introuvable : {path}")
        return {"accounts": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Erreur de lecture de {path}: {e}")
        return {"accounts": []}


def get_account(account_id: str, workspace_dir: str = None) -> Optional[EmailAccount]:
    """Récupère un compte par son ID."""
    data = load_accounts(workspace_dir)
    for acc in data.get("accounts", []):
        if acc.get("id") == account_id:
            return EmailAccount(**acc)
    return None


def list_accounts(workspace_dir: str = None) -> list:
    """Liste tous les comptes (sans les mots de passe)."""
    data = load_accounts(workspace_dir)
    safe = []
    for acc in data.get("accounts", []):
        safe.append({k: v for k, v in acc.items() if k != "password"})
    return safe


# ═══════════════════════════════════════════════════════════════════
#  Index local des emails
# ═══════════════════════════════════════════════════════════════════

def _load_index(workspace_dir: str = None) -> dict:
    """Charge l'index des emails."""
    path = _get_index_path(workspace_dir)
    if not os.path.exists(path):
        return {"emails": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"emails": []}


def _save_index(index: dict, workspace_dir: str = None):
    """Sauvegarde l'index des emails."""
    path = _get_index_path(workspace_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def _add_to_index(email_info: EmailInfo, workspace_dir: str = None):
    """Ajoute un email à l'index (ou met à jour s'il existe déjà)."""
    index = _load_index(workspace_dir)
    # Vérifier si le message_id_hash existe déjà
    for i, existing in enumerate(index["emails"]):
        if existing.get("message_id_hash") == email_info.message_id_hash:
            index["emails"][i] = asdict(email_info)
            _save_index(index, workspace_dir)
            return
    index["emails"].append(asdict(email_info))
    _save_index(index, workspace_dir)


# ═══════════════════════════════════════════════════════════════════
#  IMAP — Récupération des emails
# ═══════════════════════════════════════════════════════════════════

BLOCK_SIZE_MINUTES = 360  # Taille d'un bloc de recherche (6 heures)


def _search_between_dates(
    imap: imaplib.IMAP4_SSL,
    folder: str,
    since_dt: datetime,
    before_dt: datetime,
) -> list:
    """Recherche les emails dans un dossier IMAP entre deux dates.

    Utilise SEARCH SINCE/BEFORE pour filtrer côté serveur (granularité jour),
    puis on filtre côté client sur l'heure.

    Retourne la liste triée du plus récent au plus ancien.
    """
    since_str = since_dt.strftime("%d-%b-%Y")
    before_str = before_dt.strftime("%d-%b-%Y")
    status, msg_ids = imap.search(None, f'(SINCE {since_str} BEFORE {before_str})')
    if status != "OK":
        return []

    ids = msg_ids[0].split() if msg_ids[0] else []
    if not ids:
        return []

    # Trier du plus récent (ID élevé) au plus ancien (ID bas)
    # Note : les IDs IMAP sont croissants, le plus récent a l'ID le plus grand
    return sorted([int(b) for b in ids], reverse=True)


def _fetch_email_data(imap: imaplib.IMAP4_SSL, msg_id: int) -> Optional[bytes]:
    """Télécharge un email complet (RFC822)."""
    status, data = imap.fetch(str(msg_id), "(RFC822)")
    if status != "OK" or not data or not data[0]:
        return None
    # data[0] est un tuple (b'RFC822 {size}', b'...contenu...')
    if isinstance(data[0], tuple):
        return data[0][1]
    return None


def _parse_and_save_email(
    raw_email: bytes,
    account: EmailAccount,
    folder: str,
    workspace_dir: str,
) -> tuple[Optional[str], EmailInfo]:
    """Parse un email brut, le sauvegarde en .eml et extrait les pièces-jointes.

    Retourne :
        - (filename, EmailInfo) si succès
        - (None, EmailInfo avec erreur) si échec
    """
    try:
        msg = email.message_from_bytes(raw_email, policy=email_policy)
    except Exception as e:
        logger.error(f"Erreur de parsing email: {e}")
        return None, EmailInfo(
            filename="", message_id_hash="", date="", subject="",
            sender="", recipients=[], account_id=account.id,
            has_attachments=False, size=len(raw_email), folder=folder,
        )

    # Extraire les métadonnées
    msg_id_orig = msg.get("Message-ID", "") or f"generated-{time.time_ns()}"
    msg_id_hash = _short_hash(msg_id_orig)
    subject = _decode_header_value(msg.get("Subject", "(Sans sujet)"))
    sender = _decode_header_value(msg.get("From", ""))
    to_raw = _decode_header_value(msg.get("To", ""))
    cc_raw = _decode_header_value(msg.get("Cc", ""))

    recipients = _parse_addresses(to_raw) + _parse_addresses(cc_raw)
    dt = _datetime_from_email(msg) or datetime.now(timezone.utc)

    # Nom de fichier
    ts = dt.strftime("%Y-%m-%d_%H-%M-%S")
    sender_safe = _format_sender_for_filename(sender)
    filename = f"{ts}_{sender_safe}.eml"

    # Chemin complet
    emails_dir = _get_email_dir(workspace_dir)
    os.makedirs(emails_dir, exist_ok=True)
    filepath = os.path.join(emails_dir, filename)

    # Vérifier si le fichier existe déjà (Message-ID basé)
    existing_index = _load_index(workspace_dir)
    for entry in existing_index.get("emails", []):
        if entry.get("message_id_hash") == msg_id_hash:
            logger.info(f"Email déjà téléchargé (doublon) : {msg_id_hash}")
            return None, EmailInfo(
                filename=entry.get("filename", filename),
                message_id_hash=msg_id_hash,
                date=entry.get("date", dt.isoformat()),
                subject=entry.get("subject", subject),
                sender=entry.get("sender", sender),
                recipients=entry.get("recipients", recipients),
                account_id=account.id,
                has_attachments=entry.get("has_attachments", False),
                attachment_files=entry.get("attachment_files", []),
                size=len(raw_email),
                folder=folder,
            )

    # Sauvegarder le .eml
    with open(filepath, "wb") as f:
        f.write(raw_email)

    # Extraire les pièces-jointes
    attachments_dir = _get_attachments_dir(workspace_dir)
    attachment_files = _extract_attachments(msg, msg_id_hash, attachments_dir)
    has_attachments = len(attachment_files) > 0

    email_info = EmailInfo(
        filename=filename,
        message_id_hash=msg_id_hash,
        date=dt.isoformat(),
        subject=subject,
        sender=sender,
        recipients=recipients,
        account_id=account.id,
        has_attachments=has_attachments,
        attachment_files=attachment_files,
        size=len(raw_email),
        folder=folder,
    )

    _add_to_index(email_info, workspace_dir)
    logger.info(f"Email sauvegardé : {filename} ({len(attachment_files)} pièce(s)-jointe(s))")

    return filename, email_info


def _extract_attachments(
    msg: email.message.Message,
    msg_id_hash: str,
    attachments_dir: str,
) -> list:
    """Extrait les pièces-jointes d'un email et les sauvegarde.

    Retourne la liste des chemins relatifs des fichiers sauvegardés.
    """
    saved = []
    os.makedirs(attachments_dir, exist_ok=True)

    counter = 0
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        content_disposition = str(part.get("Content-Disposition", ""))
        content_type = part.get_content_type()

        # Ignorer le corps du message (inline text/plain, text/html)
        is_attachment = "attachment" in content_disposition
        is_inline_file = "filename" in content_disposition
        has_filename = part.get_filename() is not None

        if not (is_attachment or is_inline_file or has_filename):
            continue

        filename = part.get_filename()
        if not filename:
            continue

        counter += 1
        filename = _decode_header_value(filename)
        safe_filename = _safe_filename(filename, max_len=100)

        # Nom du fichier : {hash}__{num:03d}_{nom}
        attachment_name = f"{msg_id_hash}__{counter:03d}_{safe_filename}"
        attachment_path = os.path.join(attachments_dir, attachment_name)

        payload = part.get_payload(decode=True)
        if payload is None:
            continue

        # Vérifier la taille max
        if len(payload) > MAX_ATTACHMENT_SIZE:
            logger.warning(
                f"Pièce-jointe trop volumineuse ({len(payload)} octets) : "
                f"{attachment_name}, ignorée"
            )
            continue

        # Éviter les doublons
        if os.path.exists(attachment_path):
            logger.debug(f"Pièce-jointe déjà existante : {attachment_name}")
            saved.append(attachment_name)
            continue

        try:
            with open(attachment_path, "wb") as f:
                f.write(payload)
            saved.append(attachment_name)
            logger.debug(f"Pièce-jointe sauvegardée : {attachment_name}")
        except OSError as e:
            logger.error(f"Erreur d'écriture pièce-jointe {attachment_name}: {e}")

    return saved


def _fetch_folder(
    imap: imaplib.IMAP4_SSL,
    account: EmailAccount,
    folder: str,
    since_dt: datetime,
    workspace_dir: str,
    result: FetchResult,
):
    """Récupère les emails d'un seul dossier IMAP par blocs de 360 min,
    du plus récent au plus ancien, jusqu'à atteindre since_dt.
    """
    # Sélection du dossier — quoter le nom si nécessaire (Gandi utilise des guillemets)
    folder_quoted = f'"{folder}"' if ' ' in folder else folder
    status, data = imap.select(folder_quoted)
    if status != "OK":
        # Fallback sans quotes
        status, data = imap.select(folder)
        result.errors.append(
            f"Impossible d'ouvrir le dossier '{folder}' : {data}"
        )
        logger.error(f"IMAP select failed for '{folder}': {data}")
        return

    now = datetime.now(timezone.utc)
    block_end = now
    total_fetched_in_folder = 0

    # On itère par blocs de BLOCK_SIZE_MINUTES en remontant le temps
    while block_end > since_dt:
        block_start = max(since_dt, block_end - timedelta(minutes=BLOCK_SIZE_MINUTES))

        logger.info(
            f"[{account.label}] Bloc '{folder}' : "
            f"{block_start.strftime('%Y-%m-%d %H:%M')} → "
            f"{block_end.strftime('%Y-%m-%d %H:%M')}"
        )

        # Recherche dans ce bloc (trié du plus récent au plus ancien)
        msg_ids = _search_between_dates(imap, folder, block_start, block_end)

        if msg_ids:
            logger.info(
                f"[{account.label}] {len(msg_ids)} email(s) dans ce bloc '{folder}'"
            )

            for msg_id in msg_ids:
                try:
                    raw_data = _fetch_email_data(imap, msg_id)
                    if raw_data is None:
                        continue

                    _, email_info = _parse_and_save_email(
                        raw_data, account, folder, workspace_dir
                    )

                    if email_info.message_id_hash:
                        existing_index = _load_index(workspace_dir)
                        is_dup = any(
                            e.get("message_id_hash") == email_info.message_id_hash
                            and e.get("filename") != email_info.filename
                            for e in existing_index.get("emails", [])
                        )
                        if is_dup or email_info.filename == "":
                            result.skipped_duplicates += 1
                        else:
                            result.fetched += 1
                            result.attachments_saved += len(email_info.attachment_files)
                            total_fetched_in_folder += 1

                except Exception as e:
                    result.errors.append(
                        f"Erreur sur l'email ID {msg_id} dans '{folder}': {e}"
                    )
                    logger.exception(f"Error processing email {msg_id} in '{folder}'")

        # Passer au bloc suivant (360 min plus tôt)
        block_end = block_start

        # Petit délai pour ne pas surcharger le serveur IMAP entre les blocs
        if block_end > since_dt and len(msg_ids or []) > 0:
            import time
            time.sleep(0.5)

    if total_fetched_in_folder == 0:
        logger.info(f"[{account.label}] Aucun nouvel email dans '{folder}'")


def fetch_emails(
    account: EmailAccount,
    max_age_minutes: int = 60,
    folders: list = None,
    since_dt: Optional[datetime] = None,
    workspace_dir: str = None,
) -> FetchResult:
    """Récupère les emails récents d'un compte IMAP dans un ou plusieurs dossiers.

    La récupération se fait par blocs de 360 minutes, du plus récent au plus ancien.

    Args:
        account: Configuration du compte.
        max_age_minutes: Récupère les emails de moins de x minutes (ignoré si since_dt est fourni).
        folders: Liste des dossiers IMAP à scruter (défaut: ["INBOX"]).
        since_dt: Date de début absolue (optionnel, prime sur max_age_minutes).
        workspace_dir: Répertoire workspace (défaut: ~/.craft-agent/...).

    Retourne:
        FetchResult avec le détail des opérations.
    """
    if folders is None:
        folders = ["INBOX"]

    result = FetchResult(account_id=account.id, account_label=account.label)

    if since_dt is not None:
        actual_since = since_dt
    else:
        actual_since = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)

    logger.info(
        f"Connexion IMAP à {account.imap_server}:{account.imap_port} "
        f"pour {account.username}, dossiers={folders}, "
        f"depuis {actual_since.isoformat()}"
    )

    try:
        imap = imaplib.IMAP4_SSL(
            account.imap_server,
            account.imap_port,
            timeout=30,
        )
    except Exception as e:
        result.errors.append(f"Connexion IMAP échouée : {e}")
        logger.error(f"IMAP connection failed: {e}")
        return result

    try:
        # Login
        status, data = imap.login(account.username, account.password)
        if status != "OK":
            result.errors.append(f"Login IMAP échoué : {data}")
            logger.error(f"IMAP login failed: {data}")
            imap.logout()
            return result

        # Parcourir chaque dossier
        for folder in folders:
            _fetch_folder(imap, account, folder, actual_since, workspace_dir, result)

    except Exception as e:
        result.errors.append(f"Erreur IMAP : {e}")
        logger.exception("IMAP error")
    finally:
        try:
            imap.logout()
        except Exception:
            pass

    return result


def list_imap_folders(
    account: EmailAccount,
    workspace_dir: str = None,
) -> dict:
    """Liste les dossiers/dossiers disponibles sur le serveur IMAP.

    Retourne la liste des dossiers avec leurs attributs.
    """
    logger.info(
        f"Connexion IMAP à {account.imap_server}:{account.imap_port} "
        f"pour {account.username} — listage des dossiers"
    )

    result = {
        "account_id": account.id,
        "account_label": account.label,
        "folders": [],
        "error": None,
    }

    try:
        imap = imaplib.IMAP4_SSL(
            account.imap_server,
            account.imap_port,
            timeout=30,
        )
    except Exception as e:
        result["error"] = f"Connexion IMAP échouée : {e}"
        logger.error(f"IMAP connection failed: {e}")
        return result

    try:
        status, data = imap.login(account.username, account.password)
        if status != "OK":
            result["error"] = f"Login IMAP échoué : {data}"
            logger.error(f"IMAP login failed: {data}")
            imap.logout()
            return result

        # Commande LIST pour lister tous les dossiers
        # "" signifie la racine, "*" signifie tous les niveaux
        status, folders_data = imap.list()
        if status != "OK":
            result["error"] = f"Impossible de lister les dossiers : {folders_data}"
            imap.logout()
            return result

        parsed_folders = []
        for line in folders_data:
            if not line:
                continue
            # Format IMAP LIST : (\\Attr1 \\Attr2) "/" "INBOX"
            try:
                decoded = line.decode("utf-8", errors="replace")
            except AttributeError:
                decoded = str(line)

            import re
            # Pattern: (attributs) séparateur "nom"
            match = re.match(r'\(([^)]*)\)\s+"([^"]*)"\s+"?(.*?)"?$', decoded.strip())
            if match:
                attrs_str = match.group(1)
                separator = match.group(2)
                folder_name = match.group(3)

                attrs = [a.strip() for a in attrs_str.split() if a.strip()]
                has_children = "\\HasChildren" in attrs
                is_selectable = "\\NoSelect" not in attrs

                folder_type = "folder"
                if "\\Sent" in attrs:
                    folder_type = "sent"
                elif "\\Trash" in attrs:
                    folder_type = "trash"
                elif "\\Drafts" in attrs:
                    folder_type = "drafts"
                elif "\\Junk" in attrs or "\\Spam" in attrs:
                    folder_type = "junk"
                elif "\\Archive" in attrs:
                    folder_type = "archive"
                elif "\\All" in attrs or "\\AllMail" in attrs:
                    folder_type = "all"

                parsed_folders.append({
                    "name": folder_name,
                    "attributes": attrs,
                    "separator": separator,
                    "type": folder_type,
                    "has_children": has_children,
                    "selectable": is_selectable,
                })
            else:
                # Fallback : prendre la ligne brute
                folder_name = decoded.strip()
                if " " in folder_name:
                    folder_name = folder_name.rsplit(" ", 1)[-1].strip('"')
                parsed_folders.append({
                    "name": folder_name,
                    "attributes": [],
                    "separator": "/",
                    "type": "folder",
                    "has_children": False,
                    "selectable": True,
                })

        result["folders"] = parsed_folders
        logger.info(f"{len(parsed_folders)} dossier(s) trouvé(s) pour {account.label}")

    except Exception as e:
        result["error"] = f"Erreur IMAP : {e}"
        logger.exception("IMAP error during folder listing")
    finally:
        try:
            imap.logout()
        except Exception:
            pass

    return result


# ═══════════════════════════════════════════════════════════════════
#  SMTP — Envoi d'emails
# ═══════════════════════════════════════════════════════════════════

def send_email(
    account: EmailAccount,
    to: list,
    subject: str,
    body: str,
    cc: list = None,
    bcc: list = None,
    body_html: str = None,
    attachment_paths: list = None,
    workspace_dir: str = None,
) -> dict:
    """Envoie un email via SMTP.

    Args:
        account: Compte à utiliser pour l'envoi.
        to: Liste des destinataires (emails ou "Name <email>").
        subject: Sujet de l'email.
        body: Corps en texte brut.
        cc: Copie carbone.
        bcc: Copie carbone invisible.
        body_html: Corps en HTML (optionnel, si fourni, crée un multipart/alternative).
        attachment_paths: Chemins vers des fichiers à joindre.
        workspace_dir: Répertoire workspace.

    Retourne:
        dict avec les clés status, to, subject, saved_as.
    """
    recipients = list(to) + (cc or []) + (bcc or [])
    if not recipients:
        return {"status": "error", "detail": "Aucun destinataire spécifié"}

    msg = EmailMessage()
    msg["From"] = account.username
    msg["To"] = ", ".join(to) if isinstance(to, list) else to
    if cc:
        msg["Cc"] = ", ".join(cc) if isinstance(cc, list) else cc
    msg["Subject"] = subject
    msg["Date"] = email.utils.formatdate(localtime=True)

    # Corps du message
    if body_html:
        # Message multipart/alternative avec texte et HTML
        msg.set_content(body)
        msg.add_alternative(body_html, subtype="html")
    else:
        msg.set_content(body)

    # Pièces-jointes
    attachment_paths = attachment_paths or []
    for filepath in attachment_paths:
        filepath = os.path.expanduser(filepath)
        if not os.path.exists(filepath):
            logger.warning(f"Pièce-jointe introuvable : {filepath}")
            continue

        try:
            with open(filepath, "rb") as f:
                file_data = f.read()

            filename = os.path.basename(filepath)
            # Déterminer le type MIME
            import mimetypes
            maintype, subtype = mimetypes.guess_type(filename)
            if maintype is None:
                maintype = "application"
                subtype = "octet-stream"

            msg.add_attachment(
                file_data,
                maintype=maintype,
                subtype=subtype,
                filename=filename,
            )
        except OSError as e:
            logger.error(f"Erreur de lecture pièce-jointe {filepath}: {e}")

    # Sauvegarder une copie dans le répertoire des emails
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%d_%H-%M-%S")
    sender_safe = _safe_filename(account.username, max_len=50)
    filename = f"{ts}_sent_{sender_safe}.eml"
    emails_dir = _get_email_dir(workspace_dir)
    os.makedirs(emails_dir, exist_ok=True)
    filepath = os.path.join(emails_dir, filename)

    with open(filepath, "wb") as f:
        f.write(msg.as_bytes())

    # Envoi via SMTP
    try:
        if account.smtp_use_tls and account.smtp_port == 465:
            # SSL direct
            smtp = smtplib.SMTP_SSL(
                account.smtp_server, account.smtp_port, timeout=30
            )
        else:
            smtp = smtplib.SMTP(
                account.smtp_server, account.smtp_port, timeout=30
            )
            if account.smtp_use_tls:
                smtp.starttls()

        smtp.login(account.username, account.password)
        smtp.send_message(msg)
        smtp.quit()

        logger.info(f"Email envoyé : {subject} -> {recipients}")
        return {
            "status": "sent",
            "to": to,
            "cc": cc or [],
            "bcc": bcc or [],
            "subject": subject,
            "saved_as": filename,
        }

    except smtplib.SMTPException as e:
        logger.error(f"Erreur SMTP : {e}")
        return {"status": "error", "detail": str(e), "saved_as": filename}
    except Exception as e:
        logger.error(f"Erreur d'envoi : {e}")
        return {"status": "error", "detail": str(e), "saved_as": filename}


# ═══════════════════════════════════════════════════════════════════
#  Gestion locale —  Liste et recherche
# ═══════════════════════════════════════════════════════════════════

def list_local_emails(
    query: str = None,
    from_addr: str = None,
    limit: int = 50,
    offset: int = 0,
    workspace_dir: str = None,
) -> dict:
    """Liste et recherche les emails téléchargés localement.

    Args:
        query: Recherche textuelle dans le sujet et l'expéditeur.
        from_addr: Filtre par expéditeur.
        limit: Nombre max de résultats (défaut: 50).
        offset: Décalage pour pagination (défaut: 0).
        workspace_dir: Répertoire workspace.

    Retourne:
        dict avec total, results (liste de dicts).
    """
    index = _load_index(workspace_dir)
    emails = index.get("emails", [])

    # Filtres
    if query:
        query_lower = query.lower()
        emails = [
            e for e in emails
            if query_lower in e.get("subject", "").lower()
            or query_lower in e.get("sender", "").lower()
        ]

    if from_addr:
        from_lower = from_addr.lower()
        emails = [
            e for e in emails
            if from_lower in e.get("sender", "").lower()
        ]

    # Trier par date (plus récent d'abord)
    emails.sort(key=lambda e: e.get("date", ""), reverse=True)

    total = len(emails)
    page = emails[offset:offset + limit]

    # Ajouter le chemin complet pour chaque email
    emails_dir = _get_email_dir(workspace_dir)
    for entry in page:
        entry["filepath"] = os.path.join(emails_dir, entry.get("filename", ""))
        # Ajouter les chemins complets des pièces-jointes
        attachments_dir = _get_attachments_dir(workspace_dir)
        entry["attachment_paths"] = [
            os.path.join(attachments_dir, att)
            for att in entry.get("attachment_files", [])
        ]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": page,
    }
