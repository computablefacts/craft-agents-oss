import logging
from fastapi import FastAPI
from dotenv import load_dotenv
from emails import router as emails_router

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API Proxy")

# Include routers
app.include_router(emails_router)
