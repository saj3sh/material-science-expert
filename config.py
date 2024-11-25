from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_TOKEN = os.getenv('QDRANT_TOKEN')
