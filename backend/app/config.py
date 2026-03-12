# from pydantic import BaseModel
# import os

# class Settings(BaseModel):
#     API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
#     API_PORT: int = int(os.getenv("API_PORT", "8000"))

#     # security / limits
#     MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
#     ALLOWED_EXTS: str = os.getenv("ALLOWED_EXTS", ".pdf,.txt")

#     # RAG
#     OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
#     OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

#     # CORS
#     CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000")

# settings = Settings()

import os
from pydantic import BaseModel

class Settings(BaseModel):
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    ALLOWED_EXTS: str = os.getenv("ALLOWED_EXTS", ".pdf,.txt")

    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3:mini")

    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000")

settings = Settings()