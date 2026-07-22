# Maximum allowed size for a single file (50 MB)
MAX_FILE_SIZE: int = 50 * 1024 * 1024

# Maximum allowed total size for all uploaded files (200 MB)
MAX_TOTAL_SIZE: int = 200 * 1024 * 1024

# Allowed file types for upload
ALLOWED_TYPES: list = [".txt", ".pdf", ".docx", ".md"]

# Branding (change this whenever you land on a permanent name)
APP_NAME: str = "Verishelf"
APP_TAGLINE: str = "Verified answers, not hallucinations."
APP_MASTHEAD: str = "Est. MMXXVI &middot; Vol. I &middot; A verification ledger"
