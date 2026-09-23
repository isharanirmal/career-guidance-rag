import os
from dotenv import load_dotenv


class ConfigManager:
    """
    Loads application configuration from the .env file.
    """

    def __init__(self):
        load_dotenv()

    def get_gemini_api_key(self):
        return os.getenv("GEMINI_API_KEY")