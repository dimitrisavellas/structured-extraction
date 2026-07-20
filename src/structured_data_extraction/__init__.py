"""
Loads environment variables from the nearest .env file on import so that
config.py and all other modules see the user's settings immediately.
"""
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv(usecwd=True))
