"""
Curious Peer Bot Package
-----------------------
A chatbot designed to engage in academic discussions and analysis.

This package contains the core components for the Curious Peer Bot:
- Bot implementation (bot.py)
- Utility functions (utils.py)
- System prompts and configurations
"""

from .bot import CuriousPeerBot
from .utils import read_pdf, save_chat_history

__version__ = "1.0.0"
__author__ = "Minjung Shin"
__email__ = "mjshin77@snu.ac.kr"

__all__ = [
    'CuriousPeerBot',
    'read_pdf',
    'save_chat_history',
]