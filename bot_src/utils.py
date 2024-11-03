import os
import datetime
from pypdf import PdfReader
from typing import List, Dict

def read_pdf(file_path: str) -> str:
    """Read and extract text from PDF file"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def save_chat_history(file_name: str, chat_history: List[Dict[str, str]]) -> str:
    """Save chat history to markdown file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f"{timestamp}_{file_name}.md")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Chat History with Curious Peer Bot\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Article: {file_name}\n\n")
        
        for message in chat_history:
            role = "Human" if message['role'] == 'user' else "Bot"
            f.write(f"## {role}:\n{message['content']}\n\n")
    
    return output_file