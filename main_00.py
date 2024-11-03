import gradio as gr
from bot_src.bot import CuriousPeerBot
from bot_src.utils import read_pdf, save_chat_history
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize bot
bot = CuriousPeerBot()

def process_file(file):
    """Process uploaded PDF file"""
    if file is None:
        return "Please upload a PDF file."
    
    text = read_pdf(file.name)
    tldr = bot.generate_tldr(text)
    return f"TLDR of {os.path.basename(file.name)}:\n\n{tldr}"

def chat(message, history):
    """Handle chat interaction"""
    bot_response = bot.chat(message)
    history.append((message, bot_response))
    return "", history

def save_history():
    """Save chat history"""
    if not bot.get_chat_history():
        return "No chat history to save."
    
    file_name = "chat_session"  # Default name if no file was uploaded
    output_file = save_chat_history(file_name, bot.get_chat_history())
    return f"Chat history saved to: {output_file}"

# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".gradio-container {font-family: Arial, sans-serif}"
) as interface:
    gr.Markdown("# 📚 Curious Peer Bot")
    gr.Markdown("Upload an academic article and let's discuss it together!")
    
    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="Upload Academic Article (PDF)",
                file_types=[".pdf"]
            )
            process_btn = gr.Button("📄 Process Article", variant="primary")
            tldr_output = gr.Textbox(
                label="Article Summary",
                lines=10,
                interactive=False
            )
            
    with gr.Row():
        chatbot = gr.Chatbot(
            label="Discussion",
            height=400,
            bubble_full_width=False,
            show_copy_button=True
        )
        
    with gr.Row():
        msg_input = gr.Textbox(
            label="Your message",
            placeholder="Type your thoughts or questions here... (Ctrl+Enter to send)",
            lines=3,
            show_label=False
        )
        
    with gr.Row():
        with gr.Column(scale=1):
            send_btn = gr.Button("🚀 Send", variant="primary")
        with gr.Column(scale=1):
            save_btn = gr.Button("💾 Save Chat History", variant="secondary")
            save_status = gr.Textbox(label="Save Status", interactive=False)
    
    # Event handlers
    process_btn.click(
        fn=process_file,
        inputs=[file_input],
        outputs=[tldr_output]
    )
    
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    )
    
    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    )
    
    save_btn.click(
        fn=save_history,
        outputs=[save_status]
    )

if __name__ == "__main__":
    interface.launch(
        server_port=7860,
        share=True,
        debug=True
    )