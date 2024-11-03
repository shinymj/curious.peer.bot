from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
import os

class CuriousPeerBot:
    def __init__(self):
        self.current_file = "chat_session" 

        self.chat_model = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            temperature=0.7,
            max_tokens=4096
        )
        
        with open("bot_src/sys_prompt.txt", "r") as f:
            self.system_prompt = f.read()
            
        self.chat_history: List[Dict[str, str]] = []
        self.output_parser = StrOutputParser()
        
        # Create conversation prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create the chain
        self.chain = self.prompt | self.chat_model | self.output_parser
    
    def set_current_file(self, filename: str):
        """Set current file name"""
        self.current_file = filename
        
    def generate_tldr(self, text: str) -> str:
        """Generate TLDR summary of the article"""
        tldr_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at summarizing academic articles concisely."),
            ("human", "Please provide a TLDR summary of the following academic article. "
                     "Focus on the main findings, methodology, and significance. "
                     "Use bullet points for clarity:\n\n{text}")
        ])
        
        chain = tldr_prompt | self.chat_model | self.output_parser
        return chain.invoke({"text": text})
        
    def chat(self, user_input: str) -> str:
        """Generate response to user input"""
        # Convert chat history to message format
        messages = []
        for msg in self.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # Generate response
        response = self.chain.invoke({
            "chat_history": messages,
            "input": user_input
        })
        
        # Update chat history
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": response})
        
        return response

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Return chat history"""
        return self.chat_history
    

