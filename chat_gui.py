"""
Chat Interface for Smart AI Model Builder
========================================

This module provides a chat interface for testing generated AI models.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import customtkinter as ctk
import threading
import json
import os
import sys
from datetime import datetime

class ChatInterface:
    """Chat interface for testing AI models"""
    
    def __init__(self, app, model_path=None):
        """Initialize the chat interface"""
        self.app = app
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Initialize the GUI
        self.root = ctk.CTkToplevel()
        self.root.title("AI Model Chat Interface")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Load model if path provided
        if self.model_path and os.path.exists(self.model_path):
            self.load_model()
        
        # Create widgets
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 15))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="🤖 AI Model Chat Interface",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(side="left")
        
        model_label = ctk.CTkLabel(
            header_frame,
            text=f"Model: {self.model_path or 'Not loaded'}",
            font=ctk.CTkFont(size=12)
        )
        model_label.pack(side="right")
        
        # Chat display
        chat_frame = ctk.CTkFrame(main_frame)
        chat_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            width=70,
            height=20,
            bg="#2b2b2b",
            fg="white",
            font=("Consolas", 11)
        )
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=10)
        self.chat_display.config(state=tk.DISABLED)
        
        # Input area
        input_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        input_frame.pack(fill="x")
        
        self.input_field = ctk.CTkEntry(input_frame, height=40, font=ctk.CTkFont(size=12))
        self.input_field.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.input_field.bind("<Return>", self.send_message)
        
        send_button = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.send_message,
            width=80,
            height=40
        )
        send_button.pack(side="right")
        
        # Control buttons
        control_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        control_frame.pack(fill="x", pady=(10, 0))
        
        clear_button = ctk.CTkButton(
            control_frame,
            text="Clear Chat",
            command=self.clear_chat,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE")
        )
        clear_button.pack(side="left", padx=(0, 10))
        
        load_button = ctk.CTkButton(
            control_frame,
            text="Load Model",
            command=self.load_model_dialog,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE")
        )
        load_button.pack(side="left")
        
        # Add welcome message
        self.add_message("AI Assistant", "Hello! I'm your AI assistant. Ask me anything!", "system")
        
    def load_model(self):
        """Load the AI model"""
        try:
            # This would load the actual model in a real implementation
            self.add_message("System", f"Model loaded from: {self.model_path}", "system")
        except Exception as e:
            self.add_message("System", f"Failed to load model: {e}", "error")
    
    def load_model_dialog(self):
        """Open dialog to load a model"""
        from tkinter import filedialog
        model_dir = filedialog.askdirectory(title="Select Model Directory")
        if model_dir:
            self.model_path = model_dir
            self.load_model()
    
    def send_message(self, event=None):
        """Send a message to the AI"""
        user_input = self.input_field.get().strip()
        if not user_input:
            return
            
        # Clear input field
        self.input_field.delete(0, tk.END)
        
        # Add user message to chat
        self.add_message("You", user_input, "user")
        
        # Process with AI (simulated)
        self.process_ai_response(user_input)
    
    def process_ai_response(self, user_input):
        """Process user input and generate AI response"""
        # Simulate AI processing
        import time
        time.sleep(1)
        
        # Generate response based on input
        responses = {
            "hello": "Hello there! How can I assist you today?",
            "help": "I'm here to help! You can ask me questions or give me tasks to perform.",
            "how are you": "I'm functioning optimally, thank you for asking!",
            "what can you do": "I can answer questions, help with tasks, and assist with various topics.",
            "bye": "Goodbye! Feel free to come back if you need assistance.",
            "default": "I understand your input. In a real implementation, I would process this with an AI model and provide a relevant response."
        }
        
        # Find appropriate response
        response = responses.get(user_input.lower(), responses["default"])
        
        # Add AI response to chat
        self.add_message("AI Assistant", response, "ai")
    
    def add_message(self, sender, message, msg_type="user"):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Format message based on type
        if msg_type == "user":
            formatted_message = f"[{datetime.now().strftime('%H:%M:%S')}] 👤 {sender}: {message}\\n"
        elif msg_type == "ai":
            formatted_message = f"[{datetime.now().strftime('%H:%M:%S')}] 🤖 {sender}: {message}\\n"
        else:  # system or error
            formatted_message = f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ {sender}: {message}\\n"
        
        self.chat_display.insert(tk.END, formatted_message)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def clear_chat(self):
        """Clear the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.add_message("System", "Chat cleared", "system")

# This module can be run standalone for testing
if __name__ == "__main__":
    app = type('App', (), {'logger': type('Logger', (), {'info': print, 'error': print})})()
    chat = ChatInterface(app)
    chat.root.mainloop()