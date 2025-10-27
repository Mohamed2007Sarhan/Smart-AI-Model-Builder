"""
GUI Module for Smart AI Model Builder
=====================================

This module provides the graphical user interface for the Smart AI Model Builder application.
It handles user input collection, progress display, and interaction with other modules.
It includes a user-friendly graphical interface for collecting user input, 
displaying progress, and interacting with other modules.
- Make By Mohamed Sarhan
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import customtkinter as ctk
from datetime import datetime
import threading
import os
import json

class ModelBuilderGUI:
    """GUI class for the Smart AI Model Builder application"""
    
    def __init__(self, app):
        """Initialize the GUI"""
        self.app = app
        self.root = ctk.CTk()
        self.root.title("Smart AI Model Builder - Create AI Models from Scratch")
        self.root.geometry("950x800")
        self.root.minsize(850, 700)
        self.root.resizable(True, True)
        
        # Theme variables
        self.current_theme = "dark"
        self.current_color_theme = "blue"
        
        # Configure customtkinter appearance
        ctk.set_appearance_mode(self.current_theme)
        ctk.set_default_color_theme(self.current_color_theme)
        
        # Initialize UI components
        self.create_widgets()
        
        # Progress tracking
        self.current_step = 0
        self.total_steps = 5
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title with theme selector
        title_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 15))
        
        title_label = ctk.CTkLabel(
            title_frame, 
            text="🤖 Smart AI Model Builder", 
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(side="left")
        
        # Theme selector
        theme_frame = ctk.CTkFrame(title_frame, fg_color="transparent")
        theme_frame.pack(side="right")
        
        theme_label = ctk.CTkLabel(theme_frame, text="Theme:", font=ctk.CTkFont(size=12))
        theme_label.pack(side="left", padx=(0, 5))
        
        self.theme_var = tk.StringVar(value=self.current_theme)
        theme_combo = ctk.CTkComboBox(
            theme_frame,
            values=["dark", "light"],
            variable=self.theme_var,
            width=100,
            command=self.change_theme
        )
        theme_combo.pack(side="left")
        
        # Notebook for different steps
        self.notebook = ctk.CTkTabview(main_frame)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Create tabs
        self.create_input_tab()
        self.create_advanced_tab()
        self.create_progress_tab()
        self.create_backup_tab()
        
        # Control buttons
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.prev_button = ctk.CTkButton(
            button_frame, 
            text="Previous", 
            command=self.previous_step,
            state="disabled"
        )
        self.prev_button.pack(side="left", padx=(0, 10))
        
        self.next_button = ctk.CTkButton(
            button_frame, 
            text="Next", 
            command=self.next_step
        )
        self.next_button.pack(side="left")
        
        self.build_button = ctk.CTkButton(
            button_frame, 
            text="🚀 Build AI Model", 
            command=self.build_model,
            state="disabled",
            fg_color="#1e88e5",
            hover_color="#1976d2"
        )
        self.build_button.pack(side="right", padx=(0, 10))
        
        test_button = ctk.CTkButton(
            button_frame, 
            text="💬 Test Model Chat", 
            command=self.open_chat_interface,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE")
        )
        test_button.pack(side="right", padx=(0, 10))
        
    def change_theme(self, theme_name):
        """Change the application theme"""
        ctk.set_appearance_mode(theme_name)
        self.current_theme = theme_name
        
    def create_input_tab(self):
        """Create the input collection tab"""
        self.notebook.add("Basic Info")
        input_frame = ctk.CTkFrame(self.notebook.tab("Basic Info"))
        input_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header_label = ctk.CTkLabel(
            input_frame,
            text="Basic Model Information",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        header_label.pack(pady=(0, 20))
        
        # Model Name
        model_name_label = ctk.CTkLabel(input_frame, text="AI Model Name:", font=ctk.CTkFont(size=14))
        model_name_label.pack(anchor="w", pady=(0, 5))
        
        self.model_name_entry = ctk.CTkEntry(input_frame, width=500, height=35, font=ctk.CTkFont(size=14))
        self.model_name_entry.pack(anchor="w", pady=(0, 15))
        self.model_name_entry.insert(0, self.app.user_data.get("last_model_name", "My Custom AI Model"))
        
        # API Keys Section
        api_keys_label = ctk.CTkLabel(input_frame, text="API Keys:", font=ctk.CTkFont(size=16, weight="bold"))
        api_keys_label.pack(anchor="w", pady=(20, 10))
        
        # Hugging Face API Key
        hf_key_label = ctk.CTkLabel(input_frame, text="Hugging Face API Key:", font=ctk.CTkFont(size=14))
        hf_key_label.pack(anchor="w", pady=(0, 5))
        
        self.hf_key_entry = ctk.CTkEntry(input_frame, width=500, height=35, show="*", font=ctk.CTkFont(size=14))
        self.hf_key_entry.pack(anchor="w", pady=(0, 15))
        self.hf_key_entry.insert(0, self.app.user_data.get("hf_api_key", ""))
        
        # OpenAI API Key
        openai_key_label = ctk.CTkLabel(input_frame, text="OpenAI API Key (Optional):", font=ctk.CTkFont(size=14))
        openai_key_label.pack(anchor="w", pady=(0, 5))
        
        self.openai_key_entry = ctk.CTkEntry(input_frame, width=500, height=35, show="*", font=ctk.CTkFont(size=14))
        self.openai_key_entry.pack(anchor="w", pady=(0, 15))
        self.openai_key_entry.insert(0, self.app.user_data.get("openai_api_key", ""))
        
        # Google Gemini API Key
        gemini_key_label = ctk.CTkLabel(input_frame, text="Google Gemini API Key (Optional):", font=ctk.CTkFont(size=14))
        gemini_key_label.pack(anchor="w", pady=(0, 5))
        
        self.gemini_key_entry = ctk.CTkEntry(input_frame, width=500, height=35, show="*", font=ctk.CTkFont(size=14))
        self.gemini_key_entry.pack(anchor="w", pady=(0, 15))
        self.gemini_key_entry.insert(0, self.app.user_data.get("gemini_api_key", ""))
        
        # Project Description
        desc_label = ctk.CTkLabel(input_frame, text="Project Description:", font=ctk.CTkFont(size=14))
        desc_label.pack(anchor="w", pady=(0, 5))
        
        self.desc_textbox = ctk.CTkTextbox(input_frame, width=500, height=120, font=ctk.CTkFont(size=13))
        self.desc_textbox.pack(anchor="w", pady=(0, 15))
        self.desc_textbox.insert("1.0", self.app.user_data.get("last_description", "Describe what you want your AI model to do..."))
        
        # Additional info
        info_label = ctk.CTkLabel(
            input_frame, 
            text="💡 Tip: Be as specific as possible about your model's purpose and requirements.\n"
                 "The more detailed you are, the better we can tailor the model for you.",
            font=ctk.CTkFont(size=12),
            text_color="#bbbbbb"
        )
        info_label.pack(anchor="w", pady=(10, 0))
        
    def create_advanced_tab(self):
        """Create the advanced settings tab"""
        self.notebook.add("Advanced")
        advanced_frame = ctk.CTkFrame(self.notebook.tab("Advanced"))
        advanced_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header_label = ctk.CTkLabel(
            advanced_frame,
            text="Advanced Settings & Dataset Options",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        header_label.pack(pady=(0, 20))
        
        # Additional APIs
        api_label = ctk.CTkLabel(advanced_frame, text="Additional APIs (comma separated):", font=ctk.CTkFont(size=14))
        api_label.pack(anchor="w", pady=(0, 5))
        
        self.api_entry = ctk.CTkEntry(advanced_frame, width=500, height=35, font=ctk.CTkFont(size=14))
        self.api_entry.pack(anchor="w", pady=(0, 15))
        self.api_entry.insert(0, "OpenAI, StabilityAI")
        
        # Dataset Options
        dataset_label = ctk.CTkLabel(advanced_frame, text="Training Dataset Options:", font=ctk.CTkFont(size=16, weight="bold"))
        dataset_label.pack(anchor="w", pady=(20, 10))
        
        # Dataset type selection
        dataset_type_frame = ctk.CTkFrame(advanced_frame, fg_color="transparent")
        dataset_type_frame.pack(fill="x", pady=(0, 15))
        
        dataset_type_label = ctk.CTkLabel(dataset_type_frame, text="Dataset Type:", font=ctk.CTkFont(size=14))
        dataset_type_label.pack(anchor="w", side="left")
        
        self.dataset_type_var = tk.StringVar(value="auto_generate")
        auto_gen_radio = ctk.CTkRadioButton(
            dataset_type_frame, 
            text="Auto-generate from description", 
            variable=self.dataset_type_var, 
            value="auto_generate",
            font=ctk.CTkFont(size=13)
        )
        auto_gen_radio.pack(anchor="w", side="left", padx=(20, 15))
        
        custom_radio = ctk.CTkRadioButton(
            dataset_type_frame, 
            text="Upload custom dataset", 
            variable=self.dataset_type_var, 
            value="custom",
            font=ctk.CTkFont(size=13)
        )
        custom_radio.pack(anchor="w", side="left", padx=(0, 15))
        
        # Dataset file selection (hidden by default)
        self.dataset_file_frame = ctk.CTkFrame(advanced_frame, fg_color="transparent")
        self.dataset_file_frame.pack(fill="x", pady=(0, 15))
        self.dataset_file_frame.pack_forget()  # Hidden by default
        
        self.dataset_file_label = ctk.CTkLabel(self.dataset_file_frame, text="Dataset File:", font=ctk.CTkFont(size=14))
        self.dataset_file_label.pack(anchor="w", side="left")
        
        self.dataset_file_entry = ctk.CTkEntry(self.dataset_file_frame, width=300, font=ctk.CTkFont(size=13))
        self.dataset_file_entry.pack(anchor="w", side="left", padx=(10, 10))
        
        self.browse_button = ctk.CTkButton(
            self.dataset_file_frame, 
            text="Browse", 
            command=self.browse_dataset_file,
            width=80
        )
        self.browse_button.pack(anchor="w", side="left")
        
        # Bind radio button changes
        auto_gen_radio.configure(command=self.on_dataset_type_change)
        custom_radio.configure(command=self.on_dataset_type_change)
        
        # Model customization options
        customization_label = ctk.CTkLabel(advanced_frame, text="Model Customization:", font=ctk.CTkFont(size=16, weight="bold"))
        customization_label.pack(anchor="w", pady=(20, 10))
        
        # Model size selection
        size_frame = ctk.CTkFrame(advanced_frame, fg_color="transparent")
        size_frame.pack(fill="x", pady=(0, 15))
        
        size_label = ctk.CTkLabel(size_frame, text="Model Size:", font=ctk.CTkFont(size=14))
        size_label.pack(anchor="w", side="left")
        
        self.model_size_var = tk.StringVar(value="medium")
        small_radio = ctk.CTkRadioButton(
            size_frame, 
            text="Small (faster, less accurate)", 
            variable=self.model_size_var, 
            value="small",
            font=ctk.CTkFont(size=13)
        )
        small_radio.pack(anchor="w", side="left", padx=(20, 15))
        
        medium_radio = ctk.CTkRadioButton(
            size_frame, 
            text="Medium (balanced)", 
            variable=self.model_size_var, 
            value="medium",
            font=ctk.CTkFont(size=13)
        )
        medium_radio.pack(anchor="w", side="left", padx=(0, 15))
        
        large_radio = ctk.CTkRadioButton(
            size_frame, 
            text="Large (slower, more accurate)", 
            variable=self.model_size_var, 
            value="large",
            font=ctk.CTkFont(size=13)
        )
        large_radio.pack(anchor="w", side="left", padx=(0, 15))
        
        # Training time selection
        time_frame = ctk.CTkFrame(advanced_frame, fg_color="transparent")
        time_frame.pack(fill="x", pady=(0, 15))
        
        time_label = ctk.CTkLabel(time_frame, text="Training Time:", font=ctk.CTkFont(size=14))
        time_label.pack(anchor="w", side="left")
        
        self.training_time_var = tk.StringVar(value="moderate")
        quick_radio = ctk.CTkRadioButton(
            time_frame, 
            text="Quick (1-2 hours)", 
            variable=self.training_time_var, 
            value="quick",
            font=ctk.CTkFont(size=13)
        )
        quick_radio.pack(anchor="w", side="left", padx=(20, 15))
        
        moderate_radio = ctk.CTkRadioButton(
            time_frame, 
            text="Moderate (4-8 hours)", 
            variable=self.training_time_var, 
            value="moderate",
            font=ctk.CTkFont(size=13)
        )
        moderate_radio.pack(anchor="w", side="left", padx=(0, 15))
        
        extensive_radio = ctk.CTkRadioButton(
            time_frame, 
            text="Extensive (12+ hours)", 
            variable=self.training_time_var, 
            value="extensive",
            font=ctk.CTkFont(size=13)
        )
        extensive_radio.pack(anchor="w", side="left", padx=(0, 15))
        
    def create_progress_tab(self):
        """Create the progress display tab"""
        self.notebook.add("Progress")
        progress_frame = ctk.CTkFrame(self.notebook.tab("Progress"))
        progress_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header_label = ctk.CTkLabel(
            progress_frame,
            text="Build Progress",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        header_label.pack(pady=(0, 20))
        
        # Progress log
        log_frame = ctk.CTkFrame(progress_frame)
        log_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        log_label = ctk.CTkLabel(log_frame, text="Progress Log:", font=ctk.CTkFont(size=14))
        log_label.pack(anchor="w", padx=10, pady=(10, 5))
        
        self.log_textbox = scrolledtext.ScrolledText(
            log_frame,
            width=70,
            height=15,
            bg="#2b2b2b",
            fg="white",
            font=("Consolas", 11)
        )
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Progress bar
        progress_bar_frame = ctk.CTkFrame(progress_frame, fg_color="transparent")
        progress_bar_frame.pack(fill="x", pady=(0, 10))
        
        self.progress_bar = ctk.CTkProgressBar(progress_bar_frame, height=20)
        self.progress_bar.pack(fill="x", padx=10)
        self.progress_bar.set(0)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            progress_bar_frame,
            text="Ready to build",
            font=ctk.CTkFont(size=13)
        )
        self.status_label.pack(pady=(5, 0))
        
    def create_backup_tab(self):
        """Create the backup management tab"""
        self.notebook.add("Backups")
        backup_frame = ctk.CTkFrame(self.notebook.tab("Backups"))
        backup_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header_label = ctk.CTkLabel(
            backup_frame,
            text="Backup Management",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        header_label.pack(pady=(0, 20))
        
        # Backup list
        backup_list_label = ctk.CTkLabel(backup_frame, text="Available Backups:", font=ctk.CTkFont(size=14))
        backup_list_label.pack(anchor="w", pady=(0, 10))
        
        # Create a frame for the listbox and scrollbar
        list_frame = ctk.CTkFrame(backup_frame)
        list_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        # Create listbox for backups
        self.backup_listbox = tk.Listbox(
            list_frame,
            bg="#2b2b2b",
            fg="white",
            font=("Consolas", 11),
            selectbackground="#1e88e5"
        )
        self.backup_listbox.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        
        # Scrollbar
        scrollbar = ctk.CTkScrollbar(list_frame, command=self.backup_listbox.yview)
        scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)
        self.backup_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Refresh button
        refresh_button = ctk.CTkButton(
            backup_frame,
            text="Refresh",
            command=self.refresh_backups
        )
        refresh_button.pack(side="left", padx=(0, 10))
        
        # Restore button
        restore_button = ctk.CTkButton(
            backup_frame,
            text="Restore Selected",
            command=self.restore_selected_backup,
            fg_color="#4caf50",
            hover_color="#388e3c"
        )
        restore_button.pack(side="left", padx=(0, 10))
        
        # Create backup button
        create_backup_button = ctk.CTkButton(
            backup_frame,
            text="Create Backup",
            command=self.create_new_backup,
            fg_color="#ff9800",
            hover_color="#f57c00"
        )
        create_backup_button.pack(side="left")
        
        # Load initial backup list
        self.refresh_backups()
        
    def refresh_backups(self):
        """Refresh the backup list"""
        # Clear current list
        self.backup_listbox.delete(0, tk.END)
        
        # Get backups from app
        backups = self.app.list_backups()
        
        # Add backups to list
        for backup in backups:
            self.backup_listbox.insert(tk.END, backup)
            
        self.log_message(f"Refreshed backup list: {len(backups)} backups found", "INFO")
        
    def restore_selected_backup(self):
        """Restore the selected backup"""
        selection = self.backup_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a backup to restore.")
            return
            
        backup_name = self.backup_listbox.get(selection[0])
        
        # Confirm restore
        if messagebox.askyesno("Confirm Restore", 
                              f"Are you sure you want to restore from backup '{backup_name}'?\n"
                              "This will overwrite your current settings."):
            try:
                self.app.restore_backup(backup_name)
                self.log_message(f"Restored from backup: {backup_name}", "SUCCESS")
                messagebox.showinfo("Success", f"Successfully restored from backup: {backup_name}")
            except Exception as e:
                self.log_message(f"Failed to restore backup: {e}", "ERROR")
                messagebox.showerror("Error", f"Failed to restore backup: {e}")
                
    def create_new_backup(self):
        """Create a new backup"""
        try:
            self.app.create_backup()
            self.refresh_backups()
            self.log_message("New backup created successfully", "SUCCESS")
            messagebox.showinfo("Success", "New backup created successfully!")
        except Exception as e:
            self.log_message(f"Failed to create backup: {e}", "ERROR")
            messagebox.showerror("Error", f"Failed to create backup: {e}")
        
    def on_dataset_type_change(self):
        """Handle dataset type selection change"""
        if self.dataset_type_var.get() == "custom":
            self.dataset_file_frame.pack(fill="x", pady=(0, 15))
        else:
            self.dataset_file_frame.pack_forget()
            
    def browse_dataset_file(self):
        """Open file dialog to select dataset file"""
        file_path = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.dataset_file_entry.delete(0, tk.END)
            self.dataset_file_entry.insert(0, file_path)
            
    def next_step(self):
        """Move to the next step"""
        current_tab = self.notebook.get()
        tabs = ["Basic Info", "Advanced", "Progress", "Backups"]
        current_index = tabs.index(current_tab)
        
        if current_index < len(tabs) - 1:
            self.notebook.set(tabs[current_index + 1])
            self.update_navigation_buttons()
            
    def previous_step(self):
        """Move to the previous step"""
        current_tab = self.notebook.get()
        tabs = ["Basic Info", "Advanced", "Progress", "Backups"]
        current_index = tabs.index(current_tab)
        
        if current_index > 0:
            self.notebook.set(tabs[current_index - 1])
            self.update_navigation_buttons()
            
    def update_navigation_buttons(self):
        """Update navigation button states based on current tab"""
        current_tab = self.notebook.get()
        tabs = ["Basic Info", "Advanced", "Progress", "Backups"]
        current_index = tabs.index(current_tab)
        
        if current_index == 0:
            self.prev_button.configure(state="disabled")
            self.next_button.configure(state="normal")
            self.build_button.configure(state="disabled")
        elif current_index == len(tabs) - 1:
            self.prev_button.configure(state="normal")
            self.next_button.configure(state="disabled")
            self.build_button.configure(state="normal")
        else:
            self.prev_button.configure(state="normal")
            self.next_button.configure(state="normal")
            self.build_button.configure(state="disabled")
            
    def build_model(self):
        """Start the model building process"""
        # Validate inputs
        if not self.validate_inputs():
            return
            
        # Collect user data
        self.collect_user_data()
        
        # Disable buttons during processing
        self.build_button.configure(state="disabled")
        self.prev_button.configure(state="disabled")
        self.next_button.configure(state="disabled")
        
        # Update status
        self.status_label.configure(text="Building model... This may take a while")
        
        # Start building in a separate thread
        build_thread = threading.Thread(target=self.app.build_model_process)
        build_thread.daemon = True
        build_thread.start()
        
    def open_chat_interface(self):
        """Open the chat interface for testing models"""
        try:
            self.app.create_chat_gui()
        except Exception as e:
            self.log_message(f"Failed to open chat interface: {e}", "ERROR")
            messagebox.showerror("Error", f"Failed to open chat interface: {e}")
        
    def validate_inputs(self):
        """Validate user inputs"""
        model_name = self.model_name_entry.get().strip()
        hf_key = self.hf_key_entry.get().strip()
        openai_key = self.openai_key_entry.get().strip()
        gemini_key = self.gemini_key_entry.get().strip()
        description = self.desc_textbox.get("1.0", "end").strip()
        
        if not model_name or model_name == "My Custom AI Model":
            messagebox.showerror("Input Error", "Please enter a unique AI Model Name")
            return False
            
        # At least one API key is required
        if not hf_key and not openai_key and not gemini_key:
            messagebox.showerror("Input Error", "Please enter at least one API key (Hugging Face, OpenAI, or Google Gemini)")
            return False
            
        if not description or description == "Describe what you want your AI model to do...":
            messagebox.showerror("Input Error", "Please provide a detailed project description")
            return False
            
        # Check if custom dataset is selected but no file provided
        if self.dataset_type_var.get() == "custom":
            dataset_file = self.dataset_file_entry.get().strip()
            if not dataset_file:
                messagebox.showerror("Input Error", "Please select a dataset file or choose auto-generation")
                return False
            if not os.path.exists(dataset_file):
                messagebox.showerror("Input Error", "Dataset file not found")
                return False
                
        return True
        
    def collect_user_data(self):
        """Collect user input data"""
        self.app.user_data = {
            "model_name": self.model_name_entry.get().strip(),
            "hf_api_key": self.hf_key_entry.get().strip(),
            "openai_api_key": self.openai_key_entry.get().strip(),
            "gemini_api_key": self.gemini_key_entry.get().strip(),
            "description": self.desc_textbox.get("1.0", "end").strip(),
            "additional_apis": [
                api.strip() for api in self.api_entry.get().strip().split(",") 
                if api.strip()
            ],
            "dataset_type": self.dataset_type_var.get(),
            "dataset_file": self.dataset_file_entry.get().strip() if self.dataset_type_var.get() == "custom" else None,
            "model_size": self.model_size_var.get(),
            "training_time": self.training_time_var.get(),
            "last_model_name": self.model_name_entry.get().strip(),
            "last_description": self.desc_textbox.get("1.0", "end").strip()
        }
        
    def log_message(self, message, level="INFO"):
        """Add a message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}\n"
        self.log_textbox.insert("end", formatted_message)
        self.log_textbox.see("end")
        self.root.update_idletasks()
        
        # Update status label for important messages
        if level in ["SUCCESS", "ERROR"]:
            self.status_label.configure(text=message[:50] + "..." if len(message) > 50 else message)
        
    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.set(value)
        self.root.update_idletasks()
        
    def show_popup(self, title, message):
        """Show a popup message"""
        messagebox.showinfo(title, message)
        
    def run(self):
        """Run the GUI main loop"""
        self.root.mainloop()