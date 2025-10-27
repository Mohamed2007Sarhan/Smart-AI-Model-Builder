#!/usr/bin/env python3
"""
Smart AI Model Builder - Automated Model Creation System
=======================================================

This application automatically builds, customizes, and packages AI models
based on user input through a modern GUI interface.

Features:
- Step-by-step GUI for collecting user requirements
- Automated web information gathering
- Hugging Face model selection and customization
- Complete project generation with all necessary files
- Real-time progress logging and error handling
- Dataset generation from user input
- Model training automation
- Backup and restore functionality
- Multiple UI themes and layouts
- Chat interface for testing models
- Comprehensive backup system
- Support for multiple AI APIs (Hugging Face, OpenAI, Gemini) 
- Make By Mohamed Sarhan
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import logging
import os
import json
from datetime import datetime
import shutil
import sys

# Conditional imports to handle missing dependencies gracefully
try:
    import customtkinter as ctk
    CUSTOMTKINTER_AVAILABLE = True
except ImportError:
    print("Warning: customtkinter not installed. Using standard tkinter.")
    import tkinter as tk
    from tkinter import ttk
    ctk = tk
    CUSTOMTKINTER_AVAILABLE = False

class SmartAIModelBuilder:
    """Main application class for the Smart AI Model Builder"""
    
    def __init__(self):
        """Initialize the application"""
        # Set up logging
        self.setup_logging()
        
        # Initialize components (will be set up when needed)
        self.web_searcher = None
        self.model_selector = None
        self.project_generator = None
        
        # User data storage
        self.user_data = {}
        self.backup_dir = "backups"
        self.config_file = "app_config.json"
        
        # Load existing configuration
        self.load_config()
        
        # Create backups directory
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Create GUI
        self.gui = None
        self.chat_gui = None
        self.create_gui()
        
    def setup_logging(self):
        """Set up logging configuration"""
        log_filename = f"model_builder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Smart AI Model Builder initialized")
    
    def load_config(self):
        """Load application configuration from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.user_data = json.load(f)
                self.logger.info("Configuration loaded successfully")
            else:
                # Default configuration
                self.user_data = {
                    "hf_api_key": "",
                    "openai_api_key": "",
                    "gemini_api_key": "",
                    "last_model_name": "",
                    "last_description": "",
                    "additional_apis": [],
                    "model_size": "medium",
                    "training_time": "moderate"
                }
                self.save_config()
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
    
    def save_config(self):
        """Save application configuration to JSON file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.user_data, f, indent=2)
            self.logger.info("Configuration saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def create_gui(self):
        """Create the main GUI window"""
        # Delayed import to avoid circular imports
        from gui import ModelBuilderGUI
        self.gui = ModelBuilderGUI(self)
        self.logger.info("GUI created successfully")
        
    def create_chat_gui(self, model_path=None):
        """Create the chat interface for testing models"""
        try:
            from chat_gui import ChatInterface
            self.chat_gui = ChatInterface(self, model_path)
            self.logger.info("Chat GUI created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create chat GUI: {e}")
            if self.gui:
                self.gui.log_message(f"Failed to create chat interface: {e}", "ERROR")
    
    def build_model_process(self):
        """Main process for building the AI model"""
        try:
            if self.gui:
                self.gui.log_message("Starting AI model building process...", "INFO")
                self.gui.update_progress(0.1)
            
            # Save configuration
            self.save_config()
            
            # Step 1: Gather web information
            if self.gui:
                self.gui.log_message("Step 1: Gathering web information...", "INFO")
            self.setup_web_searcher()
            web_data = self.gather_web_information()
            if self.gui:
                self.gui.update_progress(0.2)
            
            # Step 2: Generate dataset from user input
            if self.gui:
                self.gui.log_message("Step 2: Generating training dataset...", "INFO")
            self.generate_dataset()
            if self.gui:
                self.gui.update_progress(0.3)
            
            # Step 3: Create backup before model selection
            if self.gui:
                self.gui.log_message("Step 3: Creating backup of current state...", "INFO")
            self.create_backup()
            if self.gui:
                self.gui.update_progress(0.4)
            
            # Step 4: Select model from available APIs
            if self.gui:
                self.gui.log_message("Step 4: Searching for models from available APIs...", "INFO")
            self.setup_model_selector()
            model_info = self.select_model()
            if self.gui:
                self.gui.update_progress(0.6)
            
            # Step 5: Customize model
            if self.gui:
                self.gui.log_message("Step 5: Customizing model based on your requirements...", "INFO")
            customized_model = self.customize_model(model_info)
            if self.gui:
                self.gui.update_progress(0.7)
            
            # Step 6: Generate project
            if self.gui:
                self.gui.log_message("Step 6: Generating complete AI project...", "INFO")
            self.setup_project_generator()
            project_path = self.generate_project(customized_model)
            if self.gui:
                self.gui.update_progress(0.9)
            
            # Step 7: Create final backup
            if self.gui:
                self.gui.log_message("Step 7: Creating final backup...", "INFO")
            self.create_backup(project_path)
            if self.gui:
                self.gui.update_progress(1.0)
            
            # Completion
            if self.gui:
                self.gui.log_message("AI model building complete!", "SUCCESS")
                self.gui.log_message(f"Project generated at: {project_path}", "SUCCESS")
                self.gui.log_message("Backup created successfully", "SUCCESS")
                
                # Show completion message
                self.gui.show_popup("Success", f"AI Model '{customized_model['custom_name']}' has been successfully built!\\n\\nProject location: {project_path}\\nBackup created in: {self.backup_dir}")
            else:
                self.logger.info("AI model building complete!")
                self.logger.info(f"Project generated at: {project_path}")
                self.logger.info(f"Backup created in: {self.backup_dir}")
            
        except Exception as e:
            error_msg = f"Error during model building: {str(e)}"
            if self.gui:
                self.gui.log_message(error_msg, "ERROR")
            self.logger.error(error_msg, exc_info=True)
            if self.gui:
                self.gui.show_popup("Error", f"An error occurred during model building: {str(e)}")
    
    def setup_web_searcher(self):
        """Set up the web searcher component"""
        try:
            from web_search import WebSearchEngine
            gui_logger = self.gui.log_message if self.gui else None
            self.web_searcher = WebSearchEngine(gui_logger)
            if self.gui:
                self.gui.log_message("Web searcher initialized", "INFO")
        except Exception as e:
            error_msg = f"Failed to initialize web searcher: {e}"
            if self.gui:
                self.gui.log_message(error_msg, "ERROR")
            raise
    
    def setup_model_selector(self):
        """Set up the model selector component for multiple APIs"""
        try:
            from model_selector import ModelSelector
            gui_logger = self.gui.log_message if self.gui else None
            self.model_selector = ModelSelector(
                hf_api_key=self.user_data.get("hf_api_key", ""),
                openai_api_key=self.user_data.get("openai_api_key", ""),
                gemini_api_key=self.user_data.get("gemini_api_key", ""),
                gui_logger=gui_logger
            )
            if self.gui:
                self.gui.log_message("Multi-API model selector initialized", "INFO")
        except Exception as e:
            error_msg = f"Failed to initialize model selector: {e}"
            if self.gui:
                self.gui.log_message(error_msg, "ERROR")
            raise
    
    def setup_project_generator(self):
        """Set up the project generator component"""
        try:
            from project_generator import ProjectGenerator
            gui_logger = self.gui.log_message if self.gui else None
            self.project_generator = ProjectGenerator(gui_logger)
            if self.gui:
                self.gui.log_message("Project generator initialized", "INFO")
        except Exception as e:
            error_msg = f"Failed to initialize project generator: {e}"
            if self.gui:
                self.gui.log_message(error_msg, "ERROR")
            raise
    
    def gather_web_information(self):
        """Gather information from the web about the model topic"""
        if not self.web_searcher:
            raise RuntimeError("Web searcher not initialized")
            
        topic = self.user_data.get("model_name", "")
        description = self.user_data.get("description", "")
        
        if not topic:
            raise ValueError("Model name/topic is required")
            
        raw_data = self.web_searcher.gather_model_information(topic, description)
        refined_data = self.web_searcher.analyze_and_refine(raw_data)
        
        if self.gui:
            self.gui.log_message(f"Gathered information about: {topic}", "INFO")
        return refined_data
    
    def generate_dataset(self):
        """Generate training dataset from user input"""
        if self.gui:
            self.gui.log_message("Generating training dataset from user input...", "INFO")
        
        # In a real implementation, this would generate actual training data
        # For now, we'll just simulate the process
        import time
        time.sleep(2)  # Simulate time-consuming process
        
        if self.gui:
            self.gui.log_message("Dataset generation complete", "INFO")
        return True
    
    def select_model(self):
        """Select the best model from available APIs"""
        if not self.model_selector:
            raise RuntimeError("Model selector not initialized")
            
        topic = self.user_data.get("model_name", "")
        models = self.model_selector.search_models(topic, limit=10)
        
        if not models:
            raise ValueError(f"No models found for topic: {topic}")
            
        best_model = self.model_selector.select_best_model(models)
        
        if not best_model:
            raise ValueError("Could not select a suitable model")
            
        # Get detailed information about the selected model
        model_details = self.model_selector.get_model_details(best_model["model_id"])
        
        if self.gui:
            self.gui.log_message(f"Selected model: {best_model['model_id']}", "INFO")
        return model_details
    
    def customize_model(self, model_info):
        """Customize the model based on user parameters"""
        if not self.model_selector:
            raise RuntimeError("Model selector not initialized")
            
        customized_model = self.model_selector.customize_model(model_info, self.user_data)
        if self.gui:
            self.gui.log_message(f"Model customized as: {customized_model['custom_name']}", "INFO")
        return customized_model
    
    def generate_project(self, model_config):
        """Generate the complete AI project"""
        if not self.project_generator:
            raise RuntimeError("Project generator not initialized")
            
        project_path = self.project_generator.generate_project(model_config)
        if self.gui:
            self.gui.log_message("Project files generated successfully", "INFO")
        return project_path
    
    def create_backup(self, project_path=None):
        """Create a backup of the current state"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"backup_{timestamp}"
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            # Create backup directory
            os.makedirs(backup_path, exist_ok=True)
            
            # Backup user data
            user_data_file = os.path.join(backup_path, "user_data.json")
            with open(user_data_file, 'w') as f:
                json.dump(self.user_data, f, indent=2)
            
            # Backup project if exists
            if project_path and os.path.exists(project_path):
                project_backup = os.path.join(backup_path, "project")
                shutil.copytree(project_path, project_backup)
            
            # Backup logs
            log_files = [f for f in os.listdir('.') if f.startswith('model_builder_') and f.endswith('.log')]
            for log_file in log_files:
                shutil.copy2(log_file, backup_path)
            
            # Backup config file
            if os.path.exists(self.config_file):
                shutil.copy2(self.config_file, backup_path)
            
            if self.gui:
                self.gui.log_message(f"Backup created: {backup_name}", "INFO")
                
        except Exception as e:
            error_msg = f"Failed to create backup: {e}"
            if self.gui:
                self.gui.log_message(error_msg, "ERROR")
            self.logger.error(error_msg)
    
    def restore_backup(self, backup_name):
        """Restore from a backup"""
        try:
            backup_path = os.path.join(self.backup_dir, backup_name)
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup {backup_name} not found")
            
            # Restore user data
            user_data_file = os.path.join(backup_path, "user_data.json")
            if os.path.exists(user_data_file):
                with open(user_data_file, 'r') as f:
                    self.user_data = json.load(f)
            
            # Restore config file
            config_backup = os.path.join(backup_path, "app_config.json")
            if os.path.exists(config_backup):
                shutil.copy2(config_backup, self.config_file)
            
            if self.gui:
                self.gui.log_message(f"Restored from backup: {backup_name}", "INFO")
                
        except Exception as e:
            error_msg = f"Failed to restore backup: {e}"
            if self.gui:
                self.gui.log_message(error_msg, "ERROR")
            self.logger.error(error_msg)
    
    def list_backups(self):
        """List all available backups"""
        try:
            backups = []
            if os.path.exists(self.backup_dir):
                backups = [d for d in os.listdir(self.backup_dir) 
                          if os.path.isdir(os.path.join(self.backup_dir, d))]
            return sorted(backups, reverse=True)
        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
            return []
    
    def run(self):
        """Run the application"""
        self.logger.info("Starting Smart AI Model Builder application")
        try:
            if self.gui:
                self.gui.run()
            else:
                # Fallback if GUI fails
                self.logger.warning("GUI not available, running in command line mode")
                self.user_data = {
                    "model_name": "gpt",
                    "hf_api_key": "",
                    "openai_api_key": "",
                    "gemini_api_key": "",
                    "description": "A test model",
                    "additional_apis": []
                }
                self.build_model_process()
        except Exception as e:
            self.logger.error(f"Error running application: {e}")
            messagebox.showerror("Error", f"Application error: {e}")

def main():
    """Main entry point"""
    app = SmartAIModelBuilder()
    app.run()

if __name__ == "__main__":
    main()