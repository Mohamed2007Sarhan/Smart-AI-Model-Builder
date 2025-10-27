"""
Multi-API Model Selector for Smart AI Model Builder
==================================================

This module handles model selection and customization from multiple AI APIs.
It connects to Hugging Face, OpenAI, and Google Gemini APIs to find and compare models.
- Make By Mohamed Sarhan
"""

import requests
import logging
from huggingface_hub import HfApi, ModelFilter
import time

class ModelSelector:
    """Multi-API model selector and customizer"""
    
    def __init__(self, hf_api_key="", openai_api_key="", gemini_api_key="", gui_logger=None):
        """Initialize the multi-API model selector"""
        self.hf_api_key = hf_api_key
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.logger = logging.getLogger(__name__)
        self.gui_logger = gui_logger
        
        # Initialize Hugging Face API client if key provided
        self.hf_api = HfApi() if hf_api_key else None
        
        # Headers for API requests
        self.headers = {}
        
    def log_message(self, message, level="INFO"):
        """Log a message to both console and GUI"""
        self.logger.info(message)
        if self.gui_logger:
            self.gui_logger(message, level)
            
    def search_models(self, query, limit=10):
        """
        Search for models from all available APIs
        
        Args:
            query (str): The search query
            limit (int): Maximum number of models to return per API
            
        Returns:
            list: List of model information dictionaries
        """
        self.log_message(f"Searching for models related to: {query}")
        
        all_models = []
        
        # Search Hugging Face if API key provided
        if self.hf_api_key and self.hf_api:
            try:
                hf_models = self._search_hf_models(query, limit)
                all_models.extend(hf_models)
                self.log_message(f"Found {len(hf_models)} models on Hugging Face")
            except Exception as e:
                self.log_message(f"Error searching Hugging Face models: {e}", "ERROR")
        
        # Search OpenAI if API key provided
        if self.openai_api_key:
            try:
                openai_models = self._search_openai_models(query, limit)
                all_models.extend(openai_models)
                self.log_message(f"Found {len(openai_models)} models on OpenAI")
            except Exception as e:
                self.log_message(f"Error searching OpenAI models: {e}", "ERROR")
        
        # Search Gemini if API key provided
        if self.gemini_api_key:
            try:
                gemini_models = self._search_gemini_models(query, limit)
                all_models.extend(gemini_models)
                self.log_message(f"Found {len(gemini_models)} models on Gemini")
            except Exception as e:
                self.log_message(f"Error searching Gemini models: {e}", "ERROR")
        
        self.log_message(f"Total models found: {len(all_models)}")
        return all_models
    
    def _search_hf_models(self, query, limit=10):
        """Search for models on Hugging Face"""
        try:
            # Check if Hugging Face API is available
            if not self.hf_api:
                return []
                
            # Use Hugging Face API to search models
            models = self.hf_api.list_models(
                filter=ModelFilter(model_name=query),
                sort="downloads",
                direction=-1,
                limit=limit
            )
            
            model_list = []
            for model in models:
                model_info = {
                    "model_id": model.modelId,
                    "author": model.author,
                    "downloads": getattr(model, 'downloads', 0),
                    "likes": getattr(model, 'likes', 0),
                    "library_name": getattr(model, 'library_name', ''),
                    "tags": getattr(model, 'tags', []),
                    "pipeline_tag": getattr(model, 'pipeline_tag', ''),
                    "last_modified": getattr(model, 'lastModified', ''),
                    "source": "Hugging Face"
                }
                model_list.append(model_info)
                
            return model_list
            
        except Exception as e:
            self.log_message(f"Error searching Hugging Face models: {e}", "ERROR")
            return []
    
    def _search_openai_models(self, query, limit=10):
        """Search for models on OpenAI"""
        try:
            # OpenAI models API
            url = "https://api.openai.com/v1/models"
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            models = data.get("data", [])
            
            model_list = []
            for model in models[:limit]:
                model_info = {
                    "model_id": model.get("id", ""),
                    "owned_by": model.get("owned_by", ""),
                    "created": model.get("created", ""),
                    "source": "OpenAI"
                }
                model_list.append(model_info)
                
            return model_list
            
        except Exception as e:
            self.log_message(f"Error searching OpenAI models: {e}", "ERROR")
            return []
    
    def _search_gemini_models(self, query, limit=10):
        """Search for models on Google Gemini"""
        try:
            # For Gemini, we'll simulate model discovery
            # In a real implementation, you would use the Gemini API
            gemini_models = [
                {"model_id": "gemini-pro", "source": "Google Gemini", "description": "Gemini Pro model"},
                {"model_id": "gemini-pro-vision", "source": "Google Gemini", "description": "Gemini Pro Vision model"},
                {"model_id": "gemini-ultra", "source": "Google Gemini", "description": "Gemini Ultra model"}
            ]
            
            # Filter based on query
            filtered_models = [m for m in gemini_models if query.lower() in m["model_id"].lower()]
            
            return filtered_models[:limit]
            
        except Exception as e:
            self.log_message(f"Error searching Gemini models: {e}", "ERROR")
            return []
            
    def get_model_details(self, model_id):
        """
        Get detailed information about a specific model
        
        Args:
            model_id (str): The model ID
            
        Returns:
            dict: Detailed model information
        """
        self.log_message(f"Fetching details for model: {model_id}")
        
        # Try to get details from Hugging Face first
        if self.hf_api_key and self.hf_api:
            try:
                # Get model info from Hugging Face
                model_info = self.hf_api.model_info(model_id)
                
                # Extract relevant details
                details = {
                    "model_id": model_info.modelId,
                    "author": model_info.author,
                    "downloads": getattr(model_info, 'downloads', 0),
                    "likes": getattr(model_info, 'likes', 0),
                    "library_name": getattr(model_info, 'library_name', ''),
                    "tags": getattr(model_info, 'tags', []),
                    "pipeline_tag": getattr(model_info, 'pipeline_tag', ''),
                    "last_modified": getattr(model_info, 'lastModified', ''),
                    "sha": getattr(model_info, 'sha', ''),
                    "siblings": getattr(model_info, 'siblings', []),
                    "config": getattr(model_info, 'config', {}),
                    "card_data": getattr(model_info, 'cardData', {}),
                    "source": "Hugging Face"
                }
                
                self.log_message(f"Retrieved details for model: {model_id}")
                return details
                
            except Exception as e:
                self.log_message(f"Error fetching Hugging Face model details: {e}", "ERROR")
        
        # If not found in Hugging Face, return basic info
        details = {
            "model_id": model_id,
            "source": "Unknown",
            "description": f"Model details for {model_id}"
        }
        
        self.log_message(f"Retrieved basic details for model: {model_id}")
        return details
            
    def select_best_model(self, models, criteria=None):
        """
        Select the best model based on criteria
        
        Args:
            models (list): List of model dictionaries
            criteria (dict): Selection criteria (e.g., downloads, likes, tags)
            
        Returns:
            dict: The selected model information
        """
        self.log_message("Selecting the best model based on criteria")
        
        if not models:
            self.log_message("No models available for selection", "WARNING")
            return None
            
        # Default criteria: sort by downloads (popularity) or fallback to first model
        if not criteria:
            criteria = {"sort_by": "downloads", "order": "desc"}
            
        sort_key = criteria.get("sort_by", "downloads")
        order = criteria.get("order", "desc")
        
        # Sort models based on criteria
        try:
            # For models without the sort key, use 0 as default
            sorted_models = sorted(
                models, 
                key=lambda x: x.get(sort_key, 0), 
                reverse=(order == "desc")
            )
            
            best_model = sorted_models[0] if sorted_models else None
            
            if best_model:
                self.log_message(f"Selected best model: {best_model['model_id']} from {best_model.get('source', 'Unknown')}")
            else:
                self.log_message("No model selected", "WARNING")
                
            return best_model
            
        except Exception as e:
            self.log_message(f"Error selecting best model: {e}", "ERROR")
            # Return the first model if sorting fails
            return models[0] if models else None
            
    def customize_model(self, model_info, user_parameters):
        """
        Customize model information based on user parameters
        
        Args:
            model_info (dict): Model information
            user_parameters (dict): User customization parameters
            
        Returns:
            dict: Customized model configuration
        """
        self.log_message(f"Customizing model: {model_info.get('model_id', 'Unknown')}")
        
        # Create a customized configuration based on user parameters
        customized_config = {
            "model_id": model_info.get("model_id", ""),
            "author": model_info.get("author", model_info.get("owned_by", "")),
            "source": model_info.get("source", "Unknown"),
            "custom_name": user_parameters.get("model_name", model_info.get("model_id", "")),
            "description": user_parameters.get("description", ""),
            "additional_apis": user_parameters.get("additional_apis", []),
            "features": [],
            "excluded_features": [],
            "parameters": {}
        }
        
        # Add features based on model tags (if available)
        tags = model_info.get("tags", [])
        for tag in tags:
            if tag not in ["model", "pytorch", "transformers"]:
                customized_config["features"].append(tag)
                
        # Add pipeline tag as a feature (if available)
        pipeline_tag = model_info.get("pipeline_tag")
        if pipeline_tag and pipeline_tag not in customized_config["features"]:
            customized_config["features"].append(pipeline_tag)
            
        # Add library name as a feature (if available)
        library_name = model_info.get("library_name")
        if library_name and library_name not in customized_config["features"]:
            customized_config["features"].append(library_name)
            
        self.log_message("Model customization complete")
        return customized_config