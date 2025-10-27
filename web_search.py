"""
Web Search Module for Smart AI Model Builder
============================================

This module handles automated web information gathering for the Smart AI Model Builder.
It searches for relevant information about AI models and topics using various sources.
- Make By Mohamed Sarhan
"""

import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import quote
import time

class WebSearchEngine:
    """Web search engine for gathering information about AI models and topics"""
    
    def __init__(self, gui_logger=None):
        """Initialize the web search engine"""
        self.logger = logging.getLogger(__name__)
        self.gui_logger = gui_logger
        
        # Search engines and their URLs
        self.search_engines = {
            "google": "https://www.google.com/search?q=",
            "duckduckgo": "https://duckduckgo.com/html/?q="
        }
        
        # Headers to mimic a real browser
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
    def log_message(self, message, level="INFO"):
        """Log a message to both console and GUI"""
        self.logger.info(message)
        if self.gui_logger:
            self.gui_logger(message, level)
            
    def search_web(self, query, max_results=5):
        """
        Search the web for information about a query
        
        Args:
            query (str): The search query
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of dictionaries containing search results
        """
        self.log_message(f"Searching web for: {query}")
        
        results = []
        
        try:
            # Use DuckDuckGo for search (more privacy-friendly)
            search_url = self.search_engines["duckduckgo"] + quote(query)
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Use html.parser instead of lxml
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            result_divs = soup.find_all("div", class_="result")
            
            for i, result_div in enumerate(result_divs[:max_results]):
                try:
                    title_elem = result_div.find("a", class_="result__a")
                    snippet_elem = result_div.find("a", class_="result__snippet")
                    
                    if title_elem and snippet_elem:
                        title = title_elem.get_text()
                        url = title_elem.get("href")
                        snippet = snippet_elem.get_text()
                        
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "source": "DuckDuckGo"
                        })
                        
                        self.log_message(f"Found result {i+1}: {title}")
                except Exception as e:
                    self.log_message(f"Error parsing result: {e}", "WARNING")
                    continue
                    
        except Exception as e:
            self.log_message(f"Web search error: {e}", "ERROR")
            
        return results
        
    def gather_model_information(self, model_topic, description=""):
        """
        Gather comprehensive information about an AI model topic
        
        Args:
            model_topic (str): The AI model topic to research
            description (str): Additional description/context
            
        Returns:
            dict: Dictionary containing gathered information
        """
        self.log_message(f"Gathering information for model topic: {model_topic}")
        
        # Combine topic and description for search
        search_query = f"AI model {model_topic}"
        if description:
            search_query += f" {description}"
            
        # Search for general information
        general_results = self.search_web(search_query, max_results=3)
        
        # Search for technical specifications
        tech_query = f"{model_topic} AI model technical specifications"
        tech_results = self.search_web(tech_query, max_results=3)
        
        # Search for use cases
        use_case_query = f"{model_topic} AI model use cases applications"
        use_case_results = self.search_web(use_case_query, max_results=3)
        
        # Compile all information
        gathered_info = {
            "topic": model_topic,
            "description": description,
            "general_info": general_results,
            "technical_specs": tech_results,
            "use_cases": use_case_results,
            "timestamp": time.time()
        }
        
        self.log_message(f"Information gathering complete for: {model_topic}")
        return gathered_info
        
    def analyze_and_refine(self, raw_data, parameters=None):
        """
        Analyze and refine the gathered data based on parameters
        
        Args:
            raw_data (dict): Raw data from web search
            parameters (dict): Parameters to refine the data
            
        Returns:
            dict: Refined and analyzed information
        """
        self.log_message("Analyzing and refining gathered data")
        
        # For now, we'll just structure the data better
        # In a more advanced implementation, we could use NLP to extract key points
        
        refined_data = {
            "topic": raw_data.get("topic", ""),
            "description": raw_data.get("description", ""),
            "key_points": [],
            "technical_specifications": [],
            "recommended_use_cases": [],
            "related_models": []
        }
        
        # Extract key points from general info
        for result in raw_data.get("general_info", []):
            refined_data["key_points"].append({
                "title": result.get("title", ""),
                "summary": result.get("snippet", "")[:200] + "..." if len(result.get("snippet", "")) > 200 else result.get("snippet", ""),
                "source": result.get("source", "")
            })
            
        # Extract technical specifications
        for result in raw_data.get("technical_specs", []):
            refined_data["technical_specifications"].append({
                "title": result.get("title", ""),
                "details": result.get("snippet", "")[:200] + "..." if len(result.get("snippet", "")) > 200 else result.get("snippet", ""),
                "source": result.get("source", "")
            })
            
        # Extract use cases
        for result in raw_data.get("use_cases", []):
            refined_data["recommended_use_cases"].append({
                "title": result.get("title", ""),
                "application": result.get("snippet", "")[:200] + "..." if len(result.get("snippet", "")) > 200 else result.get("snippet", ""),
                "source": result.get("source", "")
            })
            
        self.log_message("Data analysis and refinement complete")
        return refined_data