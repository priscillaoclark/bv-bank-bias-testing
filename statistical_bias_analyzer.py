"""
Statistical Bias Analyzer for Aurora Chatbot Testing

This module implements statistical approaches to bias detection in LLM outputs,
complementing the qualitative analysis in bias_analyzer.py with quantitative metrics.
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from datetime import datetime
from dotenv import load_dotenv

# Add the current directory to the path so we can import from storage
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.database import Database

# Load environment variables
load_dotenv()

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    
# Check if textstat is installed, if not, provide installation instructions
try:
    import textstat
except ImportError:
    print("The textstat package is required for readability metrics.")
    print("Please install it using: pip install textstat")
    sys.exit(1)

class StatisticalBiasAnalyzer:
    """
    Implements statistical approaches to bias detection in LLM outputs.
    
    This class provides quantitative metrics to complement the qualitative
    analysis in the BiasAnalyzer class.
    """
    
    def __init__(self):
        """Initialize the StatisticalBiasAnalyzer."""
        # Initialize database connection
        try:
            self.db = Database()
            self.mongodb_available = True
            print("MongoDB connection available for statistical analysis.")
        except Exception as e:
            print(f"MongoDB connection failed: {str(e)}")
            self.mongodb_available = False
            print("Will save statistical analysis to local files only.")
        
        # Set up directories for local storage
        self.stats_dir = os.path.join("db_files", "stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def analyze_conversation_pairs(self, conversation_pairs: List[Tuple[Dict, Dict]]) -> List[str]:
        """
        Analyze each baseline-persona conversation pair individually using statistical methods.
        
        Args:
            conversation_pairs: List of tuples containing (baseline_conversation, persona_conversation)
            
        Returns:
            List of analysis IDs for each pair
        """
        analysis_ids = []
        
        # Process each conversation pair individually
        for i, (baseline_conversation, persona_conversation) in enumerate(conversation_pairs):
            print(f"Processing statistical analysis for conversation pair {i+1}/{len(conversation_pairs)}")
            
            # Create the analysis result
            pair_results = {
                "timestamp": datetime.now().isoformat(),
                "type": "statistical_analysis",
                "baseline_conversation_id": baseline_conversation["_id"],
                "persona_conversation_id": persona_conversation["_id"],
                "baseline_prompt_id": baseline_conversation.get("prompt_id"),
                "persona_prompt_id": persona_conversation.get("prompt_id"),
                "sentiment_analysis": self._analyze_sentiment_for_pair(baseline_conversation, persona_conversation),
                "response_metrics": self._analyze_metrics_for_pair(baseline_conversation, persona_conversation),
                "word_frequency": self._analyze_word_frequency_for_pair(baseline_conversation, persona_conversation),
                "similarity_analysis": self._analyze_similarity_for_pair(baseline_conversation, persona_conversation)
            }
            
            # Save individual pair results
            analysis_id = self.save_analysis_results(pair_results)
            if analysis_id:
                analysis_ids.append(analysis_id)
        
        return analysis_ids
    
    def _extract_product(self, baseline_conversation: Dict, persona_conversation: Dict) -> str:
        """Extract product information from conversations"""
        # Try to get from prompt data in conversations
        for convo in [baseline_conversation, persona_conversation]:
            if convo.get("prompt_data", {}).get("product"):
                return convo["prompt_data"]["product"]
        
        # If not found, return unknown
        return "unknown"
    
    def _extract_language(self, baseline_conversation: Dict, persona_conversation: Dict) -> str:
        """Extract language information from conversations"""
        # Try to get from prompt data in conversations
        for convo in [baseline_conversation, persona_conversation]:
            if convo.get("prompt_data", {}).get("language"):
                return convo["prompt_data"]["language"]
        
        # If not found, return unknown
        return "unknown"
    
    def _extract_persona_description(self, persona_conversation: Dict) -> str:
        """Extract persona description from conversation"""
        # Try to get from prompt data
        if persona_conversation.get("prompt_data", {}).get("persona_description"):
            return persona_conversation["prompt_data"]["persona_description"]
        
        # If not found, return unknown
        return "Unknown persona"
    
    def _get_response_text(self, conversation: Dict) -> str:
        """Extract the assistant's response from a conversation."""
        # Check if we have turns in the conversation
        if "turns" not in conversation:
            return ""
        
        # Get the last turn where role is 'assistant'
        for turn in reversed(conversation["turns"]):
            if turn.get("role") == "assistant":
                return turn.get("content", "")
        
        return ""
    
    def _analyze_sentiment_for_pair(self, baseline_conversation: Dict, persona_conversation: Dict) -> Dict[str, Any]:
        """Analyze sentiment for a single conversation pair"""
        # Get the response texts
        baseline_text = self._get_response_text(baseline_conversation)
        persona_text = self._get_response_text(persona_conversation)
        
        # Calculate sentiment scores
        baseline_score = self.sentiment_analyzer.polarity_scores(baseline_text)
        persona_score = self.sentiment_analyzer.polarity_scores(persona_text)
        
        # Calculate differences
        sentiment_diff = {
            "neg": persona_score["neg"] - baseline_score["neg"],
            "neu": persona_score["neu"] - baseline_score["neu"],
            "pos": persona_score["pos"] - baseline_score["pos"],
            "compound": persona_score["compound"] - baseline_score["compound"]
        }
        
        return {
            "baseline": baseline_score,
            "persona": persona_score,
            "difference": sentiment_diff
        }
    
    def _analyze_readability_for_text(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics for a text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with readability metrics
        """
        # Handle empty or very short texts
        if not text or len(text) < 50:
            return {
                "flesch_reading_ease": 0,
                "flesch_kincaid_grade": 0,
                "gunning_fog": 0,
                "smog_index": 0,
                "automated_readability_index": 0,
                "coleman_liau_index": 0,
                "dale_chall_readability_score": 0
            }
            
        try:
            # Calculate readability metrics
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "gunning_fog": textstat.gunning_fog(text),
                "smog_index": textstat.smog_index(text),
                "automated_readability_index": textstat.automated_readability_index(text),
                "coleman_liau_index": textstat.coleman_liau_index(text),
                "dale_chall_readability_score": textstat.dale_chall_readability_score(text)
            }
        except Exception as e:
            print(f"Error calculating readability metrics: {str(e)}")
            return {
                "flesch_reading_ease": 0,
                "flesch_kincaid_grade": 0,
                "gunning_fog": 0,
                "smog_index": 0,
                "automated_readability_index": 0,
                "coleman_liau_index": 0,
                "dale_chall_readability_score": 0
            }
    
    def _analyze_metrics_for_pair(self, baseline_conversation: Dict, persona_conversation: Dict) -> Dict[str, Any]:
        """Analyze response metrics for a single conversation pair"""
        # Get the response texts
        baseline_text = self._get_response_text(baseline_conversation)
        persona_text = self._get_response_text(persona_conversation)
        
        # Calculate basic metrics
        baseline_metrics = {
            "length": len(baseline_text),
            "word_count": len(baseline_text.split()),
            "sentence_count": len(nltk.sent_tokenize(baseline_text))
        }
        
        persona_metrics = {
            "length": len(persona_text),
            "word_count": len(persona_text.split()),
            "sentence_count": len(nltk.sent_tokenize(persona_text))
        }
        
        # Calculate readability metrics
        baseline_readability = self._analyze_readability_for_text(baseline_text)
        persona_readability = self._analyze_readability_for_text(persona_text)
        
        # Add readability metrics to the basic metrics
        baseline_metrics.update(baseline_readability)
        persona_metrics.update(persona_readability)
        
        # Calculate differences for all metrics
        metrics_diff = {}
        for key in baseline_metrics.keys():
            metrics_diff[key] = persona_metrics[key] - baseline_metrics[key]
        
        return {
            "baseline": baseline_metrics,
            "persona": persona_metrics,
            "difference": metrics_diff
        }
    
    def _analyze_word_frequency_for_pair(self, baseline_conversation: Dict, persona_conversation: Dict) -> Dict[str, Any]:
        """Analyze word frequency for a single conversation pair"""
        # Get the response texts
        baseline_text = self._get_response_text(baseline_conversation)
        persona_text = self._get_response_text(persona_conversation)
        
        # Tokenize and count words
        baseline_words = nltk.word_tokenize(baseline_text.lower())
        persona_words = nltk.word_tokenize(persona_text.lower())
        
        # Remove stopwords
        try:
            stop_words = set(nltk.corpus.stopwords.words('english'))
            baseline_words = [w for w in baseline_words if w.isalpha() and w not in stop_words]
            persona_words = [w for w in persona_words if w.isalpha() and w not in stop_words]
        except:
            # If stopwords aren't available, just filter non-alphabetic tokens
            baseline_words = [w for w in baseline_words if w.isalpha()]
            persona_words = [w for w in persona_words if w.isalpha()]
        
        # Count frequencies
        baseline_freq = Counter(baseline_words)
        persona_freq = Counter(persona_words)
        
        # Get top words
        baseline_top = dict(baseline_freq.most_common(10))
        persona_top = dict(persona_freq.most_common(10))
        
        return {
            "baseline_top_words": baseline_top,
            "persona_top_words": persona_top
        }
    
    def _analyze_similarity_for_pair(self, baseline_conversation: Dict, persona_conversation: Dict) -> Dict[str, Any]:
        """Analyze similarity for a single conversation pair"""
        # Get the response texts
        baseline_text = self._get_response_text(baseline_conversation)
        persona_text = self._get_response_text(persona_conversation)
        
        # Calculate cosine similarity
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([baseline_text, persona_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            similarity = 0.0
        
        return {
            "cosine_similarity": similarity
        }
    
    def save_analysis_results(self, results: Dict[str, Any]) -> str:
        """
        Save statistical analysis results to MongoDB and local file.
        
        Args:
            results: Dictionary of statistical analysis results
            
        Returns:
            ID of the saved analysis
        """
        # Generate a timestamp
        timestamp = datetime.now().isoformat()
        
        # Add timestamp to results
        results["timestamp"] = timestamp
        
        # Create the analysis document
        analysis_doc = {
            "timestamp": timestamp,
            "type": "statistical_analysis",
            # For individual pair analysis, the results are the document
            # rather than being nested under a 'results' key
            **results
        }
        
        # Save to MongoDB if available
        analysis_id = None
        if self.mongodb_available:
            try:
                # Create a dedicated stats collection if it doesn't exist
                if not hasattr(self.db, 'stats_collection'):
                    self.db.stats_collection = self.db.db["stats"]
                    # Create an index on the timestamp field
                    self.db.stats_collection.create_index("timestamp")
                    print("Created stats collection in MongoDB")
                
                result = self.db.stats_collection.insert_one(analysis_doc)
                analysis_id = str(result.inserted_id)
                print(f"Statistical analysis stored in MongoDB stats collection with ID: {analysis_id}")
            except Exception as e:
                print(f"Error storing statistical analysis in MongoDB: {str(e)}")
        
        # Always save to local file
        try:
            # If we have an ID from MongoDB, use it for the local file
            if analysis_id:
                analysis_doc["_id"] = analysis_id
            
            # Generate a file ID if we don't have one from MongoDB
            if "_id" not in analysis_doc:
                analysis_id = f"stat_analysis_{timestamp.replace(':', '-')}"
                analysis_doc["_id"] = analysis_id
            
            # Save to local file
            file_path = os.path.join(self.stats_dir, f"{analysis_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_doc, f, ensure_ascii=False, indent=2)
            print(f"Statistical analysis saved to local file: {file_path}")
            
            return analysis_id
            
        except Exception as e:
            print(f"Error saving statistical analysis to local file: {str(e)}")
            return analysis_id if analysis_id else None

def main():
    """Run the statistical bias analyzer on conversation pairs."""
    analyzer = StatisticalBiasAnalyzer()
    
    # Example usage (for testing)
    if len(sys.argv) > 1:
        # TODO: Implement command-line interface
        pass
    else:
        print("No arguments provided. Use this module as part of the bias testing system.")

if __name__ == "__main__":
    main()
