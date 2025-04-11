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
from collections import defaultdict
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
            print(f"MongoDB connection not available: {str(e)}")
            print("Will use local files only for statistical analysis.")
            self.mongodb_available = False
        
        # Create the db_files/stats directory if it doesn't exist
        self.stats_dir = os.path.join("db_files", "stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # Initialize NLTK sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize stop words
        self.stop_words = set(stopwords.words('english'))
        self.portuguese_stop_words = set(stopwords.words('portuguese')) if 'portuguese' in stopwords._fileids else set()
        
        # Initialize metrics
        self.metrics = {
            "sentiment_scores": {},
            "response_length": {},
            "word_frequencies": {},
            "similarity_scores": {},
            "complexity_metrics": {}
        }
    
    def analyze_conversation_pairs(self, conversation_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, Any]:
        """
        Analyze a list of baseline-persona conversation pairs using statistical methods.
        
        Args:
            conversation_pairs: List of tuples containing (baseline_conversation, persona_conversation)
            
        Returns:
            Dictionary of statistical analysis results
        """
        results = {
            "sentiment_analysis": self.analyze_sentiment(conversation_pairs),
            "response_metrics": self.analyze_response_metrics(conversation_pairs),
            "word_frequency": self.analyze_word_frequency(conversation_pairs),
            "similarity_analysis": self.analyze_similarity(conversation_pairs),
            "aggregate_statistics": self.calculate_aggregate_statistics(conversation_pairs)
        }
        
        # Save results
        self.save_analysis_results(results)
        
        return results
    
    def analyze_sentiment(self, conversation_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, Any]:
        """
        Analyze sentiment differences between baseline and persona responses.
        
        Args:
            conversation_pairs: List of tuples containing (baseline_conversation, persona_conversation)
            
        Returns:
            Dictionary of sentiment analysis results
        """
        sentiment_results = {
            "baseline": [],
            "persona": [],
            "differences": []
        }
        
        for baseline_conv, persona_conv in conversation_pairs:
            # Extract the assistant's response
            baseline_response = self._get_assistant_response(baseline_conv)
            persona_response = self._get_assistant_response(persona_conv)
            
            # Calculate sentiment scores
            baseline_sentiment = self.sentiment_analyzer.polarity_scores(baseline_response)
            persona_sentiment = self.sentiment_analyzer.polarity_scores(persona_response)
            
            # Calculate the difference in compound sentiment
            sentiment_diff = persona_sentiment['compound'] - baseline_sentiment['compound']
            
            # Store results
            sentiment_results["baseline"].append(baseline_sentiment)
            sentiment_results["persona"].append(persona_sentiment)
            sentiment_results["differences"].append({
                "compound_diff": sentiment_diff,
                "baseline_id": baseline_conv.get("_id", ""),
                "persona_id": persona_conv.get("_id", ""),
                "baseline_compound": baseline_sentiment['compound'],
                "persona_compound": persona_sentiment['compound']
            })
        
        # Calculate aggregate statistics
        sentiment_results["statistics"] = {
            "mean_difference": np.mean([d["compound_diff"] for d in sentiment_results["differences"]]),
            "std_difference": np.std([d["compound_diff"] for d in sentiment_results["differences"]]),
            "max_difference": max([d["compound_diff"] for d in sentiment_results["differences"]], key=abs),
            "min_difference": min([d["compound_diff"] for d in sentiment_results["differences"]], key=abs)
        }
        
        return sentiment_results
    
    def analyze_response_metrics(self, conversation_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, Any]:
        """
        Analyze response length and complexity differences between baseline and persona responses.
        
        Args:
            conversation_pairs: List of tuples containing (baseline_conversation, persona_conversation)
            
        Returns:
            Dictionary of response metrics analysis results
        """
        metrics_results = {
            "length": {
                "baseline": [],
                "persona": [],
                "differences": []
            },
            "complexity": {
                "baseline": [],
                "persona": [],
                "differences": []
            }
        }
        
        for baseline_conv, persona_conv in conversation_pairs:
            # Extract the assistant's response
            baseline_response = self._get_assistant_response(baseline_conv)
            persona_response = self._get_assistant_response(persona_conv)
            
            # Calculate length metrics
            baseline_length = len(baseline_response)
            persona_length = len(persona_response)
            length_diff = persona_length - baseline_length
            length_ratio = persona_length / baseline_length if baseline_length > 0 else 0
            
            # Calculate complexity metrics (average word length as a simple metric)
            baseline_words = word_tokenize(baseline_response)
            persona_words = word_tokenize(persona_response)
            
            baseline_avg_word_length = np.mean([len(word) for word in baseline_words]) if baseline_words else 0
            persona_avg_word_length = np.mean([len(word) for word in persona_words]) if persona_words else 0
            complexity_diff = persona_avg_word_length - baseline_avg_word_length
            
            # Store length results
            metrics_results["length"]["baseline"].append(baseline_length)
            metrics_results["length"]["persona"].append(persona_length)
            metrics_results["length"]["differences"].append({
                "absolute_diff": length_diff,
                "ratio": length_ratio,
                "baseline_id": baseline_conv.get("_id", ""),
                "persona_id": persona_conv.get("_id", "")
            })
            
            # Store complexity results
            metrics_results["complexity"]["baseline"].append(baseline_avg_word_length)
            metrics_results["complexity"]["persona"].append(persona_avg_word_length)
            metrics_results["complexity"]["differences"].append({
                "diff": complexity_diff,
                "baseline_id": baseline_conv.get("_id", ""),
                "persona_id": persona_conv.get("_id", "")
            })
        
        # Calculate aggregate statistics for length
        metrics_results["length"]["statistics"] = {
            "mean_difference": np.mean([d["absolute_diff"] for d in metrics_results["length"]["differences"]]),
            "std_difference": np.std([d["absolute_diff"] for d in metrics_results["length"]["differences"]]),
            "mean_ratio": np.mean([d["ratio"] for d in metrics_results["length"]["differences"]]),
            "std_ratio": np.std([d["ratio"] for d in metrics_results["length"]["differences"]])
        }
        
        # Calculate aggregate statistics for complexity
        metrics_results["complexity"]["statistics"] = {
            "mean_difference": np.mean([d["diff"] for d in metrics_results["complexity"]["differences"]]),
            "std_difference": np.std([d["diff"] for d in metrics_results["complexity"]["differences"]])
        }
        
        return metrics_results
    
    def analyze_word_frequency(self, conversation_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, Any]:
        """
        Analyze word frequency differences between baseline and persona responses.
        
        Args:
            conversation_pairs: List of tuples containing (baseline_conversation, persona_conversation)
            
        Returns:
            Dictionary of word frequency analysis results
        """
        # Combine all baseline and persona responses
        all_baseline_responses = []
        all_persona_responses = []
        
        for baseline_conv, persona_conv in conversation_pairs:
            baseline_response = self._get_assistant_response(baseline_conv)
            persona_response = self._get_assistant_response(persona_conv)
            
            all_baseline_responses.append(baseline_response)
            all_persona_responses.append(persona_response)
        
        # Create a single text corpus for each group
        baseline_corpus = " ".join(all_baseline_responses)
        persona_corpus = " ".join(all_persona_responses)
        
        # Tokenize and remove stop words
        language = self._detect_language(baseline_corpus)
        stop_words = self.portuguese_stop_words if language == "pt" else self.stop_words
        
        baseline_words = [word.lower() for word in word_tokenize(baseline_corpus) 
                         if word.isalpha() and word.lower() not in stop_words]
        persona_words = [word.lower() for word in word_tokenize(persona_corpus) 
                        if word.isalpha() and word.lower() not in stop_words]
        
        # Count word frequencies
        baseline_freq = self._count_word_frequencies(baseline_words)
        persona_freq = self._count_word_frequencies(persona_words)
        
        # Find words with significant frequency differences
        significant_words = self._find_significant_frequency_differences(baseline_freq, persona_freq)
        
        return {
            "baseline_top_words": dict(sorted(baseline_freq.items(), key=lambda x: x[1], reverse=True)[:20]),
            "persona_top_words": dict(sorted(persona_freq.items(), key=lambda x: x[1], reverse=True)[:20]),
            "significant_differences": significant_words
        }
    
    def analyze_similarity(self, conversation_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, Any]:
        """
        Analyze text similarity between baseline and persona responses.
        
        Args:
            conversation_pairs: List of tuples containing (baseline_conversation, persona_conversation)
            
        Returns:
            Dictionary of similarity analysis results
        """
        similarity_results = []
        
        for baseline_conv, persona_conv in conversation_pairs:
            baseline_response = self._get_assistant_response(baseline_conv)
            persona_response = self._get_assistant_response(persona_conv)
            
            # Calculate cosine similarity
            vectorizer = CountVectorizer().fit_transform([baseline_response, persona_response])
            vectors = vectorizer.toarray()
            cosine_sim = cosine_similarity(vectors)[0][1]
            
            similarity_results.append({
                "baseline_id": baseline_conv.get("_id", ""),
                "persona_id": persona_conv.get("_id", ""),
                "cosine_similarity": cosine_sim
            })
        
        # Calculate aggregate statistics
        mean_similarity = np.mean([r["cosine_similarity"] for r in similarity_results])
        std_similarity = np.std([r["cosine_similarity"] for r in similarity_results])
        
        return {
            "pair_similarities": similarity_results,
            "statistics": {
                "mean_similarity": mean_similarity,
                "std_similarity": std_similarity
            }
        }
    
    def calculate_aggregate_statistics(self, conversation_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, Any]:
        """
        Calculate aggregate statistics across all metrics.
        
        Args:
            conversation_pairs: List of tuples containing (baseline_conversation, persona_conversation)
            
        Returns:
            Dictionary of aggregate statistics
        """
        # Extract persona demographics for grouping
        persona_demographics = []
        
        for _, persona_conv in conversation_pairs:
            prompt_id = persona_conv.get("prompt_id")
            
            # Try to get the prompt document to extract persona info
            persona_info = self._get_persona_info_from_prompt(prompt_id)
            
            if persona_info:
                persona_demographics.append(persona_info)
        
        # Group conversations by demographic factors
        grouped_by_gender = self._group_by_demographic(conversation_pairs, persona_demographics, "gender")
        grouped_by_age = self._group_by_demographic(conversation_pairs, persona_demographics, "age_group")
        grouped_by_education = self._group_by_demographic(conversation_pairs, persona_demographics, "education_level")
        
        # Calculate statistics for each group
        gender_stats = self._calculate_group_statistics(grouped_by_gender)
        age_stats = self._calculate_group_statistics(grouped_by_age)
        education_stats = self._calculate_group_statistics(grouped_by_education)
        
        return {
            "by_gender": gender_stats,
            "by_age": age_stats,
            "by_education": education_stats,
            "sample_size": len(conversation_pairs)
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
        timestamp = pd.Timestamp.now().isoformat()
        
        # Create the analysis document
        analysis_doc = {
            "timestamp": timestamp,
            "type": "statistical_analysis",
            "results": results
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
    
    def _get_assistant_response(self, conversation: Dict) -> str:
        """Extract the assistant's response from a conversation."""
        if "turns" not in conversation:
            return ""
        
        for turn in conversation["turns"]:
            if turn.get("role") == "assistant":
                return turn.get("content", "")
        
        return ""
    
    def _count_word_frequencies(self, words: List[str]) -> Dict[str, int]:
        """Count the frequency of each word in a list."""
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        return dict(word_freq)
    
    def _find_significant_frequency_differences(self, baseline_freq: Dict[str, int], 
                                               persona_freq: Dict[str, int]) -> List[Dict[str, Any]]:
        """Find words with significant frequency differences between baseline and persona responses."""
        significant_words = []
        
        # Get all unique words
        all_words = set(list(baseline_freq.keys()) + list(persona_freq.keys()))
        
        for word in all_words:
            baseline_count = baseline_freq.get(word, 0)
            persona_count = persona_freq.get(word, 0)
            
            # Calculate relative frequencies
            total_baseline = sum(baseline_freq.values())
            total_persona = sum(persona_freq.values())
            
            baseline_rel_freq = baseline_count / total_baseline if total_baseline > 0 else 0
            persona_rel_freq = persona_count / total_persona if total_persona > 0 else 0
            
            # Calculate the difference
            diff = persona_rel_freq - baseline_rel_freq
            
            # Only include words with a significant difference
            if abs(diff) > 0.01 and (baseline_count > 2 or persona_count > 2):
                significant_words.append({
                    "word": word,
                    "baseline_count": baseline_count,
                    "persona_count": persona_count,
                    "baseline_rel_freq": baseline_rel_freq,
                    "persona_rel_freq": persona_rel_freq,
                    "diff": diff
                })
        
        # Sort by absolute difference
        significant_words.sort(key=lambda x: abs(x["diff"]), reverse=True)
        
        return significant_words[:20]  # Return top 20 most significant differences
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is in Portuguese or English."""
        # Simple heuristic: check for Portuguese-specific characters
        portuguese_chars = set('áàâãéèêíìóòôõúùçÁÀÂÃÉÈÊÍÌÓÒÔÕÚÙÇ')
        
        # Count characters that are specific to Portuguese
        pt_char_count = sum(1 for char in text if char in portuguese_chars)
        
        # If we have several Portuguese-specific characters, assume it's Portuguese
        if pt_char_count > 5:
            return "pt"
        
        # Otherwise assume English
        return "en"
    
    def _get_persona_info_from_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Extract persona information from a prompt document."""
        if not prompt_id or not self.mongodb_available:
            return None
        
        try:
            # Get the prompt document
            prompt_doc = self.db.prompts_collection.find_one({"_id": prompt_id})
            
            if not prompt_doc or "persona" not in prompt_doc:
                return None
            
            # Extract basic demographic info
            persona = prompt_doc["persona"]
            
            # Create age group
            age = persona.get("age", 0)
            if age < 30:
                age_group = "18-29"
            elif age < 50:
                age_group = "30-49"
            elif age < 65:
                age_group = "50-64"
            else:
                age_group = "65+"
            
            # Create education level group
            education = persona.get("education", "").lower()
            if "ensino fundamental" in education or "elementary" in education:
                education_level = "elementary"
            elif "ensino médio" in education or "high school" in education or "secondary" in education:
                education_level = "high_school"
            elif "graduação" in education or "college" in education or "university" in education:
                education_level = "college"
            elif "pós-graduação" in education or "mestrado" in education or "doutorado" in education or "post" in education:
                education_level = "postgraduate"
            else:
                education_level = "unknown"
            
            return {
                "persona_id": persona.get("_id", ""),
                "gender": persona.get("gender", "").lower(),
                "age": age,
                "age_group": age_group,
                "education_level": education_level,
                "occupation": persona.get("occupation", ""),
                "location": persona.get("location", "")
            }
            
        except Exception as e:
            print(f"Error getting persona info from prompt: {str(e)}")
            return None
    
    def _group_by_demographic(self, conversation_pairs: List[Tuple[Dict, Dict]], 
                             demographics: List[Dict[str, Any]], factor: str) -> Dict[str, List[Tuple[Dict, Dict]]]:
        """Group conversation pairs by a demographic factor."""
        grouped = defaultdict(list)
        
        for i, (baseline_conv, persona_conv) in enumerate(conversation_pairs):
            if i < len(demographics) and demographics[i]:
                factor_value = demographics[i].get(factor, "unknown")
                grouped[factor_value].append((baseline_conv, persona_conv))
        
        return dict(grouped)
    
    def _calculate_group_statistics(self, grouped_conversations: Dict[str, List[Tuple[Dict, Dict]]]) -> Dict[str, Any]:
        """Calculate statistics for each demographic group."""
        group_stats = {}
        
        for group, conv_pairs in grouped_conversations.items():
            # Skip groups with too few conversations
            if len(conv_pairs) < 2:
                continue
            
            # Calculate sentiment statistics
            sentiment_analysis = self.analyze_sentiment(conv_pairs)
            mean_sentiment_diff = sentiment_analysis["statistics"]["mean_difference"]
            
            # Calculate length statistics
            response_metrics = self.analyze_response_metrics(conv_pairs)
            mean_length_ratio = response_metrics["length"]["statistics"]["mean_ratio"]
            
            # Calculate similarity statistics
            similarity_analysis = self.analyze_similarity(conv_pairs)
            mean_similarity = similarity_analysis["statistics"]["mean_similarity"]
            
            group_stats[group] = {
                "sample_size": len(conv_pairs),
                "mean_sentiment_difference": mean_sentiment_diff,
                "mean_length_ratio": mean_length_ratio,
                "mean_similarity": mean_similarity
            }
        
        return group_stats


    def _load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load a conversation by ID from MongoDB or local file."""
        # Try to load from MongoDB first
        if self.mongodb_available:
            try:
                conversation = self.db.conversations_collection.find_one({"_id": conversation_id})
                if conversation:
                    return conversation
            except Exception as e:
                print(f"Error loading conversation from MongoDB: {str(e)}")
        
        # Try to load from local file
        try:
            file_path = os.path.join("db_files", "convos", f"conversation_{conversation_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading conversation from local file: {str(e)}")
        
        return None
    
    def _find_prompt_pairs(self) -> List[Dict[str, Any]]:
        """Find pairs of baseline and persona-specific prompts."""
        prompt_pairs = []
        
        if self.mongodb_available:
            try:
                # Find all baseline prompts
                baseline_prompts = list(self.db.prompts_collection.find({"is_baseline": True}))
                
                for baseline in baseline_prompts:
                    # Convert ObjectId to string if needed
                    if '_id' in baseline and not isinstance(baseline['_id'], str):
                        baseline['_id'] = str(baseline['_id'])
                    
                    # Find persona prompts with the same product and language
                    persona_prompts = list(self.db.prompts_collection.find({
                        "is_baseline": False,
                        "product": baseline.get("product"),
                        "language": baseline.get("language")
                    }))
                    
                    # Convert ObjectId to string for all persona prompts
                    for persona in persona_prompts:
                        if '_id' in persona and not isinstance(persona['_id'], str):
                            persona['_id'] = str(persona['_id'])
                    
                    # Add the pairs to the list
                    for persona in persona_prompts:
                        prompt_pairs.append({
                            "baseline": baseline,
                            "persona": persona,
                            "product": baseline.get("product"),
                            "language": baseline.get("language")
                        })
                
                print(f"Found {len(prompt_pairs)} baseline-persona prompt pairs.")
                return prompt_pairs
            
            except Exception as mongo_e:
                print(f"Error finding prompt pairs from MongoDB: {str(mongo_e)}")
        
        # If MongoDB is not available or there was an error, try local files
        try:
            baseline_prompts = []
            persona_prompts = []
            
            prompts_dir = os.path.join("db_files", "prompts")
            if os.path.exists(prompts_dir):
                for filename in os.listdir(prompts_dir):
                    if filename.endswith(".json"):
                        file_path = os.path.join(prompts_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            prompt = json.load(f)
                            if prompt.get("is_baseline") == True:
                                baseline_prompts.append(prompt)
                            elif prompt.get("persona_id") is not None:
                                persona_prompts.append(prompt)
            
            # Match baseline and persona prompts
            for baseline in baseline_prompts:
                product = baseline.get("product")
                language = baseline.get("language")
                
                if not product or not language:
                    continue
                
                for persona in persona_prompts:
                    if (persona.get("product") == product and 
                        persona.get("language") == language):
                        
                        prompt_pairs.append({
                            "baseline": baseline,
                            "persona": persona,
                            "product": product,
                            "language": language
                        })
            
            print(f"Found {len(prompt_pairs)} baseline-persona prompt pairs from local files.")
            return prompt_pairs
            
        except Exception as e:
            print(f"Error finding prompt pairs from local files: {str(e)}")
            return []
    
    def _find_conversation_pairs(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Find all baseline-persona conversation pairs using the same approach as bias_analyzer.py."""
        conversation_pairs = []
        
        # Get prompt pairs first
        prompt_pairs = self._find_prompt_pairs()
        
        # For each prompt pair, find the corresponding conversations
        for pair in prompt_pairs:
            baseline_prompt_id = pair["baseline"]["_id"]
            persona_prompt_id = pair["persona"]["_id"]
            
            # Find conversations for these prompts
            if self.mongodb_available:
                try:
                    # Find baseline conversation
                    baseline_conv = self.db.conversations_collection.find_one({"prompt_id": baseline_prompt_id})
                    
                    # Find persona conversation
                    persona_conv = self.db.conversations_collection.find_one({"prompt_id": persona_prompt_id})
                    
                    # If both conversations exist, add them to the pairs
                    if baseline_conv and persona_conv:
                        # Convert ObjectId to string if needed
                        if '_id' in baseline_conv and not isinstance(baseline_conv['_id'], str):
                            baseline_conv['_id'] = str(baseline_conv['_id'])
                        if '_id' in persona_conv and not isinstance(persona_conv['_id'], str):
                            persona_conv['_id'] = str(persona_conv['_id'])
                        
                        conversation_pairs.append((baseline_conv, persona_conv))
                        print(f"Matched: Baseline {baseline_conv.get('_id', 'unknown')} with "
                              f"Persona {persona_conv.get('_id', 'unknown')} for "
                              f"{pair['product']} in {pair['language']}")
                
                except Exception as mongo_e:
                    print(f"Error finding conversation pairs from MongoDB: {str(mongo_e)}")
            
            # If MongoDB is not available or there was an error, try local files
            else:
                try:
                    baseline_conv = None
                    persona_conv = None
                    
                    convos_dir = os.path.join("db_files", "convos")
                    if os.path.exists(convos_dir):
                        for filename in os.listdir(convos_dir):
                            if filename.endswith(".json"):
                                file_path = os.path.join(convos_dir, filename)
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    conv = json.load(f)
                                    if conv.get("prompt_id") == baseline_prompt_id:
                                        baseline_conv = conv
                                    elif conv.get("prompt_id") == persona_prompt_id:
                                        persona_conv = conv
                    
                    if baseline_conv and persona_conv:
                        conversation_pairs.append((baseline_conv, persona_conv))
                        print(f"Matched: Baseline {baseline_conv.get('_id', 'unknown')} with "
                              f"Persona {persona_conv.get('_id', 'unknown')} for "
                              f"{pair['product']} in {pair['language']} from local files")
                
                except Exception as e:
                    print(f"Error finding conversation pairs from local files: {str(e)}")
        
        print(f"Found {len(conversation_pairs)} baseline-persona conversation pairs.")
        return conversation_pairs

def main():
    """Run the statistical bias analyzer on conversation pairs."""
    parser = argparse.ArgumentParser(description="Run statistical bias analysis on Aurora chatbot conversations")
    parser.add_argument("--baseline-id", type=str, help="ID of the baseline conversation")
    parser.add_argument("--persona-id", type=str, help="ID of the persona conversation")
    parser.add_argument("--analyze-all", action="store_true", help="Analyze all conversation pairs")
    
    args = parser.parse_args()
    
    analyzer = StatisticalBiasAnalyzer()
    
    if args.baseline_id and args.persona_id:
        # Analyze a specific conversation pair
        baseline_conv = analyzer._load_conversation(args.baseline_id)
        persona_conv = analyzer._load_conversation(args.persona_id)
        
        if baseline_conv and persona_conv:
            results = analyzer.analyze_conversation_pairs([(baseline_conv, persona_conv)])
            print(f"Analysis completed and saved with ID: {results}")
        else:
            print("Could not load the specified conversations.")
    
    elif args.analyze_all:
        # Find all conversation pairs
        pairs = analyzer._find_conversation_pairs()
        
        if pairs:
            results = analyzer.analyze_conversation_pairs(pairs)
            print(f"Analysis of {len(pairs)} conversation pairs completed.")
        else:
            print("No conversation pairs found to analyze.")
    
    else:
        print("Please specify either --baseline-id and --persona-id, or --analyze-all")


if __name__ == "__main__":
    import argparse
    main()
