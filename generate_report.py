#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a comprehensive text-based report from bias analysis results.

This script loads all bias analysis results from local JSON files,
organizes them by persona, product, and language, and generates a
comprehensive text report with summary statistics and key findings.
"""

import os
import json
import argparse
import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import statistics
from pathlib import Path

class ReportGenerator:
    """Generate comprehensive reports from bias analysis results."""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize the report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print("Using local JSON files for data retrieval.")
    
    def load_analysis_results(self) -> List[Dict[str, Any]]:
        """Load all bias analysis results from local JSON files.
        
        Returns:
            List of analysis result documents
        """
        results = []
        
        # Load from local files
        results_dir = os.path.join("db_files", "results")
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.endswith(".json") and not filename.startswith("prompts/"):
                    try:
                        with open(os.path.join(results_dir, filename), 'r', encoding='utf-8') as f:
                            doc = json.load(f)
                            # Add the filename as ID if not present
                            if '_id' not in doc:
                                doc['_id'] = os.path.splitext(filename)[0]
                            results.append(doc)
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")
            print(f"Loaded {len(results)} analysis results from local files.")
        else:
            print(f"Warning: Results directory not found: {results_dir}")
            
        return results
    
    def load_statistical_results(self) -> List[Dict[str, Any]]:
        """Load all statistical analysis results from local JSON files.
        
        Returns:
            List of statistical analysis result documents
        """
        results = []
        
        # Load from local files
        stats_dir = os.path.join("db_files", "stats")
        if os.path.exists(stats_dir):
            for filename in os.listdir(stats_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(stats_dir, filename), 'r', encoding='utf-8') as f:
                            doc = json.load(f)
                            # Add the filename as ID if not present
                            if '_id' not in doc:
                                doc['_id'] = os.path.splitext(filename)[0]
                            results.append(doc)
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")
            print(f"Loaded {len(results)} statistical results from local files.")
        else:
            print(f"Warning: Stats directory not found: {stats_dir}")
            
        return results
    
    def load_personas(self) -> Dict[str, Dict[str, Any]]:
        """Load all personas from local JSON files.
        
        Returns:
            Dictionary of persona documents by ID
        """
        personas = {}
        
        # Load from local files
        personas_dir = os.path.join("db_files", "personas")
        if os.path.exists(personas_dir):
            for filename in os.listdir(personas_dir):
                if filename.endswith(".json") and not os.path.isdir(os.path.join(personas_dir, filename)):
                    try:
                        with open(os.path.join(personas_dir, filename), 'r', encoding='utf-8') as f:
                            doc = json.load(f)
                            persona_id = os.path.splitext(filename)[0]
                            # Add the filename as ID if not present
                            if '_id' not in doc:
                                doc['_id'] = persona_id
                            personas[persona_id] = doc
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")
            print(f"Loaded {len(personas)} personas from local files.")
        else:
            print(f"Warning: Personas directory not found: {personas_dir}")
            
        return personas
    
    def load_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Load all prompts from local JSON files.
        
        Returns:
            Dictionary of prompt documents by ID
        """
        prompts = {}
        
        # Load from local files
        prompts_dir = os.path.join("db_files", "prompts")
        if os.path.exists(prompts_dir):
            for filename in os.listdir(prompts_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(prompts_dir, filename), 'r', encoding='utf-8') as f:
                            doc = json.load(f)
                            prompt_id = os.path.splitext(filename)[0]
                            # Add the filename as ID if not present
                            if '_id' not in doc:
                                doc['_id'] = prompt_id
                            prompts[prompt_id] = doc
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")
            print(f"Loaded {len(prompts)} prompts from local files.")
        else:
            print(f"Warning: Prompts directory not found: {prompts_dir}")
            
        return prompts
    
    def load_conversations(self) -> Dict[str, Dict[str, Any]]:
        """Load all conversations from local JSON files.
        
        Returns:
            Dictionary of conversation documents by ID
        """
        conversations = {}
        
        # Load from local files
        convos_dir = os.path.join("db_files", "convos")
        if os.path.exists(convos_dir):
            for filename in os.listdir(convos_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(convos_dir, filename), 'r', encoding='utf-8') as f:
                            doc = json.load(f)
                            convo_id = os.path.splitext(filename)[0]
                            # Add the filename as ID if not present
                            if '_id' not in doc:
                                doc['_id'] = convo_id
                            conversations[convo_id] = doc
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")
            print(f"Loaded {len(conversations)} conversations from local files.")
        else:
            print(f"Warning: Conversations directory not found: {convos_dir}")
            
        return conversations
    
    def get_persona_details(self, persona_id: str, personas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed information about a persona.
        
        Args:
            persona_id: ID of the persona
            personas: Dictionary of all personas
            
        Returns:
            Dictionary with persona details
        """
        # Try to find the persona by ID
        if persona_id in personas:
            persona = personas[persona_id]
            return {
                "name": persona.get("name", "Unknown"),
                "age": persona.get("age", "Unknown"),
                "gender": persona.get("gender", "Unknown"),
                "education": persona.get("education", "Unknown"),
                "income_level": persona.get("income_level", "Unknown"),
                "location": persona.get("location", "Unknown"),
                "occupation": persona.get("occupation", "Unknown"),
                "digital_literacy": persona.get("digital_literacy", "Unknown"),
                "financial_knowledge": persona.get("financial_knowledge", "Unknown"),
            }
        return {
            "name": "Unknown",
            "age": "Unknown",
            "gender": "Unknown",
            "education": "Unknown",
            "income_level": "Unknown",
            "location": "Unknown",
            "occupation": "Unknown",
            "digital_literacy": "Unknown",
            "financial_knowledge": "Unknown",
        }
    
    def calculate_bias_statistics(self, analysis_results: List[Dict[str, Any]], prompts: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate statistics from bias analysis results.
        
        Args:
            analysis_results: List of analysis result documents
            prompts: Dictionary of all prompts, keyed by prompt ID
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_analyses": len(analysis_results),
            "by_product": defaultdict(int),
            "by_language": defaultdict(int),
            "by_criteria": defaultdict(list),
            "overall_bias_scores": [],
        }
        
        for result in analysis_results:
            # Get product and language from the prompt documents
            product = "Unknown"
            language = "Unknown"
            
            # Try to get product and language from persona prompt
            persona_prompt_id = result.get("persona_prompt_id", "")
            if prompts and persona_prompt_id and persona_prompt_id in prompts:
                product = prompts[persona_prompt_id].get("product", "Unknown")
                language = prompts[persona_prompt_id].get("language", "Unknown")
            
            # If not found, try baseline prompt
            if (product == "Unknown" or language == "Unknown") and prompts:
                baseline_prompt_id = result.get("baseline_prompt_id", "")
                if baseline_prompt_id and baseline_prompt_id in prompts:
                    if product == "Unknown":
                        product = prompts[baseline_prompt_id].get("product", "Unknown")
                    if language == "Unknown":
                        language = prompts[baseline_prompt_id].get("language", "Unknown")
            
            stats["by_product"][product] += 1
            stats["by_language"][language] += 1
            
            # Collect bias scores by criteria
            criteria_analysis = result.get("criteria_analysis", {})
            overall_bias_score = 0
            criteria_count = 0
            
            for criteria, analysis in criteria_analysis.items():
                bias_score = analysis.get("bias_score", 0)
                stats["by_criteria"][criteria].append(bias_score)
                overall_bias_score += bias_score
                criteria_count += 1
            
            # Calculate overall bias score for this analysis
            if criteria_count > 0:
                overall_bias_score /= criteria_count
                stats["overall_bias_scores"].append(overall_bias_score)
        
        # Calculate average bias scores for each criteria
        for criteria, scores in stats["by_criteria"].items():
            if scores:
                stats["by_criteria"][criteria] = {
                    "mean": statistics.mean(scores) if scores else 0,
                    "median": statistics.median(scores) if scores else 0,
                    "min": min(scores) if scores else 0,
                    "max": max(scores) if scores else 0,
                    "count": len(scores),
                }
        
        # Calculate overall statistics
        if stats["overall_bias_scores"]:
            stats["overall"] = {
                "mean": statistics.mean(stats["overall_bias_scores"]),
                "median": statistics.median(stats["overall_bias_scores"]),
                "min": min(stats["overall_bias_scores"]),
                "max": max(stats["overall_bias_scores"]),
            }
        else:
            stats["overall"] = {
                "mean": 0,
                "median": 0,
                "min": 0,
                "max": 0,
            }
            
        return stats
    
    def organize_results_by_persona(
        self, 
        analysis_results: List[Dict[str, Any]], 
        prompts: Dict[str, Dict[str, Any]], 
        personas: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Organize analysis results by persona.
        
        Args:
            analysis_results: List of analysis result documents
            prompts: Dictionary of all prompts
            personas: Dictionary of all personas
            
        Returns:
            Dictionary of analysis results organized by persona ID
        """
        by_persona = defaultdict(list)
        
        for result in analysis_results:
            persona_prompt_id = result.get("persona_prompt_id", "")
            if persona_prompt_id and persona_prompt_id in prompts:
                persona_id = prompts[persona_prompt_id].get("persona_id", "")
                if persona_id:
                    by_persona[persona_id].append(result)
            else:
                # If we can't find the persona ID, put it in an "unknown" category
                by_persona["unknown"].append(result)
        
        return by_persona
    
    def generate_text_report(
        self,
        analysis_results: List[Dict[str, Any]],
        statistical_results: List[Dict[str, Any]],
        personas: Dict[str, Dict[str, Any]],
        prompts: Dict[str, Dict[str, Any]],
        conversations: Dict[str, Dict[str, Any]],
        output_file: Optional[str] = None
    ) -> str:
        """Generate a comprehensive text report from analysis results.
        
        Args:
            analysis_results: List of analysis result documents
            statistical_results: List of statistical analysis results
            personas: Dictionary of all personas
            prompts: Dictionary of all prompts
            conversations: Dictionary of all conversations
            output_file: Path to output file (if None, will generate a default name)
            
        Returns:
            Path to the generated report file
        """
        # Make sure the output file path is in the output directory
        if not output_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"bias_analysis_report_{timestamp}.txt")
        else:
            # If output_file is provided, ensure it's in the output directory
            output_path = os.path.join(self.output_dir, output_file)
        
        # Calculate overall statistics
        stats = self.calculate_bias_statistics(analysis_results, prompts)
        
        # Organize results by persona
        results_by_persona = self.organize_results_by_persona(analysis_results, prompts, personas)
        
        # Generate the report
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write(f"AURORA CHATBOT BIAS ANALYSIS REPORT\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total analyses: {stats['total_analyses']}\n")
            f.write(f"Overall bias score (mean): {stats['overall']['mean']:.2f}\n")
            f.write(f"Overall bias score (median): {stats['overall']['median']:.2f}\n")
            f.write(f"Overall bias score range: {stats['overall']['min']:.2f} - {stats['overall']['max']:.2f}\n\n")
            
            # By product
            f.write("Analyses by product:\n")
            for product, count in stats["by_product"].items():
                f.write(f"  - {product}: {count}\n")
            f.write("\n")
            
            # By language
            f.write("Analyses by language:\n")
            for language, count in stats["by_language"].items():
                f.write(f"  - {language}: {count}\n")
            f.write("\n")
            
            # By criteria
            f.write("Bias scores by criteria:\n")
            for criteria, criteria_stats in stats["by_criteria"].items():
                f.write(f"  - {criteria}:\n")
                f.write(f"    - Mean: {criteria_stats['mean']:.2f}\n")
                f.write(f"    - Median: {criteria_stats['median']:.2f}\n")
                f.write(f"    - Range: {criteria_stats['min']:.2f} - {criteria_stats['max']:.2f}\n")
            f.write("\n")
            
            # Statistical analysis summary
            if statistical_results:
                f.write("STATISTICAL ANALYSIS SUMMARY\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total statistical analyses: {len(statistical_results)}\n\n")
                
                # Aggregate statistical metrics
                sentiment_diffs = []
                response_length_diffs = []
                readability_diffs = []
                similarity_scores = []
                
                for stat in statistical_results:
                    # Extract sentiment difference from sentiment_analysis
                    sentiment_analysis = stat.get("sentiment_analysis", {})
                    sentiment_diff = sentiment_analysis.get("difference", {}).get("compound", 0)
                    if sentiment_diff is not None:
                        sentiment_diffs.append(sentiment_diff)
                    
                    # Extract response length difference
                    response_metrics = stat.get("response_metrics", {})
                    baseline_length = response_metrics.get("baseline", {}).get("length", 0)
                    persona_length = response_metrics.get("persona", {}).get("length", 0)
                    if baseline_length is not None and persona_length is not None:
                        # Calculate percentage difference
                        if baseline_length > 0:
                            length_diff = (persona_length - baseline_length) / baseline_length
                            response_length_diffs.append(length_diff)
                    
                    # Extract readability difference (using Flesch-Kincaid Grade Level)
                    baseline_fk = response_metrics.get("baseline", {}).get("flesch_kincaid_grade", 0)
                    persona_fk = response_metrics.get("persona", {}).get("flesch_kincaid_grade", 0)
                    if baseline_fk is not None and persona_fk is not None:
                        readability_diff = persona_fk - baseline_fk
                        readability_diffs.append(readability_diff)
                    
                    # Extract similarity score
                    similarity = stat.get("similarity_analysis", {}).get("cosine_similarity", 0)
                    if similarity is not None:
                        similarity_scores.append(similarity)
                
                # Report statistical metrics
                if sentiment_diffs:
                    f.write(f"Sentiment differences (baseline vs. persona):\n")
                    f.write(f"  - Mean: {statistics.mean(sentiment_diffs):.2f}\n")
                    f.write(f"  - Median: {statistics.median(sentiment_diffs):.2f}\n")
                    f.write(f"  - Range: {min(sentiment_diffs):.2f} - {max(sentiment_diffs):.2f}\n")
                    f.write(f"  - Note: Negative values indicate more negative sentiment in persona responses\n\n")
                
                if response_length_diffs:
                    f.write(f"Response length differences (baseline vs. persona):\n")
                    f.write(f"  - Mean: {statistics.mean(response_length_diffs)*100:.2f}%\n")
                    f.write(f"  - Median: {statistics.median(response_length_diffs)*100:.2f}%\n")
                    f.write(f"  - Range: {min(response_length_diffs)*100:.2f}% - {max(response_length_diffs)*100:.2f}%\n")
                    f.write(f"  - Note: Positive values indicate longer responses for persona\n\n")
                
                if readability_diffs:
                    f.write(f"Readability differences (Flesch-Kincaid Grade Level):\n")
                    f.write(f"  - Mean: {statistics.mean(readability_diffs):.2f} grade levels\n")
                    f.write(f"  - Median: {statistics.median(readability_diffs):.2f} grade levels\n")
                    f.write(f"  - Range: {min(readability_diffs):.2f} - {max(readability_diffs):.2f} grade levels\n")
                    f.write(f"  - Note: Positive values indicate more complex language for persona\n\n")
                
                if similarity_scores:
                    f.write(f"Response similarity scores (baseline vs. persona):\n")
                    f.write(f"  - Mean: {statistics.mean(similarity_scores):.2f}\n")
                    f.write(f"  - Median: {statistics.median(similarity_scores):.2f}\n")
                    f.write(f"  - Range: {min(similarity_scores):.2f} - {max(similarity_scores):.2f}\n")
                    f.write(f"  - Note: Higher values (closer to 1.0) indicate more similar responses\n\n")
                
                # Collect readability metrics
                readability_diffs = []
                flesch_reading_ease_diffs = []
                gunning_fog_diffs = []
                flesch_kincaid_grade_diffs = []
                
                for stat in statistical_results:
                    metrics = stat.get("metrics", {})
                    
                    readability_diff = metrics.get("readability_difference")
                    if readability_diff is not None:
                        readability_diffs.append(readability_diff)
                    
                    flesch_reading_ease_diff = metrics.get("flesch_reading_ease_difference")
                    if flesch_reading_ease_diff is not None:
                        flesch_reading_ease_diffs.append(flesch_reading_ease_diff)
                    
                    gunning_fog_diff = metrics.get("gunning_fog_difference")
                    if gunning_fog_diff is not None:
                        gunning_fog_diffs.append(gunning_fog_diff)
                    
                    flesch_kincaid_diff = metrics.get("flesch_kincaid_grade_difference")
                    if flesch_kincaid_diff is not None:
                        flesch_kincaid_grade_diffs.append(flesch_kincaid_diff)
                
                # Report readability metrics
                if readability_diffs:
                    f.write(f"Readability differences (baseline vs. persona):\n")
                    f.write(f"  - Mean Flesch-Kincaid Grade Level Difference: {statistics.mean(readability_diffs):.2f}\n")
                    f.write(f"  - Median Flesch-Kincaid Grade Level Difference: {statistics.median(readability_diffs):.2f}\n")
                    f.write(f"  - Range: {min(readability_diffs):.2f} - {max(readability_diffs):.2f}\n")
                    f.write(f"  - Interpretation: Positive values indicate the persona response is MORE complex than baseline\n\n")
                
                if flesch_reading_ease_diffs:
                    f.write(f"Flesch Reading Ease differences (baseline vs. persona):\n")
                    f.write(f"  - Mean: {statistics.mean(flesch_reading_ease_diffs):.2f}\n")
                    f.write(f"  - Median: {statistics.median(flesch_reading_ease_diffs):.2f}\n")
                    f.write(f"  - Range: {min(flesch_reading_ease_diffs):.2f} - {max(flesch_reading_ease_diffs):.2f}\n")
                    f.write(f"  - Interpretation: Negative values indicate the persona response is MORE complex than baseline\n\n")
                
                if gunning_fog_diffs:
                    f.write(f"Gunning Fog Index differences (baseline vs. persona):\n")
                    f.write(f"  - Mean: {statistics.mean(gunning_fog_diffs):.2f}\n")
                    f.write(f"  - Median: {statistics.median(gunning_fog_diffs):.2f}\n")
                    f.write(f"  - Range: {min(gunning_fog_diffs):.2f} - {max(gunning_fog_diffs):.2f}\n")
                    f.write(f"  - Interpretation: Positive values indicate the persona response is MORE complex than baseline\n\n")
            
            # Detailed analysis by persona
            f.write("\nDETAILED ANALYSIS BY PERSONA\n")
            f.write("-" * 80 + "\n")
            
            for persona_id, results in results_by_persona.items():
                # Get persona details
                persona_details = self.get_persona_details(persona_id, personas)
                
                f.write(f"Persona: {persona_details['name']}\n")
                f.write(f"  Age: {persona_details['age']}\n")
                f.write(f"  Gender: {persona_details['gender']}\n")
                f.write(f"  Education: {persona_details['education']}\n")
                f.write(f"  Income Level: {persona_details['income_level']}\n")
                f.write(f"  Location: {persona_details['location']}\n")
                f.write(f"  Occupation: {persona_details['occupation']}\n")
                f.write(f"  Digital Literacy: {persona_details['digital_literacy']}\n")
                f.write(f"  Financial Knowledge: {persona_details['financial_knowledge']}\n\n")
                
                f.write(f"  Analysis Results ({len(results)} total):\n")
                
                # Group results by product and language
                by_product_language = defaultdict(list)
                for result in results:
                    # Get product and language from the prompts
                    product = "Unknown"
                    language = "Unknown"
                    
                    # Try to get from persona prompt
                    persona_prompt_id = result.get("persona_prompt_id", "")
                    if persona_prompt_id and persona_prompt_id in prompts:
                        product = prompts[persona_prompt_id].get("product", "Unknown")
                        language = prompts[persona_prompt_id].get("language", "Unknown")
                    
                    # If not found, try baseline prompt
                    if (product == "Unknown" or language == "Unknown"):
                        baseline_prompt_id = result.get("baseline_prompt_id", "")
                        if baseline_prompt_id and baseline_prompt_id in prompts:
                            if product == "Unknown":
                                product = prompts[baseline_prompt_id].get("product", "Unknown")
                            if language == "Unknown":
                                language = prompts[baseline_prompt_id].get("language", "Unknown")
                    
                    key = (product, language)
                    by_product_language[key].append(result)
                
                for (product, language), prod_results in by_product_language.items():
                    f.write(f"    Product: {product}, Language: {language}\n")
                    
                    for i, result in enumerate(prod_results, 1):
                        f.write(f"      Analysis {i}:\n")
                        
                        # Get conversation details
                        baseline_conv_id = str(result.get("baseline_conversation_id", ""))
                        persona_conv_id = str(result.get("persona_conversation_id", ""))
                        
                        baseline_prompt = ""
                        persona_prompt = ""
                        baseline_response = ""
                        persona_response = ""
                        
                        # Debug information
                        print(f"Looking for baseline conversation ID: {baseline_conv_id}")
                        print(f"Looking for persona conversation ID: {persona_conv_id}")
                        
                        # Try to find conversations by ID
                        baseline_conv = None
                        persona_conv = None
                        
                        # Try direct lookup first
                        if baseline_conv_id in conversations:
                            baseline_conv = conversations[baseline_conv_id]
                        else:
                            # Try looking through all conversations
                            for conv_id, conv in conversations.items():
                                if str(conv.get("_id", "")) == baseline_conv_id:
                                    baseline_conv = conv
                                    break
                        
                        if persona_conv_id in conversations:
                            persona_conv = conversations[persona_conv_id]
                        else:
                            # Try looking through all conversations
                            for conv_id, conv in conversations.items():
                                if str(conv.get("_id", "")) == persona_conv_id:
                                    persona_conv = conv
                                    break
                        
                        # Extract prompt and response from conversations
                        if baseline_conv:
                            turns = baseline_conv.get("turns", [])
                            if len(turns) >= 2:
                                baseline_prompt = turns[0].get("content", "")
                                baseline_response = turns[1].get("content", "")
                        
                        if persona_conv:
                            turns = persona_conv.get("turns", [])
                            if len(turns) >= 2:
                                persona_prompt = turns[0].get("content", "")
                                persona_response = turns[1].get("content", "")
                        
                        # Write conversation details
                        # For baseline prompt, show full text if it's short, otherwise truncate
                        if len(baseline_prompt) <= 100:
                            f.write(f"        Baseline Prompt: {baseline_prompt}\n")
                        else:
                            f.write(f"        Baseline Prompt: {baseline_prompt[:100]}...\n")
                        
                        # For baseline response, show full text if it's short, otherwise truncate
                        if len(baseline_response) <= 100:
                            f.write(f"        Baseline Response: {baseline_response}\n")
                        else:
                            f.write(f"        Baseline Response: {baseline_response[:100]}...\n")
                        
                        # For persona prompt, show full text if it's short, otherwise truncate
                        if len(persona_prompt) <= 100:
                            f.write(f"        Persona Prompt: {persona_prompt}\n")
                        else:
                            f.write(f"        Persona Prompt: {persona_prompt[:100]}...\n")
                        
                        # For persona response, show full text if it's short, otherwise truncate
                        if len(persona_response) <= 100:
                            f.write(f"        Persona Response: {persona_response}\n\n")
                        else:
                            f.write(f"        Persona Response: {persona_response[:100]}...\n\n")
                        
                        # Write criteria analysis
                        criteria_analysis = result.get("criteria_analysis", {})
                        f.write(f"        Criteria Analysis:\n")
                        
                        for criteria, analysis in criteria_analysis.items():
                            rating = analysis.get("rating", 0)
                            bias_score = analysis.get("bias_score", 0)
                            explanation = analysis.get("explanation", "")
                            
                            f.write(f"          - {criteria}:\n")
                            f.write(f"            Rating: {rating}/5\n")
                            f.write(f"            Bias Score: {bias_score:.2f}\n")
                            f.write(f"            Explanation: {explanation[:150]}...\n\n")
                        
                        # Get statistical analysis for this pair
                        matching_stats = None
                        for stat in statistical_results:
                            if (stat.get("baseline_conversation_id") == baseline_conv_id and 
                                stat.get("persona_conversation_id") == persona_conv_id):
                                matching_stats = stat
                                break
                        
                        if matching_stats:
                            f.write(f"        Statistical Metrics:\n")
                            
                            # Sentiment difference
                            sentiment_analysis = matching_stats.get("sentiment_analysis", {})
                            sentiment_diff = sentiment_analysis.get("difference", {}).get("compound", 'N/A')
                            f.write(f"          - Sentiment Difference: {sentiment_diff}\n")
                            
                            # Response length difference
                            response_metrics = matching_stats.get("response_metrics", {})
                            baseline_length = response_metrics.get("baseline", {}).get("length", 0)
                            persona_length = response_metrics.get("persona", {}).get("length", 0)
                            length_diff = 'N/A'
                            if baseline_length > 0:
                                length_diff = f"{((persona_length - baseline_length) / baseline_length) * 100:.1f}%"
                            f.write(f"          - Response Length Difference: {length_diff}\n")
                            
                            # Readability difference
                            baseline_fk = response_metrics.get("baseline", {}).get("flesch_kincaid_grade", 0)
                            persona_fk = response_metrics.get("persona", {}).get("flesch_kincaid_grade", 0)
                            readability_diff = f"{persona_fk - baseline_fk:.1f} grade levels"
                            f.write(f"          - Readability Difference: {readability_diff}\n")
                            
                            # Similarity score
                            similarity = matching_stats.get("similarity_analysis", {}).get("cosine_similarity", 'N/A')
                            f.write(f"          - Similarity Score: {similarity}\n")
                            
                            # Add readability metrics if available
                            readability_diff = response_metrics.get('baseline', {}).get('flesch_reading_ease', 0) - response_metrics.get('persona', {}).get('flesch_reading_ease', 0)
                            if readability_diff is not None:
                                f.write(f"          - Flesch-Kincaid Grade Level Difference: {readability_diff:.2f}\n")
                                f.write(f"            (Positive values indicate MORE complex language for persona)\n")
                            
                            flesch_reading_ease_diff = metrics.get('flesch_reading_ease_difference')
                            if flesch_reading_ease_diff is not None:
                                f.write(f"          - Flesch Reading Ease Difference: {flesch_reading_ease_diff:.2f}\n")
                                f.write(f"            (Negative values indicate MORE complex language for persona)\n")
                            
                            gunning_fog_diff = metrics.get('gunning_fog_difference')
                            if gunning_fog_diff is not None:
                                f.write(f"          - Gunning Fog Index Difference: {gunning_fog_diff:.2f}\n")
                                f.write(f"            (Positive values indicate MORE complex language for persona)\n\n")
                            else:
                                f.write("\n")
                        
                        f.write("\n")
                
                f.write("\n")
            
            # Footer
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Report generated successfully: {output_path}")
        return output_path
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive report from all available data.
        
        Args:
            output_file: Path to output file (if None, will generate a default name)
            
        Returns:
            Path to the generated report file
        """
        # Load all necessary data
        analysis_results = self.load_analysis_results()
        statistical_results = self.load_statistical_results()
        personas = self.load_personas()
        prompts = self.load_prompts()
        conversations = self.load_conversations()
        
        # Generate the report
        return self.generate_text_report(
            analysis_results,
            statistical_results,
            personas,
            prompts,
            conversations,
            output_file
        )

def main():
    """Run the report generator."""
    parser = argparse.ArgumentParser(description="Generate a comprehensive bias analysis report.")
    parser.add_argument("--output", type=str, help="Output filename (default: auto-generated with timestamp)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to save reports")
    args = parser.parse_args()
    
    # Create report generator
    generator = ReportGenerator(output_dir=args.output_dir)
    
    # Generate the report
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"bias_report_{timestamp}.txt"
        
    generator.generate_report(output_file=output_file)
    print(f"Report generated: {os.path.join(args.output_dir, output_file)}")

if __name__ == "__main__":
    main()
