#!/usr/bin/env python
"""
Run Bias Test for Aurora Chatbot

This script orchestrates the entire bias testing process:
1. Generate prompts (both baseline and persona-specific)
2. Send prompts to the Aurora chatbot
3. Run statistical analysis on conversation pairs
4. Analyze responses for bias using both statistical and qualitative approaches

Usage:
    python run_bias_test.py --products 2 --personas 3
"""

import os
import sys
import json
import argparse
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from persona_generator import PersonaGenerator
from prompt_generator import PromptGenerator
from chatbot_tester import LightweightChatbotTester
from bias_analyzer import BiasAnalyzer
from statistical_bias_analyzer import StatisticalBiasAnalyzer
from utils.logging_config import get_run_logger

# Load environment variables
load_dotenv()

class BiasTester:
    """Orchestrate the entire bias testing process."""
    
    def __init__(self, logger=None):
        """Initialize the BiasTester.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.persona_generator = PersonaGenerator()
        self.prompt_generator = PromptGenerator()
        self.statistical_analyzer = StatisticalBiasAnalyzer()
        self.bias_analyzer = BiasAnalyzer()
    
    def generate_personas(self, num_personas: int, diversity_strategy: str = "mixed", temperature: float = None, enforce_diversity: bool = True) -> List[str]:
        """Generate personas for testing.
        
        Args:
            num_personas: Number of personas to generate
            diversity_strategy: Strategy for temperature diversity
            temperature: Specific temperature to use (overrides diversity_strategy)
            enforce_diversity: Whether to enforce diversity between generated personas
        """
        self.logger.info(f"Generating {num_personas} personas with '{diversity_strategy}' diversity strategy")
        if enforce_diversity:
            self.logger.info("Diversity validation is ENABLED - personas will be checked for uniqueness")
        else:
            self.logger.info("Diversity validation is DISABLED - personas may have similar characteristics")
        
        if temperature is not None:
            self.logger.info(f"Using fixed temperature: {temperature}")
            # Generate personas with a specific temperature
            if enforce_diversity:
                # Use the new ensure_diverse_persona method for each persona
                persona_ids = []
                existing_personas = []
                
                # Try to get existing personas from the database for comparison
                try:
                    if hasattr(self.persona_generator, 'db') and self.persona_generator.db:
                        existing_personas = self.persona_generator.db.get_all_personas()
                        print(f"Retrieved {len(existing_personas)} existing personas for diversity validation")
                except Exception as e:
                    print(f"Warning: Could not retrieve existing personas: {str(e)}")
                
                for i in range(num_personas):
                    print(f"\nGenerating persona {i+1}/{num_personas}...")
                    # Combine existing and newly generated personas for comparison
                    all_personas_for_comparison = existing_personas + [p for p in existing_personas if isinstance(p, dict)]
                    
                    # Generate a diverse persona
                    persona = self.persona_generator.ensure_diverse_persona(
                        all_personas_for_comparison,
                        temperature=temperature,
                        max_attempts=3
                    )
                    
                    # Save the persona and add to our list
                    persona_id = self.persona_generator.save_persona(persona)
                    if persona_id:
                        persona_ids.append(persona_id)
                        existing_personas.append(persona)  # Add to our comparison set for next iteration
                
                return persona_ids
            else:
                # Original implementation without diversity validation
                persona_ids = []
                for i in range(num_personas):
                    print(f"\nGenerating persona {i+1}/{num_personas}...")
                    persona = self.persona_generator.generate_persona(temperature=temperature)
                    persona_id = self.persona_generator.save_persona(persona)
                    if persona_id:
                        persona_ids.append(persona_id)
                return persona_ids
        else:
            # Generate personas with the specified diversity strategy using the updated method
            return self.persona_generator.generate_personas(
                num_personas, 
                diversity_strategy=diversity_strategy,
                enforce_diversity=enforce_diversity
            )
    
    def generate_prompts(self, num_products: int) -> List[str]:
        """Generate prompts for testing."""
        self.logger.info(f"Generating prompts for {num_products} products")
        return self.prompt_generator.generate_prompts_for_personas(num_products)
    
    def test_prompts(self, prompt_ids: List[str]) -> Dict[str, str]:
        """Test prompts with the Aurora chatbot."""
        self.logger.info("Testing prompts with Aurora chatbot")
        conversation_map = {}
        
        for i, prompt_id in enumerate(prompt_ids):
            self.logger.info(f"Testing prompt {i+1}/{len(prompt_ids)} (ID: {prompt_id})...")
            
            # Load the prompt
            prompt_doc = None
            try:
                # Try to load from MongoDB
                if hasattr(self.prompt_generator.db, 'prompts_collection'):
                    prompt_doc = self.prompt_generator.db.prompts_collection.find_one({"_id": prompt_id})
            except Exception as e:
                print(f"Error loading prompt from MongoDB: {str(e)}")
            
            if not prompt_doc:
                # Try to load from local file
                try:
                    file_path = os.path.join("db_files", "prompts", f"{prompt_id}.json")
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            prompt_doc = json.load(f)
                except Exception as e:
                    print(f"Error loading prompt from local file: {str(e)}")
                    continue
            
            if not prompt_doc:
                print(f"Could not load prompt {prompt_id}. Skipping...")
                continue
            
            # Extract the prompt text and language
            prompt_text = prompt_doc.get("prompt", "")
            language = prompt_doc.get("language", "en")
            
            if not prompt_text:
                print(f"Prompt {prompt_id} has no text. Skipping...")
                continue
            
            # Use the chatbot tester as a context manager to ensure proper resource cleanup
            try:
                with LightweightChatbotTester() as tester:
                    # Send the prompt to the chatbot
                    if language == "pt":
                        tester.test_prompt_pt(prompt_text)
                    else:
                        tester.test_prompt_en(prompt_text)
                    
                    # Store the conversation with the prompt ID
                    conversation_id = tester.store_conversation(prompt_id)
                    
                    if conversation_id:
                        conversation_map[prompt_id] = conversation_id
                        self.logger.info(f"Conversation for prompt {prompt_id} stored with ID: {conversation_id}")
                    
                    # Wait a bit to avoid overwhelming the chatbot
                    time.sleep(2)
            
            except Exception as e:
                print(f"Error testing prompt {prompt_id}: {str(e)}")
        
        return conversation_map
    
    def analyze_conversations(self, force_all: bool = False) -> List[str]:
        """Analyze conversations for bias using both statistical and qualitative approaches.
        
        Args:
            force_all: If True, re-analyze all conversations even if already analyzed
            
        Returns:
            List of analysis IDs from the qualitative analysis
        """
        # Step 1: Run statistical analysis
        self.logger.info("Running statistical analysis on conversation pairs")
        stats_data = self.run_statistical_analysis()
        
        # Step 2: Run qualitative analysis with statistical context
        self.logger.info("Analyzing conversations for bias with statistical context")
        if force_all:
            self.logger.info("Forcing re-analysis of all conversation pairs, even those already analyzed")
        
        # Pass the statistical data to the bias analyzer
        return self.bias_analyzer.analyze_all_conversation_pairs(force_all=force_all, stats_data=stats_data)
    
    def run_statistical_analysis(self) -> Dict[str, Any]:
        """Run statistical analysis on conversation pairs.
        
        Returns:
            Dictionary containing statistical analysis results
        """
        try:
            # Find conversation pairs
            self.logger.info("Finding conversation pairs for statistical analysis...")
            conversation_pairs = []
            
            # Get conversation pairs from the bias analyzer
            if hasattr(self.bias_analyzer, 'find_conversation_pairs'):
                pairs = self.bias_analyzer.find_conversation_pairs(skip_analyzed=False)
                for pair in pairs:
                    baseline_conv = pair.get("baseline_conversation")
                    persona_conv = pair.get("persona_conversation")
                    if baseline_conv and persona_conv:
                        conversation_pairs.append((baseline_conv, persona_conv))
            
            if not conversation_pairs:
                self.logger.warning("No conversation pairs found for statistical analysis.")
                return {}
            
            self.logger.info(f"Found {len(conversation_pairs)} conversation pairs for statistical analysis.")
            
            # Run the statistical analysis
            self.logger.info("Running statistical analysis...")
            stats_results = self.statistical_analyzer.analyze_conversation_pairs(conversation_pairs)
            
            # Save the results
            stats_data = {
                "timestamp": datetime.now().isoformat(),
                "num_pairs": len(conversation_pairs),
                "results": stats_results
            }
            
            self.logger.info("Statistical analysis complete.")
            return stats_data
            
        except Exception as e:
            self.logger.error(f"Error running statistical analysis: {str(e)}")
            return {}
    
    def count_results(self):
        """Count the number of results in each category."""
        # Count personas
        personas = []
        try:
            if hasattr(self.persona_generator, 'db') and self.persona_generator.db:
                personas = self.persona_generator.db.get_all_personas()
        except Exception as e:
            self.logger.error(f"Error counting personas: {str(e)}")
        
        # Count prompts
        prompts = []
        try:
            if hasattr(self.prompt_generator, 'db') and self.prompt_generator.db:
                prompts = list(self.prompt_generator.db.prompts_collection.find({}))
        except Exception as e:
            self.logger.error(f"Error counting prompts: {str(e)}")
        
        # Count conversations
        conversations = []
        try:
            if hasattr(self.bias_analyzer, 'db') and self.bias_analyzer.db:
                conversations = list(self.bias_analyzer.db.conversations_collection.find({}))
        except Exception as e:
            self.logger.error(f"Error counting conversations: {str(e)}")
        
        # Count analyses
        analyses = []
        try:
            if hasattr(self.bias_analyzer, 'db') and self.bias_analyzer.db:
                analyses = list(self.bias_analyzer.db.test_results_collection.find({"analysis_type": "bias_analysis"}))
        except Exception as e:
            self.logger.error(f"Error counting analyses: {str(e)}")
        
        self.logger.info("=== System Status ===")
        self.logger.info(f"Personas: {len(personas)}")
        self.logger.info(f"Prompts: {len(prompts)}")
        self.logger.info(f"Conversations: {len(conversations)}")
        self.logger.info(f"Bias analyses: {len(analyses)}")

def main():
    """Main function to run the bias tester."""
    parser = argparse.ArgumentParser(description="Run bias testing for Aurora chatbot")
    parser.add_argument("--personas", "-p", type=int, default=2, help="Number of personas to generate")
    parser.add_argument("--products", "-pr", type=int, default=3, help="Number of products to test")
    parser.add_argument("--questions", "-q", type=int, default=2, help="Number of questions per product")
    parser.add_argument("--diversity", "-d", type=str, default="mixed", 
                        choices=["mixed", "conservative", "balanced", "creative", "incremental"],
                        help="Diversity strategy for persona generation")
    parser.add_argument("--temperature", "-t", type=float, 
                        help="Specific temperature to use for persona generation (0.0 to 1.0)")
    parser.add_argument("--skip-personas", action="store_true", help="Skip persona generation")
    parser.add_argument("--skip-prompts", action="store_true", help="Skip prompt generation")
    parser.add_argument("--skip-testing", action="store_true", help="Skip chatbot testing")
    parser.add_argument("--skip-stats", action="store_true", help="Skip statistical analysis")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip qualitative bias analysis")
    parser.add_argument("--force-analysis", action="store_true", help="Force re-analysis of already analyzed conversations")
    parser.add_argument("--force-all", action="store_true", help="Force all steps to run, even if results already exist")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--run-id", type=str, help="Custom run ID for logging (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Set up logging
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    log_level = getattr(logging, args.log_level)
    logger = get_run_logger(run_id)
    logger.setLevel(log_level)
    
    logger.info(f"Starting bias test run with ID: {run_id}")
    logger.info(f"Command-line arguments: {args}")
    
    # Initialize the bias tester with the logger
    tester = BiasTester(logger=logger)
    
    if args.skip_prompts and args.skip_testing and args.skip_analysis:
        logger.error("You've chosen to skip all steps. Nothing to do.")
        sys.exit(1)
    
    if not args.skip_personas:
        # Step 1: Generate personas if needed
        personas = tester.persona_generator.load_personas()
        if len(personas) < args.personas or args.force_all:
            num_to_generate = args.personas if args.force_all else (args.personas - len(personas))
            logger.info(f"Need to generate {num_to_generate} personas...")
            tester.generate_personas(
                num_to_generate,
                diversity_strategy=args.diversity,
                temperature=args.temperature
            )
    
    if not args.skip_prompts:
        # Step 2: Generate prompts
        prompt_ids = tester.prompt_generator.generate_prompts_for_personas(
            args.products,
            questions_per_product=args.questions
        )
    
    if not args.skip_testing:
        # If we skipped prompt generation, load all prompts
        if args.skip_prompts:
            prompt_ids = []
            try:
                if hasattr(tester.prompt_generator.db, 'prompts_collection'):
                    cursor = tester.prompt_generator.db.prompts_collection.find({})
                    for doc in cursor:
                        if '_id' in doc:
                            prompt_ids.append(str(doc['_id']))
            except Exception as e:
                print(f"Error loading prompts from MongoDB: {str(e)}")
        
        # Step 3: Test prompts with the chatbot
        conversation_map = tester.test_prompts(prompt_ids)
    
    # Step 4: Run the integrated analysis (statistical + qualitative)
    if args.skip_stats and args.skip_analysis:
        logger.info("Skipping both statistical and qualitative analysis.")
    elif args.skip_stats:
        logger.info("Skipping statistical analysis, running only qualitative analysis.")
        # If force_all is specified, it overrides force_analysis
        force_analysis = args.force_analysis or args.force_all
        # Run qualitative analysis without statistical context
        analysis_ids = tester.bias_analyzer.analyze_all_conversation_pairs(force_all=force_analysis)
        logger.info(f"Completed {len(analysis_ids)} qualitative bias analyses.")
    elif args.skip_analysis:
        logger.info("Running only statistical analysis, skipping qualitative analysis.")
        # Run statistical analysis only
        stats_data = tester.run_statistical_analysis()
        if stats_data:
            logger.info("Statistical analysis completed successfully.")
    else:
        # Run both statistical and qualitative analysis
        # If force_all is specified, it overrides force_analysis
        force_analysis = args.force_analysis or args.force_all
        analysis_ids = tester.analyze_conversations(force_all=force_analysis)
        logger.info(f"Completed integrated bias analysis with {len(analysis_ids)} qualitative analyses.")
        
    # Log completion of the run
    logger.info(f"Bias test run {run_id} completed successfully")

if __name__ == "__main__":
    main()
