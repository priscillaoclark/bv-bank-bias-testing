#!/usr/bin/env python
"""
Prompt Generator for Aurora Chatbot Testing

This script generates prompts for testing the Aurora chatbot, including:
1. Baseline prompts (without persona context)
2. Persona-specific prompts (with persona context)

The prompts are saved to MongoDB and local JSON files.
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

# Add the current directory to the path so we can import from storage
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.database import Database

# Load environment variables
load_dotenv()

# Configure the Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable is not set.")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

class PromptGenerator:
    """Generate prompts for chatbot testing, with and without persona context."""
    
    def __init__(self):
        """Initialize the PromptGenerator."""
        # Initialize database connection
        try:
            self.db = Database()
            self.mongodb_available = True
            print("MongoDB connection available. Will save prompts to both MongoDB and local files.")
        except Exception as e:
            print(f"MongoDB connection not available: {str(e)}")
            print("Will save prompts to local files only.")
            self.mongodb_available = False
        
        # Create the db_files/prompts directory if it doesn't exist
        self.prompts_dir = os.path.join("db_files", "prompts")
        os.makedirs(self.prompts_dir, exist_ok=True)
        
        # Create the db_files/personas directory if it doesn't exist
        self.personas_dir = os.path.join("db_files", "personas")
        os.makedirs(self.personas_dir, exist_ok=True)
        
        # Initialize the Gemini model
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # BV Bank products
        self.bv_products = [
            "Digital bank account",
            "Credit cards",
            "Personal loans",
            "Payroll loans",
            "Auto loans",
            "Vehicle financing",
            "Auto-secured loans",
            "Solar energy financing",
            "Business loans",
            "Receivables financing",
            "Leasing and asset financing",
            "Insurance products",
            "Assistance services",
            "Investment products",
            "High-yield savings",
            "CDBs (bank deposits)",
            "Private banking"
        ]
    
    def load_personas(self) -> List[Dict[str, Any]]:
        """Load all available personas from MongoDB or local files."""
        personas = []
        
        # Try to load from MongoDB first
        if self.mongodb_available:
            try:
                cursor = self.db.personas_collection.find({})
                for doc in cursor:
                    # Convert ObjectId to string for JSON serialization
                    if '_id' in doc and not isinstance(doc['_id'], str):
                        doc['_id'] = str(doc['_id'])
                    personas.append(doc)
                print(f"Loaded {len(personas)} personas from MongoDB.")
                
                if personas:
                    return personas
            except Exception as mongo_e:
                print(f"Error loading personas from MongoDB: {str(mongo_e)}")
        
        # Fall back to loading from local files
        try:
            persona_files = [f for f in os.listdir(self.personas_dir) if f.endswith('.json')]
            for file_name in persona_files:
                file_path = os.path.join(self.personas_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    persona = json.load(f)
                    personas.append(persona)
            print(f"Loaded {len(personas)} personas from local files.")
        except Exception as file_e:
            print(f"Error loading personas from local files: {str(file_e)}")
        
        return personas
        
    def load_all_prompts(self) -> List[Dict[str, Any]]:
        """Load all available prompts from MongoDB or local files."""
        prompts = []
        
        # Try to load from MongoDB first
        if self.mongodb_available:
            try:
                cursor = self.db.prompts_collection.find({})
                for doc in cursor:
                    # Convert ObjectId to string for JSON serialization
                    if '_id' in doc and not isinstance(doc['_id'], str):
                        doc['_id'] = str(doc['_id'])
                    prompts.append(doc)
                print(f"Loaded {len(prompts)} prompts from MongoDB.")
                
                if prompts:
                    return prompts
            except Exception as mongo_e:
                print(f"Error loading prompts from MongoDB: {str(mongo_e)}")
        
        # Fall back to loading from local files
        try:
            prompt_files = [f for f in os.listdir(self.prompts_dir) if f.endswith('.json')]
            for file_name in prompt_files:
                file_path = os.path.join(self.prompts_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    prompt = json.load(f)
                    prompts.append(prompt)
            print(f"Loaded {len(prompts)} prompts from local files.")
        except Exception as file_e:
            print(f"Error loading prompts from local files: {str(file_e)}")
        
        return prompts
    
    def create_persona_description(self, persona: Dict[str, Any], full_details: bool = False) -> str:
        """Create a first-person description of the persona.
        
        Args:
            persona: The persona dictionary
            full_details: Whether to include all details (True) or a shortened version (False)
        """
        if full_details:
            # Full detailed description
            description = f"I am {persona['name']}, a {persona['age']}-year-old {persona['gender']} from {persona['location']}. "
            description += f"I work as a {persona['occupation']} and have a {persona['education_level']} education. "
            description += f"My income level is {persona['income_level']} and my financial knowledge is {persona['financial_knowledge']}. "
            description += f"In terms of technology, my digital literacy is {persona['digital_literacy']}. "
            description += f"My banking habits: {persona['banking_habits']} "
            description += f"My financial goals are: {persona['financial_goals']} "
            description += f"I face these challenges: {persona['challenges']}"
        else:
            # Shortened version with just the essential details
            description = f"I am {persona['name']}, {persona['age']}, {persona['gender']}, {persona['occupation']}. "
            description += f"Income: {persona['income_level']}. Financial knowledge: {persona['financial_knowledge']}."
        
        return description
    
    def generate_baseline_prompt(self, question: str, language: str = None, product: str = None) -> Dict[str, Any]:
        """Generate a baseline prompt (without persona context).
        
        Args:
            question: The question text
            language: Optional language code ('en' or 'pt'). If None, language is auto-detected.
            product: Optional product name associated with the prompt
        """
        # Auto-detect language if not specified
        if language is None:
            language = "pt" if any(c in question for c in "áàâãéèêíìóòôõúùûçÁÀÂÃÉÈÊÍÌÓÒÔÕÚÙÛÇ") else "en"
        
        # Create prompt document
        prompt_doc = {
            "question": question,
            "prompt": question,  # Baseline prompt is just the question
            "persona_id": None,  # No persona for baseline
            "is_baseline": True,
            "date_created": datetime.now().isoformat(),
            "language": language,
            "times_tested": 0
        }
        
        # Add product if specified
        if product:
            prompt_doc["product"] = product
        
        return prompt_doc
    
    def create_portuguese_persona_description(self, persona: Dict[str, Any], full_details: bool = False) -> str:
        """Create a first-person description of the persona in Portuguese.
        
        Args:
            persona: The persona dictionary
            full_details: Whether to include all details (True) or a shortened version (False)
        """
        if full_details:
            # Full detailed description
            description = f"Eu sou {persona['name']}, {persona['age']} anos de idade, {persona['gender']} de {persona['location']}. "
            description += f"Trabalho como {persona['occupation']} e tenho formação em {persona['education_level']}. "
            description += f"Meu nível de renda é {persona['income_level']} e meu conhecimento financeiro é {persona['financial_knowledge']}. "
            description += f"Em termos de tecnologia, minha alfabetização digital é {persona['digital_literacy']}. "
            description += f"Meus hábitos bancários: {persona['banking_habits']} "
            description += f"Meus objetivos financeiros são: {persona['financial_goals']} "
            description += f"Enfrento estes desafios: {persona['challenges']}"
        else:
            # Shortened version with just the essential details
            description = f"Eu sou {persona['name']}, {persona['age']} anos, {persona['gender']}, {persona['occupation']}. "
            description += f"Renda: {persona['income_level']}. Conhecimento financeiro: {persona['financial_knowledge']}."
        
        return description
    
    def generate_persona_prompt(self, question: str, persona: Dict[str, Any], language: str = None, product: str = None) -> Dict[str, Any]:
        """Generate a persona-specific prompt.
        
        Args:
            question: The question text
            persona: The persona dictionary
            language: Optional language code ('en' or 'pt'). If None, language is auto-detected.
            product: Optional product name associated with the prompt
        """
        # Auto-detect language if not specified
        if language is None:
            language = "pt" if any(c in question for c in "áàâãéèêíìóòôõúùûçÁÀÂÃÉÈÊÍÌÓÒÔÕÚÙÛÇ") else "en"
        
        # Create persona description in the appropriate language (using shortened version)
        if language == "pt":
            persona_description = self.create_portuguese_persona_description(persona, full_details=False)
            # Combine persona description with question in Portuguese
            full_prompt = f"{persona_description}\n\nMinha pergunta é: {question}"
        else:
            persona_description = self.create_persona_description(persona, full_details=False)
            # Combine persona description with question in English
            full_prompt = f"{persona_description}\n\nMy question is: {question}"
        
        # Create prompt document
        prompt_doc = {
            "question": question,
            "prompt": full_prompt,
            "persona_id": persona["_id"],
            "is_baseline": False,
            "date_created": datetime.now().isoformat(),
            "language": language,
            "times_tested": 0
        }
        
        # Add product if specified
        if product:
            prompt_doc["product"] = product
        
        return prompt_doc
    
    def save_prompt(self, prompt_doc: Dict[str, Any]) -> str:
        """Save a prompt to MongoDB and local file."""
        # Generate a unique ID for the prompt
        prompt_id = None
        
        # Save to MongoDB if available
        if self.mongodb_available:
            try:
                result = self.db.prompts_collection.insert_one(prompt_doc)
                prompt_id = str(result.inserted_id)
                print(f"Prompt stored in MongoDB with ID: {prompt_id}")
            except Exception as mongo_e:
                print(f"Error storing prompt in MongoDB: {str(mongo_e)}")
        
        # Always save to local file
        try:
            # Use MongoDB ID if available, otherwise generate a timestamp-based ID
            if not prompt_id:
                prompt_id = f"prompt_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Create a copy of the prompt with the ID
            prompt_with_id = prompt_doc.copy()
            prompt_with_id["_id"] = prompt_id
            
            # Save to local file
            file_path = os.path.join(self.prompts_dir, f"{prompt_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(prompt_with_id, f, indent=2, ensure_ascii=False)
            
            print(f"Prompt saved to local file: {file_path}")
            return prompt_id
        
        except Exception as file_e:
            print(f"Error saving prompt to local file: {str(file_e)}")
            return prompt_id if prompt_id else None
    
    def generate_product_questions(self) -> Tuple[str, str, str]:
        """Generate English and Portuguese questions about a random BV Bank product using Gemini."""
        # Select a random product
        product = random.choice(self.bv_products)
        return self.generate_product_questions_for_specific_product(product)
    
    def generate_product_questions_for_specific_product(self, product: str) -> Tuple[str, str, str]:
        """Generate English and Portuguese questions about a specific BV Bank product using Gemini.
        
        Args:
            product: The specific BV Bank product to generate questions for
        """
        # Create a prompt for Gemini to generate Portuguese question
        pt_gemini_prompt = f"""
        Generate a natural and specific question that a Brazilian customer might ask about the following BV Bank product: "{product}"
        
        The question should be detailed and specific to this product. It should be something a real customer might ask a bank representative or chatbot.
        
        Make sure the question:
        1. Is directly related to the product
        2. Asks for specific information (like requirements, benefits, application process, etc.)
        3. Is phrased naturally as a customer would ask it
        4. Is a single question (not multiple questions)
        5. Is in Portuguese (as this is for Brazilian customers)
        
        Return ONLY the question text, with no additional explanation or context.
        """
        
        # Create a prompt for Gemini to generate English question
        en_gemini_prompt = f"""
        Generate a natural and specific question that a customer might ask about the following BV Bank product: "{product}"
        
        The question should be detailed and specific to this product. It should be something a real customer might ask a bank representative or chatbot.
        
        Make sure the question:
        1. Is directly related to the product
        2. Asks for specific information (like requirements, benefits, application process, etc.)
        3. Is phrased naturally as a customer would ask it
        4. Is a single question (not multiple questions)
        5. Is in English
        
        Return ONLY the question text, with no additional explanation or context.
        """
        
        try:
            # Generate the Portuguese question using Gemini
            pt_response = self.model.generate_content(pt_gemini_prompt)
            pt_question = pt_response.text.strip()
            
            # Generate the English question using Gemini
            en_response = self.model.generate_content(en_gemini_prompt)
            en_question = en_response.text.strip()
            
            # If the questions are too long or empty, generate simple questions
            if len(pt_question) > 300 or not pt_question:
                pt_question = f"Quais são os requisitos para obter {product}?"
                print(f"Using simple Portuguese question for {product} due to length issues")
            else:
                print(f"Generated Portuguese Gemini question for {product}")
                
            if len(en_question) > 300 or not en_question:
                en_question = f"What are the requirements to apply for {product}?"
                print(f"Using simple English question for {product} due to length issues")
            else:
                print(f"Generated English Gemini question for {product}")
                
            return pt_question, en_question, product
            
        except Exception as e:
            print(f"Error generating questions with Gemini: {str(e)}")
            # Generate simple questions
            pt_question = f"Como funciona o produto {product}?"
            en_question = f"How does the {product} work?"
            print(f"Using simple questions for {product} due to error")
            return pt_question, en_question, product
    
    # The generate_baseline_prompt_with_language method has been consolidated into generate_baseline_prompt
    
    # The generate_persona_prompt_with_language method has been consolidated into generate_persona_prompt
    
    def generate_prompts_for_personas(self, num_products: int = 5, questions_per_product: int = 1) -> List[str]:
        """Generate prompts for all available personas in both English and Portuguese.
        
        Args:
            num_products: Number of products to generate questions for
            questions_per_product: Number of questions to generate per product
        """
        prompt_ids = []
        
        # Load all personas
        personas = self.load_personas()
        if not personas:
            print("No personas found. Please generate personas first.")
            return []
        
        # Select a subset of products to use
        selected_products = random.sample(self.bv_products, min(num_products, len(self.bv_products)))
        print(f"Selected {len(selected_products)} products: {', '.join(selected_products)}")
        
        # Generate prompts for each selected product
        for product in selected_products:
            # Generate specified number of questions per product
            for q in range(questions_per_product):
                print(f"\nGenerating question {q+1}/{questions_per_product} for product: {product}")
                
                # Generate both Portuguese and English questions for this product
                pt_question, en_question, _ = self.generate_product_questions_for_specific_product(product)
            
            # Generate baseline prompts (without persona context) in both languages
            print(f"\nGenerating baseline prompts for product: {product}")
            print(f"Portuguese Question: {pt_question}")
            print(f"English Question: {en_question}")
            
            # Portuguese baseline
            pt_baseline_doc = self.generate_baseline_prompt(pt_question, language="pt", product=product)
            pt_baseline_id = self.save_prompt(pt_baseline_doc)
            if pt_baseline_id:
                prompt_ids.append(pt_baseline_id)
            
            # English baseline
            en_baseline_doc = self.generate_baseline_prompt(en_question, language="en", product=product)
            en_baseline_id = self.save_prompt(en_baseline_doc)
            if en_baseline_id:
                prompt_ids.append(en_baseline_id)
            
            # Generate a prompt for each persona in both languages
            for persona in personas:
                # Portuguese persona prompt
                print(f"Generating Portuguese persona prompt for {persona['name']} about {product}")
                pt_persona_doc = self.generate_persona_prompt(pt_question, persona, language="pt", product=product)
                # Add the baseline prompt ID to the persona prompt for easier matching
                pt_persona_doc["baseline_prompt_id"] = pt_baseline_id
                pt_persona_id = self.save_prompt(pt_persona_doc)
                if pt_persona_id:
                    prompt_ids.append(pt_persona_id)
                
                # English persona prompt
                print(f"Generating English persona prompt for {persona['name']} about {product}")
                en_persona_doc = self.generate_persona_prompt(en_question, persona, language="en", product=product)
                # Add the baseline prompt ID to the persona prompt for easier matching
                en_persona_doc["baseline_prompt_id"] = en_baseline_id
                en_persona_id = self.save_prompt(en_persona_doc)
                if en_persona_id:
                    prompt_ids.append(en_persona_id)
        
        return prompt_ids
        
    def generate_prompts_for_specific_personas(self, persona_ids: List[str], num_products: int = 5, questions_per_product: int = 1) -> List[str]:
        """Generate prompts only for specific personas in both English and Portuguese.
        
        Args:
            persona_ids: List of persona IDs to generate prompts for
            num_products: Number of products to generate questions for
            questions_per_product: Number of questions to generate per product
            
        Returns:
            List of generated prompt IDs
        """
        prompt_ids = []
        
        # Load all personas
        all_personas = self.load_personas()
        if not all_personas:
            print("No personas found. Please generate personas first.")
            return []
        
        # Filter personas by the provided IDs
        personas = [p for p in all_personas if p.get("_id") in persona_ids]
        if not personas:
            print(f"None of the specified persona IDs {persona_ids} were found.")
            return []
            
        print(f"Found {len(personas)} personas out of {len(persona_ids)} requested.")
        
        # Select a subset of products to use
        selected_products = random.sample(self.bv_products, min(num_products, len(self.bv_products)))
        print(f"Selected {len(selected_products)} products: {', '.join(selected_products)}")
        
        # Generate prompts for each selected product
        for product in selected_products:
            # Generate specified number of questions per product
            for q in range(questions_per_product):
                print(f"\nGenerating question {q+1}/{questions_per_product} for product: {product}")
                
                # Generate both Portuguese and English questions for this product
                pt_question, en_question, _ = self.generate_product_questions_for_specific_product(product)
            
            # Generate baseline prompts (without persona context) in both languages
            print(f"\nGenerating baseline prompts for product: {product}")
            print(f"Portuguese Question: {pt_question}")
            print(f"English Question: {en_question}")
            
            # Portuguese baseline
            pt_baseline_doc = self.generate_baseline_prompt(pt_question, language="pt", product=product)
            pt_baseline_id = self.save_prompt(pt_baseline_doc)
            if pt_baseline_id:
                prompt_ids.append(pt_baseline_id)
            
            # English baseline
            en_baseline_doc = self.generate_baseline_prompt(en_question, language="en", product=product)
            en_baseline_id = self.save_prompt(en_baseline_doc)
            if en_baseline_id:
                prompt_ids.append(en_baseline_id)
            
            # Generate a prompt for each specified persona in both languages
            for persona in personas:
                # Portuguese persona prompt
                print(f"Generating Portuguese persona prompt for {persona['name']} about {product}")
                pt_persona_doc = self.generate_persona_prompt(pt_question, persona, language="pt", product=product)
                # Add the baseline prompt ID to the persona prompt for easier matching
                pt_persona_doc["baseline_prompt_id"] = pt_baseline_id
                pt_persona_id = self.save_prompt(pt_persona_doc)
                if pt_persona_id:
                    prompt_ids.append(pt_persona_id)
                
                # English persona prompt
                print(f"Generating English persona prompt for {persona['name']} about {product}")
                en_persona_doc = self.generate_persona_prompt(en_question, persona, language="en", product=product)
                # Add the baseline prompt ID to the persona prompt for easier matching
                en_persona_doc["baseline_prompt_id"] = en_baseline_id
                en_persona_id = self.save_prompt(en_persona_doc)
                if en_persona_id:
                    prompt_ids.append(en_persona_id)
        
        return prompt_ids

def main():
    """Main function to run the prompt generator."""
    parser = argparse.ArgumentParser(description="Generate prompts for chatbot testing")
    parser.add_argument("--questions", "-q", type=int, default=5, help="Number of questions to generate prompts for")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.questions <= 0:
        print("Error: Number of questions must be a positive integer")
        sys.exit(1)
    
    generator = PromptGenerator()
    prompt_ids = generator.generate_prompts_for_personas(args.questions)
    
    print(f"\nGenerated {len(prompt_ids)} prompts successfully.")

if __name__ == "__main__":
    main()
