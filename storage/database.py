"""
Database module for the Responsible AI Testing System.
Handles storage and retrieval of prompts, test results, and other data.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from bson import ObjectId
import pymongo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Database:
    """Database handler for the Responsible AI Testing System."""
    
    def __init__(self):
        """Initialize the database connection."""
        # Use MONGODB_URI as primary, fall back to MONGO_URI for backward compatibility
        self.mongo_uri = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        # Create a masked version of the URI for logging
        self.masked_uri = self._mask_connection_string(self.mongo_uri)
        # Explicitly set to rai_testing2 to match what's in MongoDB Compass
        self.db_name = "rai_testing2"  # Force the correct database name
        print(f"Using database: {self.db_name}")
        
        try:
            # Connect to MongoDB
            self.client = pymongo.MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            
            # Create collections if they don't exist
            self.prompts_collection = self.db["prompts"]
            self.test_results_collection = self.db["test_results"]
            self.personas_collection = self.db["personas"]
            self.conversations_collection = self.db["conversations"]
            self.stats_collection = self.db["stats"]
            
            # Create indexes
            self._create_indexes()
            
            print(f"Connected to MongoDB at {self.masked_uri}")
        except Exception as e:
            print(f"Error connecting to MongoDB at {self.masked_uri}: {str(e)}")
            raise
        
    def _mask_connection_string(self, connection_string):
        """Mask sensitive parts of a MongoDB connection string.
        
        Args:
            connection_string: The MongoDB connection string to mask
            
        Returns:
            A masked version of the connection string with credentials hidden
        """
        if not connection_string:
            return "None"
            
        try:
            # Handle mongodb:// and mongodb+srv:// connection strings
            if '@' in connection_string:
                # Connection string contains credentials
                protocol_part, rest = connection_string.split('://', 1)
                if '@' in rest:
                    # There are credentials in the connection string
                    credentials_part, host_part = rest.split('@', 1)
                    return f"{protocol_part}://***:***@{host_part}"
            
            # For connection strings without credentials or simple localhost connections
            return connection_string
        except Exception:
            # If any error occurs during masking, return a generic placeholder
            return "[mongodb-connection-string]"
    
    def _create_indexes(self):
        """Create indexes for the collections."""
        # Prompts collection indexes
        self.prompts_collection.create_index("attack_type")
        self.prompts_collection.create_index("language")
        self.prompts_collection.create_index("times_tested")
        self.prompts_collection.create_index("date_created")
        
        # Conversations collection indexes
        self.conversations_collection.create_index("conversation_id")
        self.conversations_collection.create_index("date_created")
        
        # Stats collection indexes
        self.stats_collection.create_index("timestamp")
        self.stats_collection.create_index("type")
        
        print(f"Created indexes for collections: prompts, test_results, personas, conversations, stats")
        print(f"Collections in database: {', '.join(self.db.list_collection_names())}")
        
        # Test results collection indexes
        self.test_results_collection.create_index("prompt_id")
        self.test_results_collection.create_index("date_tested")
        self.test_results_collection.create_index("success")
    
    def store_prompt(self, prompt_text: str, attack_type: Optional[str] = None, 
                    language: str = "english", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a prompt in the database.
        
        Args:
            prompt_text: Text of the prompt
            attack_type: Type of attack (optional)
            language: Language of the prompt (default: english)
            metadata: Additional metadata (optional)
            
        Returns:
            ID of the stored prompt
        """
        # Check if a prompt with the same text already exists
        existing_prompt = self.prompts_collection.find_one({"prompt_text": prompt_text})
        if existing_prompt:
            # Return the ID of the existing prompt
            return str(existing_prompt["_id"])
        
        # Create a new document for the prompt
        document = {
            "prompt_text": prompt_text,
            "attack_type": attack_type,
            "language": language.lower(),
            "times_tested": 0,
            "date_created": datetime.now(),
            "metadata": metadata or {}
        }
        
        # Insert the new prompt
        try:
            result = self.prompts_collection.insert_one(document)
            return str(result.inserted_id)
        except pymongo.errors.DuplicateKeyError:
            # If a duplicate key error occurs (race condition), get the existing prompt
            existing_prompt = self.prompts_collection.find_one({"prompt_text": prompt_text})
            if existing_prompt:
                return str(existing_prompt["_id"])
            # If we can't find the existing prompt (unlikely), raise the error
            raise
    
    def get_prompt(self, prompt_id: Union[str, ObjectId]) -> Optional[Dict[str, Any]]:
        """
        Get a prompt from the database.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            Prompt document or None if not found
        """
        if isinstance(prompt_id, str):
            try:
                prompt_id = ObjectId(prompt_id)
            except Exception:
                return None
        
        return self.prompts_collection.find_one({"_id": prompt_id})
    
    def get_random_untested_prompts(self, count: int = 10, 
                                   attack_type: Optional[str] = None,
                                   language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get random untested prompts from the database.
        
        Args:
            count: Number of prompts to get
            attack_type: Type of attack to filter by (optional)
            language: Language to filter by (optional)
            
        Returns:
            List of prompt documents
        """
        query = {"times_tested": 0}
        
        if attack_type:
            query["attack_type"] = attack_type
        
        if language:
            query["language"] = language.lower()
        
        pipeline = [
            {"$match": query},
            {"$sample": {"size": count}}
        ]
        
        return list(self.prompts_collection.aggregate(pipeline))
    
    def get_prompts_by_date_range(self, start_date: datetime, end_date: datetime,
                                 attack_type: Optional[str] = None,
                                 language: Optional[str] = None,
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get prompts created within a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            attack_type: Type of attack to filter by (optional)
            language: Language to filter by (optional)
            limit: Maximum number of prompts to return
            
        Returns:
            List of prompt documents
        """
        query = {
            "date_created": {
                "$gte": start_date,
                "$lte": end_date
            }
        }
        
        if attack_type:
            query["attack_type"] = attack_type
        
        if language:
            query["language"] = language.lower()
        
        return list(self.prompts_collection.find(query).limit(limit))
    
    def mark_prompt_as_tested(self, prompt_id: Union[str, ObjectId], success: bool) -> bool:
        """
        Mark a prompt as tested.
        
        Args:
            prompt_id: ID of the prompt
            success: Whether the test was successful
            
        Returns:
            True if the update was successful, False otherwise
        """
        if isinstance(prompt_id, str):
            try:
                prompt_id = ObjectId(prompt_id)
            except Exception:
                return False
        
        result = self.prompts_collection.update_one(
            {"_id": prompt_id},
            {
                "$inc": {"times_tested": 1},
                "$set": {"last_tested": datetime.now(), "last_test_success": success}
            }
        )
        
        return result.modified_count > 0
    
    def store_test_result(self, prompt_id: Union[str, ObjectId], response: str, 
                         success: bool, model_info: Dict[str, str],
                         execution_time: float = 0.0,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a test result in the database.
        
        Args:
            prompt_id: ID of the prompt
            response: Response from the chatbot
            success: Whether the test was successful
            model_info: Information about the model used
            execution_time: Time taken to execute the test (in seconds)
            metadata: Additional metadata (optional)
            
        Returns:
            ID of the stored test result
        """
        if isinstance(prompt_id, str):
            try:
                prompt_id = ObjectId(prompt_id)
            except Exception:
                # If conversion fails, use the string ID directly
                pass
        
        document = {
            "prompt_id": prompt_id,
            "response": response,
            "success": success,
            "model_info": model_info,
            "execution_time": execution_time,
            "date_tested": datetime.now(),
            "metadata": metadata or {}
        }
        
        result = self.test_results_collection.insert_one(document)
        return str(result.inserted_id)
    
    def get_test_results(self, prompt_id: Optional[Union[str, ObjectId]] = None,
                        success: Optional[bool] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get test results from the database.
        
        Args:
            prompt_id: ID of the prompt (optional)
            success: Filter by success (optional)
            limit: Maximum number of results to return
            
        Returns:
            List of test result documents
        """
        query = {}
        
        if prompt_id:
            if isinstance(prompt_id, str):
                try:
                    prompt_id = ObjectId(prompt_id)
                except Exception:
                    # If conversion fails, use the string ID directly
                    pass
            
            query["prompt_id"] = prompt_id
        
        if success is not None:
            query["success"] = success
        
        return list(self.test_results_collection.find(query).sort("date_tested", -1).limit(limit))
    
    def store_persona(self, persona: Dict[str, Any]) -> str:
        """
        Store a persona in the database.
        
        Args:
            persona: Persona data
            
        Returns:
            ID of the stored persona
        """
        # Add creation timestamp if not present
        if "date_created" not in persona:
            persona["date_created"] = datetime.now().isoformat()
        
        # Insert the persona
        result = self.personas_collection.insert_one(persona)
        return str(result.inserted_id)
    
    def get_persona(self, persona_id: Union[str, ObjectId]) -> Optional[Dict[str, Any]]:
        """
        Get a persona from the database.
        
        Args:
            persona_id: ID of the persona
            
        Returns:
            Persona document or None if not found
        """
        if isinstance(persona_id, str):
            try:
                persona_id = ObjectId(persona_id)
            except Exception:
                return None
        
        return self.personas_collection.find_one({"_id": persona_id})
    
    def get_all_personas(self) -> List[Dict[str, Any]]:
        """
        Get all personas from the database.
        
        Returns:
            List of persona documents
        """
        return list(self.personas_collection.find({}))
    
    def get_personas_by_attribute(self, attribute: str, value: Any) -> List[Dict[str, Any]]:
        """
        Get personas with a specific attribute value.
        
        Args:
            attribute: Attribute name (e.g., "gender", "age_range", etc.)
            value: Value to match
            
        Returns:
            List of matching persona documents
        """
        return list(self.personas_collection.find({attribute: value}))
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()
            print("Closed MongoDB connection")
