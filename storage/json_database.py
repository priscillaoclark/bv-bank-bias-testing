"""
JSON Database module for the Responsible AI Testing System.
Handles storage and retrieval of prompts, test results, and other data using local JSON files.
"""

import os
import sys
import json
import uuid
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

class ObjectId:
    """Simple ObjectId implementation for compatibility with existing code."""
    
    def __init__(self, id_str=None):
        """Initialize with an existing ID or generate a new one."""
        if id_str:
            self.id_str = id_str
        else:
            # Generate a unique ID based on timestamp and random number
            timestamp = hex(int(time.time()))[2:]
            random_part = hex(random.randint(0, 0xffffff))[2:]
            self.id_str = f"{timestamp}{random_part}"
    
    def __str__(self):
        """Return the string representation of the ID."""
        return self.id_str
    
    def __eq__(self, other):
        """Compare with another ObjectId or string."""
        if isinstance(other, ObjectId):
            return self.id_str == other.id_str
        elif isinstance(other, str):
            return self.id_str == other
        return False

class JSONDatabase:
    """Database handler for the Responsible AI Testing System using local JSON files."""
    
    def __init__(self):
        """Initialize the JSON database."""
        # Create the main db_files directory
        self.db_name = "json_db"
        self.db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db_files")
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Create subdirectories for each collection
        self.prompts_dir = os.path.join(self.db_dir, "prompts")
        self.test_results_dir = os.path.join(self.db_dir, "test_results")
        self.personas_dir = os.path.join(self.db_dir, "personas")
        self.conversations_dir = os.path.join(self.db_dir, "convos")
        self.stats_dir = os.path.join(self.db_dir, "stats")
        self.analysis_results_dir = os.path.join(self.db_dir, "results")
        
        # Create all directories
        for directory in [self.prompts_dir, self.test_results_dir, self.personas_dir, 
                          self.conversations_dir, self.stats_dir, self.analysis_results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Create collection references for compatibility with MongoDB code
        self.prompts_collection = self
        self.test_results_collection = self
        self.personas_collection = self
        self.conversations_collection = self
        self.stats_collection = self
        self.results_collection = self
        
        # For compatibility with MongoDB code
        self.db = {
            "prompts": self,
            "test_results": self,
            "personas": self,
            "conversations": self,
            "stats": self,
            "results": self
        }
        
        print(f"Initialized JSON database at {self.db_dir}")
    
    def _serialize_for_json(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        else:
            return obj
    
    def _get_collection_dir(self, collection_name):
        """Get the directory for a specific collection."""
        if collection_name == "prompts":
            return self.prompts_dir
        elif collection_name == "test_results":
            return self.test_results_dir
        elif collection_name == "personas":
            return self.personas_dir
        elif collection_name == "conversations":
            return self.conversations_dir
        elif collection_name == "stats":
            return self.stats_dir
        elif collection_name == "results":
            return self.analysis_results_dir
        else:
            raise ValueError(f"Unknown collection: {collection_name}")
    
    def _get_file_path(self, collection_name, doc_id):
        """Get the file path for a document."""
        collection_dir = self._get_collection_dir(collection_name)
        return os.path.join(collection_dir, f"{doc_id}.json")
    
    def _save_document(self, collection_name, document):
        """Save a document to a JSON file."""
        # Ensure the document has an _id
        if "_id" not in document:
            document["_id"] = str(ObjectId())
        
        # Convert ObjectId to string if needed
        if not isinstance(document["_id"], str):
            document["_id"] = str(document["_id"])
        
        # Prepare the document for JSON serialization
        json_doc = self._serialize_for_json(document)
        
        # Save to file
        file_path = self._get_file_path(collection_name, document["_id"])
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_doc, f, indent=2, ensure_ascii=False)
        
        return document["_id"]
    
    def _load_document(self, collection_name, doc_id):
        """Load a document from a JSON file."""
        file_path = self._get_file_path(collection_name, doc_id)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    def _list_documents(self, collection_name):
        """List all documents in a collection."""
        collection_dir = self._get_collection_dir(collection_name)
        documents = []
        
        if not os.path.exists(collection_dir):
            return []
        
        for filename in os.listdir(collection_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(collection_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        document = json.load(f)
                        documents.append(document)
                except Exception as e:
                    print(f"Error loading document {filename}: {str(e)}")
        
        return documents
    
    def _filter_documents(self, documents, query):
        """Filter documents based on a query."""
        if not query:
            return documents
        
        filtered_docs = []
        for doc in documents:
            match = True
            for key, value in query.items():
                # Handle nested fields with dot notation
                if "." in key:
                    parts = key.split(".")
                    doc_value = doc
                    for part in parts:
                        if isinstance(doc_value, dict) and part in doc_value:
                            doc_value = doc_value[part]
                        else:
                            doc_value = None
                            break
                else:
                    doc_value = doc.get(key)
                
                # Handle special operators
                if isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):
                    for op, op_value in value.items():
                        if op == "$gte":
                            if not (doc_value and doc_value >= op_value):
                                match = False
                                break
                        elif op == "$lte":
                            if not (doc_value and doc_value <= op_value):
                                match = False
                                break
                        elif op == "$eq":
                            if doc_value != op_value:
                                match = False
                                break
                        elif op == "$ne":
                            if doc_value == op_value:
                                match = False
                                break
                        elif op == "$in":
                            if not (doc_value and doc_value in op_value):
                                match = False
                                break
                else:
                    # Direct value comparison
                    if doc_value != value:
                        match = False
                        break
            
            if match:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _sort_documents(self, documents, sort_key, direction=1):
        """Sort documents by a key."""
        reverse = direction < 0
        return sorted(documents, key=lambda x: x.get(sort_key, ""), reverse=reverse)
    
    # MongoDB-compatible methods
    def insert_one(self, document, collection_name=None):
        """Insert a document into a collection."""
        # Determine the collection name
        if not collection_name:
            # Try to infer from the caller
            import inspect
            caller_frame = inspect.currentframe().f_back
            caller_locals = caller_frame.f_locals
            
            for name, value in caller_locals.items():
                if value is self and name.endswith("_collection"):
                    collection_name = name.replace("_collection", "")
                    break
        
        if not collection_name:
            raise ValueError("Collection name not specified")
        
        doc_id = self._save_document(collection_name, document)
        
        # Return a result object with inserted_id attribute
        class InsertOneResult:
            def __init__(self, inserted_id):
                self.inserted_id = inserted_id
        
        return InsertOneResult(doc_id)
    
    def find_one(self, query, collection_name=None):
        """Find a single document matching the query."""
        # Determine the collection name
        if not collection_name:
            # Try to infer from the caller
            import inspect
            caller_frame = inspect.currentframe().f_back
            caller_locals = caller_frame.f_locals
            
            for name, value in caller_locals.items():
                if value is self and name.endswith("_collection"):
                    collection_name = name.replace("_collection", "")
                    break
        
        if not collection_name:
            raise ValueError("Collection name not specified")
        
        # Handle _id queries directly
        if "_id" in query and isinstance(query, dict) and len(query) == 1:
            doc_id = query["_id"]
            if isinstance(doc_id, ObjectId):
                doc_id = str(doc_id)
            return self._load_document(collection_name, doc_id)
        
        # For other queries, load all documents and filter
        documents = self._list_documents(collection_name)
        filtered = self._filter_documents(documents, query)
        
        if filtered:
            return filtered[0]
        return None
    
    def find(self, query=None, collection_name=None):
        """Find documents matching the query."""
        # Determine the collection name
        if not collection_name:
            # Try to infer from the caller
            import inspect
            caller_frame = inspect.currentframe().f_back
            caller_locals = caller_frame.f_locals
            
            for name, value in caller_locals.items():
                if value is self and name.endswith("_collection"):
                    collection_name = name.replace("_collection", "")
                    break
        
        if not collection_name:
            raise ValueError("Collection name not specified")
        
        documents = self._list_documents(collection_name)
        if query:
            documents = self._filter_documents(documents, query)
        
        # Return a cursor-like object
        class Cursor:
            def __init__(self, docs):
                self.docs = docs
                self.limit_val = None
                self.sort_key = None
                self.sort_dir = 1
            
            def limit(self, n):
                self.limit_val = n
                return self
            
            def sort(self, key, direction=1):
                self.sort_key = key
                self.sort_dir = direction
                return self
            
            def __iter__(self):
                docs = self.docs
                
                # Apply sorting if specified
                if self.sort_key:
                    docs = sorted(docs, key=lambda x: x.get(self.sort_key, ""), reverse=(self.sort_dir < 0))
                
                # Apply limit if specified
                if self.limit_val:
                    docs = docs[:self.limit_val]
                
                return iter(docs)
            
            def __list__(self):
                return list(self)
        
        return Cursor(documents)
    
    def update_one(self, query, update, collection_name=None):
        """Update a single document matching the query."""
        # Determine the collection name
        if not collection_name:
            # Try to infer from the caller
            import inspect
            caller_frame = inspect.currentframe().f_back
            caller_locals = caller_frame.f_locals
            
            for name, value in caller_locals.items():
                if value is self and name.endswith("_collection"):
                    collection_name = name.replace("_collection", "")
                    break
        
        if not collection_name:
            raise ValueError("Collection name not specified")
        
        # Find the document to update
        doc = self.find_one(query, collection_name)
        if not doc:
            # Return a result with modified count 0
            class UpdateResult:
                def __init__(self, modified_count):
                    self.modified_count = modified_count
            
            return UpdateResult(0)
        
        # Apply the update operations
        if "$set" in update:
            for key, value in update["$set"].items():
                doc[key] = value
        
        if "$inc" in update:
            for key, value in update["$inc"].items():
                if key in doc:
                    doc[key] += value
                else:
                    doc[key] = value
        
        # Save the updated document
        self._save_document(collection_name, doc)
        
        # Return a result with modified count 1
        class UpdateResult:
            def __init__(self, modified_count):
                self.modified_count = modified_count
        
        return UpdateResult(1)
    
    def aggregate(self, pipeline, collection_name=None):
        """Perform an aggregation pipeline operation."""
        # This is a simplified implementation that only supports $match and $sample
        # Determine the collection name
        if not collection_name:
            # Try to infer from the caller
            import inspect
            caller_frame = inspect.currentframe().f_back
            caller_locals = caller_frame.f_locals
            
            for name, value in caller_locals.items():
                if value is self and name.endswith("_collection"):
                    collection_name = name.replace("_collection", "")
                    break
        
        if not collection_name:
            raise ValueError("Collection name not specified")
        
        documents = self._list_documents(collection_name)
        
        for stage in pipeline:
            if "$match" in stage:
                documents = self._filter_documents(documents, stage["$match"])
            elif "$sample" in stage:
                sample_size = min(stage["$sample"]["size"], len(documents))
                documents = random.sample(documents, sample_size)
        
        return documents
    
    def list_collection_names(self):
        """List all collection names."""
        return ["prompts", "test_results", "personas", "conversations", "stats", "results"]
    
    def create_index(self, field_name):
        """Create an index (no-op for JSON database)."""
        # This is a no-op for the JSON database
        pass
    
    def close(self):
        """Close the database connection (no-op for JSON database)."""
        # This is a no-op for the JSON database
        print("Closed JSON database connection")

    # Specific methods for the bias testing system
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
        documents = self._list_documents("prompts")
        for doc in documents:
            if doc.get("prompt_text") == prompt_text:
                return doc["_id"]
        
        # Create a new document for the prompt
        document = {
            "prompt_text": prompt_text,
            "attack_type": attack_type,
            "language": language.lower(),
            "times_tested": 0,
            "date_created": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Insert the new prompt
        result = self.insert_one(document, "prompts")
        return result.inserted_id
    
    def get_prompt(self, prompt_id: Union[str, ObjectId]) -> Optional[Dict[str, Any]]:
        """
        Get a prompt from the database.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            Prompt document or None if not found
        """
        if isinstance(prompt_id, ObjectId):
            prompt_id = str(prompt_id)
        
        return self._load_document("prompts", prompt_id)
    
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
        
        documents = self._list_documents("prompts")
        filtered = self._filter_documents(documents, query)
        
        # Randomly sample up to count documents
        sample_size = min(count, len(filtered))
        if sample_size > 0:
            return random.sample(filtered, sample_size)
        return []
    
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
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat()
            }
        }
        
        if attack_type:
            query["attack_type"] = attack_type
        
        if language:
            query["language"] = language.lower()
        
        documents = self._list_documents("prompts")
        filtered = self._filter_documents(documents, query)
        
        # Return up to limit documents
        return filtered[:limit]
    
    def mark_prompt_as_tested(self, prompt_id: Union[str, ObjectId], success: bool) -> bool:
        """
        Mark a prompt as tested.
        
        Args:
            prompt_id: ID of the prompt
            success: Whether the test was successful
            
        Returns:
            True if the update was successful, False otherwise
        """
        if isinstance(prompt_id, ObjectId):
            prompt_id = str(prompt_id)
        
        prompt = self._load_document("prompts", prompt_id)
        if not prompt:
            return False
        
        # Update the prompt
        prompt["times_tested"] = prompt.get("times_tested", 0) + 1
        prompt["last_tested"] = datetime.now().isoformat()
        prompt["last_test_success"] = success
        
        # Save the updated prompt
        self._save_document("prompts", prompt)
        return True
    
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
        if isinstance(prompt_id, ObjectId):
            prompt_id = str(prompt_id)
        
        document = {
            "prompt_id": prompt_id,
            "response": response,
            "success": success,
            "model_info": model_info,
            "execution_time": execution_time,
            "date_tested": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        result = self.insert_one(document, "test_results")
        return result.inserted_id
    
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
            if isinstance(prompt_id, ObjectId):
                prompt_id = str(prompt_id)
            query["prompt_id"] = prompt_id
        
        if success is not None:
            query["success"] = success
        
        documents = self._list_documents("test_results")
        filtered = self._filter_documents(documents, query)
        
        # Sort by date_tested in descending order
        sorted_docs = sorted(filtered, 
                            key=lambda x: x.get("date_tested", ""), 
                            reverse=True)
        
        # Return up to limit documents
        return sorted_docs[:limit]
    
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
        
        result = self.insert_one(persona, "personas")
        return result.inserted_id
    
    def get_persona(self, persona_id: Union[str, ObjectId]) -> Optional[Dict[str, Any]]:
        """
        Get a persona from the database.
        
        Args:
            persona_id: ID of the persona
            
        Returns:
            Persona document or None if not found
        """
        if isinstance(persona_id, ObjectId):
            persona_id = str(persona_id)
        
        return self._load_document("personas", persona_id)
    
    def get_all_personas(self) -> List[Dict[str, Any]]:
        """
        Get all personas from the database.
        
        Returns:
            List of persona documents
        """
        return self._list_documents("personas")
    
    def get_personas_by_attribute(self, attribute: str, value: Any) -> List[Dict[str, Any]]:
        """
        Get personas with a specific attribute value.
        
        Args:
            attribute: Attribute name (e.g., "gender", "age_range", etc.)
            value: Value to match
            
        Returns:
            List of matching persona documents
        """
        query = {attribute: value}
        documents = self._list_documents("personas")
        return self._filter_documents(documents, query)
    
    def store_conversation(self, conversation: Dict[str, Any]) -> str:
        """
        Store a conversation in the database.
        
        Args:
            conversation: Conversation data
            
        Returns:
            ID of the stored conversation
        """
        result = self.insert_one(conversation, "conversations")
        return result.inserted_id
    
    def get_conversation(self, conversation_id: Union[str, ObjectId]) -> Optional[Dict[str, Any]]:
        """
        Get a conversation from the database.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation document or None if not found
        """
        if isinstance(conversation_id, ObjectId):
            conversation_id = str(conversation_id)
        
        # Try direct lookup by ID
        conv = self._load_document("conversations", conversation_id)
        if conv:
            return conv
        
        # Try lookup by conversation_id field
        documents = self._list_documents("conversations")
        for doc in documents:
            if doc.get("conversation_id") == conversation_id:
                return doc
        
        return None
    
    def get_conversations_by_prompt(self, prompt_id: Union[str, ObjectId]) -> List[Dict[str, Any]]:
        """
        Get conversations associated with a prompt.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            List of conversation documents
        """
        if isinstance(prompt_id, ObjectId):
            prompt_id = str(prompt_id)
        
        query = {"prompt_id": prompt_id}
        documents = self._list_documents("conversations")
        return self._filter_documents(documents, query)
    
    def store_analysis_result(self, result: Dict[str, Any]) -> str:
        """
        Store an analysis result in the database.
        
        Args:
            result: Analysis result data
            
        Returns:
            ID of the stored result
        """
        result = self.insert_one(result, "results")
        return result.inserted_id
    
    def get_analysis_results(self, query: Optional[Dict[str, Any]] = None, 
                            limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get analysis results from the database.
        
        Args:
            query: Query to filter results (optional)
            limit: Maximum number of results to return
            
        Returns:
            List of analysis result documents
        """
        documents = self._list_documents("results")
        
        if query:
            filtered = self._filter_documents(documents, query)
        else:
            filtered = documents
        
        # Sort by date in descending order if available
        sorted_docs = sorted(filtered, 
                            key=lambda x: x.get("date", x.get("timestamp", "")), 
                            reverse=True)
        
        # Return up to limit documents
        return sorted_docs[:limit]
