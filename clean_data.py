#!/usr/bin/env python
"""
Clean Data Script for Aurora Chatbot Testing

This script deletes all data from MongoDB collections and local JSON files
to provide a fresh start for testing the bias testing system.
"""

import os
import sys
import shutil
from typing import List
from dotenv import load_dotenv

# Add the current directory to the path so we can import from storage
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from storage.database import Database

# Load environment variables
load_dotenv()

def clean_mongodb():
    """Delete all data from MongoDB collections."""
    try:
        db = Database()
        print("Connected to MongoDB successfully.")
        
        # Get the collections
        collections = [
            "personas_collection",
            "prompts_collection", 
            "conversations_collection", 
            "test_results_collection",
            "stats_collection"
        ]
        
        # Delete all documents from each collection
        for collection_name in collections:
            if hasattr(db, collection_name):
                collection = getattr(db, collection_name)
                result = collection.delete_many({})
                print(f"Deleted {result.deleted_count} documents from {collection_name}")
            else:
                print(f"Collection {collection_name} not found")
        
        print("MongoDB data cleaning completed.")
        return True
    except Exception as e:
        print(f"Error cleaning MongoDB: {str(e)}")
        return False

def clean_local_files():
    """Delete all local JSON files and prompt text files."""
    # Define directories to clean JSON files from
    directories = [
        os.path.join("db_files", "personas"),
        os.path.join("db_files", "prompts"),
        os.path.join("db_files", "convos"),
        os.path.join("db_files", "results"),
        os.path.join("db_files", "stats")
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            # Get all JSON files in the directory
            json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
            
            # Delete each JSON file
            for file_name in json_files:
                file_path = os.path.join(directory, file_name)
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")
            
            print(f"Deleted {len(json_files)} files from {directory}")
        else:
            print(f"Directory {directory} does not exist")
    
    # Clean persona prompt text files
    persona_prompts_dir = os.path.join("db_files", "personas", "prompts")
    if os.path.exists(persona_prompts_dir):
        # Get all text files in the persona prompts directory
        persona_txt_files = [f for f in os.listdir(persona_prompts_dir) if f.endswith('.txt')]
        
        # Delete each text file
        for file_name in persona_txt_files:
            file_path = os.path.join(persona_prompts_dir, file_name)
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
        
        print(f"Deleted {len(persona_txt_files)} persona prompt text files from {persona_prompts_dir}")
    else:
        print(f"Directory {persona_prompts_dir} does not exist")
    
    # Clean prompt text files
    prompts_dir = os.path.join("db_files", "results", "prompts")
    if os.path.exists(prompts_dir):
        # Get all text files in the prompts directory
        txt_files = [f for f in os.listdir(prompts_dir) if f.endswith('.txt')]
        
        # Delete each text file
        for file_name in txt_files:
            file_path = os.path.join(prompts_dir, file_name)
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
        
        print(f"Deleted {len(txt_files)} prompt text files from {prompts_dir}")
    else:
        print(f"Directory {prompts_dir} does not exist")
    
    print("Local file cleaning completed.")

def clean_logs(keep_last_n=5):
    """Delete log files, keeping the most recent n files."""
    log_dir = "logs"
    
    if os.path.exists(log_dir):
        # Get all log files
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        
        # Sort by modification time (newest first)
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
        
        # Keep the most recent n files
        files_to_keep = log_files[:keep_last_n]
        files_to_delete = log_files[keep_last_n:]
        
        # Delete older log files
        for file_name in files_to_delete:
            file_path = os.path.join(log_dir, file_name)
            try:
                os.remove(file_path)
                print(f"Deleted log file: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
        
        print(f"Kept {len(files_to_keep)} most recent log files and deleted {len(files_to_delete)} older log files")
    else:
        print(f"Log directory {log_dir} does not exist")

def main():
    """Main function to run the data cleaning process."""
    print("Starting data cleaning process...")
    
    # Ask for confirmation
    confirmation = input("This will delete ALL data from MongoDB and local files. Are you sure? (y/n): ")
    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Clean MongoDB
    mongodb_cleaned = clean_mongodb()
    
    # Clean local files
    clean_local_files()
    
    # Clean log files (keep the 5 most recent)
    clean_logs(keep_last_n=5)
    
    print("\nData cleaning completed successfully.")
    print("You can now run the bias testing system with a fresh start.")

if __name__ == "__main__":
    main()
