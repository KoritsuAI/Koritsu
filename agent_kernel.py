"""
Main entry point for the Agent Kernel application.

This script initializes the environment from .env file and runs the Hermes agent
with a unique session identifier.
"""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import the Hermes agent and UUID generation
from agents import hermes
from uuid import uuid4
        
# Generate a unique session identifier
uuid = str(uuid4())

# Start the Hermes agent with the generated UUID
hermes.hermes(uuid)