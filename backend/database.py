from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, List, Dict, Any
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from models import User, UserCreate, ChatSession, CreditTransaction, AgentTask
from auth import get_password_hash, verify_password
import logging

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.mongo_url = os.environ['MONGO_URL']
        self.db_name = os.environ['DB_NAME']
        self.client = AsyncIOMotorClient(self.mongo_url)
        self.db = self.client[self.db_name]
    
    async def close(self):
        """Close database connection."""
        self.client.close()
    
    # User operations
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        # Check if user already exists
        existing_user = await self.get_user_by_username(user_data.username)
        if existing_user:
            raise ValueError("Username already exists")
        
        existing_email = await self.get_user_by_email(user_data.email)
        if existing_email:
            raise ValueError("Email already exists")
        
        # Create user
        user = User(
            username=user_data.username,
            email=user_data.email,
            credits=1000  # Start with 1000 credits
        )
        user_dict = user.dict()
        user_dict["password_hash"] = get_password_hash(user_data.password)
        
        await self.db.users.insert_one(user_dict)
        return user
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        user_data = await self.db.users.find_one({"username": username})
        if user_data:
            user_data.pop("password_hash", None)  # Remove password hash
            return User(**user_data)
        return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        user_data = await self.db.users.find_one({"email": email})
        if user_data:
            user_data.pop("password_hash", None)
            return User(**user_data)
        return None
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user login."""
        user_data = await self.db.users.find_one({"username": username})
        if not user_data:
            return None
        
        if not verify_password(password, user_data["password_hash"]):
            return None
        
        # Update last login
        await self.db.users.update_one(
            {"username": username},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        user_data.pop("password_hash", None)
        return User(**user_data)
    
    async def update_user_credits(self, user_id: str, credit_change: int, description: str = "", session_id: str = None):
        """Update user credits and log transaction."""
        # Update user credits
        result = await self.db.users.update_one(
            {"id": user_id},
            {"$inc": {"credits": credit_change}}
        )
        
        if result.modified_count == 0:
            raise ValueError("User not found")
        
        # Log transaction
        transaction = CreditTransaction(
            user_id=user_id,
            amount=credit_change,
            description=description,
            session_id=session_id
        )
        await self.db.credit_transactions.insert_one(transaction.dict())
    
    async def get_user_credits(self, user_id: str) -> int:
        """Get current user credits."""
        user = await self.db.users.find_one({"id": user_id}, {"credits": 1})
        return user["credits"] if user else 0
    
    # Chat session operations
    async def create_chat_session(self, user_id: str, title: str = "New Chat", model: str = None) -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(
            user_id=user_id,
            title=title,
            model=model or "mistralai/mistral-7b-instruct:free"
        )
        await self.db.chat_sessions.insert_one(session.dict())
        return session
    
    async def get_chat_session(self, session_id: str, user_id: str) -> Optional[ChatSession]:
        """Get chat session by ID."""
        session_data = await self.db.chat_sessions.find_one({
            "id": session_id,
            "user_id": user_id
        })
        if session_data:
            return ChatSession(**session_data)
        return None
    
    async def get_user_chat_sessions(self, user_id: str, limit: int = 50) -> List[ChatSession]:
        """Get user's chat sessions."""
        cursor = self.db.chat_sessions.find(
            {"user_id": user_id}
        ).sort("updated_at", -1).limit(limit)
        
        sessions = []
        async for session_data in cursor:
            sessions.append(ChatSession(**session_data))
        return sessions
    
    async def update_chat_session(self, session_id: str, user_id: str, updates: Dict[str, Any]):
        """Update chat session."""
        updates["updated_at"] = datetime.utcnow()
        await self.db.chat_sessions.update_one(
            {"id": session_id, "user_id": user_id},
            {"$set": updates}
        )
    
    async def delete_chat_session(self, session_id: str, user_id: str):
        """Delete chat session."""
        await self.db.chat_sessions.delete_one({
            "id": session_id,
            "user_id": user_id
        })
    
    # Agent task operations
    async def create_agent_task(self, user_id: str, agent_type: str, task_description: str) -> AgentTask:
        """Create a new agent task."""
        task = AgentTask(
            user_id=user_id,
            agent_type=agent_type,
            task_description=task_description
        )
        await self.db.agent_tasks.insert_one(task.dict())
        return task
    
    async def get_agent_task(self, task_id: str, user_id: str) -> Optional[AgentTask]:
        """Get agent task by ID."""
        task_data = await self.db.agent_tasks.find_one({
            "id": task_id,
            "user_id": user_id
        })
        if task_data:
            return AgentTask(**task_data)
        return None
    
    async def update_agent_task(self, task_id: str, updates: Dict[str, Any]):
        """Update agent task."""
        updates["updated_at"] = datetime.utcnow()
        await self.db.agent_tasks.update_one(
            {"id": task_id},
            {"$set": updates}
        )
    
    async def get_user_agent_tasks(self, user_id: str, limit: int = 50) -> List[AgentTask]:
        """Get user's agent tasks."""
        cursor = self.db.agent_tasks.find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(limit)
        
        tasks = []
        async for task_data in cursor:
            tasks.append(AgentTask(**task_data))
        return tasks

# Global database instance
database = Database()

async def init_admin_user():
    """Initialize admin user if not exists."""
    try:
        admin_user = await database.get_user_by_username("Admin")
        if not admin_user:
            # Create admin user
            admin_data = UserCreate(
                username="Admin",
                email="admin@aiplatform.com",
                password="Superd1ck"
            )
            admin_user = await database.create_user(admin_data)
            
            # Set as admin with unlimited credits
            await database.db.users.update_one(
                {"id": admin_user.id},
                {"$set": {"is_admin": True, "credits": 999999999}}
            )
            logger.info("Admin user created successfully")
        else:
            # Ensure admin has unlimited credits
            await database.db.users.update_one(
                {"username": "Admin"},
                {"$set": {"is_admin": True, "credits": 999999999}}
            )
            logger.info("Admin user verified")
    except Exception as e:
        logger.error(f"Error initializing admin user: {e}")