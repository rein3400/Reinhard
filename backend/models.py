from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

# User Models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: EmailStr
    is_admin: bool = False
    credits: int = 100  # Default credits for new users
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_admin: bool
    credits: int
    created_at: datetime
    last_login: Optional[datetime] = None

# OpenRouter Models
class ChatMessage(BaseModel):
    role: str  # system, user, assistant
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "mistralai/mistral-7b-instruct:free"
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0

class ChatResponse(BaseModel):
    id: str
    content: str
    model: str
    usage: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Chat Session Models
class ChatSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str = "New Chat"
    messages: List[ChatMessage] = []
    model: str = "mistralai/mistral-7b-instruct:free"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ChatSessionCreate(BaseModel):
    title: Optional[str] = "New Chat"
    model: Optional[str] = "mistralai/mistral-7b-instruct:free"

class ChatSessionResponse(BaseModel):
    id: str
    user_id: str
    title: str
    model: str
    message_count: int
    created_at: datetime
    updated_at: datetime

# Credit Transaction Models
class CreditTransaction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    amount: int  # positive for credits added, negative for credits used
    description: str
    session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

# AI Agent Models
class AgentTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    agent_type: str  # browser, document, search, presentation
    task_description: str
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    credits_used: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class AgentTaskCreate(BaseModel):
    agent_type: str
    task_description: str

# Model Info
class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    context_length: int
    pricing: Dict[str, float]
    provider: str