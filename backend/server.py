from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Import our modules
from models import (
    User, UserCreate, UserLogin, UserResponse,
    ChatRequest, ChatResponse, ChatMessage,
    ChatSession, ChatSessionCreate, ChatSessionResponse,
    AgentTaskCreate, AgentTask, ModelInfo
)
from auth import create_access_token, verify_token
from database import database, init_admin_user
from openrouter_client import openrouter_client
from chatgpt_agent import get_agent, cleanup_agent

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Create the main app
app = FastAPI(title="AI Multi-Agent Platform", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Security
security = HTTPBearer()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global agent instances tracking
agent_instances = {}

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user."""
    username = verify_token(credentials.credentials)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await database.get_user_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

# Auth endpoints
@api_router.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """Register a new user."""
    try:
        user = await database.create_user(user_data)
        return UserResponse(**user.dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/auth/login")
async def login(user_data: UserLogin):
    """Login user."""
    user = await database.authenticate_user(user_data.username, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse(**user.dict())
    }

@api_router.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(**current_user.dict())

# OpenRouter endpoints
@api_router.get("/models")
async def get_available_models(current_user: User = Depends(get_current_user)):
    """Get available OpenRouter models."""
    try:
        models = await openrouter_client.get_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch models")

@api_router.post("/chat")
async def chat_completion(
    request: ChatRequest, 
    current_user: User = Depends(get_current_user)
):
    """Send chat completion request."""
    try:
        # Check if user has enough credits (unless admin)
        if not current_user.is_admin:
            current_credits = await database.get_user_credits(current_user.id)
            if current_credits <= 0:
                raise HTTPException(status_code=402, detail="Insufficient credits")
        
        # Prepare messages for OpenRouter
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Send request to OpenRouter
        response = await openrouter_client.chat_completion(
            messages=messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Calculate tokens used
        total_tokens = response.get("usage", {}).get("total_tokens", 1)
        
        # Deduct credits (unless admin)
        if not current_user.is_admin:
            await database.update_user_credits(
                current_user.id, 
                -total_tokens, 
                f"Chat completion using {request.model}"
            )
        
        # Return response
        return {
            "id": response.get("id"),
            "content": response["choices"][0]["message"]["content"],
            "model": response.get("model"),
            "usage": response.get("usage"),
            "created_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        if "Insufficient credits" in str(e):
            raise HTTPException(status_code=402, detail="Insufficient credits")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

# Chat session endpoints
@api_router.post("/chat/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    session_data: ChatSessionCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new chat session."""
    session = await database.create_chat_session(
        user_id=current_user.id,
        title=session_data.title,
        model=session_data.model
    )
    
    return ChatSessionResponse(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        model=session.model,
        message_count=len(session.messages),
        created_at=session.created_at,
        updated_at=session.updated_at
    )

@api_router.get("/chat/sessions", response_model=List[ChatSessionResponse])
async def get_chat_sessions(current_user: User = Depends(get_current_user)):
    """Get user's chat sessions."""
    sessions = await database.get_user_chat_sessions(current_user.id)
    return [
        ChatSessionResponse(
            id=session.id,
            user_id=session.user_id,
            title=session.title,
            model=session.model,
            message_count=len(session.messages),
            created_at=session.created_at,
            updated_at=session.updated_at
        )
        for session in sessions
    ]

@api_router.get("/chat/sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get specific chat session."""
    session = await database.get_chat_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    return session

@api_router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete chat session."""
    await database.delete_chat_session(session_id, current_user.id)
    return {"message": "Chat session deleted"}

# Agent endpoints
@api_router.post("/agents/tasks")
async def create_agent_task(
    task_data: AgentTaskCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new agent task."""
    task = await database.create_agent_task(
        user_id=current_user.id,
        agent_type=task_data.agent_type,
        task_description=task_data.task_description
    )
    return task

@api_router.get("/agents/tasks")
async def get_agent_tasks(current_user: User = Depends(get_current_user)):
    """Get user's agent tasks."""
    tasks = await database.get_user_agent_tasks(current_user.id)
    return tasks

@api_router.get("/agents/tasks/{task_id}")
async def get_agent_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get specific agent task."""
    task = await database.get_agent_task(task_id, current_user.id)
    if not task:
        raise HTTPException(status_code=404, detail="Agent task not found")
    return task

# Enhanced ChatGPT Agent endpoints
@api_router.post("/agents/chatgpt/tasks")
async def create_chatgpt_task(
    task_data: dict,
    current_user: User = Depends(get_current_user)
):
    """Create enhanced ChatGPT Agent task with cognitive capabilities."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        
        task = await agent.create_task(
            task_type=task_data.get("task_type", "general"),
            description=task_data.get("description", ""),
            goal=task_data.get("goal", ""),
            persona=task_data.get("persona", "assistant")
        )
        
        return {
            "task_id": task.id, 
            "status": task.status, 
            "message": "Enhanced ChatGPT Agent task created",
            "persona": task.persona,
            "priority_score": task.priority.priority_score if task.priority else 50
        }
        
    except Exception as e:
        logger.error(f"Enhanced ChatGPT Agent task creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent task: {str(e)}")

@api_router.post("/agents/chatgpt/preferences")
async def set_user_preferences(
    preferences: dict,
    current_user: User = Depends(get_current_user)
):
    """Set user preferences for personalized agent behavior."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        result = await agent.set_user_preferences(preferences)
        return result
    except Exception as e:
        logger.error(f"Set preferences error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set preferences: {str(e)}")

@api_router.get("/agents/chatgpt/preferences")
async def get_user_preferences(current_user: User = Depends(get_current_user)):
    """Get current user preferences."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        await agent.initialize_user_context()
        if agent.user_preferences:
            return {"preferences": agent.user_preferences.dict()}
        return {"preferences": None}
    except Exception as e:
        logger.error(f"Get preferences error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get preferences: {str(e)}")

@api_router.get("/agents/chatgpt/tasks/list")
async def list_user_tasks(
    limit: int = 50,
    status_filter: str = None,
    current_user: User = Depends(get_current_user)
):
    """List user tasks with intelligence insights."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        tasks = await agent.list_user_tasks(limit=limit, status_filter=status_filter)
        return {"tasks": tasks}
    except Exception as e:
        logger.error(f"List tasks error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@api_router.post("/agents/chatgpt/tasks/{task_id}/pause")
async def pause_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Pause task execution."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        result = await agent.pause_task(task_id)
        return result
    except Exception as e:
        logger.error(f"Pause task error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause task: {str(e)}")

@api_router.post("/agents/chatgpt/tasks/{task_id}/resume")
async def resume_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Resume paused task."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        result = await agent.resume_task(task_id)
        return result
    except Exception as e:
        logger.error(f"Resume task error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume task: {str(e)}")

@api_router.post("/agents/schedule/recurring")
async def schedule_recurring_task(
    schedule_data: dict,
    current_user: User = Depends(get_current_user)
):
    """Schedule a recurring task."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        
        # Import TaskScheduler here to avoid circular imports
        from chatgpt_agent import TaskScheduler
        
        if not hasattr(agent, 'scheduler'):
            agent.scheduler = TaskScheduler(agent)
        
        schedule_id = await agent.scheduler.schedule_recurring_task(
            task_type=schedule_data.get("task_type", "general"),
            description=schedule_data.get("description", ""),
            goal=schedule_data.get("goal", ""),
            interval_minutes=schedule_data.get("interval_minutes", 60),
            max_iterations=schedule_data.get("max_iterations")
        )
        
        return {"schedule_id": schedule_id, "message": "Recurring task scheduled"}
        
    except Exception as e:
        logger.error(f"Schedule recurring task error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule task: {str(e)}")

@api_router.delete("/agents/schedule/{schedule_id}")
async def cancel_scheduled_task(
    schedule_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a scheduled task."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        
        if hasattr(agent, 'scheduler'):
            success = await agent.scheduler.cancel_scheduled_task(schedule_id)
            if success:
                return {"message": "Scheduled task cancelled"}
            else:
                return {"error": "Scheduled task not found"}
        
        return {"error": "No scheduler found"}
        
    except Exception as e:
        logger.error(f"Cancel scheduled task error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")

@api_router.get("/agents/chatgpt/analytics")
async def get_agent_analytics(current_user: User = Depends(get_current_user)):
    """Get agent analytics and insights."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        
        # Get task statistics
        all_tasks = await agent.list_user_tasks(limit=1000)
        
        analytics = {
            "total_tasks": len(all_tasks),
            "completed_tasks": len([t for t in all_tasks if t["status"] == "completed"]),
            "failed_tasks": len([t for t in all_tasks if t["status"] == "failed"]),
            "active_tasks": len([t for t in all_tasks if t["status"] in ["executing", "planning"]]),
            "average_priority": sum(t["priority_score"] for t in all_tasks) / len(all_tasks) if all_tasks else 0,
            "success_rate": len([t for t in all_tasks if t["status"] == "completed"]) / len(all_tasks) if all_tasks else 0,
            "capabilities": {name: cap.dict() for name, cap in agent.capabilities.items()},
            "tool_success_rates": agent.tool_selector.tool_success_rates if hasattr(agent, 'tool_selector') else {}
        }
        
        return {"analytics": analytics}
        
    except Exception as e:
        logger.error(f"Get analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@api_router.post("/agents/chatgpt/reflect")
async def trigger_reflection(
    reflection_data: dict,
    current_user: User = Depends(get_current_user)
):
    """Trigger self-reflection on a specific topic or task."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        
        # Create a mock task for reflection
        from chatgpt_agent import AgentTask
        mock_task = AgentTask(
            user_id=current_user.id,
            task_type="reflection",
            description=reflection_data.get("topic", "General reflection"),
            goal="Learn and improve from experience"
        )
        
        reflection = await agent.reflection_engine.reflect_on_failure(
            mock_task,
            reflection_data.get("situation", "General situation"),
            reflection_data.get("context", {})
        )
        
        return {"reflection": reflection}
        
    except Exception as e:
        logger.error(f"Trigger reflection error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger reflection: {str(e)}")

@api_router.post("/agents/chatgpt/tasks/{task_id}/execute")
async def execute_chatgpt_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Execute ChatGPT Agent task autonomously."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        result = await agent.execute_task(task_id)
        
        return result
        
    except Exception as e:
        logger.error(f"ChatGPT Agent execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute agent task: {str(e)}")

@api_router.get("/agents/chatgpt/tasks/{task_id}/status")
async def get_chatgpt_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get ChatGPT Agent task status."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        status = await agent.get_task_status(task_id)
        
        return status
        
    except Exception as e:
        logger.error(f"ChatGPT Agent status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@api_router.post("/agents/shell/execute")
async def execute_shell_command(
    command_data: dict,
    current_user: User = Depends(get_current_user)
):
    """Execute shell command with security restrictions."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        command = command_data.get("command", "")
        
        if not command:
            raise HTTPException(status_code=400, detail="Command is required")
        
        result = await agent.shell_executor.execute(command)
        
        return result
        
    except Exception as e:
        logger.error(f"Shell execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute shell command: {str(e)}")

@api_router.post("/agents/web/browse")
async def browse_web(
    browse_data: dict,
    current_user: User = Depends(get_current_user)
):
    """Browse web and fetch content."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        url = browse_data.get("url", "")
        
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        result = await agent.web_browser.fetch_url(url)
        
        return result
        
    except Exception as e:
        logger.error(f"Web browsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to browse web: {str(e)}")

@api_router.post("/agents/web/search")
async def search_web(
    search_data: dict,
    current_user: User = Depends(get_current_user)
):
    """Search web using privacy-focused search engine."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        query = search_data.get("query", "")
        num_results = search_data.get("num_results", 5)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        results = await agent.web_browser.search_web(query, num_results)
        
        return {"success": True, "results": results}
        
    except Exception as e:
        logger.error(f"Web search error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search web: {str(e)}")

@api_router.get("/agents/capabilities")
async def get_agent_capabilities(current_user: User = Depends(get_current_user)):
    """Get available agent capabilities."""
    try:
        agent = get_agent(current_user.id, openrouter_client, database)
        capabilities = {name: cap.dict() for name, cap in agent.capabilities.items()}
        
        return {"capabilities": capabilities}
        
    except Exception as e:
        logger.error(f"Agent capabilities error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")

# Admin endpoints
@api_router.get("/admin/users")
async def get_all_users(current_user: User = Depends(get_current_user)):
    """Get all users (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    cursor = database.db.users.find({}, {"password_hash": 0})
    users = []
    async for user_data in cursor:
        users.append(UserResponse(**user_data))
    return users

@api_router.post("/admin/users/{user_id}/credits")
async def update_user_credits_admin(
    user_id: str,
    credits: int,
    current_user: User = Depends(get_current_user)
):
    """Update user credits (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    await database.update_user_credits(
        user_id,
        credits,
        f"Credits updated by admin {current_user.username}"
    )
    return {"message": f"Credits updated for user {user_id}"}

# Health check
@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@api_router.get("/")
async def root():
    return {"message": "AI Multi-Agent Platform API", "version": "1.0.0"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Starting AI Multi-Agent Platform...")
    await init_admin_user()
    logger.info("Application started successfully!")

# Global agent instances tracking
from chatgpt_agent import agent_instances

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    # Cleanup all agent instances
    for user_id in list(agent_instances.keys()):
        await cleanup_agent(user_id)
    
    await database.close()
    logger.info("Application shutdown complete.")
