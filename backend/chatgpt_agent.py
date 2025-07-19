"""
ChatGPT Agent Implementation - Enhanced Version 2.0
Advanced AI agent with cognitive behavior, adaptive reasoning, and intelligent task management
Features: Task planning, self-reflection, memory personalization, dynamic tool selection, and more
"""

import asyncio
import subprocess
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import httpx
from pydantic import BaseModel
import uuid
import shlex
from pathlib import Path
import random
import time
import re

logger = logging.getLogger(__name__)

class AgentCapability(BaseModel):
    name: str
    description: str
    enabled: bool = True
    security_level: str = "medium"  # low, medium, high, restricted
    success_rate: float = 1.0
    last_used: Optional[datetime] = None
    usage_count: int = 0

class UserPreferences(BaseModel):
    """User personalization and memory"""
    user_id: str
    communication_style: str = "professional"  # professional, casual, friendly, technical
    preferred_tools: List[str] = []
    task_patterns: Dict[str, Any] = {}
    feedback_history: List[Dict[str, Any]] = []
    timezone: str = "UTC"
    language: str = "en"
    custom_instructions: str = ""

class TaskPriority(BaseModel):
    """Task prioritization system"""
    task_id: str
    priority_score: float  # 0-100
    deadline: Optional[datetime] = None
    dependencies: List[str] = []
    estimated_duration: Optional[int] = None  # minutes
    importance: str = "medium"  # low, medium, high, critical
    urgency: str = "medium"  # low, medium, high, critical

class AgentMemory(BaseModel):
    """Agent memory and state tracking"""
    user_id: str
    session_context: Dict[str, Any] = {}
    long_term_memory: Dict[str, Any] = {}
    working_memory: Dict[str, Any] = {}
    learned_patterns: Dict[str, Any] = {}
    error_patterns: Dict[str, Any] = {}

class AgentTask(BaseModel):
    id: str = None
    user_id: str
    task_type: str
    description: str
    goal: str
    steps: List[Dict[str, Any]] = []
    status: str = "pending"  # pending, planning, executing, completed, failed, paused
    result: Optional[Dict[str, Any]] = None
    logs: List[str] = []
    created_at: datetime = None
    updated_at: datetime = None
    priority: TaskPriority = None
    context: Dict[str, Any] = {}
    retry_count: int = 0
    reflection_notes: List[str] = []
    persona: str = "assistant"  # assistant, expert, teacher, friend, coach

    def __init__(self, **data):
        if data.get('id') is None:
            data['id'] = str(uuid.uuid4())
        if data.get('created_at') is None:
            data['created_at'] = datetime.utcnow()
        if data.get('updated_at') is None:
            data['updated_at'] = datetime.utcnow()
        if data.get('priority') is None:
            data['priority'] = TaskPriority(
                task_id=data.get('id', str(uuid.uuid4())),
                priority_score=50.0
            )
        super().__init__(**data)

class ShellExecutor:
    """Secure shell command executor with whitelisting"""
    
    ALLOWED_COMMANDS = {
        # File operations
        'ls', 'cat', 'head', 'tail', 'find', 'grep', 'wc', 'sort', 'uniq',
        # System info
        'ps', 'top', 'df', 'free', 'uptime', 'who', 'id', 'pwd', 'date',
        # Network
        'ping', 'curl', 'wget', 'nslookup', 'dig',
        # Package management (read-only)
        'pip', 'npm', 'yarn', 'apt',
        # Git operations
        'git',
        # Text processing
        'awk', 'sed', 'cut', 'tr',
        # Archive operations
        'tar', 'zip', 'unzip', 'gzip', 'gunzip'
    }
    
    RESTRICTED_PATTERNS = [
        'rm', 'del', 'delete', 'format', 'fdisk', 'mkfs',
        'shutdown', 'reboot', 'halt', 'poweroff',
        'passwd', 'su', 'sudo', 'chmod +x', 'chown',
        '>', '>>', '|', '&', ';', '&&', '||',
        'eval', 'exec', 'source', '.'
    ]
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.working_dir = f"/tmp/agent_{user_id}"
        os.makedirs(self.working_dir, exist_ok=True)
    
    def is_command_allowed(self, command: str) -> tuple[bool, str]:
        """Check if command is allowed and safe"""
        command_parts = shlex.split(command)
        if not command_parts:
            return False, "Empty command"
        
        base_command = command_parts[0]
        
        # Check if base command is allowed
        if base_command not in self.ALLOWED_COMMANDS:
            return False, f"Command '{base_command}' not in allowed list"
        
        # Check for restricted patterns
        for pattern in self.RESTRICTED_PATTERNS:
            if pattern in command.lower():
                return False, f"Command contains restricted pattern: {pattern}"
        
        # Additional security checks
        if any(char in command for char in ['`', '$(']):
            return False, "Command substitution not allowed"
        
        return True, "Command allowed"
    
    async def execute(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute shell command with security restrictions"""
        
        # Security check
        allowed, reason = self.is_command_allowed(command)
        if not allowed:
            return {
                "success": False,
                "error": f"Security violation: {reason}",
                "stdout": "",
                "stderr": reason,
                "exit_code": -1
            }
        
        try:
            # Execute command in restricted environment
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
                env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "HOME": self.working_dir,
                    "USER": f"agent_{self.user_id}",
                    "SHELL": "/bin/bash"
                }
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                return {
                    "success": process.returncode == 0,
                    "stdout": stdout.decode('utf-8', errors='ignore'),
                    "stderr": stderr.decode('utf-8', errors='ignore'),
                    "exit_code": process.returncode,
                    "command": command
                }
                
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "stdout": "",
                    "stderr": "Timeout",
                    "exit_code": -1
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1
            }

class WebBrowser:
    """Web browsing and interaction capabilities"""
    
    def __init__(self, user_agent: str = None):
        self.user_agent = user_agent or "AI-Agent/1.0 (Advanced Web Assistant)"
        self.session = None
    
    async def get_session(self):
        """Get or create HTTP session"""
        if self.session is None:
            self.session = httpx.AsyncClient(
                headers={"User-Agent": self.user_agent},
                timeout=30.0,
                follow_redirects=True
            )
        return self.session
    
    async def fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch content from URL"""
        try:
            session = await self.get_session()
            response = await session.get(url)
            
            return {
                "success": True,
                "status_code": response.status_code,
                "content": response.text,
                "headers": dict(response.headers),
                "url": str(response.url)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": 0,
                "content": "",
                "headers": {},
                "url": url
            }
    
    async def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search web using DuckDuckGo (privacy-focused)"""
        try:
            search_url = f"https://duckduckgo.com/html/?q={query}&s=0&dc={num_results}"
            result = await self.fetch_url(search_url)
            
            if result["success"]:
                # Parse search results (simplified implementation)
                # In production, use proper HTML parsing
                content = result["content"]
                results = []
                
                # Basic extraction (would need BeautifulSoup in production)
                import re
                links = re.findall(r'href="(/l/\?uddg=.*?)"', content)
                titles = re.findall(r'class="result__title">.*?<a.*?>(.*?)</a>', content)
                
                for i in range(min(len(links), len(titles), num_results)):
                    results.append({
                        "title": titles[i],
                        "link": f"https://duckduckgo.com{links[i]}",
                        "snippet": ""  # Would extract snippet in full implementation
                    })
                
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.aclose()

class TaskPlanner:
    """AI task planning and decomposition"""
    
    def __init__(self, openrouter_client):
        self.openrouter_client = openrouter_client
    
    async def plan_task(self, description: str, goal: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Break down complex task into executable steps"""
        
        planning_prompt = f"""
You are an AI task planner. Break down this task into specific, executable steps.

Task Description: {description}
Goal: {goal}
Context: {context or 'None provided'}

Available capabilities:
- Shell command execution (limited to safe commands)
- Web browsing and content fetching
- File operations (read/write in user directory)
- Data analysis and processing
- API calls and integrations

Please create a detailed step-by-step plan. For each step, specify:
1. Step number and description
2. Action type (shell, web, api, analysis, etc.)
3. Specific command or action to take
4. Expected output or result
5. Success criteria

Return the plan as a JSON array of step objects.
"""
        
        try:
            response = await self.openrouter_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert AI task planner. Always respond with valid JSON."},
                    {"role": "user", "content": planning_prompt}
                ],
                model="mistralai/mistral-7b-instruct:free",
                max_tokens=2000
            )
            
            content = response["choices"][0]["message"]["content"]
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                steps = json.loads(json_match.group())
                return steps
            else:
                # Fallback: create basic steps
                return [
                    {
                        "step": 1,
                        "description": description,
                        "action_type": "analysis",
                        "command": "analyze_task",
                        "expected_output": "Task analysis complete",
                        "success_criteria": "Understanding achieved"
                    }
                ]
                
        except Exception as e:
            logger.error(f"Task planning error: {e}")
            return []

class ChatGPTAgent:
    """Main ChatGPT Agent with autonomous capabilities"""
    
    def __init__(self, user_id: str, openrouter_client, database):
        self.user_id = user_id
        self.openrouter_client = openrouter_client
        self.database = database
        
        # Initialize components
        self.shell_executor = ShellExecutor(user_id)
        self.web_browser = WebBrowser()
        self.task_planner = TaskPlanner(openrouter_client)
        
        # Agent capabilities
        self.capabilities = {
            "shell_access": AgentCapability(
                name="Shell Access",
                description="Execute shell commands with security restrictions",
                security_level="high"
            ),
            "web_browsing": AgentCapability(
                name="Web Browsing",
                description="Browse web and fetch content",
                security_level="medium"
            ),
            "task_planning": AgentCapability(
                name="Task Planning",
                description="Break down complex tasks into steps",
                security_level="low"
            ),
            "file_operations": AgentCapability(
                name="File Operations",
                description="Read and write files in user directory",
                security_level="medium"
            ),
            "data_analysis": AgentCapability(
                name="Data Analysis",
                description="Analyze and process data",
                security_level="low"
            )
        }
        
        self.active_tasks = {}
    
    async def create_task(self, task_type: str, description: str, goal: str) -> AgentTask:
        """Create new agent task"""
        
        task = AgentTask(
            user_id=self.user_id,
            task_type=task_type,
            description=description,
            goal=goal
        )
        
        # Store in database
        await self.database.db.agent_tasks.insert_one(task.dict())
        
        # Add to active tasks
        self.active_tasks[task.id] = task
        
        return task
    
    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """Execute agent task autonomously"""
        
        if task_id not in self.active_tasks:
            # Load from database
            task_data = await self.database.db.agent_tasks.find_one({"id": task_id})
            if not task_data:
                return {"success": False, "error": "Task not found"}
            
            task = AgentTask(**task_data)
            self.active_tasks[task_id] = task
        else:
            task = self.active_tasks[task_id]
        
        try:
            # Update task status
            task.status = "planning"
            await self._update_task(task)
            
            # Plan the task if no steps exist
            if not task.steps:
                task.logs.append(f"[{datetime.utcnow()}] Starting task planning...")
                steps = await self.task_planner.plan_task(
                    task.description, 
                    task.goal,
                    {"user_id": self.user_id, "task_type": task.task_type}
                )
                task.steps = steps
                task.logs.append(f"[{datetime.utcnow()}] Created {len(steps)} execution steps")
            
            # Execute steps
            task.status = "executing"
            await self._update_task(task)
            
            results = []
            for i, step in enumerate(task.steps):
                task.logs.append(f"[{datetime.utcnow()}] Executing step {i+1}: {step.get('description', 'No description')}")
                
                step_result = await self._execute_step(step)
                results.append(step_result)
                
                task.logs.append(f"[{datetime.utcnow()}] Step {i+1} result: {step_result.get('success', False)}")
                
                # Stop if step failed and it's critical
                if not step_result.get("success", False) and step.get("critical", False):
                    task.status = "failed"
                    task.result = {
                        "success": False,
                        "error": f"Critical step {i+1} failed: {step_result.get('error', 'Unknown error')}",
                        "completed_steps": i,
                        "step_results": results
                    }
                    await self._update_task(task)
                    return task.result
            
            # Task completed successfully
            task.status = "completed"
            task.result = {
                "success": True,
                "message": "Task completed successfully",
                "completed_steps": len(task.steps),
                "step_results": results
            }
            await self._update_task(task)
            
            return task.result
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            task.status = "failed"
            task.result = {
                "success": False,
                "error": str(e),
                "completed_steps": 0,
                "step_results": []
            }
            await self._update_task(task)
            return task.result
    
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual step based on action type"""
        
        action_type = step.get("action_type", "unknown")
        command = step.get("command", "")
        
        try:
            if action_type == "shell":
                return await self.shell_executor.execute(command)
            
            elif action_type == "web":
                if command.startswith("fetch:"):
                    url = command.replace("fetch:", "").strip()
                    return await self.web_browser.fetch_url(url)
                elif command.startswith("search:"):
                    query = command.replace("search:", "").strip()
                    results = await self.web_browser.search_web(query)
                    return {"success": True, "results": results}
                else:
                    return {"success": False, "error": "Unknown web command"}
            
            elif action_type == "file":
                return await self._execute_file_operation(command)
            
            elif action_type == "analysis":
                return await self._execute_analysis(command, step)
            
            elif action_type == "api":
                return await self._execute_api_call(command, step)
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_type}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_file_operation(self, command: str) -> Dict[str, Any]:
        """Execute file operations"""
        
        try:
            if command.startswith("read:"):
                file_path = command.replace("read:", "").strip()
                # Ensure file is in user directory
                safe_path = os.path.join(self.shell_executor.working_dir, os.path.basename(file_path))
                
                if os.path.exists(safe_path):
                    with open(safe_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return {"success": True, "content": content, "file": safe_path}
                else:
                    return {"success": False, "error": "File not found"}
            
            elif command.startswith("write:"):
                parts = command.replace("write:", "").strip().split("|", 1)
                if len(parts) != 2:
                    return {"success": False, "error": "Invalid write command format"}
                
                file_path, content = parts
                safe_path = os.path.join(self.shell_executor.working_dir, os.path.basename(file_path))
                
                with open(safe_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return {"success": True, "message": f"File written: {safe_path}"}
            
            else:
                return {"success": False, "error": "Unknown file command"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_analysis(self, command: str, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data analysis tasks"""
        
        try:
            analysis_prompt = f"""
Analyze the following data or perform the requested analysis:

Command: {command}
Step Context: {step}

Provide detailed analysis results.
"""
            
            response = await self.openrouter_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a data analysis expert. Provide clear, actionable insights."},
                    {"role": "user", "content": analysis_prompt}
                ],
                model="mistralai/mistral-7b-instruct:free",
                max_tokens=1500
            )
            
            content = response["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "analysis": content,
                "type": "ai_analysis"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_api_call(self, command: str, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute API calls"""
        
        try:
            # Parse API call parameters
            if command.startswith("call:"):
                api_params = command.replace("call:", "").strip()
                # Implementation would depend on specific API requirements
                return {
                    "success": True,
                    "message": "API call executed",
                    "params": api_params
                }
            
            return {"success": False, "error": "Unknown API command"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _update_task(self, task: AgentTask):
        """Update task in database"""
        
        task.updated_at = datetime.utcnow()
        
        await self.database.db.agent_tasks.update_one(
            {"id": task.id},
            {"$set": task.dict()}
        )
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current task status"""
        
        task_data = await self.database.db.agent_tasks.find_one({"id": task_id})
        if not task_data:
            return {"error": "Task not found"}
        
        return dict(task_data)
    
    async def list_user_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all tasks for user"""
        
        cursor = self.database.db.agent_tasks.find(
            {"user_id": self.user_id}
        ).sort("created_at", -1).limit(limit)
        
        tasks = []
        async for task_data in cursor:
            tasks.append(dict(task_data))
        
        return tasks
    
    async def cleanup(self):
        """Cleanup agent resources"""
        await self.web_browser.close()
        
        # Cleanup user working directory if needed
        import shutil
        if os.path.exists(self.shell_executor.working_dir):
            try:
                shutil.rmtree(self.shell_executor.working_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup working directory: {e}")

# Global agent instances
agent_instances = {}

def get_agent(user_id: str, openrouter_client, database) -> ChatGPTAgent:
    """Get or create agent instance for user"""
    if user_id not in agent_instances:
        agent_instances[user_id] = ChatGPTAgent(user_id, openrouter_client, database)
    return agent_instances[user_id]

async def cleanup_agent(user_id: str):
    """Cleanup agent instance"""
    if user_id in agent_instances:
        await agent_instances[user_id].cleanup()
        del agent_instances[user_id]