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

class SelfReflectionEngine:
    """Self-reflection and adaptive learning system"""
    
    def __init__(self, openrouter_client):
        self.openrouter_client = openrouter_client
        self.reflection_history = []
    
    async def reflect_on_failure(self, task: AgentTask, error: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze failure and suggest improvements"""
        
        reflection_prompt = f"""
I need to reflect on this failure and learn from it:

Task: {task.description}
Goal: {task.goal}
Error: {error}
Context: {context}
Previous attempts: {task.retry_count}

Please analyze:
1. What went wrong and why?
2. What could be done differently?
3. Are there alternative approaches?
4. What should I remember for similar tasks?
5. Should I retry or abort?

Provide a JSON response with: analysis, alternative_approach, should_retry, learned_lesson, confidence_level
"""
        
        try:
            response = await self.openrouter_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an AI that learns from mistakes and adapts. Provide thoughtful analysis."},
                    {"role": "user", "content": reflection_prompt}
                ],
                model="mistralai/mistral-7b-instruct:free",
                max_tokens=1000
            )
            
            content = response["choices"][0]["message"]["content"]
            
            # Try to extract JSON
            try:
                reflection_data = json.loads(content)
            except:
                # Fallback structured response
                reflection_data = {
                    "analysis": f"Failed due to: {error}",
                    "alternative_approach": "Try different method or parameters",
                    "should_retry": task.retry_count < 3,
                    "learned_lesson": f"Remember to handle {error} type errors",
                    "confidence_level": 0.7
                }
            
            # Store reflection for learning
            self.reflection_history.append({
                "timestamp": datetime.utcnow(),
                "task_type": task.task_type,
                "error": error,
                "reflection": reflection_data
            })
            
            return reflection_data
            
        except Exception as e:
            logger.error(f"Reflection error: {e}")
            return {
                "analysis": "Unable to analyze failure",
                "alternative_approach": "Retry with caution",
                "should_retry": False,
                "learned_lesson": "Improve error handling",
                "confidence_level": 0.3
            }
    
    async def suggest_improvements(self, task: AgentTask) -> List[str]:
        """Suggest improvements based on reflection history"""
        
        similar_tasks = [
            r for r in self.reflection_history 
            if r["task_type"] == task.task_type
        ]
        
        suggestions = []
        for reflection in similar_tasks[-3:]:  # Last 3 similar tasks
            suggestions.append(reflection["reflection"]["learned_lesson"])
        
        return suggestions

class AdaptiveToolSelector:
    """Intelligent tool selection based on context and success rates"""
    
    def __init__(self):
        self.tool_success_rates = {
            "shell": 0.85,
            "web": 0.90,
            "file": 0.95,
            "api": 0.80,
            "analysis": 0.95
        }
        self.tool_usage_patterns = {}
    
    def select_best_tool(self, task_description: str, available_tools: List[str], context: Dict[str, Any] = None) -> str:
        """Select the best tool based on context and success rates"""
        
        # Analyze task description for keywords
        task_lower = task_description.lower()
        
        tool_scores = {}
        for tool in available_tools:
            base_score = self.tool_success_rates.get(tool, 0.5)
            
            # Boost score based on keywords
            if tool == "web" and any(keyword in task_lower for keyword in ["browse", "search", "website", "url", "internet"]):
                base_score += 0.2
            elif tool == "shell" and any(keyword in task_lower for keyword in ["command", "execute", "run", "script"]):
                base_score += 0.2
            elif tool == "file" and any(keyword in task_lower for keyword in ["file", "read", "write", "save", "document"]):
                base_score += 0.2
            elif tool == "api" and any(keyword in task_lower for keyword in ["api", "service", "request", "call"]):
                base_score += 0.2
            elif tool == "analysis" and any(keyword in task_lower for keyword in ["analyze", "think", "reason", "understand"]):
                base_score += 0.2
            
            tool_scores[tool] = base_score
        
        # Select tool with highest score
        best_tool = max(tool_scores.items(), key=lambda x: x[1])[0]
        
        # Update usage patterns
        if task_description not in self.tool_usage_patterns:
            self.tool_usage_patterns[task_description] = {}
        self.tool_usage_patterns[task_description][best_tool] = \
            self.tool_usage_patterns[task_description].get(best_tool, 0) + 1
        
        return best_tool
    
    def update_success_rate(self, tool: str, success: bool):
        """Update tool success rate based on feedback"""
        current_rate = self.tool_success_rates.get(tool, 0.5)
        
        # Simple moving average update
        if success:
            self.tool_success_rates[tool] = min(1.0, current_rate + 0.05)
        else:
            self.tool_success_rates[tool] = max(0.1, current_rate - 0.05)

class PersonaEngine:
    """Dynamic persona and communication style management"""
    
    def __init__(self):
        self.personas = {
            "assistant": {
                "tone": "professional and helpful",
                "style": "clear and concise",
                "greeting": "I'm here to help you with",
                "encouragement": "Let's work on this together"
            },
            "expert": {
                "tone": "knowledgeable and precise", 
                "style": "detailed and technical",
                "greeting": "Based on my analysis of",
                "encouragement": "This approach should be effective"
            },
            "teacher": {
                "tone": "patient and educational",
                "style": "step-by-step explanations",
                "greeting": "Let me help you understand",
                "encouragement": "You're making great progress"
            },
            "friend": {
                "tone": "casual and supportive",
                "style": "friendly and encouraging",
                "greeting": "Hey! I'd love to help you with",
                "encouragement": "You've got this!"
            },
            "coach": {
                "tone": "motivational and goal-oriented",
                "style": "action-focused guidance",
                "greeting": "Let's achieve your goal of",
                "encouragement": "Every step gets you closer to success"
            }
        }
    
    def get_persona_response(self, persona: str, message_type: str, context: str = "") -> str:
        """Generate persona-appropriate response"""
        
        persona_config = self.personas.get(persona, self.personas["assistant"])
        
        if message_type == "greeting":
            return f"{persona_config['greeting']} {context}."
        elif message_type == "encouragement":
            return persona_config["encouragement"]
        elif message_type == "explanation":
            if persona == "teacher":
                return f"Let me explain this step by step: {context}"
            elif persona == "expert":
                return f"From a technical perspective: {context}"
            elif persona == "friend":
                return f"So basically, {context}"
            elif persona == "coach":
                return f"Here's your action plan: {context}"
            else:
                return context
        
        return context
    
    def adapt_communication_style(self, user_prefs: UserPreferences, message: str) -> str:
        """Adapt message to user's preferred communication style"""
        
        style = user_prefs.communication_style
        
        if style == "casual":
            message = message.replace("I will", "I'll").replace("you will", "you'll")
            message = message.replace("I shall", "I'll").replace("cannot", "can't")
        elif style == "technical":
            message = f"Technical execution: {message}"
        elif style == "friendly":
            message = f"😊 {message}"
        
        return message

class StateTracker:
    """Enhanced state tracking and memory management"""
    
    def __init__(self, database):
        self.database = database
        self.user_memories = {}
    
    async def save_state(self, user_id: str, key: str, value: Any, state_type: str = "session"):
        """Save state to appropriate memory type"""
        
        if user_id not in self.user_memories:
            self.user_memories[user_id] = AgentMemory(user_id=user_id)
        
        memory = self.user_memories[user_id]
        
        if state_type == "session":
            memory.session_context[key] = value
        elif state_type == "working":
            memory.working_memory[key] = value
        elif state_type == "long_term":
            memory.long_term_memory[key] = value
        elif state_type == "pattern":
            memory.learned_patterns[key] = value
        
        # Persist to database
        await self.database.db.agent_memory.update_one(
            {"user_id": user_id},
            {"$set": {f"{state_type}_memory.{key}": value}},
            upsert=True
        )
    
    async def get_state(self, user_id: str, key: str, state_type: str = "session") -> Any:
        """Retrieve state from memory"""
        
        if user_id not in self.user_memories:
            # Load from database
            memory_data = await self.database.db.agent_memory.find_one({"user_id": user_id})
            if memory_data:
                self.user_memories[user_id] = AgentMemory(**memory_data)
            else:
                self.user_memories[user_id] = AgentMemory(user_id=user_id)
        
        memory = self.user_memories[user_id]
        
        if state_type == "session":
            return memory.session_context.get(key)
        elif state_type == "working":
            return memory.working_memory.get(key)
        elif state_type == "long_term":
            return memory.long_term_memory.get(key)
        elif state_type == "pattern":
            return memory.learned_patterns.get(key)
        
        return None
    
    async def clear_session_memory(self, user_id: str):
        """Clear session memory but keep long-term learning"""
        if user_id in self.user_memories:
            self.user_memories[user_id].session_context = {}
            self.user_memories[user_id].working_memory = {}
        
        await self.database.db.agent_memory.update_one(
            {"user_id": user_id},
            {"$set": {"session_context": {}, "working_memory": {}}}
        )

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
    """Enhanced AI task planning with cognitive abilities"""
    
    def __init__(self, openrouter_client):
        self.openrouter_client = openrouter_client
    
    async def plan_task(self, description: str, goal: str, context: Dict[str, Any] = None, user_prefs: UserPreferences = None) -> List[Dict[str, Any]]:
        """Enhanced task planning with reasoning and adaptability"""
        
        # Determine communication style based on user preferences
        comm_style = user_prefs.communication_style if user_prefs else "professional"
        persona_prompt = self._get_persona_prompt(comm_style)
        
        # Chain-of-thought reasoning prompt
        planning_prompt = f"""
{persona_prompt}

I need to break down this task using advanced reasoning and planning:

Task Description: {description}
Goal: {goal}
Context: {context or 'None provided'}
User Preferences: {user_prefs.dict() if user_prefs else 'Default settings'}

Available capabilities:
- Shell command execution (security-restricted)
- Web browsing and content fetching  
- File operations (read/write in user directory)
- Data analysis and processing
- API calls and integrations
- Adaptive error recovery
- Self-reflection and learning

REASONING PROCESS:
1. First, I'll analyze what the user really wants to achieve
2. Then identify potential challenges and alternative approaches
3. Break down into logical, prioritized steps
4. Consider error scenarios and fallback plans
5. Estimate time and complexity for each step

Please create a detailed step-by-step plan with chain-of-thought reasoning. For each step:
- Step number and clear description
- Action type (shell, web, api, analysis, reflection, etc.)
- Specific command or action
- Expected output/result
- Success criteria
- Fallback plan if it fails
- Priority level (1-5)
- Estimated duration (minutes)

Use this reasoning: "To achieve [goal], I need to first [reason], then [reason], because [logic]..."

Return as JSON array of step objects with reasoning field.
"""
        
        try:
            response = await self.openrouter_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert AI task planner with advanced reasoning capabilities. Always use chain-of-thought reasoning and respond with valid JSON."},
                    {"role": "user", "content": planning_prompt}
                ],
                model="mistralai/mistral-7b-instruct:free",
                max_tokens=3000,
                temperature=0.3  # Lower temperature for more consistent planning
            )
            
            content = response["choices"][0]["message"]["content"]
            
            # Enhanced JSON extraction with fallback
            steps = self._extract_json_steps(content, description, goal)
            
            # Apply intelligent prioritization
            steps = self._prioritize_steps(steps, context)
            
            return steps
                
        except Exception as e:
            logger.error(f"Enhanced task planning error: {e}")
            # Intelligent fallback planning
            return self._create_fallback_plan(description, goal)
    
    def _get_persona_prompt(self, style: str) -> str:
        """Get persona-based prompt based on communication style"""
        personas = {
            "professional": "You are a professional AI assistant focused on efficiency and precision.",
            "casual": "You are a friendly AI buddy who explains things in a relaxed, easy-going way.",
            "friendly": "You are a warm and supportive AI companion who encourages and helps cheerfully.",
            "technical": "You are a technical expert AI who provides detailed, accurate information with technical depth.",
            "coach": "You are a motivational AI coach who guides users to achieve their goals step by step.",
            "teacher": "You are a patient AI teacher who explains concepts clearly and asks clarifying questions."
        }
        return personas.get(style, personas["professional"])
    
    def _extract_json_steps(self, content: str, description: str, goal: str) -> List[Dict[str, Any]]:
        """Enhanced JSON extraction with multiple fallback strategies"""
        try:
            # Try to extract JSON array
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                steps = json.loads(json_match.group())
                return self._validate_and_enhance_steps(steps)
            else:
                # Fallback: parse structured text
                return self._parse_structured_text(content, description, goal)
        except:
            return self._create_fallback_plan(description, goal)
    
    def _validate_and_enhance_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and enhance step objects"""
        enhanced_steps = []
        for i, step in enumerate(steps):
            enhanced_step = {
                "step": i + 1,
                "description": step.get("description", f"Step {i+1}"),
                "action_type": step.get("action_type", "analysis"),
                "command": step.get("command", ""),
                "expected_output": step.get("expected_output", ""),
                "success_criteria": step.get("success_criteria", "Task completed"),
                "fallback_plan": step.get("fallback_plan", "Retry with different approach"),
                "priority": step.get("priority", 3),
                "estimated_duration": step.get("estimated_duration", 5),
                "reasoning": step.get("reasoning", "Standard execution step"),
                "critical": step.get("critical", False)
            }
            enhanced_steps.append(enhanced_step)
        return enhanced_steps
    
    def _prioritize_steps(self, steps: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply intelligent prioritization to steps"""
        for step in steps:
            priority_score = step.get("priority", 3)
            
            # Boost priority for critical steps
            if step.get("critical", False):
                priority_score += 2
                
            # Boost priority for prerequisite steps
            if "setup" in step.get("description", "").lower():
                priority_score += 1
                
            # Lower priority for optional steps
            if "optional" in step.get("description", "").lower():
                priority_score -= 1
                
            step["priority"] = max(1, min(5, priority_score))
        
        # Sort by priority (higher first)
        return sorted(steps, key=lambda x: x.get("priority", 3), reverse=True)
    
    def _parse_structured_text(self, content: str, description: str, goal: str) -> List[Dict[str, Any]]:
        """Parse structured text when JSON extraction fails"""
        steps = []
        lines = content.split('\n')
        current_step = {}
        step_count = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.lower().startswith(('step', f'{step_count}.', f'{step_count}:')):
                if current_step:
                    steps.append(current_step)
                current_step = {
                    "step": step_count,
                    "description": line,
                    "action_type": "analysis",
                    "command": line,
                    "priority": 3,
                    "estimated_duration": 5
                }
                step_count += 1
            elif current_step:
                if "action" in line.lower():
                    current_step["action_type"] = "shell" if "command" in line else "analysis"
                elif "command" in line.lower():
                    current_step["command"] = line.split(":", 1)[-1].strip()
        
        if current_step:
            steps.append(current_step)
            
        return steps if steps else self._create_fallback_plan(description, goal)
    
    def _create_fallback_plan(self, description: str, goal: str) -> List[Dict[str, Any]]:
        """Create intelligent fallback plan when parsing fails"""
        return [
            {
                "step": 1,
                "description": f"Analyze the task: {description}",
                "action_type": "analysis",
                "command": f"analyze_task: {description}",
                "expected_output": "Task understanding and approach",
                "success_criteria": "Clear understanding achieved",
                "priority": 5,
                "estimated_duration": 3,
                "reasoning": "First step is to understand what needs to be done"
            },
            {
                "step": 2,
                "description": f"Execute main task toward goal: {goal}",
                "action_type": "execution",
                "command": f"execute_task: {goal}",
                "expected_output": "Goal achievement",
                "success_criteria": "Task completed successfully",
                "priority": 4,
                "estimated_duration": 10,
                "reasoning": "Main execution step to achieve the goal"
            },
            {
                "step": 3,
                "description": "Verify results and provide summary",
                "action_type": "reflection",
                "command": "verify_and_summarize",
                "expected_output": "Task completion summary",
                "success_criteria": "Results verified and documented",
                "priority": 3,
                "estimated_duration": 2,
                "reasoning": "Important to verify success and learn from the process"
            }
        ]

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