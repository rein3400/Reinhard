#!/usr/bin/env python3
"""
Backend API Testing Script for AI Platform
Tests authentication, OpenRouter integration, chat sessions, and credit system
"""

import asyncio
import httpx
import json
import sys
from typing import Dict, Any, Optional
import uuid

# Backend URL from environment
BACKEND_URL = "https://5d5824b1-0713-4290-9cad-02d7025a906b.preview.emergentagent.com/api"

class BackendTester:
    def __init__(self):
        self.base_url = BACKEND_URL
        self.admin_token = None
        self.test_user_token = None
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   Details: {details}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
        
    async def test_health_check(self):
        """Test basic health check endpoint"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    self.log_test("Health Check", True, f"Status: {data.get('status')}")
                    return True
                else:
                    self.log_test("Health Check", False, f"Status code: {response.status_code}")
                    return False
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False
    
    async def test_admin_login(self):
        """Test admin login with provided credentials"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                login_data = {
                    "username": "Admin",
                    "password": "Superd1ck"
                }
                response = await client.post(
                    f"{self.base_url}/auth/login",
                    json=login_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.admin_token = data.get("access_token")
                    user_info = data.get("user", {})
                    is_admin = user_info.get("is_admin", False)
                    credits = user_info.get("credits", 0)
                    
                    self.log_test(
                        "Admin Login", 
                        True, 
                        f"Admin: {is_admin}, Credits: {credits}, Token received: {bool(self.admin_token)}"
                    )
                    return True
                else:
                    self.log_test("Admin Login", False, f"Status: {response.status_code}, Response: {response.text}")
                    return False
        except Exception as e:
            self.log_test("Admin Login", False, f"Exception: {str(e)}")
            return False
    
    async def test_get_current_user(self):
        """Test getting current user info with token"""
        if not self.admin_token:
            self.log_test("Get Current User", False, "No admin token available")
            return False
            
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"Authorization": f"Bearer {self.admin_token}"}
                response = await client.get(
                    f"{self.base_url}/auth/me",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    username = data.get("username")
                    is_admin = data.get("is_admin")
                    credits = data.get("credits")
                    
                    self.log_test(
                        "Get Current User", 
                        True, 
                        f"User: {username}, Admin: {is_admin}, Credits: {credits}"
                    )
                    return True
                else:
                    self.log_test("Get Current User", False, f"Status: {response.status_code}, Response: {response.text}")
                    return False
        except Exception as e:
            self.log_test("Get Current User", False, f"Exception: {str(e)}")
            return False
    
    async def test_user_registration(self):
        """Test registering a new user"""
        try:
            # Generate unique user data
            unique_id = str(uuid.uuid4())[:8]
            user_data = {
                "username": f"testuser_{unique_id}",
                "email": f"test_{unique_id}@example.com",
                "password": "TestPassword123!"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/auth/register",
                    json=user_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    username = data.get("username")
                    credits = data.get("credits")
                    
                    self.log_test(
                        "User Registration", 
                        True, 
                        f"User: {username}, Initial Credits: {credits}"
                    )
                    
                    # Try to login with the new user
                    login_response = await client.post(
                        f"{self.base_url}/auth/login",
                        json={"username": user_data["username"], "password": user_data["password"]}
                    )
                    
                    if login_response.status_code == 200:
                        login_data = login_response.json()
                        self.test_user_token = login_data.get("access_token")
                        self.log_test("New User Login", True, "Successfully logged in with new user")
                    else:
                        self.log_test("New User Login", False, f"Login failed: {login_response.status_code}")
                    
                    return True
                else:
                    self.log_test("User Registration", False, f"Status: {response.status_code}, Response: {response.text}")
                    return False
        except Exception as e:
            self.log_test("User Registration", False, f"Exception: {str(e)}")
            return False
    
    async def test_get_models(self):
        """Test fetching available OpenRouter models"""
        if not self.admin_token:
            self.log_test("Get Models", False, "No admin token available")
            return False
            
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"Authorization": f"Bearer {self.admin_token}"}
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    model_count = len(models)
                    
                    # Check if the required model is available
                    required_model = "mistralai/mistral-7b-instruct:free"
                    model_ids = [model.get("id", "") for model in models]
                    has_required_model = required_model in model_ids
                    
                    self.log_test(
                        "Get Models", 
                        True, 
                        f"Found {model_count} models, Required model available: {has_required_model}"
                    )
                    return True
                else:
                    self.log_test("Get Models", False, f"Status: {response.status_code}, Response: {response.text}")
                    return False
        except Exception as e:
            self.log_test("Get Models", False, f"Exception: {str(e)}")
            return False
    
    async def test_chat_completion(self):
        """Test chat completion with admin account"""
        if not self.admin_token:
            self.log_test("Chat Completion", False, "No admin token available")
            return False
            
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                headers = {"Authorization": f"Bearer {self.admin_token}"}
                chat_data = {
                    "messages": [
                        {"role": "user", "content": "Hello! Please respond with 'Test successful' if you can understand this message."}
                    ],
                    "model": "mistralai/mistral-7b-instruct:free",
                    "max_tokens": 50,
                    "temperature": 0.7
                }
                
                response = await client.post(
                    f"{self.base_url}/chat",
                    headers=headers,
                    json=chat_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("content", "")
                    model = data.get("model", "")
                    usage = data.get("usage", {})
                    
                    self.log_test(
                        "Chat Completion", 
                        True, 
                        f"Model: {model}, Response length: {len(content)}, Tokens used: {usage.get('total_tokens', 'N/A')}"
                    )
                    return True
                else:
                    self.log_test("Chat Completion", False, f"Status: {response.status_code}, Response: {response.text}")
                    return False
        except Exception as e:
            self.log_test("Chat Completion", False, f"Exception: {str(e)}")
            return False
    
    async def test_create_chat_session(self):
        """Test creating a new chat session"""
        if not self.admin_token:
            self.log_test("Create Chat Session", False, "No admin token available")
            return False
            
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"Authorization": f"Bearer {self.admin_token}"}
                session_data = {
                    "title": "Test Chat Session",
                    "model": "mistralai/mistral-7b-instruct:free"
                }
                
                response = await client.post(
                    f"{self.base_url}/chat/sessions",
                    headers=headers,
                    json=session_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    session_id = data.get("id")
                    title = data.get("title")
                    model = data.get("model")
                    
                    self.log_test(
                        "Create Chat Session", 
                        True, 
                        f"Session ID: {session_id}, Title: {title}, Model: {model}"
                    )
                    return True
                else:
                    self.log_test("Create Chat Session", False, f"Status: {response.status_code}, Response: {response.text}")
                    return False
        except Exception as e:
            self.log_test("Create Chat Session", False, f"Exception: {str(e)}")
            return False
    
    async def test_get_chat_sessions(self):
        """Test listing chat sessions"""
        if not self.admin_token:
            self.log_test("Get Chat Sessions", False, "No admin token available")
            return False
            
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"Authorization": f"Bearer {self.admin_token}"}
                response = await client.get(
                    f"{self.base_url}/chat/sessions",
                    headers=headers
                )
                
                if response.status_code == 200:
                    sessions = response.json()
                    session_count = len(sessions)
                    
                    self.log_test(
                        "Get Chat Sessions", 
                        True, 
                        f"Found {session_count} chat sessions"
                    )
                    return True
                else:
                    self.log_test("Get Chat Sessions", False, f"Status: {response.status_code}, Response: {response.text}")
                    return False
        except Exception as e:
            self.log_test("Get Chat Sessions", False, f"Exception: {str(e)}")
            return False
    
    async def test_credit_system_admin(self):
        """Test that admin has unlimited credits and can make chat requests"""
        if not self.admin_token:
            self.log_test("Credit System (Admin)", False, "No admin token available")
            return False
            
        try:
            # First get current user info to check credits
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"Authorization": f"Bearer {self.admin_token}"}
                
                # Get user info
                user_response = await client.get(f"{self.base_url}/auth/me", headers=headers)
                if user_response.status_code != 200:
                    self.log_test("Credit System (Admin)", False, "Failed to get user info")
                    return False
                
                user_data = user_response.json()
                initial_credits = user_data.get("credits", 0)
                is_admin = user_data.get("is_admin", False)
                
                # Make a chat request
                chat_data = {
                    "messages": [{"role": "user", "content": "Test credit system"}],
                    "model": "mistralai/mistral-7b-instruct:free",
                    "max_tokens": 10
                }
                
                chat_response = await client.post(
                    f"{self.base_url}/chat",
                    headers=headers,
                    json=chat_data
                )
                
                if chat_response.status_code != 200:
                    self.log_test("Credit System (Admin)", False, f"Chat request failed: {chat_response.status_code}")
                    return False
                
                # Check credits after chat
                user_response_after = await client.get(f"{self.base_url}/auth/me", headers=headers)
                if user_response_after.status_code == 200:
                    user_data_after = user_response_after.json()
                    final_credits = user_data_after.get("credits", 0)
                    
                    # For admin, credits should remain high (unlimited)
                    credits_unchanged = (final_credits >= initial_credits - 100)  # Allow small deduction but should be high
                    
                    self.log_test(
                        "Credit System (Admin)", 
                        True, 
                        f"Admin: {is_admin}, Initial: {initial_credits}, Final: {final_credits}, Unlimited: {credits_unchanged}"
                    )
                    return True
                else:
                    self.log_test("Credit System (Admin)", False, "Failed to get user info after chat")
                    return False
                    
        except Exception as e:
            self.log_test("Credit System (Admin)", False, f"Exception: {str(e)}")
            return False
    
    async def test_unauthorized_access(self):
        """Test that endpoints require authentication"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Try to access protected endpoint without token
                response = await client.get(f"{self.base_url}/auth/me")
                
                if response.status_code == 401:
                    self.log_test("Unauthorized Access", True, "Correctly rejected request without token")
                    return True
                else:
                    self.log_test("Unauthorized Access", False, f"Expected 401, got {response.status_code}")
                    return False
        except Exception as e:
            self.log_test("Unauthorized Access", False, f"Exception: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Run all backend tests"""
        print(f"🚀 Starting Backend API Tests")
        print(f"Backend URL: {self.base_url}")
        print("=" * 60)
        
        # Test sequence
        tests = [
            self.test_health_check,
            self.test_unauthorized_access,
            self.test_admin_login,
            self.test_get_current_user,
            self.test_user_registration,
            self.test_get_models,
            self.test_chat_completion,
            self.test_create_chat_session,
            self.test_get_chat_sessions,
            self.test_credit_system_admin,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                result = await test()
                if result:
                    passed += 1
            except Exception as e:
                print(f"❌ FAIL {test.__name__}: Exception {str(e)}")
            print()  # Add spacing between tests
        
        print("=" * 60)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print(f"⚠️  {total - passed} tests failed")
            return False

async def main():
    """Main test runner"""
    tester = BackendTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ Backend testing completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some backend tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())