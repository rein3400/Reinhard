import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { chatAPI, agentAPI } from '../utils/api';
import { 
  UserIcon,
  EnvelopeIcon,
  CalendarIcon,
  BanknotesIcon,
  ChartBarIcon,
  CogIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

const Profile = () => {
  const { user, logout } = useAuth();
  const [stats, setStats] = useState({
    totalChats: 0,
    totalAgentTasks: 0,
    creditsUsed: 0,
    usageHistory: []
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUserStats();
  }, []);

  const fetchUserStats = async () => {
    try {
      const [sessionsResponse, tasksResponse] = await Promise.all([
        chatAPI.getSessions(),
        agentAPI.getTasks()
      ]);

      const sessions = sessionsResponse.data;
      const tasks = tasksResponse.data;
      const creditsUsed = tasks.reduce((total, task) => total + task.credits_used, 0);

      // Generate usage history for the last 7 days
      const usageHistory = [];
      for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        usageHistory.push({
          date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          chats: Math.floor(Math.random() * 5) + 1,
          agents: Math.floor(Math.random() * 3) + 1,
          credits: Math.floor(Math.random() * 100) + 10
        });
      }

      setStats({
        totalChats: sessions.length,
        totalAgentTasks: tasks.length,
        creditsUsed,
        usageHistory
      });
    } catch (error) {
      console.error('Error fetching user stats:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Profile & Settings</h1>
        <p className="text-gray-600">
          Manage your account and view your usage statistics
        </p>
      </div>

      {/* Profile Information */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Account Information</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <UserIcon className="h-5 w-5 text-gray-400" />
              <div>
                <p className="text-sm font-medium text-gray-700">Username</p>
                <p className="text-gray-900">{user?.username}</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <EnvelopeIcon className="h-5 w-5 text-gray-400" />
              <div>
                <p className="text-sm font-medium text-gray-700">Email</p>
                <p className="text-gray-900">{user?.email}</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <CalendarIcon className="h-5 w-5 text-gray-400" />
              <div>
                <p className="text-sm font-medium text-gray-700">Member Since</p>
                <p className="text-gray-900">
                  {user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A'}
                </p>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <BanknotesIcon className="h-5 w-5 text-gray-400" />
              <div>
                <p className="text-sm font-medium text-gray-700">Current Credits</p>
                <div className="flex items-center space-x-2">
                  <p className="text-2xl font-bold text-green-600">{user?.credits?.toLocaleString()}</p>
                  {user?.is_admin && (
                    <span className="px-2 py-1 bg-red-100 text-red-800 text-xs rounded-full">
                      Unlimited
                    </span>
                  )}
                </div>
              </div>
            </div>
            
            {user?.is_admin && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-md">
                <p className="text-sm font-medium text-red-800">Admin Account</p>
                <p className="text-xs text-red-600">You have administrative privileges</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Usage Statistics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <ChartBarIcon className="h-8 w-8 text-blue-600" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-600">Total Chats</p>
              <p className="text-2xl font-bold text-gray-900">{stats.totalChats}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <CogIcon className="h-8 w-8 text-purple-600" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-600">Agent Tasks</p>
              <p className="text-2xl font-bold text-gray-900">{stats.totalAgentTasks}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <BanknotesIcon className="h-8 w-8 text-green-600" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-600">Credits Used</p>
              <p className="text-2xl font-bold text-gray-900">{stats.creditsUsed}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Usage Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Daily Activity</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={stats.usageHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="chats" 
                  stroke="#3B82F6" 
                  strokeWidth={2}
                  name="Chats"
                />
                <Line 
                  type="monotone" 
                  dataKey="agents" 
                  stroke="#8B5CF6" 
                  strokeWidth={2}
                  name="Agent Tasks"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Credits Usage</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.usageHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="credits" fill="#10B981" name="Credits Used" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Account Actions */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Account Actions</h2>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
            <div>
              <h3 className="text-sm font-medium text-gray-900">API Integration</h3>
              <p className="text-sm text-gray-600">Connect with external services and APIs</p>
            </div>
            <button className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700">
              Manage APIs
            </button>
          </div>
          
          <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
            <div>
              <h3 className="text-sm font-medium text-gray-900">Export Data</h3>
              <p className="text-sm text-gray-600">Download your chat history and agent results</p>
            </div>
            <button className="px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-md hover:bg-green-700">
              Export
            </button>
          </div>
          
          <div className="flex items-center justify-between p-4 border border-red-200 rounded-lg">
            <div>
              <h3 className="text-sm font-medium text-red-900">Sign Out</h3>
              <p className="text-sm text-red-600">Sign out from your account</p>
            </div>
            <button 
              onClick={logout}
              className="px-4 py-2 bg-red-600 text-white text-sm font-medium rounded-md hover:bg-red-700"
            >
              Sign Out
            </button>
          </div>
        </div>
      </div>

      {/* Feature Roadmap */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-medium text-blue-900 mb-2">🚀 Coming Soon</h3>
        <p className="text-blue-700 mb-3">
          We're working on exciting new features based on the integrated repositories:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="text-sm text-blue-600">
            <p>• Advanced browser automation capabilities</p>
            <p>• Real-time workflow monitoring</p>
          </div>
          <div className="text-sm text-blue-600">
            <p>• Enhanced document processing</p>
            <p>• Smart presentation generation</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;