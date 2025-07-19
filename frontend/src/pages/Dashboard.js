import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { chatAPI, agentAPI } from '../utils/api';
import { 
  ChatBubbleLeftRightIcon, 
  CpuChipIcon, 
  UserGroupIcon, 
  BanknotesIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  UserIcon
} from '@heroicons/react/24/outline';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Dashboard = () => {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    totalChats: 0,
    totalAgentTasks: 0,
    creditsUsed: 0,
    recentActivity: []
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [sessionsResponse, tasksResponse] = await Promise.all([
        chatAPI.getSessions(),
        agentAPI.getTasks()
      ]);

      const sessions = sessionsResponse.data;
      const tasks = tasksResponse.data;

      // Calculate stats
      const creditsUsed = tasks.reduce((total, task) => total + task.credits_used, 0);
      
      // Generate sample activity data for chart
      const activityData = [];
      for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        activityData.push({
          date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          chats: Math.floor(Math.random() * 10) + 1,
          agents: Math.floor(Math.random() * 5) + 1,
        });
      }

      setStats({
        totalChats: sessions.length,
        totalAgentTasks: tasks.length,
        creditsUsed,
        recentActivity: activityData
      });
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
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

  const statCards = [
    {
      title: 'Total Chats',
      value: stats.totalChats,
      icon: ChatBubbleLeftRightIcon,
      color: 'blue',
      change: '+12%',
      changeType: 'increase'
    },
    {
      title: 'Agent Tasks',
      value: stats.totalAgentTasks,
      icon: CpuChipIcon,
      color: 'purple',
      change: '+8%',
      changeType: 'increase'
    },
    {
      title: 'Credits Remaining',
      value: user?.credits || 0,
      icon: BanknotesIcon,
      color: 'green',
      change: `-${stats.creditsUsed}`,
      changeType: 'decrease'
    },
    {
      title: 'Total Users',
      value: user?.is_admin ? '156' : '1',
      icon: UserGroupIcon,
      color: 'orange',
      change: '+3%',
      changeType: 'increase'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">
          Welcome back, {user?.username}! 👋
        </h1>
        <p className="text-gray-600">
          Here's what's happening with your AI platform today.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat, index) => (
          <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                <p className="text-2xl font-bold text-gray-900">{stat.value.toLocaleString()}</p>
              </div>
              <div className={`p-3 rounded-md bg-${stat.color}-100`}>
                <stat.icon className={`h-6 w-6 text-${stat.color}-600`} />
              </div>
            </div>
            <div className="mt-4 flex items-center">
              {stat.changeType === 'increase' ? (
                <ArrowUpIcon className="h-4 w-4 text-green-500 mr-1" />
              ) : (
                <ArrowDownIcon className="h-4 w-4 text-red-500 mr-1" />
              )}
              <span className={`text-sm ${
                stat.changeType === 'increase' ? 'text-green-600' : 'text-red-600'
              }`}>
                {stat.change}
              </span>
              <span className="text-sm text-gray-500 ml-1">from last week</span>
            </div>
          </div>
        ))}
      </div>

      {/* Activity Chart */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Activity Overview</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={stats.recentActivity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Area 
                type="monotone" 
                dataKey="chats" 
                stackId="1" 
                stroke="#3B82F6" 
                fill="#3B82F6" 
                fillOpacity={0.6}
                name="Chats"
              />
              <Area 
                type="monotone" 
                dataKey="agents" 
                stackId="1" 
                stroke="#8B5CF6" 
                fill="#8B5CF6" 
                fillOpacity={0.6}
                name="Agent Tasks"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-2">Start a New Chat</h3>
          <p className="text-gray-600 mb-4">Begin a conversation with AI models using OpenRouter.</p>
          <a
            href="/chat"
            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700"
          >
            <ChatBubbleLeftRightIcon className="h-4 w-4 mr-2" />
            New Chat
          </a>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-2">AI Agents</h3>
          <p className="text-gray-600 mb-4">Use specialized AI agents for automation tasks.</p>
          <a
            href="/agents"
            className="inline-flex items-center px-4 py-2 bg-purple-600 text-white text-sm font-medium rounded-md hover:bg-purple-700"
          >
            <CpuChipIcon className="h-4 w-4 mr-2" />
            View Agents
          </a>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-2">Account Settings</h3>
          <p className="text-gray-600 mb-4">Manage your profile and view usage statistics.</p>
          <a
            href="/profile"
            className="inline-flex items-center px-4 py-2 bg-gray-600 text-white text-sm font-medium rounded-md hover:bg-gray-700"
          >
            <UserIcon className="h-4 w-4 mr-2" />
            Profile
          </a>
        </div>
      </div>

      {/* Admin Panel Link */}
      {user?.is_admin && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h3 className="text-lg font-medium text-red-900 mb-2">Admin Panel</h3>
          <p className="text-red-700 mb-4">You have admin access. Manage users and system settings.</p>
          <button className="inline-flex items-center px-4 py-2 bg-red-600 text-white text-sm font-medium rounded-md hover:bg-red-700">
            <UserGroupIcon className="h-4 w-4 mr-2" />
            Admin Dashboard
          </button>
        </div>
      )}
    </div>
  );
};

export default Dashboard;