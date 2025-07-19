import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { 
  CommandLineIcon,
  GlobeAltIcon,
  PlayIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  CpuChipIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import LoadingSpinner from '../components/LoadingSpinner';
import api from '../utils/api';

const ChatGPTAgent = () => {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('autonomous');
  const [tasks, setTasks] = useState([]);
  const [capabilities, setCapabilities] = useState({});
  const [loading, setLoading] = useState(false);
  
  // Autonomous Agent
  const [taskDescription, setTaskDescription] = useState('');
  const [taskGoal, setTaskGoal] = useState('');
  const [taskType, setTaskType] = useState('general');
  
  // Shell Access
  const [shellCommand, setShellCommand] = useState('');
  const [shellHistory, setShellHistory] = useState([]);
  const shellRef = useRef(null);
  
  // Web Browsing
  const [webUrl, setWebUrl] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [webResults, setWebResults] = useState(null);

  useEffect(() => {
    fetchCapabilities();
    fetchTasks();
  }, []);

  useEffect(() => {
    if (shellRef.current) {
      shellRef.current.scrollTop = shellRef.current.scrollHeight;
    }
  }, [shellHistory]);

  const fetchCapabilities = async () => {
    try {
      const response = await api.get('/agents/capabilities');
      setCapabilities(response.data.capabilities);
    } catch (error) {
      console.error('Error fetching capabilities:', error);
    }
  };

  const fetchTasks = async () => {
    try {
      const response = await api.get('/agents/chatgpt/tasks');
      setTasks(response.data || []);
    } catch (error) {
      console.error('Error fetching tasks:', error);
    }
  };

  const createAutonomousTask = async () => {
    if (!taskDescription.trim() || !taskGoal.trim()) {
      toast.error('Please provide both description and goal');
      return;
    }

    setLoading(true);
    try {
      const response = await api.post('/agents/chatgpt/tasks', {
        task_type: taskType,
        description: taskDescription.trim(),
        goal: taskGoal.trim()
      });

      toast.success('Autonomous task created successfully!');
      setTaskDescription('');
      setTaskGoal('');
      fetchTasks();

      // Auto-execute the task
      executeTask(response.data.task_id);
    } catch (error) {
      console.error('Error creating task:', error);
      toast.error('Failed to create autonomous task');
    } finally {
      setLoading(false);
    }
  };

  const executeTask = async (taskId) => {
    try {
      toast.loading('Executing autonomous task...', { id: taskId });
      
      const response = await api.post(`/agents/chatgpt/tasks/${taskId}/execute`);
      
      if (response.data.success) {
        toast.success('Task completed successfully!', { id: taskId });
      } else {
        toast.error(`Task failed: ${response.data.error}`, { id: taskId });
      }
      
      fetchTasks();
    } catch (error) {
      console.error('Error executing task:', error);
      toast.error('Failed to execute task', { id: taskId });
    }
  };

  const executeShellCommand = async () => {
    if (!shellCommand.trim()) {
      toast.error('Please enter a command');
      return;
    }

    const command = shellCommand.trim();
    setShellHistory(prev => [...prev, { type: 'input', content: `$ ${command}`, timestamp: new Date() }]);
    setShellCommand('');

    try {
      const response = await api.post('/agents/shell/execute', { command });
      
      if (response.data.success) {
        setShellHistory(prev => [...prev, { 
          type: 'output', 
          content: response.data.stdout, 
          timestamp: new Date(),
          exitCode: response.data.exit_code
        }]);
      } else {
        setShellHistory(prev => [...prev, { 
          type: 'error', 
          content: response.data.stderr || response.data.error, 
          timestamp: new Date(),
          exitCode: response.data.exit_code
        }]);
      }
    } catch (error) {
      console.error('Error executing shell command:', error);
      setShellHistory(prev => [...prev, { 
        type: 'error', 
        content: `Error: ${error.response?.data?.detail || error.message}`, 
        timestamp: new Date() 
      }]);
    }
  };

  const browseWeb = async () => {
    if (!webUrl.trim()) {
      toast.error('Please enter a URL');
      return;
    }

    setLoading(true);
    try {
      const response = await api.post('/agents/web/browse', { url: webUrl.trim() });
      
      if (response.data.success) {
        setWebResults({
          type: 'browse',
          url: response.data.url,
          status: response.data.status_code,
          content: response.data.content.substring(0, 5000) + '...',
          headers: response.data.headers
        });
        toast.success('Web page fetched successfully!');
      } else {
        toast.error('Failed to fetch web page');
      }
    } catch (error) {
      console.error('Error browsing web:', error);
      toast.error('Failed to browse web');
    } finally {
      setLoading(false);
    }
  };

  const searchWeb = async () => {
    if (!searchQuery.trim()) {
      toast.error('Please enter a search query');
      return;
    }

    setLoading(true);
    try {
      const response = await api.post('/agents/web/search', { 
        query: searchQuery.trim(),
        num_results: 5 
      });
      
      if (response.data.success) {
        setWebResults({
          type: 'search',
          query: searchQuery.trim(),
          results: response.data.results
        });
        toast.success('Web search completed!');
      } else {
        toast.error('Failed to search web');
      }
    } catch (error) {
      console.error('Error searching web:', error);
      toast.error('Failed to search web');
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'pending':
        return <ClockIcon className="h-5 w-5 text-yellow-500" />;
      case 'planning':
      case 'executing':
        return <PlayIcon className="h-5 w-5 text-blue-500" />;
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const handleShellKeyPress = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      executeShellCommand();
    }
  };

  const tabs = [
    { id: 'autonomous', name: 'Autonomous Agent', icon: CpuChipIcon },
    { id: 'shell', name: 'Shell Access', icon: CommandLineIcon },
    { id: 'web', name: 'Web Browsing', icon: GlobeAltIcon }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">ChatGPT Agent</h1>
        <p className="text-gray-600">
          Advanced AI agent with autonomous capabilities, shell access, and web interaction
        </p>
      </div>

      {/* Capabilities Info */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center mb-2">
          <InformationCircleIcon className="h-5 w-5 text-blue-600 mr-2" />
          <h3 className="font-medium text-blue-900">Available Capabilities</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm text-blue-700">
          {Object.entries(capabilities).map(([key, cap]) => (
            <div key={key} className="flex items-center">
              <span className={`w-2 h-2 rounded-full mr-2 ${cap.enabled ? 'bg-green-500' : 'bg-red-500'}`}></span>
              {cap.name}
            </div>
          ))}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <tab.icon className="h-4 w-4 mr-2" />
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        {activeTab === 'autonomous' && (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Create Autonomous Task</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Task Type</label>
                  <select
                    value={taskType}
                    onChange={(e) => setTaskType(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="general">General Task</option>
                    <option value="research">Research & Analysis</option>
                    <option value="automation">Automation</option>
                    <option value="data_processing">Data Processing</option>
                    <option value="web_interaction">Web Interaction</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Task Description</label>
                  <textarea
                    value={taskDescription}
                    onChange={(e) => setTaskDescription(e.target.value)}
                    placeholder="Describe what you want the agent to do..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows="3"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Goal</label>
                  <textarea
                    value={taskGoal}
                    onChange={(e) => setTaskGoal(e.target.value)}
                    placeholder="What should be the end result or outcome?"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows="2"
                  />
                </div>

                <button
                  onClick={createAutonomousTask}
                  disabled={loading || !taskDescription.trim() || !taskGoal.trim()}
                  className="w-full px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Creating Task...' : 'Create & Execute Autonomous Task'}
                </button>
              </div>
            </div>

            {/* Recent Tasks */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Autonomous Tasks</h3>
              <div className="space-y-3">
                {tasks.length === 0 ? (
                  <p className="text-gray-500 text-center py-4">No autonomous tasks yet</p>
                ) : (
                  tasks.slice(0, 5).map((task) => (
                    <div key={task.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-2">
                            {getStatusIcon(task.status)}
                            <span className="font-medium text-gray-900">{task.task_type}</span>
                          </div>
                          <p className="text-sm text-gray-600 mb-1">{task.description}</p>
                          <p className="text-xs text-gray-500">Goal: {task.goal}</p>
                        </div>
                        <span className="text-xs text-gray-500">
                          {new Date(task.created_at).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'shell' && (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Shell Command Execution</h3>
              
              {/* Terminal Interface */}
              <div className="bg-black rounded-lg p-4 font-mono text-sm">
                <div 
                  ref={shellRef}
                  className="h-96 overflow-y-auto text-green-400 whitespace-pre-wrap"
                >
                  {shellHistory.length === 0 && (
                    <div className="text-gray-500">
                      Welcome to AI Agent Shell Access
                      {'\n'}Type commands to execute. Security restrictions apply.
                      {'\n'}Available commands: ls, cat, grep, find, ps, curl, git, etc.
                      {'\n\n'}
                    </div>
                  )}
                  
                  {shellHistory.map((entry, index) => (
                    <div key={index} className="mb-1">
                      <span className={
                        entry.type === 'input' ? 'text-yellow-400' :
                        entry.type === 'error' ? 'text-red-400' : 'text-green-400'
                      }>
                        {entry.content}
                      </span>
                      {entry.exitCode !== undefined && entry.exitCode !== 0 && (
                        <span className="text-red-400"> (exit code: {entry.exitCode})</span>
                      )}
                    </div>
                  ))}
                </div>
                
                {/* Command Input */}
                <div className="flex items-center mt-2 text-green-400">
                  <span className="mr-2">$</span>
                  <input
                    type="text"
                    value={shellCommand}
                    onChange={(e) => setShellCommand(e.target.value)}
                    onKeyPress={handleShellKeyPress}
                    placeholder="Enter shell command..."
                    className="flex-1 bg-transparent outline-none text-green-400 placeholder-gray-600"
                  />
                </div>
              </div>

              <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                <p className="text-sm text-yellow-800">
                  <strong>Security Notice:</strong> Only safe, whitelisted commands are allowed. 
                  Destructive operations (rm, chmod, etc.) are blocked.
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'web' && (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Web Browsing & Search</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Web Browsing */}
                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Browse URL</h4>
                  <div className="space-y-3">
                    <input
                      type="url"
                      value={webUrl}
                      onChange={(e) => setWebUrl(e.target.value)}
                      placeholder="https://example.com"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <button
                      onClick={browseWeb}
                      disabled={loading || !webUrl.trim()}
                      className="w-full px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loading ? 'Browsing...' : 'Browse URL'}
                    </button>
                  </div>
                </div>

                {/* Web Search */}
                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Web Search</h4>
                  <div className="space-y-3">
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search query..."
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <button
                      onClick={searchWeb}
                      disabled={loading || !searchQuery.trim()}
                      className="w-full px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loading ? 'Searching...' : 'Search Web'}
                    </button>
                  </div>
                </div>
              </div>

              {/* Web Results */}
              {webResults && (
                <div className="mt-6 border border-gray-200 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3">
                    {webResults.type === 'search' ? 'Search Results' : 'Web Content'}
                  </h4>
                  
                  {webResults.type === 'search' ? (
                    <div className="space-y-3">
                      <p className="text-sm text-gray-600">Query: "{webResults.query}"</p>
                      {webResults.results.map((result, index) => (
                        <div key={index} className="border-l-4 border-blue-500 pl-4">
                          <h5 className="font-medium text-blue-700">{result.title}</h5>
                          <p className="text-sm text-gray-600">{result.link}</p>
                          {result.snippet && (
                            <p className="text-sm text-gray-500 mt-1">{result.snippet}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="flex items-center space-x-4 text-sm text-gray-600">
                        <span>URL: {webResults.url}</span>
                        <span>Status: {webResults.status}</span>
                      </div>
                      <div className="bg-gray-50 p-3 rounded text-sm font-mono max-h-64 overflow-y-auto">
                        {webResults.content}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatGPTAgent;