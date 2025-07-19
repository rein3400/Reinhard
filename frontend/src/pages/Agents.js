import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { agentAPI } from '../utils/api';
import { 
  GlobeAltIcon,
  DocumentTextIcon,
  MagnifyingGlassIcon,
  PresentationChartBarIcon,
  PlayIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

const Agents = () => {
  const { user } = useAuth();
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [taskDescription, setTaskDescription] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const agentTypes = [
    {
      id: 'browser',
      name: 'Browser Automation',
      description: 'Automate web browsing tasks, scraping, and form filling',
      icon: GlobeAltIcon,
      color: 'blue',
      examples: [
        'Extract data from websites',
        'Fill out forms automatically',
        'Monitor website changes',
        'Take screenshots of pages'
      ]
    },
    {
      id: 'document',
      name: 'Document Generation',
      description: 'Generate and process documents, manuscripts, and reports',
      icon: DocumentTextIcon,
      color: 'green',
      examples: [
        'Generate research papers',
        'Create technical documentation',
        'Process PDF files',
        'Format documents'
      ]
    },
    {
      id: 'search',
      name: 'Information Seeking',
      description: 'Search and gather information from various sources',
      icon: MagnifyingGlassIcon,
      color: 'purple',
      examples: [
        'Research topics in depth',
        'Find specific information',
        'Compare different sources',
        'Fact-check claims'
      ]
    },
    {
      id: 'presentation',
      name: 'Presentation AI',
      description: 'Create presentations and visual content',
      icon: PresentationChartBarIcon,
      color: 'orange',
      examples: [
        'Generate slide presentations',
        'Create visual layouts',
        'Design charts and graphs',
        'Format presentation content'
      ]
    }
  ];

  useEffect(() => {
    fetchTasks();
  }, []);

  const fetchTasks = async () => {
    try {
      const response = await agentAPI.getTasks();
      setTasks(response.data);
    } catch (error) {
      console.error('Error fetching tasks:', error);
      toast.error('Failed to fetch agent tasks');
    } finally {
      setLoading(false);
    }
  };

  const createTask = async () => {
    if (!selectedAgent || !taskDescription.trim()) {
      toast.error('Please select an agent and provide a task description');
      return;
    }

    setSubmitting(true);
    try {
      const response = await agentAPI.createTask({
        agent_type: selectedAgent.id,
        task_description: taskDescription.trim()
      });

      setTasks([response.data, ...tasks]);
      setTaskDescription('');
      setSelectedAgent(null);
      toast.success('Agent task created successfully!');
    } catch (error) {
      console.error('Error creating task:', error);
      toast.error('Failed to create agent task');
    } finally {
      setSubmitting(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'pending':
        return <ClockIcon className="h-5 w-5 text-yellow-500" />;
      case 'running':
        return <PlayIcon className="h-5 w-5 text-blue-500" />;
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'pending':
        return 'bg-yellow-100 text-yellow-800';
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
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
        <h1 className="text-2xl font-bold text-gray-900">AI Agents</h1>
        <p className="text-gray-600">
          Use specialized AI agents to automate various tasks
        </p>
      </div>

      {/* Agent Selection */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Create New Task</h2>
        
        {/* Agent Types Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {agentTypes.map((agent) => (
            <div
              key={agent.id}
              onClick={() => setSelectedAgent(agent)}
              className={`cursor-pointer p-4 rounded-lg border-2 transition-all ${
                selectedAgent?.id === agent.id
                  ? `border-${agent.color}-500 bg-${agent.color}-50`
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center mb-3">
                <agent.icon className={`h-6 w-6 text-${agent.color}-600 mr-2`} />
                <h3 className="font-medium text-gray-900">{agent.name}</h3>
              </div>
              <p className="text-sm text-gray-600 mb-3">{agent.description}</p>
              <div className="space-y-1">
                <p className="text-xs font-medium text-gray-700">Examples:</p>
                {agent.examples.slice(0, 2).map((example, index) => (
                  <p key={index} className="text-xs text-gray-500">• {example}</p>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Task Description */}
        {selectedAgent && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Task Description for {selectedAgent.name}
              </label>
              <textarea
                value={taskDescription}
                onChange={(e) => setTaskDescription(e.target.value)}
                placeholder={`Describe what you want the ${selectedAgent.name.toLowerCase()} to do...`}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows="4"
              />
            </div>
            <div className="flex justify-end">
              <button
                onClick={createTask}
                disabled={submitting || !taskDescription.trim()}
                className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {submitting ? 'Creating Task...' : 'Create Task'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Task History */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Task History</h2>
        
        {tasks.length === 0 ? (
          <div className="text-center py-8">
            <CpuChipIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No tasks yet</h3>
            <p className="mt-1 text-sm text-gray-500">
              Create your first agent task above
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {tasks.map((task) => {
              const agent = agentTypes.find(a => a.id === task.agent_type);
              return (
                <div key={task.id} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3">
                      {agent && <agent.icon className={`h-6 w-6 text-${agent.color}-600 mt-0.5`} />}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className="text-sm font-medium text-gray-900">
                            {agent?.name || task.agent_type}
                          </h3>
                          <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(task.status)}`}>
                            {task.status}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 mb-2">{task.task_description}</p>
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <span>Created: {new Date(task.created_at).toLocaleDateString()}</span>
                          {task.credits_used > 0 && (
                            <span>Credits used: {task.credits_used}</span>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center">
                      {getStatusIcon(task.status)}
                    </div>
                  </div>
                  
                  {task.result && (
                    <div className="mt-3 p-3 bg-gray-50 rounded-md">
                      <p className="text-xs font-medium text-gray-700 mb-1">Result:</p>
                      <pre className="text-xs text-gray-600 whitespace-pre-wrap">
                        {JSON.stringify(task.result, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Coming Soon Features */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
        <h3 className="text-lg font-medium text-yellow-900 mb-2">🚧 Coming Soon</h3>
        <p className="text-yellow-700 mb-3">
          Advanced AI agent features are being integrated from the repositories you specified:
        </p>
        <ul className="text-sm text-yellow-600 space-y-1">
          <li>• Real browser automation with UI-TARS integration</li>
          <li>• Advanced workflow automation with deer-flow</li>
          <li>• Enhanced document processing with ai-manus</li>
          <li>• Intelligent web searching with agenticSeek</li>
        </ul>
      </div>
    </div>
  );
};

export default Agents;