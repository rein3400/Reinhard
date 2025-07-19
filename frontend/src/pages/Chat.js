import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { chatAPI } from '../utils/api';
import { 
  PaperAirplaneIcon,
  CogIcon,
  PlusIcon,
  TrashIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import LoadingSpinner from '../components/LoadingSpinner';

const Chat = () => {
  const { user, updateUserCredits } = useAuth();
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('mistralai/mistral-7b-instruct:free');
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(true);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetchModels();
    fetchSessions();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchModels = async () => {
    try {
      const response = await chatAPI.getModels();
      setModels(response.data.models || []);
    } catch (error) {
      console.error('Error fetching models:', error);
      toast.error('Failed to fetch available models');
    } finally {
      setModelsLoading(false);
    }
  };

  const fetchSessions = async () => {
    try {
      const response = await chatAPI.getSessions();
      setSessions(response.data);
    } catch (error) {
      console.error('Error fetching sessions:', error);
    }
  };

  const createNewSession = async () => {
    try {
      const response = await chatAPI.createSession({
        title: 'New Chat',
        model: selectedModel
      });
      
      const newSession = response.data;
      setSessions([newSession, ...sessions]);
      setCurrentSession(newSession);
      setMessages([]);
      toast.success('New chat session created');
    } catch (error) {
      console.error('Error creating session:', error);
      toast.error('Failed to create new session');
    }
  };

  const selectSession = async (session) => {
    try {
      const response = await chatAPI.getSession(session.id);
      setCurrentSession(session);
      setMessages(response.data.messages || []);
      setSelectedModel(response.data.model);
    } catch (error) {
      console.error('Error loading session:', error);
      toast.error('Failed to load session');
    }
  };

  const deleteSession = async (sessionId, e) => {
    e.stopPropagation();
    try {
      await chatAPI.deleteSession(sessionId);
      setSessions(sessions.filter(s => s.id !== sessionId));
      if (currentSession?.id === sessionId) {
        setCurrentSession(null);
        setMessages([]);
      }
      toast.success('Session deleted');
    } catch (error) {
      console.error('Error deleting session:', error);
      toast.error('Failed to delete session');
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    // Check credits
    if (!user?.is_admin && user?.credits <= 0) {
      toast.error('Insufficient credits. Please contact admin.');
      return;
    }

    // Create session if none exists
    if (!currentSession) {
      await createNewSession();
    }

    const userMessage = { role: 'user', content: inputMessage.trim() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputMessage('');
    setLoading(true);

    try {
      const response = await chatAPI.sendMessage({
        messages: newMessages,
        model: selectedModel,
        max_tokens: 1000,
        temperature: 0.7
      });

      const assistantMessage = { role: 'assistant', content: response.data.content };
      setMessages([...newMessages, assistantMessage]);

      // Update user credits
      if (response.data.usage?.total_tokens && !user?.is_admin) {
        const newCredits = user.credits - response.data.usage.total_tokens;
        updateUserCredits(newCredits);
      }

    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = error.response?.data?.detail || 'Failed to send message';
      toast.error(errorMessage);
      
      // Remove the user message if send failed
      setMessages(messages);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatModelName = (modelId) => {
    const parts = modelId.split('/');
    const modelName = parts[parts.length - 1];
    return modelName.replace(/-/g, ' ').replace(/:/g, ' - ');
  };

  const getModelProvider = (modelId) => {
    const providers = {
      'mistralai': 'Mistral AI',
      'meta-llama': 'Meta',
      'anthropic': 'Anthropic',
      'openai': 'OpenAI',
      'google': 'Google',
      'microsoft': 'Microsoft'
    };
    
    const provider = modelId.split('/')[0];
    return providers[provider] || provider;
  };

  return (
    <div className="flex h-[calc(100vh-theme(spacing.24))] bg-white rounded-lg shadow-sm border border-gray-200">
      {/* Sidebar */}
      <div className="w-64 bg-gray-50 border-r border-gray-200 flex flex-col">
        {/* New Chat Button */}
        <div className="p-4 border-b border-gray-200">
          <button
            onClick={createNewSession}
            className="w-full flex items-center justify-center px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700"
          >
            <PlusIcon className="h-4 w-4 mr-2" />
            New Chat
          </button>
        </div>

        {/* Model Selector */}
        <div className="p-4 border-b border-gray-200">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            AI Model
          </label>
          {modelsLoading ? (
            <div className="flex justify-center">
              <LoadingSpinner size="sm" />
            </div>
          ) : (
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {formatModelName(model.id)}
                </option>
              ))}
            </select>
          )}
          <p className="text-xs text-gray-500 mt-1">
            Provider: {getModelProvider(selectedModel)}
          </p>
        </div>

        {/* Chat Sessions */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-4">
            <h3 className="text-sm font-medium text-gray-700 mb-3">Recent Chats</h3>
            <div className="space-y-2">
              {sessions.map((session) => (
                <div
                  key={session.id}
                  onClick={() => selectSession(session)}
                  className={`group flex items-center justify-between p-3 rounded-md cursor-pointer ${
                    currentSession?.id === session.id
                      ? 'bg-blue-100 text-blue-700'
                      : 'hover:bg-gray-100'
                  }`}
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">
                      {session.title}
                    </p>
                    <p className="text-xs text-gray-500">
                      {session.message_count} messages
                    </p>
                  </div>
                  <button
                    onClick={(e) => deleteSession(session.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-100 hover:text-red-600"
                  >
                    <TrashIcon className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">
              {currentSession?.title || 'Select or create a chat session'}
            </h2>
            {currentSession && (
              <p className="text-sm text-gray-500">
                Using {formatModelName(selectedModel)}
              </p>
            )}
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">
              Credits: {user?.credits?.toLocaleString() || 0}
            </span>
            {user?.is_admin && (
              <span className="px-2 py-1 bg-red-100 text-red-800 text-xs rounded-full">
                Unlimited
              </span>
            )}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl px-4 py-2 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 px-4 py-2 rounded-lg">
                <LoadingSpinner size="sm" />
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-gray-200">
          <div className="flex space-x-4">
            <div className="flex-1">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={currentSession ? "Type your message..." : "Create a new chat to start messaging"}
                disabled={loading}
                className="w-full px-3 py-2 border border-gray-300 rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                rows="3"
              />
            </div>
            <button
              onClick={sendMessage}
              disabled={loading || !inputMessage.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <PaperAirplaneIcon className="h-5 w-5" />
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
};

export default Chat;