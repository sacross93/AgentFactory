import React, { useState, useRef, useEffect } from 'react';
import { format } from 'date-fns';
import './ChatbotUI.css';

const ChatbotUI = () => {
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('theme') === 'dark');
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [selectedModel, setSelectedModel] = useState(() => localStorage.getItem('selectedModel') || 'Gemma3');
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState(() => {
    const savedMessages = localStorage.getItem('chatMessages');
    return savedMessages ? JSON.parse(savedMessages) : [
      {
        id: Date.now(),
        type: 'bot',
        name: '제플몰 챗봇',
        content: '안녕하세요! 제플몰 상담 챗봇입니다. 어떻게 도와드릴까요?',
        time: new Date().toISOString(),
        reactions: { like: 0, dislike: 0 },
      },
    ];
  });
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const chatSectionRef = useRef(null);

  const toggleDarkMode = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    document.documentElement.setAttribute('data-theme', newMode ? 'dark' : 'light');
    localStorage.setItem('theme', newMode ? 'dark' : 'light');
  };

  const selectModel = (model) => {
    setSelectedModel(model);
    setShowModelDropdown(false);
    localStorage.setItem('selectedModel', model);
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const newUserMessage = {
      id: Date.now(),
      type: 'user',
      name: '사용자',
      content: inputMessage,
      time: new Date().toISOString(),
      reactions: { like: 0, dislike: 0 },
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      const botResponse = {
        id: Date.now(),
        type: 'bot',
        name: '제플몰 챗봇',
        content: `"${inputMessage}"에 대한 답변입니다.`,
        time: new Date().toISOString(),
        reactions: { like: 0, dislike: 0 },
      };
      setMessages((prev) => [...prev, botResponse]);
    } catch (error) {
      console.error('메시지 전송 오류:', error);
      const errorMessage = {
        id: Date.now(),
        type: 'bot',
        name: '제플몰 챗봇',
        content: '오류가 발생했습니다. 다시 시도해주세요.',
        time: new Date().toISOString(),
        reactions: { like: 0, dislike: 0 },
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const copyMessage = (content) => {
    navigator.clipboard.writeText(content);
    alert('메시지가 클립보드에 복사되었습니다.');
  };

  const exportChat = () => {
    const chatText = messages.map((msg) => `[${formatTime(msg.time)}] ${msg.name}: ${msg.content}`).join('\n\n');
    const blob = new Blob([chatText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${format(new Date(), 'yyyy-MM-dd')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const clearChat = () => {
    if (window.confirm('대화 내용을 모두 지우시겠습니까?')) {
      const welcomeMessage = {
        id: Date.now(),
        type: 'bot',
        name: '제플몰 챗봇',
        content: '안녕하세요! 제플몰 상담 챗봇입니다. 어떻게 도와드릴까요?',
        time: new Date().toISOString(),
        reactions: { like: 0, dislike: 0 },
      };
      setMessages([welcomeMessage]);
    }
  };

  const toggleVoiceInput = () => {
    if (!isRecording) {
      setIsRecording(true);
      const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
      recognition.lang = 'ko-KR';
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInputMessage((prev) => prev + transcript);
        setIsRecording(false);
      };
      recognition.onerror = () => {
        alert('음성 인식에 실패했습니다.');
        setIsRecording(false);
      };
      recognition.start();
    } else {
      setIsRecording(false);
    }
  };

  const reactToMessage = (id, reaction) => {
    setMessages((prev) => {
      const newMessages = [...prev];
      const messageIndex = newMessages.findIndex((msg) => msg.id === id);
      const currentReaction = newMessages[messageIndex].reactions[reaction];
      newMessages[messageIndex].reactions[reaction] = currentReaction === 1 ? 0 : 1;
      return newMessages;
    });
  };

  const formatTime = (isoString) => {
    const date = new Date(isoString);
    const now = new Date();
    const diffMinutes = Math.floor((now - date) / (1000 * 60));

    if (diffMinutes < 1) return '방금 전';
    if (diffMinutes < 60) return `${diffMinutes}분 전`;
    if (diffMinutes < 1440) return `${Math.floor(diffMinutes / 60)}시간 전`;

    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${month}월 ${day}일`;
  };

  useEffect(() => {
    localStorage.setItem('chatMessages', JSON.stringify(messages));
    if (chatSectionRef.current) {
      chatSectionRef.current.scrollTop = chatSectionRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  return (
    <div className={`chatbot-ui ${darkMode ? 'dark' : 'light'}`}>
      <div className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <button className="sidebar-toggle" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
          {isSidebarOpen ? '✕' : '☰'}
        </button>
        <div className="sidebar-icon">
          <img src="/logo.png" alt="Logo" />
        </div>
        <div className="sidebar-icon">🏠</div>
        <div className="sidebar-icon">📊</div>
        <div className="sidebar-icon">
          <img src="/avatar-40.png" alt="Avatar" />
        </div>
      </div>

      <div className="main">
        <div className="chat-wrapper">
          <div className="header">
            <div className="header-left">
              <div className="model-selector" onClick={() => setShowModelDropdown(!showModelDropdown)}>
                <span>{selectedModel}</span>
                <span>▼</span>
                {showModelDropdown && (
                  <div className="dropdown">
                    <div className="dropdown-item" onClick={() => selectModel('GPT-3.5')}>GPT-3.5</div>
                    <div className="dropdown-item" onClick={() => selectModel('Gemma3')}>Gemma3</div>
                    <div className="dropdown-item" onClick={() => selectModel('Claude')}>Claude</div>
                    <div className="dropdown-item" onClick={() => selectModel('Gemini')}>Gemini</div>
                  </div>
                )}
              </div>
              <div>제플몰 종합 상담 챗봇</div>
            </div>
            <div className="header-right">
              <button className="button" onClick={exportChat}>
                <span>💾</span> 내보내기
              </button>
              <button className="button" onClick={clearChat}>
                <span>🗑️</span> 초기화
              </button>
              <button className="button" onClick={toggleDarkMode}>
                <span>{darkMode ? '🌙' : '☀️'}</span>
              </button>
            </div>
          </div>

          <div className="chat-container">
            <div className="chat-messages" ref={chatSectionRef}>
              {messages.map((msg) => (
                <div key={msg.id} className={`message-row ${msg.type}`}>
                  <div className={`avatar ${msg.type === 'bot' ? 'bot-avatar' : 'user-avatar'}`}>
                    <img
                      src={msg.type === 'bot' ? '/chatbot.png' : '/user.png'}
                      alt={msg.type === 'bot' ? '챗봇' : '사용자'}
                    />
                  </div>
                  <div className="message-content">
                    <div className="message-header">
                      <div className="message-name">{msg.name}</div>
                      <div className="message-time">{formatTime(msg.time)}</div>
                    </div>
                    <div className={`message-bubble ${msg.type === 'bot' ? 'bot-bubble' : 'user-bubble'}`}>
                      {msg.content}
                    </div>
                    {msg.type === 'bot' && (
                      <div className="message-actions">
                        <button className="action-button" onClick={() => reactToMessage(msg.id, 'like')}>
                          <span>👍</span> {msg.reactions.like > 0 ? msg.reactions.like : ''}
                        </button>
                        <button className="action-button" onClick={() => reactToMessage(msg.id, 'dislike')}>
                          <span>👎</span> {msg.reactions.dislike > 0 ? msg.reactions.dislike : ''}
                        </button>
                        <button className="action-button" onClick={() => copyMessage(msg.content)}>
                          <span>📋</span>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="message-row bot">
                  <div className="avatar bot-avatar">
                    <img src="/chatbot.png" alt="챗봇" />
                  </div>
                  <div className="message-content">
                    <div className="message-header">
                      <div className="message-name">제플몰 챗봇</div>
                      <div className="message-time">방금 전</div>
                    </div>
                    <div className="message-bubble bot-bubble">
                      <div className="loading-dots">
                        <div className="dot"></div>
                        <div className="dot"></div>
                        <div className="dot"></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="input-container">
              <div className="input-wrapper">
                <input
                  type="text"
                  placeholder="메시지를 입력하세요..."
                  className="input"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                />
                <button className="voice-button" onClick={toggleVoiceInput}>
                  {isRecording ? '🎙️' : '🎤'}
                </button>
                <button className="send-button" onClick={sendMessage}>
                  ➤
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatbotUI;