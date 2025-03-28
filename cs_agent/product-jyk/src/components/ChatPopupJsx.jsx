import React, { useState, useRef, useEffect } from 'react';
import './ChatPopupJsx.css';

const API_BASE_URL = 'http://localhost:8000';

function ChatPopupJsx() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const chatAreaRef = useRef(null);
  
  // 초기 메시지 추가
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([{
        id: 'welcome',
        role: 'assistant',
        content: '안녕하세요! 제플몰 AI 상담봇입니다. 무엇을 도와드릴까요?',
        time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
      }]);
    }
  }, []);
  
  // 채팅 영역 스크롤 처리
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages]);

  // 메시지 출력 포맷팅
  const formatAIResponse = (text) => {
    if (!text) return '';
    
    // 마크다운 스타일 변환 (간단한 처리)
    let formatted = text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>')
      .replace(/`(.*?)`/g, '<code>$1</code>');
    
    // 줄바꿈 처리
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
  };
  
  // 디버그 로그
  const logDebug = (message, data) => {
    console.log(`[Debug] ${message}`, data);
  };
  
  // 서버 상태 확인
  const checkServerHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return response.ok;
    } catch (error) {
      console.error('서버 연결 확인 오류:', error);
      return false;
    }
  };
  
  // 타임아웃 처리가 있는 fetch
  const fetchWithTimeout = (url, options, timeout = 10000) => {
    return Promise.race([
      fetch(url, options),
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('요청 시간이 초과되었습니다.')), timeout)
      )
    ]);
  };
  
  // 쿼리 요청 제출
  const submitQuery = async (query, history) => {
    try {
      const isServerOnline = await checkServerHealth();
      if (!isServerOnline) {
        throw new Error('서버에 연결할 수 없습니다.');
      }
      
      logDebug("쿼리 제출 시작", query);
      logDebug("대화 기록 전송", history);
      
      const response = await fetchWithTimeout(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: query,
          session_id: 'web-popup-jsx',
          timeout: 180, // 3분 타임아웃
          chat_history: history
        })
      }, 10000);
      
      if (!response.ok) {
        throw new Error(`API 오류: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      logDebug("쿼리 제출 실패", error);
      throw error;
    }
  };
  
  // 쿼리 상태 확인
  const checkQueryStatus = async (requestId) => {
    try {
      const response = await fetchWithTimeout(`${API_BASE_URL}/status/${requestId}`, {
        method: 'GET'
      }, 5000);
      
      if (!response.ok) {
        throw new Error(`상태 확인 오류: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      logDebug("상태 확인 실패", error);
      throw error;
    }
  };
  
  // 쿼리 결과 가져오기
  const getQueryResult = async (requestId) => {
    try {
      const response = await fetchWithTimeout(`${API_BASE_URL}/result/${requestId}`, {
        method: 'GET'
      }, 5000);
      
      if (!response.ok) {
        throw new Error(`결과 가져오기 오류: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      logDebug("결과 가져오기 실패", error);
      throw error;
    }
  };
  
  // 메시지 전송 처리
  const sendMessage = async () => {
    const text = inputMessage.trim();
    if (!text) return;
    
    // 메시지 입력창 초기화
    setInputMessage('');
    
    // 사용자 메시지 추가
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: text,
      time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
    };
    setMessages(prev => [...prev, userMessage]);
    
    // 로딩 메시지 추가
    const loadingId = `loading-${Date.now()}`;
    setMessages(prev => [...prev, {
      id: loadingId,
      role: 'assistant',
      content: '<div class="loading-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>',
      time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
    }]);
    
    setIsLoading(true);
    
    try {
      // 현재 사용자 메시지를 대화 기록에 추가
      const updatedHistory = [
        ...chatHistory,
        { role: "user", content: text }
      ];
      setChatHistory(updatedHistory);
      
      // 1. 쿼리 제출하여 요청 ID 받기
      const response = await submitQuery(text, updatedHistory);
      const requestId = response.request_id;
      
      // 2. 요청 상태 주기적으로 확인
      let isComplete = false;
      let maxAttempts = 180; // 최대 180회 시도 (2초 간격으로 약 6분까지 대기)
      let attempts = 0;
      let longWaitNotified = false;
      
      while (!isComplete && attempts < maxAttempts) {
        attempts++;
        await new Promise(resolve => setTimeout(resolve, 2000)); // 2초 대기
        
        // 오래 걸리는 작업일 경우 사용자에게 알림
        if (attempts === 60 && !longWaitNotified) { // 2분 경과 시점
          longWaitNotified = true;
          // 로딩 메시지 업데이트
          setMessages(prev => prev.map(msg => 
            msg.id === loadingId
              ? {
                  ...msg,
                  content: `<p>요청 처리에 시간이 조금 걸리고 있습니다. 곧 답변이 준비될 예정입니다.</p>
                          <p>잠시만 더 기다려주세요...</p>`
                }
              : msg
          ));
        }
        
        try {
          const statusResponse = await checkQueryStatus(requestId);
          
          if (statusResponse.status === 'complete') {
            // 3. 결과 가져오기
            const resultResponse = await getQueryResult(requestId);
            
            // 로딩 메시지 제거
            setMessages(prev => prev.filter(msg => msg.id !== loadingId));
            
            // AI 응답을 대화 기록에 추가
            setChatHistory(prev => [
              ...prev,
              { role: "assistant", content: resultResponse.answer }
            ]);
            
            // AI 응답 표시
            setMessages(prev => [...prev, {
              id: Date.now(),
              role: 'assistant',
              content: formatAIResponse(resultResponse.answer),
              time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
            }]);
            
            isComplete = true;
            break;
          }
        } catch (error) {
          console.error("상태 확인 오류:", error);
          // 오류가 발생해도 계속 시도
        }
      }
      
      // 최대 시도 횟수를 초과했는데 완료되지 않은 경우
      if (!isComplete) {
        throw new Error("요청 처리 시간이 너무 오래 걸립니다. 잠시 후 다시 시도해주세요.");
      }
      
    } catch (error) {
      console.error('API 통신 오류:', error);
      
      // 로딩 메시지 제거
      setMessages(prev => prev.filter(msg => msg.id.toString().startsWith('loading')));
      
      // 오류 메시지 표시
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'assistant',
        content: `<p>죄송합니다. 요청 처리 중 오류가 발생했습니다:</p><p>${error.message}</p>`,
        time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
      }]);
    } finally {
      setIsLoading(false);
    }
  };
  
  // 엔터키 처리
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="jsx-chat-container">
      <div className="jsx-chat-header">
        <div className="jsx-chat-logo">
          <img src="/chatbot.png" alt="제플몰 AI" />
          <h2>제플몰 AI 상담</h2>
        </div>
      </div>
      
      <div className="jsx-chat-messages" ref={chatAreaRef}>
        {messages.map(message => (
          <div key={message.id} className={`jsx-message ${message.role}-message`}>
            <div className="jsx-message-avatar">
              <div className={`${message.role}-icon`}>
                <img 
                  src={message.role === 'assistant' ? '/chatbot.png' : '/user.png'} 
                  alt={message.role === 'assistant' ? '제플몰 AI' : '사용자'} 
                />
              </div>
            </div>
            <div className="jsx-message-content">
              <div className="jsx-message-header">
                <span className="jsx-sender-name">
                  {message.role === 'assistant' ? '제플몰' : '사용자'}
                </span>
                <span className="jsx-message-time">{message.time}</span>
              </div>
              <div className="jsx-message-bubble">
                <div 
                  className="jsx-message-text" 
                  dangerouslySetInnerHTML={{ __html: message.content }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="jsx-chat-input">
        <textarea
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="메시지를 입력하세요..."
          disabled={isLoading}
          className="jsx-message-input"
        />
        <button 
          onClick={sendMessage} 
          disabled={isLoading || !inputMessage.trim()}
          className="jsx-send-button"
        >
          전송
        </button>
      </div>
    </div>
  );
}

export default ChatPopupJsx; 