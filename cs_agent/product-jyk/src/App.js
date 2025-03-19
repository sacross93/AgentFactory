import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import ChatbotUI from './components/ChatbotUI';
import './App.css';

function App() {
  const openChatPopup = () => {
    window.open(
      '/chat-popup.html', 
      'ChatPopup', 
      'width=375,height=600,resizable=yes'
    );
  };

  const openChatPopup2 = () => {
    window.open("/chat-popup2.html", "ChatPopup2", "width=375,height=600,resizable=yes")
  }
  
  return (
    <BrowserRouter>
      <div className="App">
        <nav className="main-nav">
          <Link to="/" className="nav-link">채팅</Link>
          <button 
            className="nav-link" 
            onClick={openChatPopup}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'white', padding: '10px 15px' }}
          >
            모바일 채팅 UI
          </button>
        </nav>
        
        <Routes>
          <Route path="/" element={<ChatbotUI />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
