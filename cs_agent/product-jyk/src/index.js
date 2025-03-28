import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import ChatPopupJsx from './components/ChatPopupJsx';

// 메인 앱 렌더링
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// 전역 객체에 ChatPopupJsx 노출 (팝업 접근용)
window.React = React;
window.ReactDOM = ReactDOM;
window.ChatPopupJsx = ChatPopupJsx;

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
