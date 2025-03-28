import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import ChatPopupJsx from './components/ChatPopupJsx';

// 팝업용 진입점
function init() {
  // DOM이 준비되었는지 확인
  if (document.getElementById('root')) {
    const root = createRoot(document.getElementById('root'));
    root.render(
      <StrictMode>
        <ChatPopupJsx />
      </StrictMode>
    );
  }
}

// 페이지 로드 시 초기화
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

export default ChatPopupJsx; 