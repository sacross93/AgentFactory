# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)

# 제플몰 AI 상담 시스템

## 설치 및 실행 방법

### 백엔드 (FastAPI 서버)

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 서버 실행 (포트 8000):
```bash
cd cs_agent/product-jyk
python server.py
```

3. 서버가 올바르게 실행되었는지 확인:
   - 브라우저에서 http://localhost:8000/health 접속 후 상태 확인

### 프론트엔드 (React 앱)

1. 필요한 패키지 설치:
```bash
cd cs_agent/product-jyk
npm install
```

2. 개발 서버 실행 (포트 3000):
```bash
npm start
```

3. 단독 챗봇 인터페이스 실행:
   - 브라우저에서 http://localhost:3000/chat-popup.html 접속

## 통신 문제 해결

프론트엔드와 백엔드 통신 문제가 발생하는 경우:

1. 포트 확인:
   - 백엔드 서버가 8000 포트에서 실행 중인지 확인
   - chat-popup.html 파일에서 API_BASE_URL이 'http://localhost:8000'으로 설정되어 있는지 확인

2. CORS 이슈:
   - 브라우저 콘솔에서 CORS 오류가 발생하는 경우 server.py의 CORS 설정 확인

3. 네트워크 로그 확인:
   - 브라우저 개발자 도구의 Network 탭에서 API 요청과 응답 확인
