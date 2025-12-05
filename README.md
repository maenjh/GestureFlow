# Magic Hand Filter (GestureFlow)

웹캠을 통해 실시간으로 손 제스처를 인식하고, 제스처에 따라 다양한 비디오 필터를 적용하는 웹 애플리케이션입니다. Flask, OpenCV, MediaPipe를 사용하여 개발되었습니다.

## 📸 주요 기능

1.  **실시간 비디오 스트리밍**: 웹캠 영상을 웹 브라우저에서 지연 없이 확인할 수 있습니다.
2.  **손 제스처 인식 및 필터 적용**:
    *   🖐 **Palm (보자기)**: 기본 화면 (Normal)
    *   ✊ **Fist (주먹)**: 흑백 필터 (Grayscale)
    *   ✌️ **Peace (가위)**: 스케치 필터 (Canny Edge Detection)
3.  **제스처 기반 자동 촬영**:
    *   **주먹(Fist)**을 쥐고 있으면 3초 카운트다운 후 자동으로 사진이 촬영됩니다.
4.  **수동 촬영 및 갤러리**:
    *   UI의 'Capture' 버튼을 눌러 즉시 촬영할 수 있습니다.
    *   촬영된 사진은 우측 갤러리 패널에서 실시간으로 확인할 수 있습니다.

## 🛠 기술 스택

*   **Language**: Python 3.9+
*   **Web Framework**: Flask
*   **Computer Vision**: OpenCV, MediaPipe
*   **Frontend**: HTML, CSS, JavaScript

## 🚀 설치 및 실행 방법

### 1. 환경 설정 (Conda 권장)

```bash
conda create -n cam python=3.11
conda activate cam
```

### 2. 프로젝트 클론 및 이동

```bash
cd GestureFlow
```

### 3. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 애플리케이션 실행

```bash
python app.py
```

### 5. 웹 브라우저 접속

브라우저를 열고 아래 주소로 접속하세요.

`http://127.0.0.1:5001`

## 📂 프로젝트 구조

```
GestureFlow/
├── app.py              # 메인 Flask 애플리케이션 및 로직
├── requirements.txt    # 필요 라이브러리 목록
├── templates/
│   └── index.html      # 웹 인터페이스 (UI)
└── output/             # 촬영된 이미지가 저장되는 폴더
```

## 📝 사용 가이드

1.  웹캠이 연결된 상태에서 앱을 실행합니다.
2.  카메라를 향해 손을 들어 제스처를 취해보세요.
    *   가위(Peace)를 하면 화면이 스케치처럼 변합니다.
    *   주먹(Fist)을 쥐면 화면이 흑백으로 변하고, 3초 뒤에 사진이 찍힙니다.
3.  찍힌 사진은 화면 오른쪽 갤러리에 자동으로 나타납니다.
4.  사진을 클릭하면 원본 크기로 볼 수 있습니다.
