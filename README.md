# MedSAM Web UI — 의료 이미지 세그멘테이션 MVP

SAM(Segment Anything Model)을 의료 이미지에 적용한 인터랙티브 웹 데모입니다.  
흉부 X-ray, 뇌 CT, 복부 CT, 피부 병변 이미지를 박스 드래그 한 번으로 세그멘테이션합니다.

---

## 화면 구성

| 탭 | 설명 |
|----|------|
| **세그멘테이션 데모** | 직접 이미지 업로드 또는 샘플 선택 → 드래그로 ROI 지정 → 결과 오버레이 표시 |
| **논문 비교** | 의료 샘플 이미지로 MedSAM 결과 확인 + 논문(Med-SA)과 차이점 시각적 비교 |

---

## 기술 스택

- **백엔드**: Python 3.10 / Flask
- **모델**: [MedSAM](https://github.com/bowang-lab/MedSAM) — SAM ViT-B (358MB)
- **참고 논문**: [Medical SAM Adapter (Med-SA)](https://doi.org/10.1016/j.media.2025.103547) — Medical Image Analysis 2025
- **프론트엔드**: Vanilla JS / HTML Canvas (드래그 박스 프롬프트)

---

## 설치 및 실행

### 1. 저장소 클론

```bash
git clone <이 저장소 URL>
cd ocr-vision
```

### 2. 가상환경 생성 및 패키지 설치

```bash
python -m venv venv
# Windows
source venv/Scripts/activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e MedSAM/
```

### 3. MedSAM 소스 클론 (서브모듈로 포함되어 있지 않을 경우)

```bash
git clone https://github.com/bowang-lab/MedSAM.git
```

### 4. 모델 가중치 다운로드

```bash
mkdir -p checkpoints
curl -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
     -o checkpoints/medsam_vit_b.pth
```

> 파일 크기: 약 358MB

### 5. 샘플 의료 이미지 생성

```bash
python generate_samples.py
```

### 6. 서버 실행

```bash
python app.py
```

브라우저에서 `http://localhost:5000` 접속

---

## 사용 방법

1. 브라우저에서 `http://localhost:5000` 열기
2. **모델 로드하기** 버튼 클릭 (최초 1회, 30초~1분 소요)
3. **세그멘테이션 데모** 탭
   - 좌측 샘플 이미지 클릭 또는 직접 업로드
   - 이미지 위에서 마우스 드래그 → ROI 박스 지정
   - 세그멘테이션 결과 오버레이 및 신뢰도 점수 확인
4. **논문 비교** 탭
   - 상단 버튼에서 의료 이미지 유형 선택
   - 드래그로 ROI 지정 → MedSAM 결과 확인
   - 우측에서 논문(Med-SA)과의 차이점 및 성능 비교 확인

---

## 프로젝트 구조

```
ocr-vision/
├── app.py                    # Flask 백엔드 (OCR 엔드포인트, 모델 로딩)
├── templates/
│   └── index.html            # 프론트엔드 UI (Canvas 드래그, 탭 전환)
├── checkpoints/
│   └── medsam_vit_b.pth      # SAM ViT-B 모델 가중치 (git 제외)
├── MedSAM/                   # MedSAM 소스 (bowang-lab)
│   └── assets/
│       └── samples/          # 의료 샘플 이미지 (생성 스크립트로 생성)
├── uploads/                  # 사용자 업로드 임시 저장
└── venv/                     # 가상환경 (git 제외)
```

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 메인 UI |
| GET | `/status` | 모델 로드 상태 및 디바이스 확인 |
| POST | `/load_model` | 모델 로드 실행 |
| GET | `/samples` | 기본 샘플 이미지 목록 |
| GET | `/sample_image/<filename>` | 샘플 이미지 서빙 |
| GET | `/medical_samples` | 의료 샘플 메타데이터 (라벨, 권장 박스 등) |
| GET | `/medical_image/<filename>` | 의료 샘플 이미지 서빙 |
| POST | `/segment` | 세그멘테이션 실행 (base64 이미지 + 박스 좌표) |

### `/segment` 요청/응답 예시

```json
// 요청
{
  "image": "<base64 PNG>",
  "box": [x1, y1, x2, y2]
}

// 응답
{
  "overlay": "<base64 PNG 오버레이>",
  "score": 0.923,
  "area_px": 14523,
  "area_pct": 5.62,
  "mask_size": [512, 512]
}
```

---

## 논문 비교: MedSAM vs Med-SA

| 항목 | MedSAM (현재 구현) | Med-SA (논문) |
|------|-------------------|--------------|
| 방식 | SAM 직접 사용 (Zero-shot) | Adapter 모듈 삽입 (PEFT) |
| 학습 파라미터 | 없음 | 전체의 2% (13M) |
| 3D CT/MRI | 미지원 | SD-Trans로 지원 |
| 프롬프트 | 박스만 | 박스 + 포인트 (HyP-Adpt) |
| BTCV 성능 | 기준선 | +9.4% 향상 |
| 검증 Task | - | 17가지 의료 이미지 |

> 논문: *Wu et al., Medical SAM Adapter, Medical Image Analysis 102 (2025)*

---

## 환경 요건

| 항목 | 최소 | 권장 |
|------|------|------|
| Python | 3.9+ | 3.10 |
| RAM | 8GB | 16GB |
| GPU | 없어도 동작 | CUDA GPU (처리 속도 대폭 향상) |
| 처리 시간 | CPU: 30초~1분/장 | GPU: 1~3초/장 |

---

## .gitignore 권장 항목

```
venv/
checkpoints/
uploads/
__pycache__/
*.pyc
*.pth
```
