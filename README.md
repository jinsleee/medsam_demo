# SAM vs MedSAM — 논문 발표 비교 데모

**같은 이미지, 같은 박스**로 SAM(일반)과 MedSAM(의료 파인튜닝)의 세그멘테이션 결과를 나란히 비교하는 인터랙티브 웹 데모입니다.

> Wu et al. (2025) *"Medical SAM Adapter"* 발표용 — Medical Image Analysis

---

## 화면 구성

| 패널 | 설명 |
|------|------|
| **입력** | 이미지 위에서 마우스 드래그 → ROI 박스 지정 |
| **SAM 결과** | 원본 SAM(Meta, 2023) 추론 결과 + IoU 점수 |
| **MedSAM 결과** | 100만+ 의료영상 파인튜닝 모델 추론 결과 + IoU 점수 |
| **비교 지표** | Dice 유사도, 면적 차이 — 두 모델의 차이를 정량화 |

---

## 기술 스택

- **백엔드**: Python 3.10 / Flask
- **모델**: SAM ViT-B + [MedSAM](https://github.com/bowang-lab/MedSAM) ViT-B
- **참고 논문**: [Medical SAM Adapter (Med-SA)](https://doi.org/10.1016/j.media.2025.103547) — Medical Image Analysis 2025
- **프론트엔드**: Vanilla JS / HTML Canvas

---

## 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/jinsleee/medsam_demo.git
cd medsam_demo
```

### 2. 가상환경 및 패키지 설치

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. MedSAM 소스 클론

```bash
git clone https://github.com/bowang-lab/MedSAM.git
pip install -e MedSAM/
```

### 4. 모델 가중치 다운로드

`checkpoints/` 폴더에 두 파일 모두 필요합니다.

```bash
mkdir checkpoints

# SAM ViT-B (원본)
curl -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
     -o checkpoints/sam_vit_b.pth

# MedSAM ViT-B (의료 파인튜닝)
# https://drive.google.com/file/d/1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ 에서 다운로드
# → checkpoints/medsam_vit_b.pth 로 저장
```

### 5. 이미지 준비

`image/` 폴더에 PNG/JPG 파일을 넣으면 사이드바에 자동으로 표시됩니다.

```bash
mkdir image
# 흉부 X-ray, CT 등 의료 이미지를 image/ 폴더에 복사
```

### 6. 서버 실행

```bash
python app.py
```

브라우저에서 `http://localhost:5020` 접속

---

## 사용 방법

1. `http://localhost:5020` 열기
2. **모델 로드** 버튼 클릭 (페이지 열면 자동 시도, CPU 기준 30초~1분)
3. 왼쪽 사이드바에서 이미지 클릭
4. 입력 패널 위에서 마우스 드래그 → ROI 박스 지정
5. SAM / MedSAM 결과 및 Dice 비교 지표 확인

> 사이드바는 `‹` / `›` 버튼으로 열고 닫을 수 있습니다.

---

## 프로젝트 구조

```
medsam_demo/
├── app.py                  # Flask 백엔드 (SAM + MedSAM 추론, 비교 지표)
├── templates/
│   └── index.html          # 프론트엔드 UI (3분할 비교 레이아웃)
├── checkpoints/            # 모델 가중치 (git 제외)
│   ├── sam_vit_b.pth
│   └── medsam_vit_b.pth
├── MedSAM/                 # bowang-lab MedSAM 소스 (git 제외)
├── image/                  # 입력 이미지 폴더 (git 제외)
├── requirements.txt
└── .gitignore
```

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 메인 UI |
| GET | `/status` | 모델 로드 상태 및 디바이스 |
| POST | `/load_model` | SAM + MedSAM 로드 |
| GET | `/list_images` | `image/` 폴더 이미지 목록 |
| GET | `/image/<filename>` | 이미지 서빙 |
| POST | `/segment` | SAM + MedSAM 동시 추론 |

### `/segment` 요청/응답

```json
// 요청
{ "image": "<base64 PNG>", "box": [x1, y1, x2, y2] }

// 응답
{
  "sam":    { "overlay": "<base64>", "score": 0.87, "area_px": 12400, "area_pct": 4.8 },
  "medsam": { "overlay": "<base64>", "score": 0.93, "area_px": 13100, "area_pct": 5.1 },
  "comparison": { "dice_between": 0.82, "area_diff_px": 700, "area_diff_pct": 0.27 }
}
```

---

## SAM vs MedSAM 비교 포인트

| 항목 | SAM (원본) | MedSAM (파인튜닝) |
|------|-----------|-----------------|
| 학습 데이터 | 일반 이미지 1B 마스크 | 의료 이미지 100만+ |
| 의료 특화 | 없음 | 있음 |
| 박스 프롬프트 | 지원 | 지원 |
| 처리 해상도 | 원본 그대로 | 1024×1024 리사이즈 |

> **논문 핵심:** 두 모델의 Dice 차이가 크다 = 일반 이미지 학습만으론 의료 특화 지식 부족.  
> Med-SA는 Adapter(13M, 전체의 2%)만 추가 학습해서 이를 해결합니다.

---

## 환경 요건

| 항목 | 최소 | 권장 |
|------|------|------|
| Python | 3.9+ | 3.10 |
| RAM | 8GB | 16GB |
| GPU | 불필요 (CPU 동작) | CUDA GPU |
| 추론 속도 | CPU: 30초~1분/장 | GPU: 1~3초/장 |
