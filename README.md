# SDXL LoRA Training Project

이 프로젝트는 SDXL 기반 모델을 LoRA를 사용하여 학습하는 파이프라인입니다.

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 학습 시작 (예: Animagine XL)
CUDA_VISIBLE_DEVICES=0 ./scripts/training/train_animagine.sh

# 3. 이미지 생성
CUDA_VISIBLE_DEVICES=0 python scripts/inference/generate_samples.py \
  --merged_model_path output/jingliu_sdxl_20_ep_dataset/animagine/trial1/merged/checkpoint-final
```

## 📁 프로젝트 구조

```
├── configs/          # 모델 설정 파일
├── scripts/          # 학습 및 추론 스크립트
│   ├── training/     # 학습 관련
│   └── inference/    # 추론 관련
├── docs/             # 상세 문서
├── tools/            # 유틸리티 도구
├── dataset/          # 학습 데이터
├── output/           # 결과물
└── test_images/      # 테스트 이미지
```

## 📚 문서

자세한 사용법은 [docs/README.md](docs/README.md)를 참고하세요.

### 주요 문서

- **[상세 README](docs/README.md)**: 전체 사용 가이드
- **[Image-to-Image 가이드](docs/IMG2IMG_GUIDE.md)**: 이미지 변환 가이드
- **[학습 일지](docs/TRAINING_LOG_animagine_trial1.md)**: 실험 기록

## 🎯 주요 기능

- **3가지 베이스 모델 지원**: Animagine XL, SDXL Base, HiDream
- **LoRA 학습**: 효율적인 파인튜닝
- **Text-to-Image**: 프롬프트로 이미지 생성
- **Image-to-Image**: 스타일 변환

## 🚀 실행 예시

### 학습

```bash
# 프로젝트 루트에서 실행
python scripts/training/train_lora.py --config configs/config_animagine.json
```

### 추론

```bash
# Text-to-Image
python scripts/inference/generate_samples.py \
  --merged_model_path output/jingliu_sdxl_20_ep_dataset/animagine/trial1/merged/checkpoint-final

# Image-to-Image
python scripts/inference/img2img_comparison.py \
  --input_image test_images/test_image_001.jpg \
  --merged_model_path output/jingliu_sdxl_20_ep_dataset/animagine/trial1/merged/checkpoint-final \
  --prompt "jingliu, 1girl, blue theme"
```

## 📊 시스템 요구사항

- **GPU**: NVIDIA GPU (16GB+ VRAM 권장)
- **RAM**: 32GB+ 권장
- **Storage**: 50GB+
