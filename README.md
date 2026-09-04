# 정체성 보존 눈썹 국소 변형 파이프라인

얼굴 형태·피부색·화질 등 정체성은 그대로 유지한 채 **눈썹 영역만** 원하는 스타일로 자연스럽게 교체하는 파이프라인입니다. 얼굴 전체를 재생성하면 화질 저하와 피부색 변화가 발생하므로, 눈썹 영역만 국소적으로 수정하는 것을 핵심 제약으로 삼았습니다.

원래 Google Drive(Colab) 환경에서 작업하던 프로젝트를 GitHub로 옮긴 저장소입니다. 아래 "실행 준비" 항목에 Drive 환경과 달라지는 부분을 정리했습니다.

## 파이프라인 구조

```
원본 이미지
  → MediaPipe FaceLandmarker 랜드마크 추출 (+ BiSeNet 합집합 마스크, 선택)
  → 눈썹 중심 512×512 정사각 크롭
  → STEP 2: LaMa 인페인팅으로 기존 눈썹 제거
  → STEP 3: SD1.5(epiCRealism) Inpaint + 연예인 눈썹 LoRA로 눈썹 생성
  → STEP 4: 원본 해상도로 복원 (마스크 밖 영역은 원본 픽셀 그대로 유지)
```

## 저장소 구조

```
conditional_gen/
├── codes/
│   ├── testing.ipynb          # 추론(시연) 코드
│   ├── train.ipynb            # LoRA 학습 코드
│   └── 과거에 사용했던 코드/     # 이전 버전 노트북 모음 (참고용, 현재 미사용)
├── data/
│   ├── actor_raw_data/         # 연예인별 원본 학습 이미지 (<celeb>_llm 폴더 단위)
│   └── raw_face_data/
├── lora_checkpoint/            # 연예인/캐릭터별 학습된 LoRA 가중치
├── mediapipe_bisenet_utils/    # 눈썹 마스킹용 유틸 (face_landmarker.task, resnet18.onnx)
├── test_img/                   # 테스트용 원본 이미지
└── test_result/                # 테스트 결과 이미지 저장 위치
```

## 실행 준비 (Drive → GitHub 전환 시 주의사항)

노트북 코드 자체는 Drive에서 쓰던 것과 동일하며, `testing.ipynb`의 `ROOT_PATH`와 `train.ipynb`의 `root` 변수가 여전히 `"/content/drive/MyDrive/eyebrows"`로 고정되어 있습니다. 아래 중 편한 방식으로 맞추면 됩니다.

- **Option A (Drive에서 그대로 실행)**: 이 저장소를 Google Drive의 `MyDrive/eyebrows` 경로에 clone(또는 폴더째 업로드) → 노트북 수정 없이 기존과 동일하게 실행
- **Option B (다른 경로에서 실행)**: 원하는 위치에 clone한 뒤, 노트북 상단의 `ROOT_PATH`(`testing.ipynb`) / `root`(`train.ipynb`) 값을 실제 clone 경로로 수정

> `lora_checkpoint/`, `data/actor_raw_data/` 등 이미지·가중치 파일은 Git LFS로 관리됩니다. clone 후 `git lfs pull`을 실행해야 실제 파일이 받아집니다.


## 1. codes/

### testing.ipynb — 추론(시연) 코드

노트북을 위에서부터 순서대로 실행합니다.

**테스트 이미지 변경**: Drive에서는 `test_img` 폴더 안 이미지(`test5.png` 등)로 경로만 바꿔서 사용했는데, GitHub에서도 방식은 동일합니다. STEP 1 셀의 아래 줄에서 파일명만 바꾸면 됩니다.

```python
S = step1_prepare(os.path.join(ROOT_PATH, "test_img/test.jpg"), use_bisenet=USE_BISENET)
```

→ `test_img/test.jpg`를 `test_img/test5.png` 등 `test_img` 폴더 안의 다른 이미지 경로로 바꿔주면 됩니다.

**사용할 LoRA(연예인) 변경**: "여기서 모델을 바꿔주세요!!" 셀 아래 `resolve_lora_checkpoint_path()` 함수의 `candidates` 리스트에 사용 가능한 LoRA 이름이 모두 주석 처리되어 있습니다. 사용하고 싶은 이름의 주석(`#`)만 해제하고 나머지는 주석 처리한 상태로 두면, 그 이름의 `lora_checkpoint/<이름>/` 폴더에 있는 가중치가 로드됩니다.

```python
candidates = [
    #'cara_llm'
    #"celebfit_eyebrows",
    #"chaeunwoo_llm"
    #"shinsekyung_llm"
    #"v_llm"
    'hermione_llm'          # ← 이 줄만 주석 해제된 상태면 hermione_llm 가중치를 사용
    #"iu_llm"
    #"kimwoobin_llm"
    #"goyounjung_llm"
]
```

마지막 STEP 3 실행 시에는 `연예인이름을 영어로 쓰세요` 프롬프트가 뜨는데, 이때 입력하는 이름은 LoRA 트리거 워드(예: `hermione`)로 위에서 선택한 LoRA와 맞춰서 입력하면 됩니다.

### train.ipynb — LoRA 학습 코드

- 학습할 대상은 셀 하나에서만 입력하면 되도록 되어 있습니다.
  ```python
  celeb = input('학습할 연예인 명을 영어로 입력(소문자로): ')
  ```
- 학습 데이터: `data/actor_raw_data/<celeb>_llm/` 폴더의 원본 이미지를 사용합니다.
- 결과물: `lora_checkpoint/<celeb>_llm/`에 `unet/`, `text_encoder/` 어댑터 가중치가 저장되고, 학습 중간에도 `checkpoints/checkpoint-<step>/`에 주기적으로 저장됩니다.
- 여기서도 `root` 변수가 저장소 경로를 올바로 가리키는지 먼저 확인해야 합니다.

## 2. lora_checkpoint/

연예인/캐릭터별로 학습된 LoRA 가중치 모음입니다. `testing.ipynb`가 `resolve_lora_checkpoint_path()`를 통해 이 중 하나를 선택해 로드합니다.

## 3. mediapipe_bisenet_utils/

눈썹 영역을 마스킹하기 위해 불러오는 유틸/가중치입니다.
- `face_landmarker.task`: MediaPipe FaceLandmarker 모델
- `resnet18.onnx`: BiSeNet(face-parsing) 보조 마스크용 백본

## 4. test_img/

`testing.ipynb`에서 사용하는 테스트용 원본 사진 모음입니다.

## 5. test_result/

테스트 실행 결과 이미지가 저장되는 폴더입니다.
