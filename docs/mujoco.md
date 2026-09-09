# MuJoCo 실행과 강화학습

MuJoCo 3.12.0의 네이티브 CPU 엔진을 사용합니다. 경기장 치수는 기존 `arena_spec.json`에서 읽고, 재질과 접착 설정은 `arena_mujoco/materials.py`에서 관리합니다. Python 3.11 이상을 권장합니다.

## 설치와 실행

저장소 최상위 폴더에서 실행합니다.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-mujoco.txt
.venv/bin/python -m arena_mujoco run --seconds 0.1
.venv/bin/python -m arena_mujoco render --seconds 0 --output arena.png
```

대화형 뷰어는 Linux에서 `.venv/bin/python -m arena_mujoco view`, macOS에서 `.venv/bin/mjpython -m arena_mujoco view`로 엽니다. 기본 로봇은 정지 상태입니다. `--drive 0.3 0.3`을 추가하면 앞으로 움직입니다.

`--config senior_preliminary`가 기본 배치입니다. 사진 배치는 `photo_reference`, 울타리 포함 배치는 `framed_reference`로 선택합니다. `--no-robot`은 로봇을 제외합니다.

```sh
.venv/bin/python -m arena_mujoco build --config senior_preliminary
.venv/bin/python -m arena_mujoco run --profile profiles/measured.json --randomize --seed 42
```

빌드 명령은 외부 메시 파일이 필요 없는 MJCF XML과 같은 이름의 JSON 메타데이터를 저장합니다. 기본 경로는 `output/mujoco/`입니다. XML만 일반 MuJoCo 뷰어로 열면 초기 접착력은 유지되지만, 손상·마모·접착 이력과 속도에 따른 마찰 갱신에는 이 저장소의 `ArenaSimulation` 실행 루프가 필요합니다.

## 물리 모델

| 항목 | 구현 |
| --- | --- |
| 바닥 | 실제 크기의 유한한 강체 받침, 위쪽 Z=0. 경기장 밖으로 떨어질 수 있습니다. |
| 블록과 빔 | 실제 크기의 자유 강체. 부피와 밀도로 질량·관성 모멘트를 계산합니다. |
| 실험실 판 | 구멍 3개를 보존하는 204개 볼록 충돌 메시. 구멍 윤곽 오차는 최대 약 0.036 mm입니다. |
| 마찰 | 나무/종이, 나무/PVC, 고무/종이, 고무/PVC 등을 별도로 설정합니다. 네 벽은 각각 배율을 가집니다. |
| 접촉 | 미끄럼, 비틀림, 구름 저항과 유한 접촉 순응성을 사용합니다. 접촉 시간 상수와 감쇠를 조정할 수 있습니다. |
| 테이프 | 폭 20 mm, 총 두께 0.15 mm의 2차원 삼각형 flex. 인장과 굽힘 탄성이 있으며 실제 면적에 따라 질량을 배분합니다. |
| 접착 | 아래쪽 절점마다 면적에 비례한 접착 접촉을 둡니다. 벌어짐·전단 이동이 누적되면 접착력이 감소합니다. 윗면은 마른 PVC 접촉입니다. |
| 시공 상태 | 한 겹으로 맞댄 테이프 배치에 끝단 들뜸, 잔류 곡률과 오염을 설정할 수 있습니다. |
| 접착 이력 | 접착 시간, 속도 의존 강도, 마모와 재접착을 선택적으로 설정합니다. 각 계수는 실측 후 사용합니다. |
| 로봇 | 질량 1 kg의 예제 차동 구동 로봇. 바퀴·캐스터·범퍼, 모터 토크/속도 한계, 관절 저항, 명령 지연과 모터 응답 지연을 포함합니다. |

기본 테이프는 1,323개 절점을 사용하며 물리 시간 간격은 0.2 ms입니다. 절점의 접착 접촉이 바닥 지지를 담당합니다. 연속 flex의 바닥 접촉을 동시에 켜면 지지가 중복되어 불안정해질 수 있습니다. 바퀴·블록과 테이프의 접촉, 테이프끼리의 접촉은 유지합니다.

두 겹으로 겹친 접합부는 안정성 검증을 통과하지 못해 실행 설정에서 제외했습니다. `joins="overlap"`은 빌드 단계에서 오류로 알려줍니다. 겹침 영역을 계산하는 기하 함수와 실패 시험의 근거는 후속 모델 개선을 위해 보존했습니다.

마찰 배열의 순서는 `[정지 계수, 운동 계수, 비틀림 길이(m), 구름 길이(m)]`입니다. 네이티브 접촉 원뿔에 속도 기반 계수 전환을 추가했습니다. 강체끼리는 접촉 쌍별로 갱신하고, flex와 접촉한 강체는 그 형상의 최대 접촉 미끄럼 속도로 PVC 계수를 갱신합니다. 이 계수 전환은 실측 마찰 법칙의 근사입니다.

## Gymnasium 사용

```python
from arena_mujoco import ArenaEnv

env = ArenaEnv(tape_mode="flex", randomize=True)
observation, info = env.reset(seed=42)
for _ in range(20):
    observation, reward, terminated, truncated, info = env.step([0.2, 0.2])
    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

동작은 왼쪽·오른쪽 바퀴 속도 명령이며 범위는 각각 -1부터 1까지입니다. 기본 제어 주기는 10 ms입니다. 관측에는 로봇과 물체의 위치·자세·속도, 바퀴 상태와 목표점이 들어갑니다. 물체의 정확한 상태를 포함하는 특권 관측입니다. 실제 카메라 정책에는 이미지 관측과 센서 보정이 필요합니다. `render_mode="rgb_array"`로 렌더링할 수 있습니다.

예제 보상은 빨간 원기둥 하나를 의료 구역의 목표점으로 미는 과제입니다. 대회 전체 채점 규칙은 별도로 구현해야 합니다. 위치 이탈, 전복, 과도한 테이프 손상과 시간 제한을 감지합니다. `env.get_state()`와 `env.set_state(state)`는 물리 상태, 접착 이력, 난수 상태와 지연된 동작을 함께 보존합니다. 서로 다른 환경은 재질 값을 갱신하므로 각각 독립된 `MjModel`을 사용해야 합니다.

`tape_mode="rigid"`는 변형 없는 0.15 mm 테이프를 쓰는 빠른 근사이며, `"none"`은 테이프를 제거합니다. 최종 정책 검증에는 `"flex"`를 사용합니다. A100 런타임과 GPU 백엔드의 실제 호환성 결과는 [Colab 기록](colab_runtime.md)에 있습니다.

## 실제 로봇으로 바꾸기

`arena_mujoco/robot.py`의 `add_robot()`을 실제 로봇의 MJCF 구조로 교체합니다. 바퀴 지름·폭·축간 거리, 무게중심·질량·관성, 범퍼 위치, 바퀴 재질, 모터·감속기 한계를 측정해 입력합니다. `motor_names`, `wheel_joints` 메타데이터와 환경의 관측 구성을 함께 맞춥니다. 경기장만 필요한 경우 `build_arena(robot=False)`가 생성한 XML에 로봇을 결합하고, 같은 메타데이터로 `TapeController`를 초기화합니다.

## 보정과 적용 범위

접착제와 표면 마찰은 아직 실제 경기장 부품에 맞춰 보정되지 않았습니다. [측정·보정 안내](mujoco_calibration.md)를 따라 값을 입력합니다. [물성 출처](mujoco_physics_sources.md)에는 제조사 자료와 모델 가정이 구분되어 있습니다.

접착 모델은 네이티브 접촉의 인장 한계를 상태에 따라 줄이는 방식입니다. `fracture_energy_j_m2`는 연화 거리의 설정에 쓰이며, 실제 소산 에너지와 같다고 보장하지 않습니다. 박리 힘은 메시 간격·당기는 각도·속도에 민감하므로 실물 시험과 메시/시간 간격 수렴 확인이 필요합니다. PVC의 소성 변형·찢어짐, 종이 박리, 나무 파손은 구현되어 있지 않습니다. 기본 모델은 실내에서 이 재료들의 파손이 일어나지 않는 로봇 조작 범위를 대상으로 합니다.

기본 잔류 곡률과 마모 계수는 0입니다. 테이프가 저절로 말리는 조건은 실제로 떼어낸 테이프의 곡률을 측정해 설정합니다. 기본 모터와 센서 구성도 예제 로봇 값입니다. 센서 노출·렌즈 왜곡·모션 블러, 통신 패킷 손실과 배터리 변화가 학습에 영향을 준다면 해당 하드웨어 자료로 추가해야 합니다.

## 검증

```sh
.venv/bin/python -m unittest discover -s tests/mujoco -p 'test_*.py' -v
.venv/bin/python scripts/validate_mujoco.py
.venv/bin/python scripts/validate_mujoco.py --drive --seconds 0.8 --output output/mujoco/wheel_crossing.json
```

첫 명령은 로봇 구동과 Gym 계약, 재현 가능한 에피소드·상태 복원, 접착 시험을 확인합니다. 나머지 명령은 컴파일된 형상·질량·구멍과 실제 물리 스텝을 검사합니다. 전체 flex 장면은 세밀한 시간 간격 때문에 실시간보다 느립니다. 검증 보고서에 시뮬레이션 시간과 실제 실행 시간을 함께 기록합니다.
