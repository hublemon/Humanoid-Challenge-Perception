# GitHub Upload

이 워크스페이스는 `.git` 경로가 읽기전용 마운트로 잡혀 있어서 일반 `git` 명령이 실패할 수 있습니다.
대신 실제 Git 저장소는 `.git-real/`에 만들었고, `./gitws` 래퍼로 관리합니다.

## Check Status

```bash
cd /home/parkum/robotis_ros2_ws
./gitws status
```

## Commit

처음 한 번만 Git 작성자 정보를 설정하세요.

```bash
./gitws config user.name "<your-name>"
./gitws config user.email "<your-email>"
./gitws commit -m "Initial ROS 2 Jazzy perception workspace"
```

## Push To GitHub

GitHub에서 빈 repository를 만든 뒤 remote를 추가하세요.

SSH:

```bash
./gitws remote add origin git@github.com:<your-id>/<repo-name>.git
./gitws push -u origin main
```

HTTPS:

```bash
./gitws remote add origin https://github.com/<your-id>/<repo-name>.git
./gitws push -u origin main
```

## Excluded From Git

- `build/`, `install/`, `log/`
- `ocr_venv/`, `yolo_venv/`
- `paddlex_cache/`, `_incoming/`
- zip backups, wheel bundles, Python caches
- model weights such as `*.pt`

모델 파일은 GitHub Release, Git LFS, Hugging Face, Google Drive, 또는 사내 스토리지로 따로 관리하는 것을 권장합니다.
