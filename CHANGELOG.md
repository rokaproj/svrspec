# 변경 이력

형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)를 따르고,
버전은 [유의적 버전](https://semver.org/lang/ko/)을 따른다.

## [0.1.0] — 2026-07-30

첫 배포.

### 산정 엔진

- CPU 추론의 두 병목을 분리한 루프라인 모델 — 토큰 생성은 메모리 대역폭 바운드,
  프롬프트 처리는 연산 바운드(AVX2 / AVX-512 / AMX 벡터 폭)
- 알람 도착·스톰을 이산사건 큐로 시뮬레이션. 평상시 지연 SLA와 스톰 소진 목표를 분리 판정
- 판정은 예측 불확실도만큼 성능을 깎은 **불리한 추정** 기준
- RAM 산정(가중치 + KV 캐시 + 컴퓨트 버퍼 + OS)과 실제 GGUF 파일 대조 검증
- 최소 / 권장 / 여유 3단 스펙 산출

### 카탈로그

- 모델 32종 (0.5B~70B, MoE 2종). 전부 실제 HuggingFace `config.json`에서 수집
- CPU 28종 (Xeon SP 3·4·5세대, Xeon E-2400, EPYC Milan·Genoa·Siena).
  AMD 10종은 벤더 데이터시트, Intel 18종은 Wikipedia SKU 표 + PassMark
- 메모리 35종. 채널당 2 DIMM 속도 하락 반영
- 효율 계수 9종. 각 행이 실측 / 실측유도 / 추정 중 무엇인지 표시
- 모든 행에 출처가 필수이고, 미확인 행은 리포트에 별도로 나열된다
- PassMark CPU Mark로 전사된 스펙을 감사 (`catalog validate`)

### 화면

- 네이티브 데스크톱 창 (Edge WebView2, 서버·포트 없음)
- 브라우저 GUI (`svrspec gui`, 리눅스·에어갭 서버용)
- 토큰 전달 시뮬레이터 — 구간별 시간과 예측 속도 실시간 재생
- 작업관리자 — 하드웨어 고정, 알람 개수별 CPU·RAM·대역폭 사용량 비교
- 단일 파일 HTML 납품 리포트 + CSV + JSON
- 다크/라이트, 키보드 조작, 토큰 기반 디자인 시스템

### 배포

- Windows 설치 프로그램(`setup.exe`)과 MSI
- 에어갭 서버용 zip 번들 (`svrspec bundle`) — 표준 라이브러리만 쓰므로 `python3`만 있으면 실행
- GitHub 릴리스에서 업데이트 확인 및 설치 (SHA-256 검증 후 실행)
