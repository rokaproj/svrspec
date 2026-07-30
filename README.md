# svrspec

CPU 전용 LLM 서버 스펙 산정 시뮬레이터. **모델을 넣으면 서버 스펙을 추천한다.**

관제 솔루션이 발생시킨 알람을 sLLM/LLM이 정제·분석해 Microsoft Teams로 자동 전송하는
파이프라인에서, 중간 LLM 단계를 어떤 서버에 올려야 하는지를 근거와 함께 산정한다.

**순수 해석적 시뮬레이터다.** 모델을 실제로 돌리지 않고, 로컬 CPU에 부하를 주지도 않는다.
손에 없는 서버를 산정하는 도구이므로 지금 돌고 있는 머신을 벤치마킹하는 것은 애초에 다른
질문에 답하는 일이다. 모든 예측은 공개된 하드웨어 스펙 + 출처가 붙은 효율 계수에서 나온다.
표준 라이브러리만 쓰므로 산정 대상인 에어갭 CPU+RAM 서버에서 그대로 돌아간다.

```bash
svrspec gui                                               # 브라우저 GUI (권장)
python3 -m svrspec.cli list models                        # 모델 카탈로그 (70B 이하)
python3 -m svrspec.cli recommend --model qwen2.5-7b-instruct   # CPU 후보 28종 전체 산정
python3 -m svrspec.cli fit --cpu xeon-silver-4410y        # 이 서버에 어디까지 올라가나
python3 -m svrspec.cli size --model exaone-3.5-2.4b-instruct --cpu epyc-9354  # 상세 내역
```

## 설치

[**Releases**](https://github.com/rokaproj/svrspec/releases)에서 내려받는다.

| 파일 | 용도 |
|---|---|
| `svrspec-<버전>-setup.exe` | Windows 설치 프로그램. 사용자 단위 설치라 UAC를 띄우지 않는다 |
| `svrspec-<버전>-win64.msi` | MSI. 그룹 정책 배포용 |
| `svrspec-<버전>-airgap.zip` | 인터넷 없는 리눅스 서버용. `python3`만 있으면 실행 |
| `SHA256SUMS` | 위 파일들의 SHA-256. 앱의 자동 업데이트가 이 값으로 무결성을 검증한다 |

둘 다 같은 내용을 설치한다 — 네이티브 창 앱 `svrspec.exe`와 콘솔 도구 `svrspec-cli.exe`.
`setup.exe`는 사용자 단위로 설치해 UAC를 띄우지 않고, MSI는 그룹 정책 배포용이다.
Windows 11이면 필요한 WebView2 런타임이 이미 들어 있다.

**데스크톱 앱은 웹서버를 띄우지 않는다.** HTML을 Edge WebView2 네이티브 창에 문자열로 넘기고,
JavaScript는 `fetch` 대신 pywebview 브리지로 파이썬을 직접 호출한다. 열려 있는 포트도,
방화벽 프롬프트도, 설명해야 할 localhost URL도 없다.

### 자동 업데이트

앱이 시작할 때 GitHub Releases를 한 번 확인하고, 새 버전이 있으면 헤더에 버튼이 나타난다.
누르면 설치 파일을 내려받아 **릴리스에 게시된 `SHA256SUMS`와 대조한 뒤에만** 실행한다 —
받은 파일을 실행하는 경로이므로 해시가 없거나 어긋나면 자동 설치를 중단하고 릴리스 페이지를
안내한다.

확인은 짧은 타임아웃으로 한 번만 하고 실패하면 조용히 넘어간다. 완전히 끄려면:

```
SVRSPEC_NO_UPDATE_CHECK=1
```

에어갭 서버가 연결되지 않을 소켓에서 멈추는 일이 없어야 하기 때문이다.

### 릴리스 만들기

버전을 올리고 태그를 밀면 CI가 빌드·검증·게시까지 한다.

```bash
# 1) svrspec/__init__.py 의 __version__ 과 CHANGELOG.md 갱신
git commit -am "release: v0.2.0"
git tag v0.2.0
git push origin main --tags
```

`release.yml`이 windows-latest에서 MSI와 setup.exe를 만들고, **패키징된 앱을 실제로 실행해
창이 열리는지 확인한 뒤**, 체크섬과 함께 Releases에 올린다. 태그와
`svrspec.__version__`이 다르면 빌드를 거부한다 — 버전만 안 올리고 태그를 미는 실수를 막는다.

### 직접 빌드

Windows에서만 된다(크로스 컴파일 불가):

```powershell
python installer/setup_cxfreeze.py build_exe    # 폴더 + exe 2개
python installer/setup_cxfreeze.py bdist_msi    # MSI
ISCC.exe installer/svrspec.iss                  # setup.exe
```

`bdist_msi`는 TEMP 경로에 비ASCII 문자가 있으면 `FCI error 4`로 죽는다(msilib의 CAB 생성기
한계). 한글 사용자명 환경에서는 TEMP를 ASCII 경로로 바꿔야 한다 — `installer` 빌드 스크립트가
그렇게 한다.

## 브라우저 GUI (리눅스·에어갭 서버용)

```bash
svrspec app                 # 네이티브 창 (Windows, 서버 없음)
svrspec gui                 # 브라우저 방식, http://127.0.0.1:8765
svrspec gui --port 9000 --no-browser
```

왼쪽에 조건(모델·양자화·알람 부하·프롬프트 토큰·소켓/DIMM), 오른쪽에 결과(권장 스펙 3단 카드,
CPU 후보 28종 표, 이 산정에 쓰인 효율 계수와 근거, 주의사항)가 놓인다. **입력을 바꾸면 즉시
다시 계산된다** — 28종 전수 산정이 약 45 ms라서 디바운스나 실행 버튼이 필요 없다. 알람 수를
올려보며 권장 스펙이 어느 지점에서 넘어가는지 직접 볼 수 있다.

표준 라이브러리 `http.server`로 서빙하고 페이지가 자체 완결(외부 CDN·폰트·스크립트 0)이라
에어갭 서버에서도 그대로 돈다. 다크/라이트는 시스템 설정을 따르고 헤더에서 수동 전환도 된다.
리포트 저장·CSV 버튼은 CLI와 동일한 산출물을 내려준다(데스크톱 앱에서는 네이티브 저장 창이 뜬다).

### 토큰 전달 시뮬레이터

표에서 CPU를 클릭하면 그 하드웨어로 알람 1건이 **언제 도달하는지**를 시간 축으로 보여준다 —
프롬프트 처리 / 토큰 생성 / 전송 세 구간의 실제 초와 각 구간의 병목(연산 바운드인지 대역폭
바운드인지). 재생 버튼을 누르면 예측된 tok/s로 **실시간 재생**되어 토큰 카운터가 올라가고
샘플 응답이 그 속도로 흐른다. 숫자를 읽는 것과 속도를 체감하는 것은 다르기 때문이다.

### 작업관리자

같은 하드웨어를 고정한 채 알람 개수를 100 / 200 / 300(변경 가능)으로 놓고 리소스 사용량을
한 화면에서 비교한다.

| 지표 | 의미 |
|---|---|
| CPU 사용률 | 하루 중 서버가 실제로 일한 시간의 비율. llama.cpp는 요청이 있으면 모든 스레드를 점유하므로 유휴가 아니면 사실상 포화다 |
| RAM | 실사용(가중치+KV+컴퓨트+OS) 대비 장착량 |
| 평균 대역폭 | 하루 평균 실효 대역폭 사용량 / 그 서버의 실효 대역폭 |
| 최대 큐 / p95 / 스톰 소진 | 시뮬레이션 결과 |
| 하루 작업시간 | 서버가 일한 총 분 |

기본은 `127.0.0.1` 바인딩이다. 인증이 없는 로컬 분석 도구이므로 `--host 0.0.0.0`은 사내망에
노출된다는 뜻이고, 필요할 때만 쓰는 게 맞다.

## 무엇이 들어 있나

| 카탈로그 | 내용 |
|---|---|
| 모델 | 0.5B~70B(CPU 추론 대상이라 70B 초과는 제외), 파라미터 규모 구간 전부. GQA 구조(`n_kv_head`)와 MoE 활성 파라미터 포함. 전부 실제 HuggingFace `config.json`에서 수집 |
| CPU 28종 | Xeon SP 3·4·5세대, Xeon E-2400, EPYC Milan·Genoa·Siena. 코어·**실제 전코어 터보**·ISA(AVX2/AVX-512/AMX)·메모리 채널·DDR 등급·PassMark CPU Mark |
| 메모리 33종 | DDR4-2666~DDR5-6400, MRDIMM-8800. **채널당 2 DIMM 시 속도 하락(derate)을 실제로 모델링** |
| 양자화 8종 | Q2_K~F16. 실효 bits-per-weight는 실제 GGUF 파일로 검증 |
| 효율 계수 8종 | 루프라인 천장 중 llama.cpp가 실제로 달성하는 비율. 각 행이 실측/실측유도/추정 중 무엇인지 표시 |

## 왜 코어 수만 보면 틀리는가

CPU 추론은 두 개의 서로 다른 병목으로 갈린다. 이 프로그램의 모든 예측이 여기서 나온다.

**토큰 생성은 메모리 대역폭 바운드.** 토큰 하나마다 모델 가중치 전체를 DRAM에서 읽는다.
연산은 거의 놀고, 메모리 채널 수와 DDR 등급이 성능을 지배한다.

```
tok/s = min(채널수 × DDR_MT/s × 8/1000 × eta_bw,  코어수 × 코어당대역폭) / (활성가중치 + KV읽기)
```

**프롬프트 처리는 연산 바운드.** 벡터 폭이 지배하므로 ISA가 배수로 갈린다.

```
tok/s = eta_c × 코어수 × 전코어클럭 × FLOP_per_cycle / (2 × 활성파라미터)
FLOP_per_cycle:  AVX2 32   AVX-512 64   AMX-BF16 1024
```

그래서 예컨대 EPYC 9354(12채널, 461 GB/s, AMX 없음)와 Xeon Gold 6438Y+(8채널, AMX 있음)의
선택은 **프롬프트 대 생성 토큰 비율에서 갈린다.** 생성 위주면 Genoa, 프롬프트 위주면 AMX Xeon.

## 하루 150건이 스펙을 결정하지 않는다

시간당 6건은 사실상 무부하다. 실제로 서버를 결정하는 것은 **스톰** — 장비 한 대가 죽어
30초에 40건이 쏟아지는 상황이다. 그래서 처리량을 부하로 나누는 대신 이산사건 큐
시뮬레이션을 돌린다.

두 SLA는 분리해서 판정한다. 스톰 알람은 정의상 큐가 쌓이므로(30초에 40건이 각각 30초 안에
끝날 수는 없다) **평상시 알람에는 지연 SLA를, 버스트에는 소진 시간 목표를** 적용한다.
섞으면 스톰이 많은 하루가 평상시엔 충분한 서버를 부당하게 탈락시킨다.

판정은 항상 **불리한 추정** 기준이다. 같은 하루를 예측 불확실도만큼 성능을 깎아 한 번 더
돌리고, 그 결과로 통과/미달을 가른다. 모든 추정이 낙관적으로 맞아떨어질 때만 성립하는
산정서는 산정서가 아니다.

## 정직성

납품 문서의 근거가 되므로 모르는 것은 모른다고 표시한다.

- 카탈로그의 모든 행은 출처(`source`)를 갖고, `unverified`가 아니면 `source_url`이 **필수**다
  (로더가 강제). 리포트는 미확인 행을 별도 섹션에 나열한다.
- Intel ARK와 AMD 제품 페이지는 WAF로 403을 뱉는다. 그래서 CPU 스펙은 **Wikipedia의 세대별
  SKU 표**(전코어 터보·DDR 속도·TDP·MSRP를 SKU별로 싣고 각 행이 ARK를 인용한다)와
  **PassMark cpubenchmark.net**(코어·클럭·TDP·소켓·CPU Mark)에서 전사하고 `third_party_db`로
  표시한다. 벤더 자체 페이지는 아니지만 실제로 열어 읽었고 재확인이 가능한 출처다.
- **PassMark CPU Mark는 이 카탈로그에서 유일하게 실측된 숫자**다. 그래서
  `catalog validate`가 이걸로 나머지를 감사한다 — `CPU Mark ÷ (코어수 × 전코어클럭)`이
  중위값에서 35% 넘게 벗어나면 스펙 전사 오류를 의심해 표시한다. 측정된 값이 측정할 수 없는
  값들을 감시하는 구조다. 예측에는 쓰지 않는다 — CPU Mark는 GEMM이 아닌 혼합 워크로드이고
  AVX-512·AMX를 llama.cpp처럼 쓰지 않으므로, 예측에 넣으면 문서화된 추정을 문서화되지 않은
  추정으로 바꾸는 셈이다.
- 효율 계수도 마찬가지로 근거 수준을 갖는다. 추정 계수가 들어간 예측은 리포트에 경고로
  표시되고 오차 범위가 넓어진다. AMX 계수가 가장 약한 값이고, 그래서 Intel 4·5세대 Xeon의
  프롬프트 처리 예측이 가장 불확실하다.

## 효율 계수와 그 근거

루프라인은 천장을 계산하고, llama.cpp는 그 일부만 달성한다. 그 비율이 효율 계수이고,
`catalog/coefficients.json`에 근거와 함께 들어 있다.

```bash
python3 -m svrspec.cli list coefficients
```

각 행은 근거 수준을 밝힌다 — **실측**(해당 등급 하드웨어에서 llama.cpp를 실제로 측정),
**실측유도**(측정값 + 명시된 가정), **추정**(문헌·추론만). 리포트의 오차 범위는 그 산정에
실제로 쓰인 계수들의 근거 수준에서 계산되므로, 추정값이 실측값처럼 보이는 일이 없다.

서버급 계수는 [llama.cpp discussion #11733](https://github.com/ggml-org/llama.cpp/discussions/11733)의
공개 실측치에서 유도했다. EPYC 9374F / 9175F / 9654 세 플랫폼의 STREAM 대역폭과
Llama-3.1-70B F16 pp512·tg128 측정값이 실려 있고, 70B F16은 토큰당 141.2 GB를 읽으므로
단일 소켓 tg128이 곧 달성 대역폭이 된다.

**이 데이터가 제 추정값 중 세 개를 반증했다:**

| 계수 | 추정값 | 실측 유도값 | 무엇이 틀렸나 |
|---|---|---|---|
| `eta_bw` DDR5 | 0.65 | **0.72** | 너무 비관적. 실측은 0.67~0.81 |
| `per_core_bw_gbs` | 20.0 | **25.0** | 16코어 EPYC이 332 GB/s(20.7/코어)를 뽑았는데 20×16=320은 그보다 낮다 |
| `dual_socket_efficiency` | 0.60 | **0.91** | 크게 틀렸다. llama.cpp는 큰 행렬을 소켓 간에 잘 분배한다(dense 1.8~1.9배 확장) |

추가로 **MoE는 소켓 확장이 dense보다 훨씬 나쁘다**는 것도 데이터에 있다(Mixtral 1.46~1.85배,
소켓당 0.73~0.92). 전문가 행렬이 작아 동기화 오버헤드가 지배하기 때문이다. 그래서
`dual_socket_efficiency`를 dense/moe로 나눠 두었다 — MoE 배포를 dense 수치로 산정하면
과대 약속이 된다.

아직 추정으로 남은 것은 **AMX와 MRDIMM 두 개**다. AMX 계수는 공개된 llama.cpp pp512
측정치를 찾지 못해, AMX의 1024 flop/cycle 타일 피크가 아니라 AVX-512 실측값에 앵커했다
(AVX-512 대비 3배 가정, 보고된 AMX prefill 이득 2~4배 범위). 이전 값 0.10은 6.8배를
함의해서 방어할 수 없었다. 그래서 Intel 4·5세대 Xeon 예측의 오차 범위가 가장 넓다.

메모리 산정 모델은 실제 GGUF 파일로 검증할 수 있다(파일 읽기만 하며 추론하지 않는다).

```bash
python3 -m svrspec.cli verify
```

표의 bits-per-weight로 계산한 파일 크기 vs 실제 크기(5% 이내), 그리고 KV 캐시·컴퓨트 버퍼
계산값을 출력한다. 후자는 llama-server 기동 로그의 `KV self size` / `compute buffer size`
줄과 직접 대조할 수 있다.

## 에어갭 반입

```bash
python3 -m svrspec.cli bundle --out svrspec-bundle.zip
```

외부 의존성이 없어 zip을 풀고 `python3`만 있으면 실행된다. 모델 파일도, 벤치마크도 필요 없다.

## 개발

```bash
uv tool install --editable .               # svrspec 실행 파일 설치
uv run --with pytest python -m pytest      # 128개 테스트
```

디자인은 토큰 한 벌(`svrspec/theme.py`)을 GUI와 납품 리포트가 공유한다 — 두 화면이 다른
제품처럼 갈라지지 않게 하기 위해서다. 컴포넌트 CSS에 색을 하드코딩하지 않는 것을 테스트로
강제한다.

핵심 검증 두 개:

- `test_reproduces_the_one_real_measurement` — 서버 CPU로 외삽하는 그 루프라인이, 먼저
  계수를 유도한 실측값(2채널 DDR4-2667에서 Llama-3.1-8B Q4_K_M 6.07 tok/s)을 재현해야 한다.
- `test_nothing_in_the_cli_loads_the_local_cpu` — CLI 소스에 벤치마크 실행 경로가 다시
  기어들어오는 것을 막는다. 고객 서버 산정이 이 머신을 갈구는 일에 의존해서는 안 된다.
