# Time-LLM을 이용한 프로젝트 (수정중)  

**논문:**  
*Time-LLM: Time Series Forecasting by Reprogramming Large Language Models* (ICLR 2024)

**공식 코드:**  
https://github.com/KimMeen/Time-LLM

**논문 리뷰:**  
https://hipposdata.tistory.com/134

## 공식 Time-LLM과 프로젝트 코드의 주요 차이점

| 항목               | 공식 코드                              | 현재 프로젝트 코드          |
|--------------------|-------------------------------------|-----------------------------|
| **LLM 백본 모델**   | LLaMA-7B, GPT-2, BERT 지원 (기본: LLaMA-7B) | DistilGPT-2 단일 사용       |
| **정규화 (RevIN)**  | RevIN (subtract_last 옵션 지원)        | RevIN (평균/표준편차 기반 커스텀) |
| **프롬프트 생성**    | 통계 기반 동적 프롬프트 (min, max, median, trend, lag 자동 계산) | ML 분석 결과 기반의 정적 프롬프트 |
| **패치 임베딩**      | ReplicationPad1d + TokenEmbedding (Conv1D 기반) | nn.functional.pad + nn.Linear 기반 |
| **프로토타입 임베딩** | num_tokens = 1,000 (고정)               | num_prototypes = 256 (가변)   |
| **출력 처리**        | 복잡한 reshape 및 마지막 패치만 사용      | flatten 후 linear 변환    |

## 주요 구조적 차이

- **LLM 모델 용량**  
  - 공식: LLaMA-7B (4,096 dim) — 대규모 언어 능력, 연산 부담 큼  
  - 현재: DistilGPT-2 (768 dim) — 경량화, 빠른 연산, 다소 학습 한계 존재

- **정규화 방식**  
  - subtract_last 옵션 방식 vs mean/std 평균·표준편차 기반 정규화

- **프롬프트 품질**  
  - 동적 통계정보 제공 vs 사전 ML 분석 결과(XAI 설명력 향상) 기반 프롬프트  
  → 동적 프롬프트가 데이터 특성 반영에 유리함

- **패치 임베딩**  
  - Conv1D(공식): 시계열 지역 패턴 학습에 강점  
  - Linear(현재): 전역적 변환으로 구조 단순화, 지역 정보 상대적 부족

- **프로토타입 수**  
  - 공식: 1,000개 (고정)  
  - 현재: 256개 (가변 조정 가능)

- **차원 처리**

| 차원 처리 단계   | 공식 TIME-LLM                                                    | 현재 프로젝트 코드                |
|------------------|------------------------------------------------------------------|----------------------------------|
| 입력             | [B, seq_len, enc_in]                                             | [B, seq_len, enc_in]             |
| 패치 생성        | [B, enc_in, patch_num, patch_len]                                | [B, F*num_patches, d_model]      |
| LLM 후 출력      | [B, enc_in, patch_num, d_ff]                                     | [B, F*num_patches, d_ff]         |
| 중간 처리        | 마지막 pred_len개 패치만: [B, enc_in, pred_len, d_ff]           | 전체 flatten: [B, F*num_patches*d_ff] |
| 헤드 처리        | 변수별 독립 헤드: enc_in개 헤드 사용                             | 통합 헤드: 1개 헤드 사용          |
| 헤드 출력        | [B, enc_in, target_window] → permute → [B, target_window, enc_in] | [B, pred_len, 1]                 |
| 최종 출력        | [B, target_window, c_out] (c_out=1, 단일 타겟만)                 | [B, pred_len, 1] (y만 예측)       |
