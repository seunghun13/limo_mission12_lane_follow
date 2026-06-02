[report.md](https://github.com/user-attachments/files/28494156/report.md)
# RTT Codeword Autopsy Localization

Symbolic-Geometric Candidate Duel for Indoor Positioning

12234002 정승훈  
스마트모빌리티공학실험2 1분반

## 1. 모티베이션 & 인트로

이번 기말 프로젝트의 목표는 WiFi RTT 기반 거리 관측값으로부터 실내 사용자 위치를 추정하는 것이다. 입력은 사용자별 거리 관측 행렬 `d_hat`과 anchor 좌표 `BS_positions`이며, 출력은 각 사용자 위치 추정값 `p_hat`이다. 중간발표 이전에는 `d_hat`을 좌표로 직접 회귀하거나, 몇 개의 후보 위치 중 residual이 작은 후보를 단순 선택하는 방향을 먼저 검토했다. 이 접근은 평균 오차를 어느 정도 줄였지만, 큰 오차가 남는 tail-risk 문제를 충분히 해결하지 못했다.

중간발표 이후의 핵심 판단은 WiFi RTT를 단순한 noisy distance로만 볼 수 없다는 점이었다. RTT 관측에는 positive bias, NLOS 영향, anchor별 distortion이 섞일 수 있다. 따라서 하나의 `d_hat` 벡터가 하나의 좌표만을 안정적으로 가리킨다고 보기 어렵고, 여러 위치가 동시에 그럴듯해 보이는 경우가 생긴다. 이 때문에 문제를 "`d_hat`을 바로 좌표로 회귀한다"가 아니라 "여러 후보 위치를 만들고, 그중 현재 샘플에서 믿을 후보를 고른다"로 재구성했다.

최종 주제명은 RTT Codeword Autopsy Localization으로 정했다. 부제는 Symbolic-Geometric Candidate Duel for Indoor Positioning이다. 이 방법은 보수적인 Safety candidate, symbolic pattern을 사용하는 Genome candidate, 기하 정보를 직접 사용하는 Survivor candidate를 동시에 생성한 뒤, Margin-Aware Candidate Duel Judge로 최종 후보를 선택한다. v1-B2 OOF validation에서 확인한 결론은 후보를 더 많이 만드는 것만으로는 충분하지 않고, 후보 간 선택 규칙이 성능과 tail-risk를 좌우한다는 점이었다.

최종 제출 artifact는 `model.pkl`이다. 이 파일은 sklearn pipeline 객체가 아니라 numeric/config dictionary pickle이다. 내부에는 계수, 절편, 통계량, codebook, training reference array, threshold 같은 추론에 필요한 값만 저장된다. 채점 시 `main.py`는 `model.pkl`, `d_hat`, `BS_positions`를 읽어 hidden `p` 없이 inference-only로 동작하고 `p_hat`을 반환한다.

## 2. 알고리즘 설명

`main.py`의 기본 입력 파일은 `DH_FR1.mat`이다. 이 파일에서 공식 anchor key인 `BS_positions`와 거리 관측값 `d_hat`을 읽는다. 과거 문서 호환성을 위해 legacy `p_bs` fallback도 유지하지만, 기본 기준은 `BS_positions`이다. `main()`은 import 후 호출해도 동작하며, 반환값은 `numpy.ndarray`이고 shape은 `(2, num_user)`이다. 여기서 `num_user`는 고정 숫자가 아니라 `d_hat.shape[1]`에서 동적으로 결정된다.

`train.py`는 제공된 training target `p`를 사용해 artifact를 만든다. 반면 `main.py`는 hidden `p`를 읽거나 사용하지 않는다. `main.py`는 `pickle.load`로 `model.pkl`을 읽고, payload가 dictionary인지와 필수 key가 있는지를 확인한 뒤, numpy 기반 연산으로 후보 생성과 duel judge를 재현한다. 따라서 최종 추론에는 학습, k-fold 재학습, hyperparameter search가 들어가지 않는다.

anchor `i`의 관측 거리값을 `d_i`, anchor 좌표를 `b_i`라고 하자. 구현은 보정된 거리 인터페이스를 유지한다. 개념적으로 보정 거리는 `r_i = max(r_min, calibration_i(d_i))`로 볼 수 있다. 최종 제출 후보에서는 training 단계에서 선택된 RTT 표현을 사용하며, 해당 설정과 관련 통계는 `model.pkl`에 저장된다.

각 사용자 샘플에 대해 세 종류의 후보 위치를 만든다.

| candidate | 역할 | 핵심 아이디어 |
|---|---|---|
| Safety candidate | 보수적인 기준 후보 | 저장된 RTT 표현 공간에서 가까운 training sample을 찾고, 그 training 위치를 거리 기반 가중 평균한다. |
| Genome candidate | symbolic evidence 후보 | anchor별 거리값을 bin으로 양자화해 codeword를 만들고, training codebook에서 symbolic nearest neighbor를 찾아 평균한다. |
| Survivor candidate | geometric evidence 후보 | anchor geometry에서 가능한 circle-intersection 및 residual이 낮은 위치를 만들고, 좋은 survivor들을 aggregate한다. |

Safety candidate는 현재 `d_hat`이 기존 training signature와 비슷할 때 안정적이다. Genome candidate는 거리의 절대값이 distortion을 받더라도 anchor별 bin pattern이 의미를 유지하는 경우를 노린다. Survivor candidate는 기하 구조를 직접 활용하므로 일부 샘플에서 좋은 복구 능력을 보이지만, 단순 residual만 믿으면 큰 오차 후보를 자신 있게 고를 수 있다. 그래서 최종 방법은 어느 한 candidate family를 무조건 신뢰하지 않는다.

각 후보에 대해 residual score, candidate 간 거리, survivor spread, survivor residual risk, genome symbolic distance, genome margin, pairwise 비교 feature를 계산한다. 후보 `c_a`, `c_b`가 있을 때 training 시점의 오차는 `e_a = ||c_a - p||_2`, `e_b = ||c_b - p||_2`이다. duel label은 margin-aware 방식으로 만든다. `e_a + margin < e_b`일 때만 `a`가 `b`보다 낫다고 보고, `e_b + margin < e_a`일 때만 `b`가 `a`보다 낫다고 본다. margin 안에 있는 거의 동률 사례는 no-contest로 처리해, 의미 없는 미세 차이를 judge가 학습하지 않도록 했다.

Margin-Aware Candidate Duel Judge는 safety, genome, survivor 사이의 pairwise logistic comparison 집합이다. `train.py`는 각 pair에 대해 feature fill median, feature mean, feature scale, coefficient, intercept를 저장한다. `main.py`는 sklearn 객체를 import하거나 로드하지 않고, 저장된 숫자만 사용해 표준화된 feature와 logistic form을 계산한다. 계산은 `probability = 1 / (1 + exp(-(x_scaled dot coef + intercept)))` 형태이다.

pairwise probability는 각 candidate의 league score로 누적된다. 어떤 pair에서 candidate가 이길 확률만큼 점수를 받고, 반대쪽 candidate는 그 보수 확률만큼 점수를 받는다. league score가 가장 큰 candidate가 우선 선택된다. 다만 confidence가 낮거나 선택된 survivor가 spread 또는 residual 관점에서 위험해 보이면 Safety candidate로 fallback한다. 이 fallback은 hidden label을 이용한 보정이 아니라, validation에서 확인한 실행 규칙이다.

`model.pkl` dictionary에는 다음 범주의 정보가 저장된다.

| category | 저장 내용 |
|---|---|
| Candidate references | anchor positions, training positions, raw and calibrated training distances |
| Symbolic genome | bin edges, codebook, symbol costs, neighbor configuration |
| Survivor configuration | bounds, spread and residual risk thresholds, survivor scoring parameters |
| Duel judge | candidate names, pair indices, feature names, coefficients, intercepts, confidence threshold |
| Feature preprocessing | fill medians, means, scales |
| Audit metadata | version, selected method name, anchor source |

이 구조는 training과 inference를 분리한다. 학습 과정에서는 scikit-learn logistic regression을 사용하지만, 최종 artifact에는 estimator 객체를 저장하지 않는다. 추론에 필요한 numeric/config 값만 `model.pkl`에 남기므로, `main.py`는 artifact dictionary와 numpy 연산만으로 같은 decision function을 실행한다.

기존 연구는 WiFi RTT가 LOS/NLOS 조건, positive bias, ranging compensation, geometric RTT positioning 문제의 영향을 받는다는 배경을 이해하는 데 참고했다. 그러나 본 프로젝트는 reference 논문들의 알고리즘을 그대로 구현하지 않았다. 본 프로젝트의 독자적 부분은 제공된 `d_hat`과 `BS_positions`만을 사용해 Safety, Genome, Survivor 후보를 생성하고, Margin-Aware Candidate Duel Judge로 후보를 선택하는 symbolic-geometric candidate selection 구조를 직접 설계한 점이다.

## 3. Agent AI 활용 방안

프로젝트 진행 중 ChatGPT와 Codex를 실험 설계 및 구현 검증을 보조하는 도구로 사용했다. AI는 후보 아이디어 정리, README 규칙 확인, 실험 명령 구성, audit checklist 작성, 파일 구조 점검, smoke test 절차 정리에 도움을 주었다. Codex는 특히 후보 폴더를 분리해 점검하고, 표준 Python 환경에서 contract 검증과 문서 검증을 수행하는 데 사용했다.

AI가 제안한 결과를 그대로 최종 근거로 사용하지는 않았다. 제안된 방법과 문구는 공식 README, TA 답변, v1-B2 OOF validation output, 최종 제출 후보의 smoke test 및 contract audit 결과와 대조했다. 성능 수치는 OOF validation 기준인지, smoke test는 실행 안정성 검증인지, hidden test 결과를 암시하지 않는지 별도로 확인했다. 따라서 본 보고서에서는 OOF validation 결과와 smoke test 결과를 명확히 구분하여 해석한다.

본인이 맡은 역할은 최종 주제 선택, 실험 결과의 신뢰도 판단, failure case 해석, 과장 표현 제거, 제출 구조 판단이었다. 예를 들어 최종 방법을 RTT Codeword Autopsy Localization으로 정한 것, `main.py`를 inference-only로 유지한 것, artifact를 `model.pkl`로 제출하기로 한 것, 그리고 performance discussion에서 training-set self-check 값을 제외한 것은 본인의 판단에 따른 결정이다.

따라서 AI는 구현과 검증을 보조한 도구였고, 최종 알고리즘 방향, 보고서에서의 주장 강도, 제출 파일 구조, validation 결과 해석은 사람이 책임지고 결정했다.

## 4. 결과 도출 & 디스커션

아래 성능 수치는 v1-B2 OOF validation 기준이다. 이는 최종 알고리즘 선택을 위한 local validation 근거이며, hidden test 결과가 아니다. hidden test 분포는 알 수 없으므로 같은 수치가 hidden에서 재현된다고 보장할 수 없다. 또한 training-set self-check 값은 성능 근거로 사용하지 않았다.

| method | mean error (m) | median error (m) | p90 error (m) | max error (m) | oracle gap mean (m) | error >20m count |
|---|---:|---:|---:|---:|---:|---:|
| v1-A simple review | 7.3013 | 5.3483 | 16.7372 | 38.1498 | 3.2942 | 44 |
| v1-B hybrid duel | 5.7442 | 4.2449 | 13.1576 | 29.1402 | 1.7371 | 16 |
| Oracle best candidate | 4.0071 | 2.9137 | 9.4410 | 23.3662 | 0.0000 | 1 |

Oracle best candidate는 validation 이후 true position을 이용해 candidate pool 안에서 가장 좋은 후보를 고른 diagnostic upper bound이다. 이는 `main.py`가 실제 제출에서 사용하는 방법이 아니며, candidate pool 자체가 가질 수 있는 상한을 해석하기 위한 기준이다. 실제 제출 방법은 v1-B hybrid duel 구조이며, hidden `p` 없이 candidate quality feature와 저장된 duel coefficient만으로 선택한다.

v1-B hybrid duel은 mean error뿐 아니라 median, p90, max error, oracle gap mean, error >20m count를 모두 개선했다. 특히 기존 방식의 핵심 약점이 평균 bias가 아니라 tail-risk였기 때문에, p90과 large-error count 감소가 중요하다.

| tail-risk item | v1-A simple review | v1-B hybrid duel |
|---|---:|---:|
| error >20m count | 44 | 16 |
| survivor selected error >20m count | 39 | 3 |
| genome oracle case selection rate | 0.0191 | 0.3567 |

두 번째 표는 왜 candidate selection 구조가 필요한지를 보여준다. Survivor candidate는 일부 샘플에서 유용하지만, 단순 선택에서는 큰 오차를 많이 만들었다. Hybrid duel은 survivor가 선택된 사례 중 error >20m인 경우를 39개에서 3개로 줄였다. 동시에 genome candidate가 oracle-best였던 case에서 실제로 genome을 선택하는 비율은 0.0191에서 0.3567로 증가했다. 이는 symbolic evidence가 단독으로 항상 충분하다는 뜻이 아니라, duel judge가 genome evidence를 필요한 경우 더 잘 활용했다는 뜻이다.

최종 제출 후보는 실행 안정성과 공식 `main()` contract도 확인했다. 아래 항목은 성능 평가가 아니라 smoke test와 contract validation이다.

| check | result | detail |
|---|---|---|
| import 후 `main()` 호출 | PASS | shape `(2, 700)`, runtime about 3.712s |
| direct `main.py` 실행 | PASS | shape `(2, 700)`, runtime about 3.861s |
| `p` 제거 MAT | PASS | hidden target 없이 inference 동작 |
| `BS_positions` only | PASS | 공식 anchor key 처리 |
| legacy `p_bs` fallback | PASS | 과거 변수명 호환 처리 |
| subset `num_user=23` | PASS | shape `(2, 23)` 반환 |
| transposed `d_hat` | PASS | 입력 방향 보정 후 shape 유지 |
| `model.pkl` audit | PASS | dict, key count 41, sklearn object 0, torch object 0 |

최종 방법의 장점은 symbolic evidence와 geometric evidence를 함께 쓰되, hidden label 없이 후보 선택을 수행한다는 점이다. 또한 `model.pkl`은 dictionary 기반 numeric/config artifact이므로, 제출 시 `main.py`가 sklearn 객체나 torch 객체에 의존하지 않는다. contract audit 기준으로 `main.py`는 `numpy.ndarray` shape `(2, num_user)`를 반환하고, `p`를 사용하지 않으며, user 수를 700으로 고정하지 않는다.

한계도 분명하다. 첫째, hidden test set의 분포는 공개되어 있지 않으므로 OOF validation 결과만으로 hidden 성능을 보장할 수 없다. 둘째, v1-B2 validation에서도 error >20m 샘플이 16개 남아 있어 tail-risk가 제거된 것은 아니다. 셋째, threshold와 fallback policy는 제공된 training distribution에서 선택되었기 때문에 distribution shift에 취약할 수 있다. 넷째, 세 candidate가 모두 나쁜 샘플에서는 duel judge가 후보 pool 안의 덜 나쁜 선택만 할 수 있다. 향후 개선은 후보 선택뿐 아니라 candidate generation 자체를 강화하는 방향이 필요하다.

단순 기하 방식이나 직접 회귀 방식을 부당하게 약한 baseline으로 단정하려는 것은 아니다. 이 프로젝트의 validation 결과는, 제공된 RTT-like 관측에서는 단일 좌표 회귀보다 후보 생성 후 후보 선택으로 문제를 나누는 구성이 observed failure mode에 더 잘 맞았다는 근거로 해석해야 한다.

## 5. Reference

아래 표는 각 reference에서 참고한 내용과 본 프로젝트에서 직접 설계 또는 구현한 부분을 구분한 것이다. 기존 연구는 WiFi RTT, LOS/NLOS, ranging bias, 보정, geometric RTT positioning의 문제 배경을 이해하기 위해 참고했으며, 본 프로젝트의 최종 방법은 제공된 수업 데이터에 맞춘 후보 생성 및 후보 선택 구조로 별도 설계했다.

| Reference | 참고한 내용 | 본 프로젝트에서 직접 설계/구현한 부분 |
|---|---|---|
| Cao et al. [1] | WiFi RTT에서 LOS/NLOS 조건과 ranging compensation이 positioning 성능에 중요하다는 배경을 참고했다. | 해당 논문의 SVM 기반 NLOS recognition이나 Bayesian trusted NLOS 모델을 그대로 구현하지 않고, 제공된 `d_hat`에서 candidate quality와 duel judge 구조를 직접 설계했다. |
| Ibrahim et al. [2] | WiFi FTM/RTT 측정값이 실제 환경에서 오차와 bias를 가질 수 있다는 실험적 배경을 참고했다. | raw RTT-like `d_hat`의 positive bias와 tail failure를 분석하고, Safety/Genome/Survivor 후보 및 selection judge를 구성했다. |
| Guo et al. [3] | WiFi RTT 기반 실내 측위와 RTT/RSS hybrid ranging의 연구 배경을 참고했다. | RSS를 사용하지 않고, 주어진 `d_hat`과 `BS_positions`만으로 symbolic-geometric candidate selection을 수행했다. |
| Dong et al. [4] | WiFi RTT/RSS에서 NLOS/LOS identification이 positioning 안정성에 영향을 준다는 점을 참고했다. | 별도의 LOS/NLOS classifier를 구현하지 않고, survivor risk, residual, candidate disagreement를 duel feature로 사용했다. |
| Feng et al. [5] | RTT 기반 LOS detection과 line-of-sight 여부가 실내 측위에 중요하다는 배경을 참고했다. | LOS detection 모델이 아니라, candidate별 evidence를 비교하는 Margin-Aware Candidate Duel Judge를 구현했다. |
| Han et al. [6] | WiFi RTT에서 NLOS/LOS 조건에 따라 RTT bias가 발생하고, 이를 기하적 관계로 다룰 수 있다는 배경을 참고했다. | 해당 논문처럼 PDR이나 user trajectory를 사용하지 않고, 제공된 `BS_positions`와 `d_hat`만으로 Survivor geometric candidate를 만들고, Safety/Genome 후보와 Margin-Aware Candidate Duel Judge로 비교하는 구조를 설계했다. |
| Banin et al. [7] | WiFi FTM sensor 기반 positioning과 cooperative FTM sensor 개념을 참고했다. | FTM sensor system 자체를 구현하지 않고, 수업 제공 데이터의 anchor geometry와 RTT pattern을 활용한 candidate duel 구조를 구현했다. |

[1] H. Cao, Y. Wang, J. Bi, Y. Zhang, G. Yao, Y. Feng, and M. Si, "LOS compensation and trusted NLOS recognition assisted WiFi RTT indoor positioning algorithm," Expert Systems with Applications, vol. 243, Article 122867, 2024. doi: 10.1016/j.eswa.2023.122867.

[2] M. Ibrahim, H. Liu, A. Jawahar, V. Nguyen, M. Gruteser, R. Howard, B. Yu, and F. Bai, "Verification: Accuracy Evaluation of WiFi Fine Time Measurements on an Open Platform," 2018. doi: 10.1145/3241539.3241555.

[3] G. Guo, R. Chen, F. Ye, X. Peng, Z. Liu, and Y. Pan, "Indoor Smartphone Localization: A Hybrid WiFi RTT-RSS Ranging Approach," IEEE Access, vol. 7, pp. 176767-176781, 2019. doi: 10.1109/ACCESS.2019.2957753.

[4] Y. Dong, T. Arslan, and Y. Yang, "Real-time NLOS/LOS Identification for Smartphone-based Indoor Positioning Systems Using WiFi RTT and RSS," IEEE Sensors Journal, vol. 22, no. 6, pp. 5199-5209, 2022. doi: 10.1109/JSEN.2021.3119234.

[5] X. Feng, K. A. Nguyen, and Z. Luo, "WiFi Access Points Line-of-Sight Detection for Indoor Positioning Using the Signal Round Trip Time," Remote Sensing, vol. 14, no. 23, 2022. doi: 10.3390/rs14236052.

[6] K. Han, S. M. Yu, S.-L. Kim, and S.-W. Ko, "Exploiting User Mobility for WiFi RTT Positioning: A Geometric Approach," IEEE Internet of Things Journal, vol. 8, no. 19, pp. 14589-14606, 2021. doi: 10.1109/JIOT.2021.3070367.

[7] L. Banin, O. Bar-Shalom, N. Dvorecki, and Y. Amizur, "Scalable Wi-Fi Client Self-positioning Using Cooperative FTM-Sensors," IEEE Transactions on Instrumentation and Measurement, vol. 68, no. 10, pp. 3686-3698, 2019.
