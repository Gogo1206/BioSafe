# BioSafe — Future Improvements

## Core Problem: Multi-class auth is the wrong framing

Identity verification is a **one-class** problem: "Is this the enrolled user?" — not "Which of 3 known categories is this?"

The current Random Forest multi-class classifier pre-defines enemy classes (`user` vs `others` vs `ai`). A real attacker with a novel typing style will be silently misclassified into a known category. The "others" synthetic data is derived from the user's own percentiles, making this a **percentile-based anomaly detector wrapped in a random forest** — not a meaningful ML pipeline.

## Priority Roadmap

### P0 — Fundamental Architectural Fix

| Task | Description |
|------|-------------|
| **Switch to one-class auth** | Replace multi-class RF with one-class SVM, Isolation Forest, or Gaussian Mixture Model + log-likelihood threshold. Learn only the legitimate user's envelope. Output: accept/reject, not a class label. |
| **Session-diverse enrollment** | Collect 50–100 typing samples across 3+ separate sessions (different days, times). Session-internal evaluation produces inflated, meaningless accuracy numbers. |
| **Proper biometric metrics** | Report False Accept Rate (FAR — security) and False Reject Rate (FRR — usability). "95% accuracy" means nothing for authentication when class distributions are imbalanced. |

### P1 — Signal Quality

| Task | Description |
|------|-------------|
| **More keystrokes per sample** | 4-digit PIN gives only 7 real features (3 inter-key intervals + 4 hold times). Keystroke dynamics research uses free text or passphrases of 10–15+ characters. Consider continuous auth: authenticate across the first 100+ characters typed after login, not just the PIN. |
| **Per-key modeling** | Train distributions per individual key hold time, not per whole-PIN sequence. More data per parameter, more robust to PIN changes. |
| **Score fusion** | Combine independent weak signals: hold-time score + inter-key score + overall rhythm score. Individual weak signals → stronger together. |
| **Multi-modal input** (mobile) | If targeting phones/tablets, add touch pressure and touch area. Touchscreens give far more signal per digit than timing alone. |

### P2 — Robustness

| Task | Description |
|------|-------------|
| **Temporal normalization** | People type faster in the morning, slower when tired, different on phone vs keyboard. Normalize features per-session to reduce session-level drift. |
| **Adaptive threshold** | Threshold should tighten as more samples accumulate (more confidence), loosen after long gaps (drift). Static threshold is either too strict or too permissive. |
| **Error handling** | Add `try/except` on all file I/O paths. `open()` with no existence check crashes silently. Handle missing data directory, corrupt CSV, keyboard listener failures. |
| **Replace `eval()` on CSV** | `eval()` is a security risk and fragile. Parse numeric values with `float()` or `int()`. Validate column count before indexing. |

### P3 — Engineering Quality

| Task | Description |
|------|-------------|
| **Parameterize key count** | `num_keys = 4` hardcoded everywhere. Make it configurable to support longer passphrases. |
| **Add tests** | Unit tests for enrollment (feature extraction, statistics computation), scoring (distance/threshold behavior), and auth decision (accept/reject logic). |
| **CLI interface** | Add `argparse` to main scripts: `--enroll`, `--verify`, `--threshold`, `--data-dir`. Currently every path/PIN/param is hardcoded. |
| **Session metadata** | Store timestamp + keyboard type with each sample. Enables later drift analysis and per-keyboard calibration. |

### P4 — Research & Validation

| Task | Description |
|------|-------------|
| **Cross-session evaluation** | Split data by session, not random 80/20. Train on sessions 1–2, test on session 3. This is the only evaluation that predicts real-world performance. |
| **Impostor effort levels** | Test against: zero-effort (random typing), informed (watched target type PIN), practiced (trained on target's rhythm). Report FAR at each level. |
| **Benchmark against baselines** | Compare one-class approaches against: (a) Mahalanobis distance (simplest), (b) GMM (classic biometric baseline), (c) Isolation Forest, (d) One-class SVM. |

## Alternative: Abandon keystroke dynamics for 4-digit PIN

For a 4-digit PIN specifically, there may simply not be enough signal for reliable keystroke biometrics regardless of method. Alternatives to consider:

- **Continuous auth on free text** — authenticate across everything the user types after login, not just the PIN
- **Multi-modal** — keyboard pressure sensors, accelerometer, typing style on surrounding text
- **Behavioral 2FA** — keystroke dynamics as one factor among several, not standalone auth

## Model Comparison (for one-class authentication)

| Method | Pros | Cons |
|--------|------|------|
| **Mahalanobis distance** | Simple, no training, fast scoring, explainable | Assumes Gaussian feature distribution |
| **Gaussian Mixture Model** | Models multi-modal typing variation, probabilistic output | Needs more samples for stable fit |
| **One-class SVM** | Good with small samples, non-linear boundaries | Kernel + nu parameter tuning required |
| **Isolation Forest** | No distribution assumptions, handles outliers well | Less interpretable scores |

**Recommendation**: Start with Mahalanobis distance as baseline, then compare against GMM. Only reach for SVM/Isolation Forest if the simple methods fail to separate user from impostor in cross-session evaluation.
