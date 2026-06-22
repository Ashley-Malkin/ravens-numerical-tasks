# Ravens numerical experiments (Modal / vLLM)

Auto-appended by `baby_reasoning_eval/modal_eval.py` after each run.

## 2026-06-22 — compare (all tasks)

_Logged from first successful compare run (Modal vLLM). `--n-examples 1`._

### EleutherAI/pythia-70m-deduped
- **Overall:** 21.7% (26/120)
- **By task:** combine 20.0% (4/20) · constancy 15.0% (3/20) · intersection 30.0% (6/20) · pattern 15.0% (3/20) · pattern_tuple 25.0% (5/20) · progression 25.0% (5/20)

### Qwen/Qwen3-8B
- **Overall:** 54.2% (65/120)
- **By task:** combine 35.0% (7/20) · constancy 90.0% (18/20) · intersection 45.0% (9/20) · pattern 65.0% (13/20) · pattern_tuple 30.0% (6/20) · progression 60.0% (12/20)

**Δ (Qwen3 − Pythia):** +32.5 pp


---
## 2026-06-22 — run_compare_ravens (max_tasks=120)

_Logged 2026-06-22 22:11 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; `--attention-backend FLASH_ATTN`; `--n-examples 0`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 24.2% (29/120)
- **By task:** combine 25.0% (5/20) · constancy 15.0% (3/20) · intersection 20.0% (4/20) · pattern 35.0% (7/20) · pattern_tuple 25.0% (5/20) · progression 25.0% (5/20)

### Qwen/Qwen3-8B
- **Overall:** 38.3% (46/120)
- **By task:** combine 40.0% (8/20) · constancy 65.0% (13/20) · intersection 5.0% (1/20) · pattern 50.0% (10/20) · pattern_tuple 20.0% (4/20) · progression 50.0% (10/20)

**Δ (Qwen3 − Pythia):** +14.2 pp

---
## 2026-06-22 — run_compare_ravens (max_tasks=120)

_Logged 2026-06-22 22:19 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; `--attention-backend FLASH_ATTN`; `--n-examples 3`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 17.5% (21/120)
- **By task:** combine 20.0% (4/20) · constancy 5.0% (1/20) · intersection 30.0% (6/20) · pattern 20.0% (4/20) · pattern_tuple 10.0% (2/20) · progression 20.0% (4/20)

### Qwen/Qwen3-8B
- **Overall:** 58.3% (70/120)
- **By task:** combine 40.0% (8/20) · constancy 80.0% (16/20) · intersection 35.0% (7/20) · pattern 100.0% (20/20) · pattern_tuple 50.0% (10/20) · progression 45.0% (9/20)

**Δ (Qwen3 − Pythia):** +40.8 pp

---
