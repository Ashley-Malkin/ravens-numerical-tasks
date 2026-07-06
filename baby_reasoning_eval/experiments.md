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
## 2026-06-22 — run_compare_ravens (max_tasks=10)

_Logged 2026-06-22 23:14 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; `--attention-backend FLASH_ATTN`; `--n-examples 0`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 10.0% (1/10)
- **By task:** combine 0.0% (0/1) · constancy 0.0% (0/2) · intersection 0.0% (0/1) · pattern 50.0% (1/2) · pattern_tuple 0.0% (0/2) · progression 0.0% (0/2)

### Qwen/Qwen3-8B
- **Overall:** 80.0% (8/10)
- **By task:** combine 100.0% (1/1) · constancy 100.0% (2/2) · intersection 0.0% (0/1) · pattern 100.0% (2/2) · pattern_tuple 100.0% (2/2) · progression 50.0% (1/2)

**Δ (Qwen3 − Pythia):** +70.0 pp

---
## 2026-06-23 — run_ravens_eval (EleutherAI--pythia-70m-deduped) (max_tasks=10)

_Logged 2026-06-23 22:04 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 30.0% (3/10)
- **By task:** combine 100.0% (1/1) · constancy 50.0% (1/2) · intersection 0.0% (0/1) · pattern 0.0% (0/2) · pattern_tuple 50.0% (1/2) · progression 0.0% (0/2)

---
## 2026-06-23 — run_ravens_eval (EleutherAI--pythia-70m-deduped) (max_tasks=10)

_Logged 2026-06-23 22:07 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 30.0% (3/10)
- **By task:** combine 0.0% (0/1) · constancy 50.0% (1/2) · intersection 0.0% (0/1) · pattern 0.0% (0/2) · pattern_tuple 50.0% (1/2) · progression 50.0% (1/2)

---
## 2026-06-23 — run_ravens_eval (EleutherAI--pythia-410m-deduped) (max_tasks=10)

_Logged 2026-06-23 22:09 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 20.0% (2/10)
- **By task:** combine 100.0% (1/1) · constancy 0.0% (0/2) · intersection 0.0% (0/1) · pattern 0.0% (0/2) · pattern_tuple 50.0% (1/2) · progression 0.0% (0/2)

---
## 2026-06-23 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped) (max_tasks=10)

_Logged 2026-06-23 22:12 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 20.0% (2/10)
- **By task:** combine 100.0% (1/1) · constancy 0.0% (0/2) · intersection 0.0% (0/1) · pattern 0.0% (0/2) · pattern_tuple 50.0% (1/2) · progression 0.0% (0/2)

---
## 2026-06-23 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped) (max_tasks=10)

_Logged 2026-06-23 22:14 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 80.0% (8/10)
- **By task:** combine 100.0% (1/1) · constancy 50.0% (1/2) · intersection 100.0% (1/1) · pattern 100.0% (2/2) · pattern_tuple 50.0% (1/2) · progression 100.0% (2/2)

---
## 2026-06-23 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped) (max_tasks=10)

_Logged 2026-06-23 22:17 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 40.0% (4/10)
- **By task:** combine 100.0% (1/1) · constancy 0.0% (0/2) · intersection 0.0% (0/1) · pattern 100.0% (2/2) · pattern_tuple 50.0% (1/2) · progression 0.0% (0/2)

---
## 2026-06-23 — run_ravens_eval (Qwen--Qwen3-0.6B) (max_tasks=10)

_Logged 2026-06-23 22:19 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### Qwen/Qwen3-0.6B
- **Overall:** 20.0% (2/10)
- **By task:** combine 100.0% (1/1) · constancy 50.0% (1/2) · intersection 0.0% (0/1) · pattern 0.0% (0/2) · pattern_tuple 0.0% (0/2) · progression 0.0% (0/2)

---
## 2026-06-23 — run_ravens_eval (Qwen--Qwen3-1.7B) (max_tasks=10)

_Logged 2026-06-23 22:22 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### Qwen/Qwen3-1.7B
- **Overall:** 30.0% (3/10)
- **By task:** combine 100.0% (1/1) · constancy 0.0% (0/2) · intersection 0.0% (0/1) · pattern 50.0% (1/2) · pattern_tuple 50.0% (1/2) · progression 0.0% (0/2)

---
## 2026-06-23 — run_ravens_eval (Qwen--Qwen3-4B) (max_tasks=10)

_Logged 2026-06-23 22:25 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### Qwen/Qwen3-4B
- **Overall:** 80.0% (8/10)
- **By task:** combine 100.0% (1/1) · constancy 100.0% (2/2) · intersection 100.0% (1/1) · pattern 100.0% (2/2) · pattern_tuple 50.0% (1/2) · progression 50.0% (1/2)

---
## 2026-06-23 — run_ravens_eval (Qwen--Qwen3-8B) (max_tasks=10)

_Logged 2026-06-23 22:28 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### Qwen/Qwen3-8B
- **Overall:** 90.0% (9/10)
- **By task:** combine 100.0% (1/1) · constancy 100.0% (2/2) · intersection 100.0% (1/1) · pattern 100.0% (2/2) · pattern_tuple 100.0% (2/2) · progression 50.0% (1/2)

---
## 2026-06-23 — run_ravens_eval (EleutherAI--pythia-70m-deduped) (max_tasks=120)

_Logged 2026-06-23 22:31 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 31.7% (38/120)
- **By task:** combine 20.0% (4/20) · constancy 15.0% (3/20) · intersection 40.0% (8/20) · pattern 45.0% (9/20) · pattern_tuple 35.0% (7/20) · progression 35.0% (7/20)

---
## 2026-06-23 — run_ravens_eval (EleutherAI--pythia-410m-deduped) (max_tasks=120)

_Logged 2026-06-23 22:33 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 31.7% (38/120)
- **By task:** combine 45.0% (9/20) · constancy 30.0% (6/20) · intersection 30.0% (6/20) · pattern 40.0% (8/20) · pattern_tuple 20.0% (4/20) · progression 25.0% (5/20)

---
## 2026-06-23 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped) (max_tasks=120)

_Logged 2026-06-23 22:36 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 34.2% (41/120)
- **By task:** combine 45.0% (9/20) · constancy 20.0% (4/20) · intersection 40.0% (8/20) · pattern 40.0% (8/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-06-23 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped) (max_tasks=120)

_Logged 2026-06-23 22:39 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 35.8% (43/120)
- **By task:** combine 45.0% (9/20) · constancy 40.0% (8/20) · intersection 15.0% (3/20) · pattern 45.0% (9/20) · pattern_tuple 25.0% (5/20) · progression 45.0% (9/20)

---
## 2026-06-23 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped) (max_tasks=120)

_Logged 2026-06-23 22:45 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 37.5% (45/120)
- **By task:** combine 50.0% (10/20) · constancy 25.0% (5/20) · intersection 45.0% (9/20) · pattern 45.0% (9/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-06-23 — run_ravens_eval (Qwen--Qwen3-0.6B) (max_tasks=120)

_Logged 2026-06-23 22:49 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### Qwen/Qwen3-0.6B
- **Overall:** 25.0% (30/120)
- **By task:** combine 35.0% (7/20) · constancy 20.0% (4/20) · intersection 25.0% (5/20) · pattern 35.0% (7/20) · pattern_tuple 10.0% (2/20) · progression 25.0% (5/20)

---
## 2026-06-23 — run_ravens_eval (Qwen--Qwen3-1.7B) (max_tasks=120)

_Logged 2026-06-23 22:52 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### Qwen/Qwen3-1.7B
- **Overall:** 35.0% (42/120)
- **By task:** combine 35.0% (7/20) · constancy 40.0% (8/20) · intersection 20.0% (4/20) · pattern 40.0% (8/20) · pattern_tuple 45.0% (9/20) · progression 30.0% (6/20)

---
## 2026-06-23 — run_ravens_eval (Qwen--Qwen3-4B) (max_tasks=120)

_Logged 2026-06-23 22:56 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### Qwen/Qwen3-4B
- **Overall:** 60.8% (73/120)
- **By task:** combine 65.0% (13/20) · constancy 85.0% (17/20) · intersection 40.0% (8/20) · pattern 90.0% (18/20) · pattern_tuple 55.0% (11/20) · progression 30.0% (6/20)

---
## 2026-06-23 — run_ravens_eval (Qwen--Qwen3-8B) (max_tasks=120)

_Logged 2026-06-23 23:02 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`_

### Qwen/Qwen3-8B
- **Overall:** 60.0% (72/120)
- **By task:** combine 50.0% (10/20) · constancy 95.0% (19/20) · intersection 35.0% (7/20) · pattern 100.0% (20/20) · pattern_tuple 35.0% (7/20) · progression 45.0% (9/20)

---
## 2026-06-24 — run_ravens_eval (EleutherAI--pythia-70m-deduped, completion) (max_tasks=10)

_Logged 2026-06-24 17:14 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 80.0% (8/10)
- **By task:** combine 100.0% (1/1) · constancy 100.0% (2/2) · intersection 0.0% (0/1) · pattern 100.0% (2/2) · pattern_tuple 50.0% (1/2) · progression 100.0% (2/2)

---
## 2026-06-24 — run_ravens_eval (EleutherAI--pythia-410m-deduped, completion) (max_tasks=10)

_Logged 2026-06-24 17:16 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 100.0% (10/10)
- **By task:** combine 100.0% (1/1) · constancy 100.0% (2/2) · intersection 100.0% (1/1) · pattern 100.0% (2/2) · pattern_tuple 100.0% (2/2) · progression 100.0% (2/2)

---
## 2026-06-24 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, completion) (max_tasks=10)

_Logged 2026-06-24 17:18 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 100.0% (10/10)
- **By task:** combine 100.0% (1/1) · constancy 100.0% (2/2) · intersection 100.0% (1/1) · pattern 100.0% (2/2) · pattern_tuple 100.0% (2/2) · progression 100.0% (2/2)

---
## 2026-06-24 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, completion) (max_tasks=10)

_Logged 2026-06-24 17:19 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 100.0% (10/10)
- **By task:** combine 100.0% (1/1) · constancy 100.0% (2/2) · intersection 100.0% (1/1) · pattern 100.0% (2/2) · pattern_tuple 100.0% (2/2) · progression 100.0% (2/2)

---
## 2026-06-24 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, completion) (max_tasks=10)

_Logged 2026-06-24 17:21 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 100.0% (10/10)
- **By task:** combine 100.0% (1/1) · constancy 100.0% (2/2) · intersection 100.0% (1/1) · pattern 100.0% (2/2) · pattern_tuple 100.0% (2/2) · progression 100.0% (2/2)

---
## 2026-06-24 — run_ravens_eval (Qwen--Qwen3-0.6B, completion) (max_tasks=10)

_Logged 2026-06-24 17:26 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-0.6B
- **Overall:** 100.0% (10/10)
- **By task:** combine 100.0% (1/1) · constancy 100.0% (2/2) · intersection 100.0% (1/1) · pattern 100.0% (2/2) · pattern_tuple 100.0% (2/2) · progression 100.0% (2/2)

---
## 2026-06-24 — run_ravens_eval (Qwen--Qwen3-1.7B, completion) (max_tasks=10)

_Logged 2026-06-24 17:32 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-1.7B
- **Overall:** 100.0% (10/10)
- **By task:** combine 100.0% (1/1) · constancy 100.0% (2/2) · intersection 100.0% (1/1) · pattern 100.0% (2/2) · pattern_tuple 100.0% (2/2) · progression 100.0% (2/2)

---
## 2026-06-24 — run_ravens_eval (Qwen--Qwen3-4B, completion) (max_tasks=10)

_Logged 2026-06-24 17:34 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-4B
- **Overall:** 100.0% (10/10)
- **By task:** combine 100.0% (1/1) · constancy 100.0% (2/2) · intersection 100.0% (1/1) · pattern 100.0% (2/2) · pattern_tuple 100.0% (2/2) · progression 100.0% (2/2)

---
## 2026-06-24 — run_ravens_eval (Qwen--Qwen3-8B, completion) (max_tasks=10)

_Logged 2026-06-24 17:36 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-8B
- **Overall:** 100.0% (10/10)
- **By task:** combine 100.0% (1/1) · constancy 100.0% (2/2) · intersection 100.0% (1/1) · pattern 100.0% (2/2) · pattern_tuple 100.0% (2/2) · progression 100.0% (2/2)

---
## 2026-06-24 — run_ravens_eval (EleutherAI--pythia-70m-deduped, completion) (max_tasks=120)

_Logged 2026-06-24 17:45 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 56.7% (68/120)
- **By task:** combine 0.0% (0/20) · constancy 100.0% (20/20) · intersection 10.0% (2/20) · pattern 100.0% (20/20) · pattern_tuple 35.0% (7/20) · progression 95.0% (19/20)

---
## 2026-06-24 — run_ravens_eval (EleutherAI--pythia-410m-deduped, completion) (max_tasks=120)

_Logged 2026-06-24 17:47 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 90.0% (108/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 80.0% (16/20) · progression 100.0% (20/20)

---
## 2026-06-24 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, completion) (max_tasks=120)

_Logged 2026-06-24 17:50 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 94.2% (113/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-24 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, completion) (max_tasks=120)

_Logged 2026-06-24 17:57 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 95.0% (114/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 70.0% (14/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-24 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, completion) (max_tasks=120)

_Logged 2026-06-24 18:06 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 93.3% (112/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-24 — run_ravens_eval (Qwen--Qwen3-0.6B, completion) (max_tasks=120)

_Logged 2026-06-24 18:09 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-0.6B
- **Overall:** 95.0% (114/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 75.0% (15/20) · pattern 100.0% (20/20) · pattern_tuple 95.0% (19/20) · progression 100.0% (20/20)

---
## 2026-06-24 — run_ravens_eval (Qwen--Qwen3-1.7B, completion) (max_tasks=120)

_Logged 2026-06-24 18:14 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-1.7B
- **Overall:** 92.5% (111/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 90.0% (18/20) · progression 100.0% (20/20)

---
## 2026-06-24 — run_ravens_eval (Qwen--Qwen3-4B, completion) (max_tasks=120)

_Logged 2026-06-24 18:18 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-4B
- **Overall:** 95.8% (115/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 75.0% (15/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-24 — run_ravens_eval (Qwen--Qwen3-8B, completion) (max_tasks=120)

_Logged 2026-06-24 18:23 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-8B
- **Overall:** 95.0% (114/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 70.0% (14/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-25 — run_ravens_eval (EleutherAI--pythia-70m-deduped, instruction) (max_tasks=120)

_Logged 2026-06-25 22:52 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 28.3% (34/120)
- **By task:** combine 20.0% (4/20) · constancy 15.0% (3/20) · intersection 40.0% (8/20) · pattern 30.0% (6/20) · pattern_tuple 35.0% (7/20) · progression 30.0% (6/20)

---
## 2026-06-25 — run_ravens_eval (EleutherAI--pythia-160m-deduped, instruction) (max_tasks=120)

_Logged 2026-06-25 22:55 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### EleutherAI/pythia-160m-deduped
- **Overall:** 19.2% (23/120)
- **By task:** combine 25.0% (5/20) · constancy 10.0% (2/20) · intersection 20.0% (4/20) · pattern 25.0% (5/20) · pattern_tuple 20.0% (4/20) · progression 15.0% (3/20)

---
## 2026-06-25 — run_ravens_eval (EleutherAI--pythia-410m-deduped, instruction) (max_tasks=120)

_Logged 2026-06-25 22:59 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 13.3% (16/120)
- **By task:** combine 25.0% (5/20) · constancy 20.0% (4/20) · intersection 0.0% (0/20) · pattern 20.0% (4/20) · pattern_tuple 5.0% (1/20) · progression 10.0% (2/20)

---
## 2026-06-25 — run_ravens_eval (EleutherAI--pythia-1b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-25 23:03 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### EleutherAI/pythia-1b-deduped
- **Overall:** 31.7% (38/120)
- **By task:** combine 35.0% (7/20) · constancy 15.0% (3/20) · intersection 40.0% (8/20) · pattern 40.0% (8/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-06-25 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-25 23:08 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 30.8% (37/120)
- **By task:** combine 35.0% (7/20) · constancy 15.0% (3/20) · intersection 40.0% (8/20) · pattern 40.0% (8/20) · pattern_tuple 35.0% (7/20) · progression 20.0% (4/20)

---
## 2026-06-25 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-25 23:13 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 20.0% (24/120)
- **By task:** combine 10.0% (2/20) · constancy 20.0% (4/20) · intersection 25.0% (5/20) · pattern 30.0% (6/20) · pattern_tuple 20.0% (4/20) · progression 15.0% (3/20)

---
## 2026-06-25 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-25 23:19 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 30.0% (36/120)
- **By task:** combine 20.0% (4/20) · constancy 15.0% (3/20) · intersection 45.0% (9/20) · pattern 40.0% (8/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-06-25 — run_ravens_eval (EleutherAI--pythia-12b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-25 23:31 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### EleutherAI/pythia-12b-deduped
- **Overall:** 31.7% (38/120)
- **By task:** combine 35.0% (7/20) · constancy 15.0% (3/20) · intersection 40.0% (8/20) · pattern 40.0% (8/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-06-25 — run_ravens_eval (Qwen--Qwen3-0.6B, instruction) (max_tasks=120)

_Logged 2026-06-25 23:36 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### Qwen/Qwen3-0.6B
- **Overall:** 35.8% (43/120)
- **By task:** combine 45.0% (9/20) · constancy 35.0% (7/20) · intersection 30.0% (6/20) · pattern 25.0% (5/20) · pattern_tuple 20.0% (4/20) · progression 60.0% (12/20)

---
## 2026-06-25 — run_ravens_eval (Qwen--Qwen3-1.7B, instruction) (max_tasks=120)

_Logged 2026-06-25 23:40 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### Qwen/Qwen3-1.7B
- **Overall:** 29.2% (35/120)
- **By task:** combine 25.0% (5/20) · constancy 25.0% (5/20) · intersection 20.0% (4/20) · pattern 40.0% (8/20) · pattern_tuple 45.0% (9/20) · progression 20.0% (4/20)

---
## 2026-06-25 — run_ravens_eval (Qwen--Qwen3-4B, instruction) (max_tasks=120)

_Logged 2026-06-25 23:43 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### Qwen/Qwen3-4B
- **Overall:** 33.3% (40/120)
- **By task:** combine 25.0% (5/20) · constancy 70.0% (14/20) · intersection 35.0% (7/20) · pattern 15.0% (3/20) · pattern_tuple 40.0% (8/20) · progression 15.0% (3/20)

---
## 2026-06-25 — run_ravens_eval (Qwen--Qwen3-8B, instruction) (max_tasks=120)

_Logged 2026-06-25 23:49 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### Qwen/Qwen3-8B
- **Overall:** 42.5% (51/120)
- **By task:** combine 50.0% (10/20) · constancy 95.0% (19/20) · intersection 40.0% (8/20) · pattern 10.0% (2/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-06-25 — run_ravens_eval (Qwen--Qwen3-14B, instruction) (max_tasks=120)

_Logged 2026-06-25 23:57 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`_

### Qwen/Qwen3-14B
- **Overall:** 33.3% (40/120)
- **By task:** combine 15.0% (3/20) · constancy 55.0% (11/20) · intersection 5.0% (1/20) · pattern 50.0% (10/20) · pattern_tuple 35.0% (7/20) · progression 40.0% (8/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-70m-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 01:24 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 20.0% (24/120)
- **By task:** combine 10.0% (2/20) · constancy 20.0% (4/20) · intersection 35.0% (7/20) · pattern 5.0% (1/20) · pattern_tuple 25.0% (5/20) · progression 25.0% (5/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-160m-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 01:26 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### EleutherAI/pythia-160m-deduped
- **Overall:** 23.3% (28/120)
- **By task:** combine 35.0% (7/20) · constancy 20.0% (4/20) · intersection 15.0% (3/20) · pattern 10.0% (2/20) · pattern_tuple 25.0% (5/20) · progression 35.0% (7/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-410m-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 01:28 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 25.8% (31/120)
- **By task:** combine 20.0% (4/20) · constancy 20.0% (4/20) · intersection 40.0% (8/20) · pattern 5.0% (1/20) · pattern_tuple 25.0% (5/20) · progression 45.0% (9/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-1b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 01:31 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### EleutherAI/pythia-1b-deduped
- **Overall:** 24.2% (29/120)
- **By task:** combine 15.0% (3/20) · constancy 20.0% (4/20) · intersection 40.0% (8/20) · pattern 5.0% (1/20) · pattern_tuple 20.0% (4/20) · progression 45.0% (9/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 01:34 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 25.0% (30/120)
- **By task:** combine 10.0% (2/20) · constancy 20.0% (4/20) · intersection 40.0% (8/20) · pattern 5.0% (1/20) · pattern_tuple 20.0% (4/20) · progression 55.0% (11/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 01:39 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 25.0% (30/120)
- **By task:** combine 10.0% (2/20) · constancy 30.0% (6/20) · intersection 45.0% (9/20) · pattern 5.0% (1/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 01:45 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 29.2% (35/120)
- **By task:** combine 15.0% (3/20) · constancy 25.0% (5/20) · intersection 40.0% (8/20) · pattern 30.0% (6/20) · pattern_tuple 40.0% (8/20) · progression 25.0% (5/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-12b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 01:50 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### EleutherAI/pythia-12b-deduped
- **Overall:** 26.7% (32/120)
- **By task:** combine 35.0% (7/20) · constancy 30.0% (6/20) · intersection 40.0% (8/20) · pattern 10.0% (2/20) · pattern_tuple 20.0% (4/20) · progression 25.0% (5/20)

---
## 2026-06-26 — run_ravens_eval (Qwen--Qwen3-0.6B, instruction) (max_tasks=120)

_Logged 2026-06-26 01:53 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### Qwen/Qwen3-0.6B
- **Overall:** 22.5% (27/120)
- **By task:** combine 35.0% (7/20) · constancy 10.0% (2/20) · intersection 45.0% (9/20) · pattern 0.0% (0/20) · pattern_tuple 35.0% (7/20) · progression 10.0% (2/20)

---
## 2026-06-26 — run_ravens_eval (Qwen--Qwen3-1.7B, instruction) (max_tasks=120)

_Logged 2026-06-26 01:57 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### Qwen/Qwen3-1.7B
- **Overall:** 34.2% (41/120)
- **By task:** combine 40.0% (8/20) · constancy 30.0% (6/20) · intersection 20.0% (4/20) · pattern 40.0% (8/20) · pattern_tuple 50.0% (10/20) · progression 25.0% (5/20)

---
## 2026-06-26 — run_ravens_eval (Qwen--Qwen3-4B, instruction) (max_tasks=120)

_Logged 2026-06-26 02:03 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### Qwen/Qwen3-4B
- **Overall:** 59.2% (71/120)
- **By task:** combine 15.0% (3/20) · constancy 100.0% (20/20) · intersection 20.0% (4/20) · pattern 95.0% (19/20) · pattern_tuple 80.0% (16/20) · progression 45.0% (9/20)

---
## 2026-06-26 — run_ravens_eval (Qwen--Qwen3-8B, instruction) (max_tasks=120)

_Logged 2026-06-26 02:10 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### Qwen/Qwen3-8B
- **Overall:** 59.2% (71/120)
- **By task:** combine 25.0% (5/20) · constancy 95.0% (19/20) · intersection 15.0% (3/20) · pattern 95.0% (19/20) · pattern_tuple 85.0% (17/20) · progression 40.0% (8/20)

---
## 2026-06-26 — run_ravens_eval (Qwen--Qwen3-14B, instruction) (max_tasks=120)

_Logged 2026-06-26 02:16 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`_

### Qwen/Qwen3-14B
- **Overall:** 45.8% (55/120)
- **By task:** combine 30.0% (6/20) · constancy 100.0% (20/20) · intersection 10.0% (2/20) · pattern 55.0% (11/20) · pattern_tuple 40.0% (8/20) · progression 40.0% (8/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-70m-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 02:25 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 26.7% (32/120)
- **By task:** combine 35.0% (7/20) · constancy 25.0% (5/20) · intersection 35.0% (7/20) · pattern 20.0% (4/20) · pattern_tuple 25.0% (5/20) · progression 20.0% (4/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-160m-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 02:27 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### EleutherAI/pythia-160m-deduped
- **Overall:** 18.3% (22/120)
- **By task:** combine 35.0% (7/20) · constancy 20.0% (4/20) · intersection 5.0% (1/20) · pattern 35.0% (7/20) · pattern_tuple 15.0% (3/20) · progression 0.0% (0/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-410m-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 02:30 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 23.3% (28/120)
- **By task:** combine 15.0% (3/20) · constancy 20.0% (4/20) · intersection 25.0% (5/20) · pattern 5.0% (1/20) · pattern_tuple 30.0% (6/20) · progression 45.0% (9/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-1b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 02:33 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### EleutherAI/pythia-1b-deduped
- **Overall:** 25.8% (31/120)
- **By task:** combine 25.0% (5/20) · constancy 20.0% (4/20) · intersection 30.0% (6/20) · pattern 5.0% (1/20) · pattern_tuple 30.0% (6/20) · progression 45.0% (9/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 02:37 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 23.3% (28/120)
- **By task:** combine 10.0% (2/20) · constancy 20.0% (4/20) · intersection 35.0% (7/20) · pattern 5.0% (1/20) · pattern_tuple 25.0% (5/20) · progression 45.0% (9/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 02:40 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 25.0% (30/120)
- **By task:** combine 35.0% (7/20) · constancy 30.0% (6/20) · intersection 20.0% (4/20) · pattern 5.0% (1/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 02:45 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 26.7% (32/120)
- **By task:** combine 25.0% (5/20) · constancy 30.0% (6/20) · intersection 45.0% (9/20) · pattern 5.0% (1/20) · pattern_tuple 35.0% (7/20) · progression 20.0% (4/20)

---
## 2026-06-26 — run_ravens_eval (EleutherAI--pythia-12b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-26 02:51 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### EleutherAI/pythia-12b-deduped
- **Overall:** 30.0% (36/120)
- **By task:** combine 30.0% (6/20) · constancy 30.0% (6/20) · intersection 35.0% (7/20) · pattern 5.0% (1/20) · pattern_tuple 40.0% (8/20) · progression 40.0% (8/20)

---
## 2026-06-26 — run_ravens_eval (Qwen--Qwen3-0.6B, instruction) (max_tasks=120)

_Logged 2026-06-26 02:55 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### Qwen/Qwen3-0.6B
- **Overall:** 22.5% (27/120)
- **By task:** combine 40.0% (8/20) · constancy 15.0% (3/20) · intersection 15.0% (3/20) · pattern 5.0% (1/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-06-26 — run_ravens_eval (Qwen--Qwen3-0.6B, instruction) (max_tasks=120)

_Logged 2026-06-26 17:08 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### Qwen/Qwen3-0.6B
- **Overall:** 22.5% (27/120)
- **By task:** combine 40.0% (8/20) · constancy 15.0% (3/20) · intersection 15.0% (3/20) · pattern 5.0% (1/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-06-26 — run_ravens_eval (Qwen--Qwen3-1.7B, instruction) (max_tasks=120)

_Logged 2026-06-26 17:13 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### Qwen/Qwen3-1.7B
- **Overall:** 29.2% (35/120)
- **By task:** combine 20.0% (4/20) · constancy 60.0% (12/20) · intersection 20.0% (4/20) · pattern 10.0% (2/20) · pattern_tuple 50.0% (10/20) · progression 15.0% (3/20)

---
## 2026-06-26 — run_ravens_eval (Qwen--Qwen3-4B, instruction) (max_tasks=120)

_Logged 2026-06-26 17:18 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### Qwen/Qwen3-4B
- **Overall:** 60.8% (73/120)
- **By task:** combine 40.0% (8/20) · constancy 100.0% (20/20) · intersection 20.0% (4/20) · pattern 65.0% (13/20) · pattern_tuple 90.0% (18/20) · progression 50.0% (10/20)

---
## 2026-06-26 — run_ravens_eval (Qwen--Qwen3-8B, instruction) (max_tasks=120)

_Logged 2026-06-26 17:25 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### Qwen/Qwen3-8B
- **Overall:** 71.7% (86/120)
- **By task:** combine 60.0% (12/20) · constancy 100.0% (20/20) · intersection 25.0% (5/20) · pattern 90.0% (18/20) · pattern_tuple 100.0% (20/20) · progression 55.0% (11/20)

---
## 2026-06-26 — run_ravens_eval (Qwen--Qwen3-14B, instruction) (max_tasks=120)

_Logged 2026-06-26 17:30 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`_

### Qwen/Qwen3-14B
- **Overall:** 45.0% (54/120)
- **By task:** combine 30.0% (6/20) · constancy 95.0% (19/20) · intersection 30.0% (6/20) · pattern 55.0% (11/20) · pattern_tuple 30.0% (6/20) · progression 30.0% (6/20)

---
## 2026-06-29 — run_ravens_eval (EleutherAI--pythia-70m-deduped, instruction) (max_tasks=120)

_Logged 2026-06-29 21:41 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 21.7% (26/120)
- **By task:** combine 5.0% (1/20) · constancy 15.0% (3/20) · intersection 30.0% (6/20) · pattern 20.0% (4/20) · pattern_tuple 40.0% (8/20) · progression 20.0% (4/20)

---
## 2026-06-29 — run_ravens_eval (EleutherAI--pythia-160m-deduped, instruction) (max_tasks=120)

_Logged 2026-06-29 21:44 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-160m-deduped
- **Overall:** 32.5% (39/120)
- **By task:** combine 25.0% (5/20) · constancy 15.0% (3/20) · intersection 40.0% (8/20) · pattern 50.0% (10/20) · pattern_tuple 35.0% (7/20) · progression 30.0% (6/20)

---
## 2026-06-29 — run_ravens_eval (EleutherAI--pythia-410m-deduped, instruction) (max_tasks=120)

_Logged 2026-06-29 21:49 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 21.7% (26/120)
- **By task:** combine 10.0% (2/20) · constancy 25.0% (5/20) · intersection 15.0% (3/20) · pattern 15.0% (3/20) · pattern_tuple 40.0% (8/20) · progression 25.0% (5/20)

---
## 2026-06-29 — run_ravens_eval (EleutherAI--pythia-1b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-29 21:56 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-1b-deduped
- **Overall:** 21.7% (26/120)
- **By task:** combine 10.0% (2/20) · constancy 30.0% (6/20) · intersection 20.0% (4/20) · pattern 10.0% (2/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-06-29 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-29 22:07 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 30.8% (37/120)
- **By task:** combine 35.0% (7/20) · constancy 20.0% (4/20) · intersection 40.0% (8/20) · pattern 35.0% (7/20) · pattern_tuple 35.0% (7/20) · progression 20.0% (4/20)

---
## 2026-06-29 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-29 22:19 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 22.5% (27/120)
- **By task:** combine 10.0% (2/20) · constancy 35.0% (7/20) · intersection 35.0% (7/20) · pattern 15.0% (3/20) · pattern_tuple 20.0% (4/20) · progression 20.0% (4/20)

---
## 2026-06-29 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-29 22:41 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 31.7% (38/120)
- **By task:** combine 35.0% (7/20) · constancy 15.0% (3/20) · intersection 40.0% (8/20) · pattern 40.0% (8/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-06-29 — run_ravens_eval (EleutherAI--pythia-12b-deduped, instruction) (max_tasks=120)

_Logged 2026-06-29 22:56 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-12b-deduped
- **Overall:** 31.7% (38/120)
- **By task:** combine 20.0% (4/20) · constancy 10.0% (2/20) · intersection 35.0% (7/20) · pattern 45.0% (9/20) · pattern_tuple 50.0% (10/20) · progression 30.0% (6/20)

---
## 2026-06-29 — run_ravens_eval (Qwen--Qwen3-0.6B, instruction) (max_tasks=120)

_Logged 2026-06-29 22:59 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-0.6B
- **Overall:** 31.7% (38/120)
- **By task:** combine 65.0% (13/20) · constancy 25.0% (5/20) · intersection 15.0% (3/20) · pattern 5.0% (1/20) · pattern_tuple 55.0% (11/20) · progression 25.0% (5/20)

---
## 2026-06-29 — run_ravens_eval (Qwen--Qwen3-1.7B, instruction) (max_tasks=120)

_Logged 2026-06-29 23:03 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-1.7B
- **Overall:** 65.0% (78/120)
- **By task:** combine 85.0% (17/20) · constancy 100.0% (20/20) · intersection 15.0% (3/20) · pattern 90.0% (18/20) · pattern_tuple 70.0% (14/20) · progression 30.0% (6/20)

---
## 2026-06-29 — run_ravens_eval (Qwen--Qwen3-4B, instruction) (max_tasks=120)

_Logged 2026-06-29 23:07 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-4B
- **Overall:** 75.0% (90/120)
- **By task:** combine 55.0% (11/20) · constancy 100.0% (20/20) · intersection 50.0% (10/20) · pattern 100.0% (20/20) · pattern_tuple 85.0% (17/20) · progression 60.0% (12/20)

---
## 2026-06-29 — run_ravens_eval (Qwen--Qwen3-8B, instruction) (max_tasks=120)

_Logged 2026-06-29 23:10 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-8B
- **Overall:** 84.2% (101/120)
- **By task:** combine 75.0% (15/20) · constancy 100.0% (20/20) · intersection 45.0% (9/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 85.0% (17/20)

---
## 2026-06-29 — run_ravens_eval (Qwen--Qwen3-14B, instruction) (max_tasks=120)

_Logged 2026-06-29 23:14 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-14B
- **Overall:** 90.0% (108/120)
- **By task:** combine 90.0% (18/20) · constancy 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 90.0% (18/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-70m-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 01:47 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 56.7% (68/120)
- **By task:** combine 0.0% (0/20) · constancy 100.0% (20/20) · intersection 10.0% (2/20) · pattern 100.0% (20/20) · pattern_tuple 35.0% (7/20) · progression 95.0% (19/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-160m-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 01:49 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-160m-deduped
- **Overall:** 70.8% (85/120)
- **By task:** combine 0.0% (0/20) · constancy 100.0% (20/20) · intersection 55.0% (11/20) · pattern 100.0% (20/20) · pattern_tuple 70.0% (14/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-410m-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 01:51 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 90.0% (108/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 80.0% (16/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-1b-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 01:54 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-1b-deduped
- **Overall:** 90.8% (109/120)
- **By task:** combine 90.0% (18/20) · constancy 100.0% (20/20) · intersection 70.0% (14/20) · pattern 100.0% (20/20) · pattern_tuple 85.0% (17/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 01:58 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 94.2% (113/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 02:05 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 95.0% (114/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 70.0% (14/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 02:11 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 93.3% (112/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-12b-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 02:16 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-12b-deduped
- **Overall:** 94.2% (113/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-0.6B, completion) (max_tasks=120)

_Logged 2026-06-30 02:19 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-0.6B
- **Overall:** 95.0% (114/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 75.0% (15/20) · pattern 100.0% (20/20) · pattern_tuple 95.0% (19/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-1.7B, completion) (max_tasks=120)

_Logged 2026-06-30 02:24 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-1.7B
- **Overall:** 92.5% (111/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 90.0% (18/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-4B, completion) (max_tasks=120)

_Logged 2026-06-30 02:29 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-4B
- **Overall:** 95.8% (115/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 75.0% (15/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-8B, completion) (max_tasks=120)

_Logged 2026-06-30 02:36 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-8B
- **Overall:** 95.0% (114/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 70.0% (14/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-14B, completion) (max_tasks=120)

_Logged 2026-06-30 02:43 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-14B
- **Overall:** 94.2% (113/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-0.6B, completion) (max_tasks=120)

_Logged 2026-06-30 17:07 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### Qwen/Qwen3-0.6B
- **Overall:** 71.7% (86/120)
- **By task:** combine 60.0% (12/20) · constancy 100.0% (20/20) · intersection 55.0% (11/20) · pattern 100.0% (20/20) · pattern_tuple 20.0% (4/20) · progression 95.0% (19/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-1.7B, completion) (max_tasks=120)

_Logged 2026-06-30 17:12 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### Qwen/Qwen3-1.7B
- **Overall:** 79.2% (95/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 20.0% (4/20) · progression 95.0% (19/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-4B, completion) (max_tasks=120)

_Logged 2026-06-30 17:16 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### Qwen/Qwen3-4B
- **Overall:** 92.5% (111/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 55.0% (11/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-8B, completion) (max_tasks=120)

_Logged 2026-06-30 17:21 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### Qwen/Qwen3-8B
- **Overall:** 94.2% (113/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 75.0% (15/20) · pattern 100.0% (20/20) · pattern_tuple 90.0% (18/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-14B, completion) (max_tasks=120)

_Logged 2026-06-30 17:27 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### Qwen/Qwen3-14B
- **Overall:** 95.8% (115/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 80.0% (16/20) · pattern 100.0% (20/20) · pattern_tuple 95.0% (19/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-70m-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 17:38 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 55.8% (67/120)
- **By task:** combine 0.0% (0/20) · constancy 100.0% (20/20) · intersection 20.0% (4/20) · pattern 85.0% (17/20) · pattern_tuple 45.0% (9/20) · progression 85.0% (17/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-160m-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 17:39 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-160m-deduped
- **Overall:** 66.7% (80/120)
- **By task:** combine 0.0% (0/20) · constancy 100.0% (20/20) · intersection 35.0% (7/20) · pattern 100.0% (20/20) · pattern_tuple 65.0% (13/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-410m-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 17:42 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 78.3% (94/120)
- **By task:** combine 35.0% (7/20) · constancy 100.0% (20/20) · intersection 55.0% (11/20) · pattern 100.0% (20/20) · pattern_tuple 80.0% (16/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-1b-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 17:45 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-1b-deduped
- **Overall:** 61.7% (74/120)
- **By task:** combine 20.0% (4/20) · constancy 100.0% (20/20) · intersection 55.0% (11/20) · pattern 100.0% (20/20) · pattern_tuple 0.0% (0/20) · progression 95.0% (19/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 17:49 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 85.8% (103/120)
- **By task:** combine 70.0% (14/20) · constancy 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 80.0% (16/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 17:52 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 71.7% (86/120)
- **By task:** combine 75.0% (15/20) · constancy 100.0% (20/20) · intersection 50.0% (10/20) · pattern 100.0% (20/20) · pattern_tuple 5.0% (1/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 17:57 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 79.2% (95/120)
- **By task:** combine 95.0% (19/20) · constancy 100.0% (20/20) · intersection 50.0% (10/20) · pattern 100.0% (20/20) · pattern_tuple 30.0% (6/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (EleutherAI--pythia-12b-deduped, completion) (max_tasks=120)

_Logged 2026-06-30 18:02 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-12b-deduped
- **Overall:** 92.5% (111/120)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 95.0% (19/20) · progression 100.0% (20/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-0.6B, instruction) (max_tasks=120)

_Logged 2026-06-30 18:11 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-0.6B
- **Overall:** 35.8% (43/120)
- **By task:** combine 35.0% (7/20) · constancy 75.0% (15/20) · intersection 20.0% (4/20) · pattern 45.0% (9/20) · pattern_tuple 10.0% (2/20) · progression 30.0% (6/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-1.7B, instruction) (max_tasks=120)

_Logged 2026-06-30 18:14 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-1.7B
- **Overall:** 46.7% (56/120)
- **By task:** combine 40.0% (8/20) · constancy 95.0% (19/20) · intersection 30.0% (6/20) · pattern 75.0% (15/20) · pattern_tuple 10.0% (2/20) · progression 30.0% (6/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-4B, instruction) (max_tasks=120)

_Logged 2026-06-30 18:16 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-4B
- **Overall:** 44.2% (53/120)
- **By task:** combine 60.0% (12/20) · constancy 75.0% (15/20) · intersection 35.0% (7/20) · pattern 85.0% (17/20) · pattern_tuple 5.0% (1/20) · progression 5.0% (1/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-8B, instruction) (max_tasks=120)

_Logged 2026-06-30 18:19 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-8B
- **Overall:** 64.2% (77/120)
- **By task:** combine 95.0% (19/20) · constancy 100.0% (20/20) · intersection 50.0% (10/20) · pattern 100.0% (20/20) · pattern_tuple 20.0% (4/20) · progression 20.0% (4/20)

---
## 2026-06-30 — run_ravens_eval (Qwen--Qwen3-14B, instruction) (max_tasks=120)

_Logged 2026-06-30 18:21 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-14B
- **Overall:** 85.0% (102/120)
- **By task:** combine 75.0% (15/20) · constancy 100.0% (20/20) · intersection 55.0% (11/20) · pattern 95.0% (19/20) · pattern_tuple 100.0% (20/20) · progression 85.0% (17/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-70m-deduped, instruction) (max_tasks=140)

_Logged 2026-07-01 20:25 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 20.0% (28/140)
- **By task:** combine 5.0% (1/20) · constancy 15.0% (3/20) · constancy_row 10.0% (2/20) · intersection 30.0% (6/20) · pattern 20.0% (4/20) · pattern_tuple 40.0% (8/20) · progression 20.0% (4/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-160m-deduped, instruction) (max_tasks=140)

_Logged 2026-07-01 20:29 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-160m-deduped
- **Overall:** 31.4% (44/140)
- **By task:** combine 25.0% (5/20) · constancy 15.0% (3/20) · constancy_row 25.0% (5/20) · intersection 40.0% (8/20) · pattern 50.0% (10/20) · pattern_tuple 35.0% (7/20) · progression 30.0% (6/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-410m-deduped, instruction) (max_tasks=140)

_Logged 2026-07-01 20:34 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 20.7% (29/140)
- **By task:** combine 10.0% (2/20) · constancy 25.0% (5/20) · constancy_row 15.0% (3/20) · intersection 15.0% (3/20) · pattern 15.0% (3/20) · pattern_tuple 40.0% (8/20) · progression 25.0% (5/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-1b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-01 20:42 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-1b-deduped
- **Overall:** 20.7% (29/140)
- **By task:** combine 10.0% (2/20) · constancy 30.0% (6/20) · constancy_row 15.0% (3/20) · intersection 20.0% (4/20) · pattern 10.0% (2/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-01 20:53 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 28.6% (40/140)
- **By task:** combine 35.0% (7/20) · constancy 20.0% (4/20) · constancy_row 15.0% (3/20) · intersection 40.0% (8/20) · pattern 35.0% (7/20) · pattern_tuple 35.0% (7/20) · progression 20.0% (4/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-01 21:03 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 19.3% (27/140)
- **By task:** combine 15.0% (3/20) · constancy 30.0% (6/20) · constancy_row 15.0% (3/20) · intersection 35.0% (7/20) · pattern 15.0% (3/20) · pattern_tuple 10.0% (2/20) · progression 15.0% (3/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-01 21:23 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 30.0% (42/140)
- **By task:** combine 35.0% (7/20) · constancy 15.0% (3/20) · constancy_row 20.0% (4/20) · intersection 40.0% (8/20) · pattern 40.0% (8/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-12b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-01 21:35 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-12b-deduped
- **Overall:** 30.7% (43/140)
- **By task:** combine 20.0% (4/20) · constancy 10.0% (2/20) · constancy_row 25.0% (5/20) · intersection 35.0% (7/20) · pattern 45.0% (9/20) · pattern_tuple 50.0% (10/20) · progression 30.0% (6/20)

---
## 2026-07-01 — run_ravens_eval (Qwen--Qwen3-0.6B, instruction) (max_tasks=140)

_Logged 2026-07-01 21:38 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-0.6B
- **Overall:** 31.4% (44/140)
- **By task:** combine 65.0% (13/20) · constancy 25.0% (5/20) · constancy_row 30.0% (6/20) · intersection 15.0% (3/20) · pattern 5.0% (1/20) · pattern_tuple 55.0% (11/20) · progression 25.0% (5/20)

---
## 2026-07-01 — run_ravens_eval (Qwen--Qwen3-1.7B, instruction) (max_tasks=140)

_Logged 2026-07-01 21:40 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-1.7B
- **Overall:** 63.6% (89/140)
- **By task:** combine 85.0% (17/20) · constancy 100.0% (20/20) · constancy_row 55.0% (11/20) · intersection 15.0% (3/20) · pattern 90.0% (18/20) · pattern_tuple 70.0% (14/20) · progression 30.0% (6/20)

---
## 2026-07-01 — run_ravens_eval (Qwen--Qwen3-4B, instruction) (max_tasks=140)

_Logged 2026-07-01 21:42 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-4B
- **Overall:** 78.6% (110/140)
- **By task:** combine 55.0% (11/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 50.0% (10/20) · pattern 100.0% (20/20) · pattern_tuple 85.0% (17/20) · progression 60.0% (12/20)

---
## 2026-07-01 — run_ravens_eval (Qwen--Qwen3-8B, instruction) (max_tasks=140)

_Logged 2026-07-01 21:44 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-8B
- **Overall:** 86.4% (121/140)
- **By task:** combine 75.0% (15/20) · constancy 100.0% (20/20) · constancy_row 95.0% (19/20) · intersection 45.0% (9/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 90.0% (18/20)

---
## 2026-07-01 — run_ravens_eval (Qwen--Qwen3-14B, instruction) (max_tasks=140)

_Logged 2026-07-01 21:48 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-14B
- **Overall:** 92.1% (129/140)
- **By task:** combine 95.0% (19/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 90.0% (18/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-70m-deduped, completion) (max_tasks=140)

_Logged 2026-07-01 21:51 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 58.6% (82/140)
- **By task:** combine 0.0% (0/20) · constancy 100.0% (20/20) · constancy_row 70.0% (14/20) · intersection 10.0% (2/20) · pattern 100.0% (20/20) · pattern_tuple 35.0% (7/20) · progression 95.0% (19/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-160m-deduped, completion) (max_tasks=140)

_Logged 2026-07-01 21:53 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-160m-deduped
- **Overall:** 75.0% (105/140)
- **By task:** combine 0.0% (0/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 55.0% (11/20) · pattern 100.0% (20/20) · pattern_tuple 70.0% (14/20) · progression 100.0% (20/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-410m-deduped, completion) (max_tasks=140)

_Logged 2026-07-01 21:56 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 91.4% (128/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 80.0% (16/20) · progression 100.0% (20/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-1b-deduped, completion) (max_tasks=140)

_Logged 2026-07-01 21:59 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-1b-deduped
- **Overall:** 92.1% (129/140)
- **By task:** combine 90.0% (18/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 70.0% (14/20) · pattern 100.0% (20/20) · pattern_tuple 85.0% (17/20) · progression 100.0% (20/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, completion) (max_tasks=140)

_Logged 2026-07-01 22:03 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 95.0% (133/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, completion) (max_tasks=140)

_Logged 2026-07-01 22:06 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 95.7% (134/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 70.0% (14/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, completion) (max_tasks=140)

_Logged 2026-07-01 22:11 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 94.3% (132/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-01 — run_ravens_eval (EleutherAI--pythia-12b-deduped, completion) (max_tasks=140)

_Logged 2026-07-01 22:16 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### EleutherAI/pythia-12b-deduped
- **Overall:** 95.0% (133/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-01 — run_ravens_eval (Qwen--Qwen3-0.6B, completion) (max_tasks=140)

_Logged 2026-07-01 22:20 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-0.6B
- **Overall:** 95.7% (134/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 75.0% (15/20) · pattern 100.0% (20/20) · pattern_tuple 95.0% (19/20) · progression 100.0% (20/20)

---
## 2026-07-01 — run_ravens_eval (Qwen--Qwen3-1.7B, completion) (max_tasks=140)

_Logged 2026-07-01 22:25 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-1.7B
- **Overall:** 93.6% (131/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 90.0% (18/20) · progression 100.0% (20/20)

---
## 2026-07-01 — run_ravens_eval (Qwen--Qwen3-4B, completion) (max_tasks=140)

_Logged 2026-07-01 22:29 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-4B
- **Overall:** 95.7% (134/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 70.0% (14/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-01 — run_ravens_eval (Qwen--Qwen3-8B, completion) (max_tasks=140)

_Logged 2026-07-01 22:35 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-8B
- **Overall:** 95.7% (134/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 70.0% (14/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-01 — run_ravens_eval (Qwen--Qwen3-14B, completion) (max_tasks=140)

_Logged 2026-07-01 22:41 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-14B
- **Overall:** 95.0% (133/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-70m-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 17:01 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 30.7% (43/140)
- **By task:** combine 35.0% (7/20) · constancy 20.0% (4/20) · constancy_row 20.0% (4/20) · intersection 45.0% (9/20) · pattern 40.0% (8/20) · pattern_tuple 30.0% (6/20) · progression 25.0% (5/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-160m-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 17:04 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-160m-deduped
- **Overall:** 28.6% (40/140)
- **By task:** combine 30.0% (6/20) · constancy 15.0% (3/20) · constancy_row 20.0% (4/20) · intersection 35.0% (7/20) · pattern 40.0% (8/20) · pattern_tuple 30.0% (6/20) · progression 30.0% (6/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-410m-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 17:09 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 27.9% (39/140)
- **By task:** combine 35.0% (7/20) · constancy 15.0% (3/20) · constancy_row 20.0% (4/20) · intersection 25.0% (5/20) · pattern 30.0% (6/20) · pattern_tuple 45.0% (9/20) · progression 25.0% (5/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-1b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 17:16 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-1b-deduped
- **Overall:** 25.7% (36/140)
- **By task:** combine 35.0% (7/20) · constancy 20.0% (4/20) · constancy_row 30.0% (6/20) · intersection 15.0% (3/20) · pattern 5.0% (1/20) · pattern_tuple 30.0% (6/20) · progression 45.0% (9/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 17:27 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 30.7% (43/140)
- **By task:** combine 35.0% (7/20) · constancy 15.0% (3/20) · constancy_row 20.0% (4/20) · intersection 40.0% (8/20) · pattern 45.0% (9/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 17:37 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 25.0% (35/140)
- **By task:** combine 15.0% (3/20) · constancy 30.0% (6/20) · constancy_row 30.0% (6/20) · intersection 35.0% (7/20) · pattern 15.0% (3/20) · pattern_tuple 15.0% (3/20) · progression 35.0% (7/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 17:56 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 25.0% (35/140)
- **By task:** combine 35.0% (7/20) · constancy 15.0% (3/20) · constancy_row 15.0% (3/20) · intersection 30.0% (6/20) · pattern 30.0% (6/20) · pattern_tuple 35.0% (7/20) · progression 15.0% (3/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-12b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 18:10 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-12b-deduped
- **Overall:** 26.4% (37/140)
- **By task:** combine 25.0% (5/20) · constancy 20.0% (4/20) · constancy_row 15.0% (3/20) · intersection 40.0% (8/20) · pattern 30.0% (6/20) · pattern_tuple 30.0% (6/20) · progression 25.0% (5/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-0.6B, instruction) (max_tasks=140)

_Logged 2026-07-02 18:13 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-0.6B
- **Overall:** 35.7% (50/140)
- **By task:** combine 35.0% (7/20) · constancy 75.0% (15/20) · constancy_row 35.0% (7/20) · intersection 20.0% (4/20) · pattern 45.0% (9/20) · pattern_tuple 10.0% (2/20) · progression 30.0% (6/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-1.7B, instruction) (max_tasks=140)

_Logged 2026-07-02 18:17 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-1.7B
- **Overall:** 45.7% (64/140)
- **By task:** combine 40.0% (8/20) · constancy 95.0% (19/20) · constancy_row 40.0% (8/20) · intersection 30.0% (6/20) · pattern 75.0% (15/20) · pattern_tuple 10.0% (2/20) · progression 30.0% (6/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-4B, instruction) (max_tasks=140)

_Logged 2026-07-02 18:19 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-4B
- **Overall:** 48.6% (68/140)
- **By task:** combine 60.0% (12/20) · constancy 80.0% (16/20) · constancy_row 60.0% (12/20) · intersection 40.0% (8/20) · pattern 85.0% (17/20) · pattern_tuple 5.0% (1/20) · progression 10.0% (2/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-8B, instruction) (max_tasks=140)

_Logged 2026-07-02 18:21 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-8B
- **Overall:** 67.1% (94/140)
- **By task:** combine 95.0% (19/20) · constancy 100.0% (20/20) · constancy_row 85.0% (17/20) · intersection 50.0% (10/20) · pattern 100.0% (20/20) · pattern_tuple 20.0% (4/20) · progression 20.0% (4/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-14B, instruction) (max_tasks=140)

_Logged 2026-07-02 18:25 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-14B
- **Overall:** 87.1% (122/140)
- **By task:** combine 75.0% (15/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 55.0% (11/20) · pattern 95.0% (19/20) · pattern_tuple 100.0% (20/20) · progression 85.0% (17/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-70m-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 18:37 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 53.6% (75/140)
- **By task:** combine 0.0% (0/20) · constancy 100.0% (20/20) · constancy_row 40.0% (8/20) · intersection 20.0% (4/20) · pattern 85.0% (17/20) · pattern_tuple 45.0% (9/20) · progression 85.0% (17/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-160m-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 18:39 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-160m-deduped
- **Overall:** 71.4% (100/140)
- **By task:** combine 0.0% (0/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 35.0% (7/20) · pattern 100.0% (20/20) · pattern_tuple 65.0% (13/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-410m-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 18:42 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 81.4% (114/140)
- **By task:** combine 35.0% (7/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 55.0% (11/20) · pattern 100.0% (20/20) · pattern_tuple 80.0% (16/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-1b-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 18:45 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-1b-deduped
- **Overall:** 67.1% (94/140)
- **By task:** combine 20.0% (4/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 55.0% (11/20) · pattern 100.0% (20/20) · pattern_tuple 0.0% (0/20) · progression 95.0% (19/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 18:49 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 87.9% (123/140)
- **By task:** combine 70.0% (14/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 80.0% (16/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 18:52 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 75.7% (106/140)
- **By task:** combine 75.0% (15/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 50.0% (10/20) · pattern 100.0% (20/20) · pattern_tuple 5.0% (1/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 18:57 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 82.1% (115/140)
- **By task:** combine 95.0% (19/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 50.0% (10/20) · pattern 100.0% (20/20) · pattern_tuple 30.0% (6/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-12b-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 19:02 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### EleutherAI/pythia-12b-deduped
- **Overall:** 93.6% (131/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 95.0% (19/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-0.6B, completion) (max_tasks=140)

_Logged 2026-07-02 19:07 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### Qwen/Qwen3-0.6B
- **Overall:** 75.7% (106/140)
- **By task:** combine 60.0% (12/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 55.0% (11/20) · pattern 100.0% (20/20) · pattern_tuple 20.0% (4/20) · progression 95.0% (19/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-1.7B, completion) (max_tasks=140)

_Logged 2026-07-02 19:12 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### Qwen/Qwen3-1.7B
- **Overall:** 82.1% (115/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 20.0% (4/20) · progression 95.0% (19/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-4B, completion) (max_tasks=140)

_Logged 2026-07-02 19:17 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### Qwen/Qwen3-4B
- **Overall:** 93.6% (131/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 55.0% (11/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-8B, completion) (max_tasks=140)

_Logged 2026-07-02 19:23 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### Qwen/Qwen3-8B
- **Overall:** 95.0% (133/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 75.0% (15/20) · pattern 100.0% (20/20) · pattern_tuple 90.0% (18/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-14B, completion) (max_tasks=140)

_Logged 2026-07-02 19:29 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 0`; `--prompt-type completion`_

### Qwen/Qwen3-14B
- **Overall:** 96.4% (135/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 80.0% (16/20) · pattern 100.0% (20/20) · pattern_tuple 95.0% (19/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-70m-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 20:25 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 23.6% (33/140)
- **By task:** combine 20.0% (4/20) · constancy 20.0% (4/20) · constancy_row 10.0% (2/20) · intersection 35.0% (7/20) · pattern 15.0% (3/20) · pattern_tuple 25.0% (5/20) · progression 40.0% (8/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-160m-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 20:29 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-160m-deduped
- **Overall:** 32.9% (46/140)
- **By task:** combine 35.0% (7/20) · constancy 35.0% (7/20) · constancy_row 20.0% (4/20) · intersection 40.0% (8/20) · pattern 45.0% (9/20) · pattern_tuple 30.0% (6/20) · progression 25.0% (5/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-410m-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 20:35 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 25.0% (35/140)
- **By task:** combine 15.0% (3/20) · constancy 40.0% (8/20) · constancy_row 15.0% (3/20) · intersection 45.0% (9/20) · pattern 10.0% (2/20) · pattern_tuple 30.0% (6/20) · progression 20.0% (4/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-1b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 20:44 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-1b-deduped
- **Overall:** 25.0% (35/140)
- **By task:** combine 25.0% (5/20) · constancy 35.0% (7/20) · constancy_row 15.0% (3/20) · intersection 35.0% (7/20) · pattern 10.0% (2/20) · pattern_tuple 30.0% (6/20) · progression 25.0% (5/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 20:55 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 22.1% (31/140)
- **By task:** combine 35.0% (7/20) · constancy 35.0% (7/20) · constancy_row 15.0% (3/20) · intersection 15.0% (3/20) · pattern 5.0% (1/20) · pattern_tuple 25.0% (5/20) · progression 25.0% (5/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 21:04 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 20.7% (29/140)
- **By task:** combine 35.0% (7/20) · constancy 20.0% (4/20) · constancy_row 20.0% (4/20) · intersection 15.0% (3/20) · pattern 0.0% (0/20) · pattern_tuple 10.0% (2/20) · progression 45.0% (9/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 21:23 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 30.0% (42/140)
- **By task:** combine 35.0% (7/20) · constancy 15.0% (3/20) · constancy_row 20.0% (4/20) · intersection 40.0% (8/20) · pattern 40.0% (8/20) · pattern_tuple 35.0% (7/20) · progression 25.0% (5/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-12b-deduped, instruction) (max_tasks=140)

_Logged 2026-07-02 21:38 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### EleutherAI/pythia-12b-deduped
- **Overall:** 29.3% (41/140)
- **By task:** combine 35.0% (7/20) · constancy 15.0% (3/20) · constancy_row 25.0% (5/20) · intersection 30.0% (6/20) · pattern 35.0% (7/20) · pattern_tuple 40.0% (8/20) · progression 25.0% (5/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-0.6B, instruction) (max_tasks=140)

_Logged 2026-07-02 21:43 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-0.6B
- **Overall:** 40.0% (56/140)
- **By task:** combine 65.0% (13/20) · constancy 40.0% (8/20) · constancy_row 30.0% (6/20) · intersection 15.0% (3/20) · pattern 5.0% (1/20) · pattern_tuple 95.0% (19/20) · progression 30.0% (6/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-1.7B, instruction) (max_tasks=140)

_Logged 2026-07-02 21:47 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-1.7B
- **Overall:** 72.1% (101/140)
- **By task:** combine 90.0% (18/20) · constancy 100.0% (20/20) · constancy_row 55.0% (11/20) · intersection 20.0% (4/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 40.0% (8/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-4B, instruction) (max_tasks=140)

_Logged 2026-07-02 21:50 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-4B
- **Overall:** 84.3% (118/140)
- **By task:** combine 70.0% (14/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 70.0% (14/20) · pattern 100.0% (20/20) · pattern_tuple 95.0% (19/20) · progression 55.0% (11/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-8B, instruction) (max_tasks=140)

_Logged 2026-07-02 21:53 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-8B
- **Overall:** 88.6% (124/140)
- **By task:** combine 85.0% (17/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 55.0% (11/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 80.0% (16/20)

---
## 2026-07-02 — run_ravens_eval (Qwen--Qwen3-14B, instruction) (max_tasks=140)

_Logged 2026-07-02 21:56 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-14B
- **Overall:** 96.4% (135/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 75.0% (15/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-70m-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 23:28 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### EleutherAI/pythia-70m-deduped
- **Overall:** 56.4% (79/140)
- **By task:** combine 5.0% (1/20) · constancy 100.0% (20/20) · constancy_row 75.0% (15/20) · intersection 20.0% (4/20) · pattern 90.0% (18/20) · pattern_tuple 35.0% (7/20) · progression 70.0% (14/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-160m-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 23:30 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### EleutherAI/pythia-160m-deduped
- **Overall:** 80.0% (112/140)
- **By task:** combine 20.0% (4/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 40.0% (8/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-410m-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 23:33 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### EleutherAI/pythia-410m-deduped
- **Overall:** 94.3% (132/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 60.0% (12/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-1b-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 23:36 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### EleutherAI/pythia-1b-deduped
- **Overall:** 96.4% (135/140)
- **By task:** combine 95.0% (19/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 80.0% (16/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-1.4b-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 23:41 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### EleutherAI/pythia-1.4b-deduped
- **Overall:** 95.7% (134/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 70.0% (14/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-2.8b-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 23:44 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### EleutherAI/pythia-2.8b-deduped
- **Overall:** 96.4% (135/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 75.0% (15/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-02 — run_ravens_eval (EleutherAI--pythia-6.9b-deduped, completion) (max_tasks=140)

_Logged 2026-07-02 23:50 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### EleutherAI/pythia-6.9b-deduped
- **Overall:** 95.0% (133/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-03 — run_ravens_eval (EleutherAI--pythia-12b-deduped, completion) (max_tasks=140)

_Logged 2026-07-03 00:00 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 1024`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### EleutherAI/pythia-12b-deduped
- **Overall:** 95.0% (133/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 65.0% (13/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-03 — run_ravens_eval (Qwen--Qwen3-0.6B, completion) (max_tasks=140)

_Logged 2026-07-03 00:04 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### Qwen/Qwen3-0.6B
- **Overall:** 96.4% (135/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 75.0% (15/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-03 — run_ravens_eval (Qwen--Qwen3-1.7B, completion) (max_tasks=140)

_Logged 2026-07-03 00:09 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### Qwen/Qwen3-1.7B
- **Overall:** 96.4% (135/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 75.0% (15/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-03 — run_ravens_eval (Qwen--Qwen3-4B, completion) (max_tasks=140)

_Logged 2026-07-03 00:13 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### Qwen/Qwen3-4B
- **Overall:** 98.6% (138/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 90.0% (18/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-03 — run_ravens_eval (Qwen--Qwen3-8B, completion) (max_tasks=140)

_Logged 2026-07-03 00:20 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### Qwen/Qwen3-8B
- **Overall:** 100.0% (140/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 100.0% (20/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-03 — run_ravens_eval (Qwen--Qwen3-14B, completion) (max_tasks=140)

_Logged 2026-07-03 00:25 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 1024`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 3`; `--prompt-type completion`_

### Qwen/Qwen3-14B
- **Overall:** 100.0% (140/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 100.0% (20/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-med-small-1M-1, completion) (max_tasks=10)

_Logged 2026-07-06 18:24 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-med-small-1M-1
- **Overall:** 40.0% (4/10)
- **By task:** combine 100.0% (1/1) · constancy 50.0% (1/2) · constancy_row 0.0% (0/2) · intersection 100.0% (1/1) · pattern 0.0% (0/2) · pattern_tuple 100.0% (1/1) · progression 0.0% (0/1)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-med-small-1M-2, completion) (max_tasks=10)

_Logged 2026-07-06 18:24 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-med-small-1M-2
- **Overall:** 50.0% (5/10)
- **By task:** combine 100.0% (1/1) · constancy 50.0% (1/2) · constancy_row 0.0% (0/2) · intersection 0.0% (0/1) · pattern 100.0% (2/2) · pattern_tuple 100.0% (1/1) · progression 0.0% (0/1)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-med-small-1M-3, completion) (max_tasks=10)

_Logged 2026-07-06 18:25 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-med-small-1M-3
- **Overall:** 20.0% (2/10)
- **By task:** combine 100.0% (1/1) · constancy 0.0% (0/2) · constancy_row 0.0% (0/2) · intersection 0.0% (0/1) · pattern 0.0% (0/2) · pattern_tuple 100.0% (1/1) · progression 0.0% (0/1)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-10M-1, completion) (max_tasks=10)

_Logged 2026-07-06 18:25 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-10M-1
- **Overall:** 0.0% (0/10)
- **By task:** combine 0.0% (0/1) · constancy 0.0% (0/2) · constancy_row 0.0% (0/2) · intersection 0.0% (0/1) · pattern 0.0% (0/2) · pattern_tuple 0.0% (0/1) · progression 0.0% (0/1)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-10M-2, completion) (max_tasks=10)

_Logged 2026-07-06 18:26 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-10M-2
- **Overall:** 30.0% (3/10)
- **By task:** combine 0.0% (0/1) · constancy 0.0% (0/2) · constancy_row 50.0% (1/2) · intersection 100.0% (1/1) · pattern 50.0% (1/2) · pattern_tuple 0.0% (0/1) · progression 0.0% (0/1)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-10M-3, completion) (max_tasks=10)

_Logged 2026-07-06 18:26 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-10M-3
- **Overall:** 0.0% (0/10)
- **By task:** combine 0.0% (0/1) · constancy 0.0% (0/2) · constancy_row 0.0% (0/2) · intersection 0.0% (0/1) · pattern 0.0% (0/2) · pattern_tuple 0.0% (0/1) · progression 0.0% (0/1)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-100M-1, completion) (max_tasks=10)

_Logged 2026-07-06 18:27 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-100M-1
- **Overall:** 30.0% (3/10)
- **By task:** combine 100.0% (1/1) · constancy 50.0% (1/2) · constancy_row 0.0% (0/2) · intersection 0.0% (0/1) · pattern 50.0% (1/2) · pattern_tuple 0.0% (0/1) · progression 0.0% (0/1)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-100M-2, completion) (max_tasks=10)

_Logged 2026-07-06 18:27 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-100M-2
- **Overall:** 50.0% (5/10)
- **By task:** combine 100.0% (1/1) · constancy 50.0% (1/2) · constancy_row 0.0% (0/2) · intersection 100.0% (1/1) · pattern 0.0% (0/2) · pattern_tuple 100.0% (1/1) · progression 100.0% (1/1)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-100M-3, completion) (max_tasks=10)

_Logged 2026-07-06 18:28 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-100M-3
- **Overall:** 20.0% (2/10)
- **By task:** combine 0.0% (0/1) · constancy 50.0% (1/2) · constancy_row 0.0% (0/2) · intersection 0.0% (0/1) · pattern 0.0% (0/2) · pattern_tuple 0.0% (0/1) · progression 100.0% (1/1)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-1B-1, completion) (max_tasks=10)

_Logged 2026-07-06 18:28 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-1B-1
- **Overall:** 30.0% (3/10)
- **By task:** combine 100.0% (1/1) · constancy 50.0% (1/2) · constancy_row 0.0% (0/2) · intersection 0.0% (0/1) · pattern 50.0% (1/2) · pattern_tuple 0.0% (0/1) · progression 0.0% (0/1)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-1B-2, completion) (max_tasks=10)

_Logged 2026-07-06 18:29 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-1B-2
- **Overall:** 70.0% (7/10)
- **By task:** combine 100.0% (1/1) · constancy 50.0% (1/2) · constancy_row 100.0% (2/2) · intersection 100.0% (1/1) · pattern 50.0% (1/2) · pattern_tuple 100.0% (1/1) · progression 0.0% (0/1)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-1B-3, completion) (max_tasks=10)

_Logged 2026-07-06 18:29 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-1B-3
- **Overall:** 30.0% (3/10)
- **By task:** combine 100.0% (1/1) · constancy 0.0% (0/2) · constancy_row 50.0% (1/2) · intersection 0.0% (0/1) · pattern 0.0% (0/2) · pattern_tuple 100.0% (1/1) · progression 0.0% (0/1)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-med-small-1M-1, completion) (max_tasks=21)

_Logged 2026-07-06 18:30 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-med-small-1M-1
- **Overall:** 47.6% (10/21)
- **By task:** combine 33.3% (1/3) · constancy 66.7% (2/3) · constancy_row 0.0% (0/3) · intersection 100.0% (3/3) · pattern 0.0% (0/3) · pattern_tuple 100.0% (3/3) · progression 33.3% (1/3)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-med-small-1M-2, completion) (max_tasks=21)

_Logged 2026-07-06 18:30 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-med-small-1M-2
- **Overall:** 47.6% (10/21)
- **By task:** combine 66.7% (2/3) · constancy 66.7% (2/3) · constancy_row 0.0% (0/3) · intersection 33.3% (1/3) · pattern 100.0% (3/3) · pattern_tuple 66.7% (2/3) · progression 0.0% (0/3)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-med-small-1M-3, completion) (max_tasks=21)

_Logged 2026-07-06 18:30 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-med-small-1M-3
- **Overall:** 9.5% (2/21)
- **By task:** combine 33.3% (1/3) · constancy 0.0% (0/3) · constancy_row 0.0% (0/3) · intersection 0.0% (0/3) · pattern 0.0% (0/3) · pattern_tuple 33.3% (1/3) · progression 0.0% (0/3)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-10M-1, completion) (max_tasks=21)

_Logged 2026-07-06 18:30 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-10M-1
- **Overall:** 9.5% (2/21)
- **By task:** combine 33.3% (1/3) · constancy 0.0% (0/3) · constancy_row 0.0% (0/3) · intersection 0.0% (0/3) · pattern 0.0% (0/3) · pattern_tuple 33.3% (1/3) · progression 0.0% (0/3)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-10M-2, completion) (max_tasks=21)

_Logged 2026-07-06 18:31 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-10M-2
- **Overall:** 33.3% (7/21)
- **By task:** combine 66.7% (2/3) · constancy 33.3% (1/3) · constancy_row 33.3% (1/3) · intersection 33.3% (1/3) · pattern 33.3% (1/3) · pattern_tuple 33.3% (1/3) · progression 0.0% (0/3)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-10M-3, completion) (max_tasks=21)

_Logged 2026-07-06 18:31 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-10M-3
- **Overall:** 14.3% (3/21)
- **By task:** combine 33.3% (1/3) · constancy 0.0% (0/3) · constancy_row 0.0% (0/3) · intersection 0.0% (0/3) · pattern 0.0% (0/3) · pattern_tuple 33.3% (1/3) · progression 33.3% (1/3)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-100M-1, completion) (max_tasks=21)

_Logged 2026-07-06 18:31 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-100M-1
- **Overall:** 38.1% (8/21)
- **By task:** combine 33.3% (1/3) · constancy 33.3% (1/3) · constancy_row 33.3% (1/3) · intersection 33.3% (1/3) · pattern 66.7% (2/3) · pattern_tuple 33.3% (1/3) · progression 33.3% (1/3)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-100M-2, completion) (max_tasks=21)

_Logged 2026-07-06 18:32 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-100M-2
- **Overall:** 33.3% (7/21)
- **By task:** combine 33.3% (1/3) · constancy 33.3% (1/3) · constancy_row 0.0% (0/3) · intersection 33.3% (1/3) · pattern 0.0% (0/3) · pattern_tuple 100.0% (3/3) · progression 33.3% (1/3)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-100M-3, completion) (max_tasks=21)

_Logged 2026-07-06 18:32 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-100M-3
- **Overall:** 19.0% (4/21)
- **By task:** combine 0.0% (0/3) · constancy 33.3% (1/3) · constancy_row 0.0% (0/3) · intersection 0.0% (0/3) · pattern 0.0% (0/3) · pattern_tuple 66.7% (2/3) · progression 33.3% (1/3)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-1B-1, completion) (max_tasks=21)

_Logged 2026-07-06 18:32 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-1B-1
- **Overall:** 28.6% (6/21)
- **By task:** combine 66.7% (2/3) · constancy 66.7% (2/3) · constancy_row 0.0% (0/3) · intersection 0.0% (0/3) · pattern 33.3% (1/3) · pattern_tuple 33.3% (1/3) · progression 0.0% (0/3)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-1B-2, completion) (max_tasks=21)

_Logged 2026-07-06 18:33 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-1B-2
- **Overall:** 57.1% (12/21)
- **By task:** combine 33.3% (1/3) · constancy 66.7% (2/3) · constancy_row 100.0% (3/3) · intersection 33.3% (1/3) · pattern 33.3% (1/3) · pattern_tuple 100.0% (3/3) · progression 33.3% (1/3)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-1B-3, completion) (max_tasks=21)

_Logged 2026-07-06 18:33 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-1B-3
- **Overall:** 23.8% (5/21)
- **By task:** combine 66.7% (2/3) · constancy 0.0% (0/3) · constancy_row 33.3% (1/3) · intersection 0.0% (0/3) · pattern 0.0% (0/3) · pattern_tuple 66.7% (2/3) · progression 0.0% (0/3)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-med-small-1M-1, completion) (max_tasks=140)

_Logged 2026-07-06 18:36 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-med-small-1M-1
- **Overall:** 30.7% (43/140)
- **By task:** combine 30.0% (6/20) · constancy 40.0% (8/20) · constancy_row 40.0% (8/20) · intersection 25.0% (5/20) · pattern 10.0% (2/20) · pattern_tuple 55.0% (11/20) · progression 15.0% (3/20)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-med-small-1M-2, completion) (max_tasks=140)

_Logged 2026-07-06 18:37 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-med-small-1M-2
- **Overall:** 25.0% (35/140)
- **By task:** combine 30.0% (6/20) · constancy 35.0% (7/20) · constancy_row 15.0% (3/20) · intersection 20.0% (4/20) · pattern 30.0% (6/20) · pattern_tuple 30.0% (6/20) · progression 15.0% (3/20)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-med-small-1M-3, completion) (max_tasks=140)

_Logged 2026-07-06 18:37 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-med-small-1M-3
- **Overall:** 25.0% (35/140)
- **By task:** combine 35.0% (7/20) · constancy 15.0% (3/20) · constancy_row 25.0% (5/20) · intersection 15.0% (3/20) · pattern 25.0% (5/20) · pattern_tuple 45.0% (9/20) · progression 15.0% (3/20)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-10M-1, completion) (max_tasks=140)

_Logged 2026-07-06 18:38 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-10M-1
- **Overall:** 20.7% (29/140)
- **By task:** combine 25.0% (5/20) · constancy 20.0% (4/20) · constancy_row 20.0% (4/20) · intersection 0.0% (0/20) · pattern 15.0% (3/20) · pattern_tuple 40.0% (8/20) · progression 25.0% (5/20)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-10M-2, completion) (max_tasks=140)

_Logged 2026-07-06 18:39 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-10M-2
- **Overall:** 22.9% (32/140)
- **By task:** combine 25.0% (5/20) · constancy 15.0% (3/20) · constancy_row 25.0% (5/20) · intersection 25.0% (5/20) · pattern 25.0% (5/20) · pattern_tuple 30.0% (6/20) · progression 15.0% (3/20)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-10M-3, completion) (max_tasks=140)

_Logged 2026-07-06 18:39 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-10M-3
- **Overall:** 13.6% (19/140)
- **By task:** combine 10.0% (2/20) · constancy 5.0% (1/20) · constancy_row 15.0% (3/20) · intersection 5.0% (1/20) · pattern 20.0% (4/20) · pattern_tuple 25.0% (5/20) · progression 15.0% (3/20)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-100M-1, completion) (max_tasks=140)

_Logged 2026-07-06 18:40 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-100M-1
- **Overall:** 25.7% (36/140)
- **By task:** combine 25.0% (5/20) · constancy 45.0% (9/20) · constancy_row 15.0% (3/20) · intersection 5.0% (1/20) · pattern 25.0% (5/20) · pattern_tuple 45.0% (9/20) · progression 20.0% (4/20)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-100M-2, completion) (max_tasks=140)

_Logged 2026-07-06 18:40 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-100M-2
- **Overall:** 20.0% (28/140)
- **By task:** combine 15.0% (3/20) · constancy 15.0% (3/20) · constancy_row 20.0% (4/20) · intersection 40.0% (8/20) · pattern 0.0% (0/20) · pattern_tuple 45.0% (9/20) · progression 5.0% (1/20)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-100M-3, completion) (max_tasks=140)

_Logged 2026-07-06 18:41 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-100M-3
- **Overall:** 21.4% (30/140)
- **By task:** combine 15.0% (3/20) · constancy 5.0% (1/20) · constancy_row 5.0% (1/20) · intersection 40.0% (8/20) · pattern 15.0% (3/20) · pattern_tuple 45.0% (9/20) · progression 25.0% (5/20)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-1B-1, completion) (max_tasks=140)

_Logged 2026-07-06 18:41 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-1B-1
- **Overall:** 20.7% (29/140)
- **By task:** combine 25.0% (5/20) · constancy 20.0% (4/20) · constancy_row 20.0% (4/20) · intersection 5.0% (1/20) · pattern 15.0% (3/20) · pattern_tuple 40.0% (8/20) · progression 20.0% (4/20)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-1B-2, completion) (max_tasks=140)

_Logged 2026-07-06 18:42 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-1B-2
- **Overall:** 35.0% (49/140)
- **By task:** combine 35.0% (7/20) · constancy 50.0% (10/20) · constancy_row 45.0% (9/20) · intersection 5.0% (1/20) · pattern 35.0% (7/20) · pattern_tuple 65.0% (13/20) · progression 10.0% (2/20)

---
## 2026-07-06 — run_ravens_eval (nyu-mll--roberta-base-1B-3, completion) (max_tasks=140)

_Logged 2026-07-06 18:43 UTC. Modal HF transformers (`modal_eval.py`); PLL scoring; max_len=512; T4; `--n-examples 1`; `--prompt-type completion`_

### nyu-mll/roberta-base-1B-3
- **Overall:** 11.4% (16/140)
- **By task:** combine 25.0% (5/20) · constancy 10.0% (2/20) · constancy_row 10.0% (2/20) · intersection 0.0% (0/20) · pattern 0.0% (0/20) · pattern_tuple 35.0% (7/20) · progression 0.0% (0/20)

---
## 2026-07-06 — run_ravens_eval (Qwen--Qwen3-8B-Base, instruction) (max_tasks=140)

_Logged 2026-07-06 20:18 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-8B-Base
- **Overall:** 70.0% (98/140)
- **By task:** combine 60.0% (12/20) · constancy 100.0% (20/20) · constancy_row 80.0% (16/20) · intersection 50.0% (10/20) · pattern 90.0% (18/20) · pattern_tuple 60.0% (12/20) · progression 50.0% (10/20)

---
## 2026-07-06 — run_ravens_eval (Qwen--Qwen3-8B-Base, completion) (max_tasks=140)

_Logged 2026-07-06 20:29 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-8B-Base
- **Overall:** 93.6% (131/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 55.0% (11/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
## 2026-07-06 — run_ravens_eval (Qwen--Qwen3-8B, instruction) (max_tasks=140)

_Logged 2026-07-06 20:36 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type instruction`; `--ravens-prompt-mode choice_only`_

### Qwen/Qwen3-8B
- **Overall:** 85.7% (120/140)
- **By task:** combine 75.0% (15/20) · constancy 100.0% (20/20) · constancy_row 95.0% (19/20) · intersection 45.0% (9/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 85.0% (17/20)

---
## 2026-07-06 — run_ravens_eval (Qwen--Qwen3-8B, completion) (max_tasks=140)

_Logged 2026-07-06 20:44 UTC. Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`; `--n-examples 1`; `--prompt-type completion`_

### Qwen/Qwen3-8B
- **Overall:** 95.7% (134/140)
- **By task:** combine 100.0% (20/20) · constancy 100.0% (20/20) · constancy_row 100.0% (20/20) · intersection 70.0% (14/20) · pattern 100.0% (20/20) · pattern_tuple 100.0% (20/20) · progression 100.0% (20/20)

---
