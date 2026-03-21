# MARBLE Project - Dependency Analysis Report

**Analysis Date**: March 21, 2026  
**Scope**: Analyzed imports and usage across marble/, scripts/, tests/, multiagentbench/ directories

---

## 1. DEPENDENCIES ACTIVELY USED

### Core LLM & AI Packages
| Dependency | Version Spec | Usage Found | Files |
|---|---|---|---|
| **litellm** | >=1.52.1,<2 | ✅ USED | llms/model_prompting.py, llms/text_embedding.py, llms/error_handler.py, memory/short_term_memory.py, memory/long_term_memory.py, agent/base_agent.py, scripts/database/batch_eval.py, tests/test_*.py |
| **beartype** | (no version spec) | ✅ USED | llms/model_prompting.py, llms/text_embedding.py, llms/error_handler.py, environments/research_utils/profile_collector.py, environments/research_utils/paper_collector.py |
| **pydantic** | >=2.9.2,<3 | ✅ USED | llms/error_handler.py, environments/research_utils/paper_collector.py |

### Data Processing & ML
| Dependency | Version Spec | Usage Found | Files |
|---|---|---|---|
| **scikit-learn** | >=1.5.2,<2 | ✅ USED | memory/long_term_memory.py (cosine_similarity) |
| **tqdm** | >=4.67.0,<5 | ✅ USED | environments/research_utils/paper_collector.py, configs/test_config_minecraft/test_case_generator.py, scripts/database/batch_eval.py |

### Web & API
| Dependency | Version Spec | Usage Found | Files |
|---|---|---|---|
| **requests** | >=2.32.3,<3 | ✅ USED | tools/web_search.py, environments/web_env.py, environments/research_env.py, environments/minecraft_utils/minecraft_client.py, environments/db_env.py, environments/db_env_docker/prometheus_abnormal_metric.py, environments/db_env_docker/anomaly_trigger/promethues.py, environments/research_utils/paper_collector.py |
| **flask** | >=3.0.3,<4 | ✅ USED | environments/minecraft_utils/minecraft_server.py |
| **waitress** | >=3.0.1,<4 | ✅ USED | environments/minecraft_utils/minecraft_server.py (serve) |

### Research & Document Processing
| Dependency | Version Spec | Usage Found | Files |
|---|---|---|---|
| **arxiv** | >=2.1.3,<3 | ✅ USED | environments/research_utils/paper_collector.py (arXiv search API) |
| **semanticscholar** | >=0.8.4,<1 | ✅ USED | environments/research_utils/profile_collector.py |
| **beautifulsoup4** | >=4.12.3,<5 | ✅ USED | environments/web_env.py, environments/research_utils/paper_collector.py (HTML parsing) |
| **bs4** | >=0.0.2 | ✅ USED | environments/web_env.py, environments/research_utils/paper_collector.py (HTML parsing - same as beautifulsoup4) |
| **keypython** | >=0.8.5,<1 | ✅ USED | environments/research_utils/paper_collector.py (keyword extraction) |
| **pypdf2** | >=3.0.1,<4 | ✅ USED | environments/research_utils/paper_collector.py (PDF reading) |

### Database & Game Environments
| Dependency | Version Spec | Usage Found | Files |
|---|---|---|---|
| **psycopg2-binary** | >=2.9.10,<3 | ✅ USED | environments/db_utils/slow_query.py, environments/db_env_docker/anomaly_trigger/* (multiple PostgreSQL-related files) |
| **pymysql** | >=1.1.1,<2 | ✅ USED | environments/db_env_docker/anomaly_trigger/utils/database.py |

### Utilities
| Dependency | Version Spec | Usage Found | Files |
|---|---|---|---|
| **names** | >=0.3.0,<1 | ✅ USED | environments/werewolf_env.py, environments/minecraft_utils/minecraft_server.py |
| **levenshtein** | >=0.26.1,<1 | ✅ USED | environments/minecraft_utils/env_api.py (string similarity) |
| **colorlog** | >=6.9.0,<7 | ✅ USED | environments/minecraft_utils/utils.py (colored logging) |
| **javascript** | >=1.2.1 | ✅ USED | environments/minecraft_utils/build_judger.py |

### Type Hints (Used implicitly via runtime dependencies)
| Dependency | Version Spec | Usage Found | Files |
|---|---|---|---|
| **types-pyyaml** | ==6.0.12.20240917 | ✅ USED | Type hints for yaml module (used throughout codebase) |
| **types-requests** | ==2.32.0.20240914 | ✅ USED | Type hints for requests module (used throughout codebase) |

---

## 2. CRITICAL ISSUES - MISSING DEPENDENCIES

### ⚠️ HIGH PRIORITY
| Package | Where Used | Why Missing |
|---|---|---|
| **PyYAML** | EXTENSIVELY USED - agent/, marble/utils/, evaluator/, environments/, db_env_docker/anomaly_trigger/ | **CRITICAL**: Only `types-pyyaml` is in dependencies (which provides TYPE STUBS only, not the actual module) |
| **ruamel.yaml** | marble/evaluator/evaluator.py, environments/coding_utils/coder.py, environments/coding_utils/reviewer.py, scripts/reasoning/update_reasoning_config.py, scripts/coding/utils/update_coding_config.py | **CRITICAL**: Not listed in dependencies at all |

**Files importing PyYAML (yaml module):**
- marble/agent/werewolf_agent.py:7
- marble/utils/milestone.py:3  
- marble/run_reasoning_ablation.py:10
- marble/evaluator/werewolf_evaluator.py:11
- marble/environments/werewolf_env.py:12
- marble/environments/db_env_docker/yaml_utils.py:1
- marble/environments/db_env_docker/anomaly_trigger/vacuum_multi.py:11
- marble/environments/db_env_docker/anomaly_trigger/vacuum.py:11
- marble/environments/db_env_docker/anomaly_trigger/script2code.py (multiple imports)
- marble/environments/db_env_docker/anomaly_trigger/reindex_multi.py:11
- marble/environments/db_env_docker/anomaly_trigger/reindex.py:11
- marble/environments/db_env_docker/anomaly_trigger/multi_anomalies.py:13
- marble/environments/db_env_docker/anomaly_trigger/miss_multi.py:11
- marble/environments/db_env_docker/anomaly_trigger/miss.py:11
- marble/environments/db_env_docker/anomaly_trigger/lock_multi.py:11
- scripts/pipeline/run_ollama_pipeline.py:18
- scripts/pipeline/run_vllm_pipeline.py:22
- multiagentbench/jsonl2yaml.py:17

---

## 3. QUESTIONABLE/REDUNDANT DEPENDENCIES

### mypy (Listed as runtime dependency, but used as dev tool)
| Status | Analysis |
|---|---|
| **Listed as:** | Main dependency (mypy>=1.13.0,<2) |
| **Actual Usage:** | Not imported in any code; only used in `[tool.mypy]` configuration section of pyproject.toml |
| **Recommendation:** | **MOVE TO DEV DEPENDENCIES** - mypy is a type checker tool, not a runtime library |
| **Files that reference it:** | pyproject.toml only (configuration) |

---

## 4. DEV/TEST DEPENDENCIES

| Dependency | Version Spec | Status | Usage |
|---|---|---|---|
| **pytest** | (no version) | ✅ USED | tests/test_communication.py and likely other test files |
| **pytest-asyncio** | (no version) | Listed - Not confirmed used | Listed in test section |
| **pre-commit** | (no version) | Dev tool | Configuration framework for git hooks |
| **nbmake** | (no version) | Dev tool | Jupyter notebook testing |
| **types-setuptools** | (no version) | Type hints | Development type stubs |

---

## 5. RECOMMENDATIONS

### Immediate Actions Required
1. **Add PyYAML to main dependencies** - Currently only has type stubs (`types-pyyaml`)
   - Suggestion: `PyYAML>=6.0,<7` or similar
   
2. **Add ruamel.yaml to main dependencies** - Currently missing entirely
   - Suggestion: `ruamel.yaml>=0.17.0` or similar

3. **Move mypy to dev dependencies**
   - Remove from main `dependencies` list
   - Add to `[project.optional-dependencies] dev = [..., "mypy>=1.13.0,<2", ...]`
   - It's a type checker tool, not a runtime library

### Optional Cleanups
- **bs4 vs beautifulsoup4**: `bs4` appears to be a compatibility wrapper/alias for `beautifulsoup4`. Consider keeping only `beautifulsoup4` and removing the redundant `bs4` dependency.

---

## Summary Statistics

- **Total main dependencies in pyproject.toml:** 23
- **Actively used in code:** 21
- **Missing but required:** 2 (PyYAML, ruamel.yaml) 
- **Questionable for main dependencies:** 1 (mypy)
- **Completely unused:** 0

**Critical Issues Found:** 2 (missing PyYAML and ruamel.yaml)  
**Configuration Issues Found:** 1 (mypy in wrong section)
