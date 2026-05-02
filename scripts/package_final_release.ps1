# Package Final Release — TRUST Agent V10

> Script đóng gói TRUST_AGENT_FINAL_PACKAGE.zip chứa source code, benchmarks và tài liệu.

## Cách chạy

```powershell
# Từ thư mục gốc của dự án
cd D:\Work\project\mutiAgentfakeNew
powershell .\scripts\package_final_release.ps1
```

## Nội dung sau khi đóng gói

```
TRUST_AGENT_FINAL_PACKAGE.zip
├── src/
│   ├── trust_agents/          # Multi-Agent pipeline (S-V-O, Zero-Tolerance)
│   ├── fake_news_detector/    # Legacy fallback + UI
│   └── shared_fact_checking/  # Retrieval policy & service
├── scripts/                    # Benchmark runners, smoke test
├── benchmarks/final/           # standard_benchmark_v4.json
├── METHODOLOGY.md              # S-V-O + Zero-Tolerance explainer
├── COMPARE_WITH_FRIEND.md      # 20 bẫy test cases + blank columns
├── EVOLUTION_V7_TO_V10.md      # Evolution report cho slides
└── V10_GUIDE.md                # Hướng dẫn chạy tối ưu
```