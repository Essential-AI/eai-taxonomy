```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN 32B → 0.5B DISTILLATION PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│   Common Crawl  │
│   104.6M docs   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────────────────────────────────┐
│  Doc Sampling   │     │ If doc > 30k chars:                         │
│  & Chunking     │────▶│ • Take first 10k chars                      │
└────────┬────────┘     │ • Random 10k from middle                    │
         │              │ • Last 10k chars                            │
         │              └─────────────────────────────────────────────┘
         ▼
┌─────────────────┐
│  Qwen2.5-32B    │     ┌─────────────────────────────────────────────┐
│  Instruct       │────▶│ Pass 1: Annotate 8 categories               │
│  Annotation     │     │ • Prompt: 2104 tokens                       │
│                 │     │ • Output: ~791 tokens                       │
└────────┬────────┘     └─────────────────────────────────────────────┘
         │                                    │
         │              ┌─────────────────────────────────────────────┐
         │              │ Pass 2: Annotate 4 more categories          │
         └─────────────▶│ • Additional taxonomy dimensions            │
                        └─────────────────────────────────────────────┘
                                              │
                        ┌─────────────────────▼─────────────────────┐
                        │        Synthetic Training Data            │
                        │    • 104.6M annotated documents           │
                        │    • 12 taxonomy categories total         │
                        └─────────────────────┬─────────────────────┘
                                              │
                        ┌─────────────────────▼─────────────────────┐
                        │         Format Transformation             │
                        │  ┌─────────────┐    ┌─────────────┐       │
                        │  │   Original  │───▶│  Condensed  │       │
                        │  │ Long format │    │   Format    │       │
                        │  └─────────────┘    └─────────────┘       │
                        │  • Remove prompt     • Keep labels only   │
                        │  • 3828→985 tokens   • JSON-like output   │
                        └─────────────────────┬─────────────────────┘
                                              │
┌─────────────────┐     ┌─────────────────────▼─────────────────────┐     ┌─────────────────┐
│  Qwen2.5-0.5B   │     │           Fine-Tuning Process             │     │ EAI-Taxonomy    │
│  Instruct       │────▶│                                           │────▶│ 0.5B            │
│  (Base Model)   │     │  • 82B training tokens                    │     │ (Final Model)   │
└─────────────────┘     │  • Batch size: 2M tokens                  │     └─────────────────┘
                        │  • LR: 1e-4 → 1e-5 → 0                    │
                        │  • Seq length: 16,384                     │
                        │  • AdamW optimizer                        │
                        │  • 2B warmup → cosine → linear decay      │
                        └───────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PERFORMANCE METRICS                                    │
│  ┌─────────────────────┐        ┌─────────────────────┐       ┌─────────────────┐   │
│  │ Inter-Category NMI  │        │  Cohen's Kappa      │       │  Speed Gain     │   │
│  │   < 0.10            │        │  0.71 (Random)      │       │  ~50x faster    │   │
│  │                     │        │  0.73 (STEM)        │       │  inference      │   │
│  └─────────────────────┘        └─────────────────────┘       └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```