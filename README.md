# Smart Energy Grid AI Fault Management

A multi-agent AI system that simulates fault detection and recovery in a power grid,
comparing four approaches: Rule-Based, LLM-Only, RAG-Only, and LLM+RAG.

## Project Structure
```
llm_m2m/
├── grid_controller.py      # Main controller with LLM+RAG fault recovery
├── fault_generator.py      # Dynamic fault generation (library + LLM-generated)
├── rag_pipeline.py         # Section-aware RAG with ChromaDB
├── prompts.py              # Prompt templates for Mistral
├── run_eval.py             # Multi-run evaluation script
├── dashboard.py            # Streamlit dashboard
├── agents/
│   ├── powerplant.py
│   ├── transmission.py
│   └── substation.py
├── data/
│   └── energy_manual.txt   # 7-section power engineering manual
└── evaluation/
    ├── baseline_eval.py    # Benchmark all 4 approaches
    ├── metrics.py          # Compute metrics from results.csv
    └── logger.py           # CSV event logger
```

## Requirements

- Python 3.11+
- Ollama (https://ollama.com) with these models pulled:
  - `ollama pull mistral:7b-instruct-q4_0`
  - `ollama pull nomic-embed-text`
- Mosquitto MQTT broker (https://mosquitto.org)



## Running the live system

Open 4 terminals:
```bash
# Terminal 1
python agents/powerplant.py

# Terminal 2
python agents/substation.py

# Terminal 3
python agents/transmission.py

# Terminal 4
python grid_controller.py
```

## Running the benchmark evaluation
```bash
python evaluation/baseline_eval.py
```

## Running the dashboard
```bash
streamlit run dashboard.py
```

