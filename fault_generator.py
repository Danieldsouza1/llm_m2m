"""
fault_generator.py
------------------
Generates faults two ways:
  1. From a predefined library  → reproducible benchmarking
  2. Via Mistral at runtime     → novel / unseen faults

Every fault dict has the shape:
  {
    "fault_id":       str,        # unique id
    "error_message":  str,        # what the agent publishes (e.g. "ERROR: Line overload in sector B")
    "fault_type":     str,        # category label
    "agent":          str,        # which agent this fault belongs to
    "correct_action": str,        # RETRY | REDIRECT | SHUTDOWN  (ground truth for scoring)
    "source":         str,        # "library" | "llm_generated"
  }
"""

import random
import json
import re
import ollama

MODEL = "mistral:7b-instruct-q4_0"

# ─────────────────────────────────────────────
# PREDEFINED FAULT LIBRARY
# ─────────────────────────────────────────────
FAULT_LIBRARY = [
    # Transmission faults
    {
        "fault_id": "F001",
        "error_message": "ERROR: Line overload in sector B",
        "fault_type": "line_overload",
        "agent": "Transmission1",
        "correct_action": "REDIRECT",
        "source": "library",
    },
    {
        "fault_id": "F002",
        "error_message": "ERROR: Transmission line disconnected at node 7",
        "fault_type": "line_disconnect",
        "agent": "Transmission1",
        "correct_action": "REDIRECT",
        "source": "library",
    },
    {
        "fault_id": "F003",
        "error_message": "ERROR: Phase imbalance detected on line 3",
        "fault_type": "phase_imbalance",
        "agent": "Transmission1",
        "correct_action": "RETRY",
        "source": "library",
    },
    # Substation faults
    {
        "fault_id": "F004",
        "error_message": "ERROR: Transformer overheating at substation 2",
        "fault_type": "transformer_fault",
        "agent": "Substation1",
        "correct_action": "SHUTDOWN",
        "source": "library",
    },
    {
        "fault_id": "F005",
        "error_message": "ERROR: Circuit breaker failed to trip at bus 4",
        "fault_type": "breaker_failure",
        "agent": "Substation1",
        "correct_action": "SHUTDOWN",
        "source": "library",
    },
    {
        "fault_id": "F006",
        "error_message": "ERROR: Voltage sag detected at distribution bus 6",
        "fault_type": "voltage_sag",
        "agent": "Substation1",
        "correct_action": "RETRY",
        "source": "library",
    },
    # PowerPlant faults
    {
        "fault_id": "F007",
        "error_message": "ERROR: Generator output frequency deviation",
        "fault_type": "frequency_deviation",
        "agent": "Plant1",
        "correct_action": "RETRY",
        "source": "library",
    },
    {
        "fault_id": "F008",
        "error_message": "ERROR: Turbine emergency shutdown initiated",
        "fault_type": "turbine_shutdown",
        "agent": "Plant1",
        "correct_action": "SHUTDOWN",
        "source": "library",
    },
    {
        "fault_id": "F009",
        "error_message": "ERROR: Partial load loss on generator unit 3",
        "fault_type": "partial_load_loss",
        "agent": "Plant1",
        "correct_action": "REDIRECT",
        "source": "library",
    },
    {
        "fault_id": "F010",
        "error_message": "ERROR: Ground fault on feeder line F12",
        "fault_type": "ground_fault",
        "agent": "Transmission1",
        "correct_action": "RETRY",
        "source": "library",
    },
]


# ─────────────────────────────────────────────
# LLM-GENERATED NOVEL FAULT
# ─────────────────────────────────────────────

FAULT_GEN_PROMPT = """
You are a power systems engineer designing fault scenarios for testing AI grid controllers.

Generate ONE realistic, novel power grid fault that is different from these common ones:
- Line overload, transformer overheating, circuit breaker failure, voltage sag, ground fault

The fault must be realistic and technically plausible for a power grid.

Return ONLY valid JSON, no explanation, no markdown:
{{
  "error_message": "ERROR: <realistic fault description>",
  "fault_type": "<short_snake_case_label>",
  "agent": "<one of: Plant1 | Transmission1 | Substation1>",
  "correct_action": "<one of: RETRY | REDIRECT | SHUTDOWN>",
  "reasoning": "<one sentence why that action is correct>"
}}

Rules:
- error_message must start with "ERROR: "
- correct_action must be exactly RETRY, REDIRECT, or SHUTDOWN
- Be specific: include sector/unit/bus numbers
"""


def generate_llm_fault(fault_id: str) -> dict:
    """Ask Mistral to create a novel fault scenario."""
    print(f"🤖 Generating novel fault via LLM (id={fault_id})...")
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": FAULT_GEN_PROMPT}],
            options={"num_ctx": 1024, "num_predict": 200},
        )
        text = response["message"]["content"]
        text = text.replace("```json", "").replace("```", "")
        
        # Add these lines to clean common Mistral JSON mistakes:
        text = text.replace("'", '"')          # single quotes → double quotes
        text = re.sub(r',\s*}', '}', text)     # trailing commas before }
        text = re.sub(r',\s*]', ']', text)     # trailing commas before ]
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            data["fault_id"] = fault_id
            data["source"] = "llm_generated"
            # validate required keys
            required = {"error_message", "fault_type", "agent", "correct_action"}
            if required.issubset(data.keys()):
                if data["correct_action"] in ("RETRY", "REDIRECT", "SHUTDOWN"):
                    if data["agent"] in ("Plant1", "Transmission1", "Substation1"):
                        print(f"✅ Novel fault generated: {data['fault_type']}")
                        return data
        print("⚠️  LLM fault generation failed validation, using library fallback")
    except Exception as e:
        print(f"⚠️  LLM fault generation error: {e}")
    # fall back to a random library fault with a new id
    fault = random.choice(FAULT_LIBRARY).copy()
    fault["fault_id"] = fault_id
    fault["source"] = "library_fallback"
    return fault


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def get_library_fault(fault_id: str = None) -> dict:
    """Return a random fault from the predefined library."""
    fault = random.choice(FAULT_LIBRARY).copy()
    if fault_id:
        fault["fault_id"] = fault_id
    return fault


def get_random_fault(fault_id: str = "DYN_001", llm_ratio: float = 0.5) -> dict:
    """
    Return a fault that is either from the library or LLM-generated.
    llm_ratio: probability of using LLM generation (0.0 = always library, 1.0 = always LLM)
    """
    if random.random() < llm_ratio:
        return generate_llm_fault(fault_id)
    return get_library_fault(fault_id)


def get_all_library_faults() -> list:
    """Return all predefined faults — used for full benchmark runs."""
    return [f.copy() for f in FAULT_LIBRARY]


if __name__ == "__main__":
    print("=== Library fault ===")
    f = get_library_fault("TEST_L")
    print(json.dumps(f, indent=2))

    print("\n=== LLM-generated fault ===")
    f2 = generate_llm_fault("TEST_G")
    print(json.dumps(f2, indent=2))
