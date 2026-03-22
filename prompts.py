"""
prompts.py
----------
Prompt templates optimised for Mistral 7B.

Key improvements over phi3 prompts:
  - PLANNING: explicit validation that agent name must match the list exactly
  - RECOVERY: chain-of-thought step before JSON to improve reasoning quality
  - Both prompts hard-constrain output to valid JSON only
"""

# ─────────────────────────────────────────────
# PLANNING PROMPT
# Used by: grid_controller.trigger_next()
# ─────────────────────────────────────────────
PLANNING_PROMPT = """You are an intelligent energy grid controller. Your job is to assign tasks to agents.

Task to perform: {task}

Available agents (use the EXACT name from this list):
{agents}

Rules:
- You MUST pick one agent from the list above — do not invent agent names
- The agent you pick must be capable of the task
- Return ONLY valid JSON with no explanation, no markdown fences

Required output format:
{{"agent": "<exact agent name>", "action": "{task}"}}
"""

# ─────────────────────────────────────────────
# RECOVERY PROMPT
# Used by: grid_controller.on_message() on ERROR
# ─────────────────────────────────────────────
RECOVERY_PROMPT = """You are a fault recovery system for a smart energy grid.

A fault has occurred:
ERROR: {error}

Relevant documentation context:
{context}

Your task:
1. Read the error and context carefully
2. Choose the most appropriate recovery action from: RETRY, REDIRECT, SHUTDOWN
3. Use this decision guide:
   - RETRY: fault is likely transient (frequency deviation, voltage sag, temporary overload)
   - REDIRECT: fault blocks the current path but power can be rerouted (line overload, line disconnect, partial loss)
   - SHUTDOWN: fault is severe and unsafe to continue (transformer overheating, breaker failure, turbine emergency)

Return ONLY valid JSON. No explanation. No markdown. No extra text.

Required output format:
{{"action": "<RETRY or REDIRECT or SHUTDOWN>", "reason": "<one sentence>"}}
"""

# ─────────────────────────────────────────────
# FAULT GENERATION PROMPT
# Used by: fault_generator.generate_llm_fault()
# (defined in fault_generator.py — kept here for reference)
# ─────────────────────────────────────────────
FAULT_GEN_PROMPT = """You are a power systems engineer designing fault scenarios.

Generate ONE realistic novel power grid fault not in this list:
line overload, transformer overheating, circuit breaker failure, voltage sag, ground fault.

Return ONLY valid JSON:
{{"error_message": "ERROR: <description>", "fault_type": "<snake_case>", "agent": "<Plant1|Transmission1|Substation1>", "correct_action": "<RETRY|REDIRECT|SHUTDOWN>", "reasoning": "<one sentence>"}}
"""
