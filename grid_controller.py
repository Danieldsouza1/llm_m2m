"""
grid_controller.py  —  FIXED & IMPROVED
----------------------------------------
Fixes in this version:
  1. REDIRECT now has a max attempts limit (MAX_REDIRECTS = 2)
     previously only RETRY was limited — REDIRECT looped infinitely
  2. REDIRECT_MAP now maps by TASK not by agent
     previously 'transmit_power' redirected to Plant1 which can't handle it
  3. Added CUDA error handling in call_llm — catches ResponseError and
     waits before retrying rather than crashing the whole pipeline
  4. Combined recovery attempt counter covers both RETRY and REDIRECT
     so total recovery attempts never exceed MAX_RETRIES + MAX_REDIRECTS
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paho.mqtt.client as mqtt
import json, time, re, ollama, httpx

from rag_pipeline import RAG
from prompts import PLANNING_PROMPT, RECOVERY_PROMPT
from evaluation.logger import init_logger, log_event

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BROKER        = "localhost"
MODEL         = "mistral:7b-instruct-q4_0"
MAX_RETRIES   = 2    # max RETRY attempts per step
MAX_REDIRECTS = 2    # max REDIRECT attempts per step  ← NEW

AGENT_LIST = ["Plant1", "Transmission1", "Substation1"]

AGENTS_DESC = """- Plant1: generates power
- Substation1: distributes power
- Transmission1: transmits / routes power"""

TASK_AGENT_MAP = {
    "generate_power":   "Plant1",
    "transmit_power":   "Transmission1",
    "distribute_power": "Substation1",
}

# FIX 2: redirect by TASK — send to agent that can actually handle the task
# If primary agent fails, try the next-best agent for that specific task
TASK_REDIRECT_MAP = {
    "generate_power":   "Substation1",    # substation can partially compensate
    "transmit_power":   "Substation1",    # substation handles distribution as fallback
    "distribute_power": "Transmission1",  # transmission as fallback distributor
}

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MANUAL_PATH = os.path.join(BASE_DIR, "data", "energy_manual.txt")

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
tasks          = ["generate_power", "transmit_power", "distribute_power"]
current_step   = 0
retry_count    = 0
redirect_count = 0    # ← NEW separate counter for redirects
start_time     = time.time()
fault_schedule: dict = {}

# ─────────────────────────────────────────────
# RAG INGEST
# ─────────────────────────────────────────────
rag = RAG()
rag.ingest(MANUAL_PATH)

try:
    httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": "nomic-embed-text", "keep_alive": 0},
        timeout=10
    )
    print("🔄 Embedding model unloaded from memory")
except Exception:
    pass

# ─────────────────────────────────────────────
# MQTT CLIENT
# ─────────────────────────────────────────────
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)


# ─────────────────────────────────────────────
# LLM HELPERS
# ─────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    """
    FIX 3: catches CUDA / ResponseError and waits before retrying.
    Returns empty string on failure so caller can use fallback.
    """
    for attempt in range(3):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"num_ctx": 1024, "num_predict": 150, "temperature": 0.1},
                keep_alive=0
            )
            return response["message"]["content"]
        except Exception as e:
            err = str(e)
            if "CUDA error" in err or "llama runner" in err:
                print(f"⚠️  LLM hardware error (attempt {attempt+1}/3), waiting 5s...")
                time.sleep(5)
            else:
                print(f"⚠️  LLM error: {err}")
                break
    print("⚠️  LLM unavailable after retries, using fallback")
    return ""


def extract_json(text: str) -> dict | None:
    try:
        text = text.replace("```json", "").replace("```", "")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"⚠️  JSON parse error: {e}")
    return None


def validate_agent(agent_name: str, task: str) -> str:
    if agent_name not in AGENT_LIST:
        correct = TASK_AGENT_MAP.get(task, "Plant1")
        print(f"⚠️  LLM picked unknown agent '{agent_name}', using '{correct}'")
        return correct
    expected = TASK_AGENT_MAP.get(task)
    if expected and agent_name != expected:
        print(f"⚠️  LLM picked '{agent_name}' for '{task}', correcting to '{expected}'")
        return expected
    return agent_name


# ─────────────────────────────────────────────
# TASK TRIGGER
# ─────────────────────────────────────────────
def trigger_next():
    global current_step, start_time, retry_count, redirect_count

    start_time     = time.time()
    retry_count    = 0     # reset counters for each new step
    redirect_count = 0

    if current_step >= len(tasks):
        print("\n🎉 All tasks complete!")
        return

    task = tasks[current_step]
    print(f"\n🔁 Step {current_step}: {task}")

    # Fault injection
    if current_step in fault_schedule:
        fault           = fault_schedule[current_step]
        agent_for_fault = TASK_AGENT_MAP[task]  # always inject into whoever gets the task
        error_msg       = fault["error_message"]
        inject_topic    = f"agents/{agent_for_fault}/task"
        print(f"💉 Injecting fault into {agent_for_fault}: {error_msg}")
        client.publish(inject_topic, f"INJECT_FAULT:{error_msg}")
        time.sleep(0.3)

    # LLM planning
    prompt   = PLANNING_PROMPT.format(task=task, agents=AGENTS_DESC)
    response = call_llm(prompt)
    print(f"🤖 LLM RAW (planning): {response.strip()}")

    data = extract_json(response)
    if data is None:
        print("⚠️  LLM parse failed, using fallback mapping")
        data = {"agent": TASK_AGENT_MAP.get(task, "Plant1"), "action": task}

    agent = validate_agent(data.get("agent", ""), task)
    topic = f"agents/{agent}/task"
    print(f"📤 Sending '{task}' to {agent} on {topic}")
    client.publish(topic, task)


# ─────────────────────────────────────────────
# RECOVERY EXECUTION
# ─────────────────────────────────────────────
def execute_recovery(decision: str, task: str):
    """
    FIX 1: REDIRECT now has its own counter and limit.
    FIX 2: redirect target chosen by task, not by failed agent.
    Once both RETRY and REDIRECT limits are exhausted → SHUTDOWN.
    """
    global retry_count, redirect_count, current_step

    print(f"\n⚙️  Executing recovery: {decision}")

    if decision == "RETRY":
        if retry_count < MAX_RETRIES:
            retry_count += 1
            current_agent = TASK_AGENT_MAP.get(task, "Plant1")
            print(f"🔄 RETRY {retry_count}/{MAX_RETRIES} — re-sending '{task}' to {current_agent}")
            time.sleep(1)
            client.publish(f"agents/{current_agent}/task", task)
        else:
            print(f"❌ Max retries ({MAX_RETRIES}) reached — escalating to SHUTDOWN")
            _shutdown()

    elif decision == "REDIRECT":
        if redirect_count < MAX_REDIRECTS:
            redirect_count += 1
            # FIX 2: use task-based redirect map, not agent-based
            alt_agent = TASK_REDIRECT_MAP.get(task, "Substation1")
            print(f"↪️  REDIRECT {redirect_count}/{MAX_REDIRECTS} — sending '{task}' to {alt_agent}")
            time.sleep(0.3)
            client.publish(f"agents/{alt_agent}/task", task)
        else:
            print(f"❌ Max redirects ({MAX_REDIRECTS}) reached — escalating to SHUTDOWN")
            _shutdown()

    elif decision == "SHUTDOWN":
        _shutdown()

    else:
        print(f"⚠️  Unknown decision '{decision}', defaulting to SHUTDOWN")
        _shutdown()


def _shutdown():
    print("🛑 Pipeline halted — disconnecting")
    try:
        client.disconnect()
    except Exception:
        pass


# ─────────────────────────────────────────────
# MQTT MESSAGE HANDLER
# ─────────────────────────────────────────────
def on_message(client_ref, userdata, msg):
    global current_step, retry_count, redirect_count

    message = msg.payload.decode().strip()
    print(f"\n📩 Received: {message}")

    task = tasks[min(current_step, len(tasks) - 1)]

    # ── SUCCESS ───────────────────────────────
    if message.startswith("SUCCESS"):
        print("✅ Step success")
        retry_count    = 0
        redirect_count = 0
        log_event(
            task=task,
            agent=TASK_AGENT_MAP.get(task, "unknown"),
            status="SUCCESS",
            error="",
            decision="",
            correct_action="",
            fault_type="none",
            start_time=start_time,
        )
        current_step += 1
        trigger_next()

    # ── FAILURE ───────────────────────────────
    elif message.startswith("ERROR"):
        print("🚨 Failure detected")

        context  = rag.query(message)
        prompt   = RECOVERY_PROMPT.format(error=message, context=context)
        response = call_llm(prompt)
        print(f"🤖 LLM RAW (recovery): {response.strip()}")

        decision_data = extract_json(response)

        if decision_data is None:
            print("⚠️  Failed to parse recovery JSON, using SHUTDOWN fallback")
            decision_data = {"action": "SHUTDOWN", "reason": "parse failure"}

        action = decision_data.get("action", "SHUTDOWN").upper()
        if action not in ("RETRY", "REDIRECT", "SHUTDOWN"):
            print(f"⚠️  Invalid action '{action}', defaulting to SHUTDOWN")
            action = "SHUTDOWN"

        print(f"✅ Recovery decision: {action}  |  {decision_data.get('reason', '')}")

        fault          = fault_schedule.get(current_step, {})
        correct_action = fault.get("correct_action", "")
        fault_type     = fault.get("fault_type", "unknown")

        # Score correctness
        is_correct = "✅" if action == correct_action else "❌"
        if correct_action:
            print(f"📊 Correct answer was: {correct_action} — {is_correct}")

        log_event(
            task=task,
            agent=TASK_AGENT_MAP.get(task, "unknown"),
            status="FAILURE",
            error=message,
            decision=action,
            correct_action=correct_action,
            fault_type=fault_type,
            start_time=start_time,
        )

        execute_recovery(action, task)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
client.on_message = on_message
client.connect(BROKER, 1883)
client.subscribe("agents/status")

print("🚀 Starting Improved Grid Controller")
print(f"   Model        : {MODEL}")
print(f"   Manual       : {MANUAL_PATH}")
print(f"   Max retries  : {MAX_RETRIES}")
print(f"   Max redirects: {MAX_REDIRECTS}")
time.sleep(1)
init_logger()

from fault_generator import get_random_fault
fault_schedule[1] = get_random_fault(fault_id="LIVE_001", llm_ratio=0.5)
print(f"\n🎲 Fault scheduled at step 1:")
print(f"   Type   : {fault_schedule[1]['fault_type']}")
print(f"   Error  : {fault_schedule[1]['error_message']}")
print(f"   Correct: {fault_schedule[1]['correct_action']}")
print(f"   Source : {fault_schedule[1]['source']}\n")

trigger_next()
client.loop_forever()
