"""
agents/substation.py
---------------------
Substation agent. Faults are injected dynamically — not hardcoded.
"""

import paho.mqtt.client as mqtt

BROKER = "localhost"
AGENT_NAME = "Substation1"

active_fault = None


def handle_task(task: str) -> str:
    global active_fault
    if active_fault:
        error = active_fault
        active_fault = None
        return error
    task_map = {
        "distribute_power": "SUCCESS: Power distributed via Substation1",
    }
    return task_map.get(task, f"ERROR: Unknown task '{task}' for Substation")


def on_message(client, userdata, msg):
    global active_fault
    payload = msg.payload.decode().strip()

    if payload.startswith("INJECT_FAULT:"):
        active_fault = payload[len("INJECT_FAULT:"):]
        print(f"⚡ Substation fault injected: {active_fault}")
        return

    print(f"⚡ Substation received task: {payload}")
    result = handle_task(payload)
    client.publish("agents/status", result)
    print(f"📤 Sent: {result}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message
client.connect(BROKER, 1883)
client.subscribe(f"agents/{AGENT_NAME}/task")

print(f"✅ Substation connected | subscribed to agents/{AGENT_NAME}/task")
client.loop_forever()
