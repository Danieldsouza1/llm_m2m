"""
agents/powerplant.py
--------------------
PowerPlant agent. Handles tasks sent by the grid controller.
Faults are injected dynamically by the controller via a special
"INJECT_FAULT:<error_message>" message — no hardcoded failures.
"""

import paho.mqtt.client as mqtt

BROKER = "localhost"
AGENT_NAME = "Plant1"

# Holds the currently injected fault (None = no active fault)
active_fault = None


def handle_task(task: str) -> str:
    global active_fault
    if active_fault:
        error = active_fault
        active_fault = None  # fault fires once then clears
        return error
    task_map = {
        "generate_power": "SUCCESS: Power generated at Plant1",
    }
    return task_map.get(task, f"ERROR: Unknown task '{task}' for PowerPlant")


def on_message(client, userdata, msg):
    global active_fault
    payload = msg.payload.decode().strip()

    # Fault injection message from controller
    if payload.startswith("INJECT_FAULT:"):
        active_fault = payload[len("INJECT_FAULT:"):]
        print(f"⚡ PowerPlant fault injected: {active_fault}")
        return

    print(f"⚡ PowerPlant received task: {payload}")
    result = handle_task(payload)
    client.publish("agents/status", result)
    print(f"📤 Sent: {result}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message
client.connect(BROKER, 1883)
client.subscribe(f"agents/{AGENT_NAME}/task")

print(f"✅ PowerPlant connected | subscribed to agents/{AGENT_NAME}/task")
client.loop_forever()
