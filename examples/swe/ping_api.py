import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def http(method: str, path: str, body: dict | None = None):
    url = f"{BASE}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def main():
    print("health:", http("GET", "/health"))

    # 1) create
    create = http("POST", "/envs", {
        "backend": "docker",
        "verbose": True,
        "scaffold": "r2egym"
    })
    env_id = create["env_id"]
    print("created env:", env_id)

    try:
        # 2) reset
        reset = http("POST", f"/envs/{env_id}/reset")
        print("instruction (truncated):", reset["instruction"][:200], "...")

        # 3) optional: list commands
        cmds = http("GET", f"/envs/{env_id}/commands")
        print("commands (sample):", cmds.get("commands", [])[:10])

        # 4) step: run a harmless command
        step = http("POST", f"/envs/{env_id}/step", {
            "function_name": "execute_bash",
            "parameters": {"cmd": "pwd && ls -1 | head -20"}
        })
        print("step done=", step["done"], "reward=", step["reward"]) 
        print("observation (truncated):", step["observation"][:400], "...")

        # 5) compute reward (invokes container tests; can take time)
        # comment out if you just want a quick ping
        # reward = http("POST", f"/envs/{env_id}/compute_reward")
        # print("final reward:", reward)
    finally:
        # 6) cleanup
        _ = http("DELETE", f"/envs/{env_id}")
        print("deleted env")


if __name__ == "__main__":
    main()


