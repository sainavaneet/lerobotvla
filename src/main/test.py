from gr00t.eval.service import ExternalRobotInferenceClient
from typing import Dict, Any

raw_obs_dict: Dict[str, Any] = {} # fill in the blanks

policy = ExternalRobotInferenceClient(host="155.230.134.196", port=5555)

print("Policy initialized")
raw_action_chunk: Dict[str, Any] = policy.get_action(raw_obs_dict)
print(raw_action_chunk)

