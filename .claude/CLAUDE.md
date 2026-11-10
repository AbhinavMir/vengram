# Vengram

Vengram is a Vapi custom LLM. It serves a local model, grades each finished call with Claude, and lets a Claude editor change the model's weights when a call goes wrong. Each kept change is a model version, and the live model switches to it in one step.

- `server.py`: HTTP routes. `/p/<token>/v1/chat/completions` is the model endpoint, `/p/<token>/webhook` is the Vapi webhook, and the dashboard and `/api/*` answer only from the local computer.
- `brain.py`: the two model copies (live and staging), versions, the grader, and the editor tools.
- `engine.py`: model loading and weight edits.
- `config.json`: the model, the grader model, the port, and an optional public address. Change the model here and restart.

To set up Vengram for a user, run `/setup`.
