# <img src="logo.png" alt="" width="38" align="absmiddle"> Vengram

Vengram serves a local language model to [Vapi](https://vapi.ai) as a custom LLM, and it changes the weights of that model after calls that go wrong. It does not train the model and it does not change the prompt.

## 1. Method

1. A Vapi assistant uses the Vengram model through an OpenAI-compatible endpoint.
2. Vapi sends its server messages to the Vengram webhook. Vengram forwards each message at once to the customer's own webhook, if one is set.
3. At the end of a call, Claude grades the call against the assistant's system prompt. If the model itself must change, the grade contains `update_model_instructions`: what to do differently and a corrected reply for one turn.
4. A Claude editor reads the model at the points where its reply differs from the corrected reply (logit lens and per-layer effects). It then applies rank-one weight edits, $W' = W + \alpha\, u v^{\top}$, to a staging copy of the model, evaluates each edit, and keeps an edit only if regression agreement on other prompts stays at or above 0.95.
5. Kept edits become a new model version. The live copy of the model is replaced with that version in one locked step, so calls are not blocked while the editor works. Any earlier version, or the base model, can be made live again.

## 2. Installation

    pip install -r requirements.txt

Put an Anthropic API key in `~/.engram/credentials.json` with mode 0600:

    {"anthropic": "sk-ant-..."}

Install [ngrok](https://ngrok.com) and sign in to it. Then start the server:

    python server.py --model /path/to/a/local/model

The server opens an ngrok tunnel and prints the public address. On a host that already has a public address, add `--public-url https://...` and the server does not open a tunnel.

## 3. Use

Open the dashboard at http://127.0.0.1:8790. The dashboard answers only on the local computer, and it has no sign-in.

1. Create a project. Each project gets a secret Custom LLM URL and a secret Server URL.
2. In Vapi, set the assistant's model to Custom LLM with the Custom LLM URL, and set its Server URL with End Of Call Report turned on.
3. Calls appear under Calls with their grade, any `update_model_instructions`, and the editor's steps. Versions appear under Model.

## 4. Limits

- Each call costs one Claude grade. Each model update costs up to 30 Claude requests. Model updates are limited to 10 a day by default.
- All projects share one model, so an update from one project changes the model for every project.
- Data stays in `~/.engram/live/`: projects, calls, transcripts, and versions. A hosted grader receives the call transcript.

## 5. Related paper

The rank-one edit is the edit that the paper below evaluates.

> Abhinav Srivastava. *Can Weight-Space Programming Lead to Better Outputs in Medical AI?* Short paper submitted to the 11th International Workshop on Health Intelligence (W3PHIAI-27) at AAAI-27, under review.
