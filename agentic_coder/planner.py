import os, json

SYSTEM_TEMPLATE = """<SYSTEM>
You follow AGENT.md. Output only the protocol blocks.
</SYSTEM>"""

USER_TEMPLATE = """<GOAL>
{goal}
</GOAL>
<CONTEXT>
repo={repo}
</CONTEXT>
"""

def make_prompt(goal: str, repo: str) -> str:
    # Plan-mode aware system header
    plan_path = os.path.join(repo, ".agentic_plan.json")
    plan_on = False
    try:
        if os.path.exists(plan_path):
            data = json.load(open(plan_path,"r",encoding="utf-8"))
            plan_on = bool(data.get("active"))
    except Exception:
        plan_on = False
    sys_hdr = SYSTEM_TEMPLATE
    if plan_on:
        sys_hdr += "\n<SYSTEM-PLAN-MODE>\nPlan mode is active. Produce a step-by-step plan ONLY.\nDo NOT emit ACTION lines. Wait for end_plan.\n</SYSTEM-PLAN-MODE>\n"
    return sys_hdr + "\n" + USER_TEMPLATE.format(goal=goal, repo=repo)