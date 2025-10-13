import json
import re
import time
from typing import List, Tuple, TypedDict

from openai import OpenAI

from agentlightning.litagent import rollout


class Room(TypedDict):
    id: str
    capacity: int
    equipment: List[str]
    accessible: bool
    distance_m: int
    booked: List[Tuple[str, str, int]]


class RoomStatus(Room):
    free: bool


class AvailableRooms(TypedDict):
    rooms: List[RoomStatus]


class RoomRequirement(TypedDict):
    date: str
    time: str
    duration_min: int
    attendees: int
    needs: List[str]
    accessible_required: bool


class RoomSelectionTask(TypedDict):
    task_input: RoomRequirement
    expected_choice: str


@rollout
def room_selector(task_input: RoomSelectionTask):
    client = OpenAI()

    system = (
        "You are a scheduling assistant.\n"
        "Hard constraints: free for slot, capacity >= attendees, includes all required equipment, "
        "accessible==True if requested.\n"
        "Tie-break scoring (lower is better):\n"
        "  1) capacity_slack = capacity - attendees (minimize)\n"
        "  2) extra_equipment = provided_equipment_count - required_equipment_count (minimize)\n"
        "  3) distance_m (minimize)\n"
        "  4) fewer total booked blocks that day (minimize)\n"
        "Return strictly:\n"
        "final_choice: <ROOM_ID>\nreason: <one line stating the decisive criteria>\n"
    )

    task_input = {
        "date": "2025-10-13",
        "time": "10:30",
        "duration_min": 45,
        "attendees": 6,
        "needs": ["whiteboard", "projector"],
        "accessible_required": True,
    }

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Find a room on {task_input['date']} at {task_input['time']} for {task_input['duration_min']} minutes, "
                f"{task_input['attendees']} attendees. Needs: {', '.join(task_input['needs']) or 'none'}. "
                f"Accessible required: {task_input['accessible_required']}"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-5",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    messages.append(resp.choices[0].message)

    for tc in resp.choices[0].message.tool_calls or []:
        if tc.function.name == "get_rooms_and_availability":
            args = json.loads(tc.function.arguments)
            tool_output = get_rooms_and_availability(args["date"], args["time"], args["duration_min"])
            # Attach a lightweight view of daily booking counts for tie-break #4
            for r in tool_output["rooms"]:
                r["booked_blocks_today"] = sum(
                    1 for (d, _, _) in next(x for x in ROOMS if x["id"] == r["id"])["booked"] if d == args["date"]
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(tool_output),
                }
            )

    final = client.chat.completions.create(
        model="gpt-5",
        messages=messages,
    )
    answer_text = final.choices[0].message.content
    print("=== Model Answer ===\n", answer_text)

    # ---- 4) Judge exact choice against expected
    expected_choice = "Atlas"  # Candidates: Atlas (6 cap, proj+wb, acc, free) vs Lyra (10) vs Nova (12). Atlas wins by smallest slack and shortest distance.

    def extract_choice(text):
        m = re.search(r"final_choice:\s*([A-Za-z0-9_-]+)", text)
        return m.group(1) if m else ""

    model_choice = extract_choice(answer_text)

    judge_prompt = f"""Task output:
    {answer_text}

    Task expected answer:
    final_choice: {expected_choice}

    Score the match on a 0-1 scale. Return JSON: {{"score": <0..1>, "reason": "<brief>"}}
    """
    judge = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Be a strict grader of exact room choice."},
            {"role": "user", "content": judge_prompt},
        ],
    )
    print("=== Judge ===\n", judge.choices[0].message.content)


# ---- 5) Batch evaluation over dataset.jsonl
# def run_one(sample):
#     t = sample["task_input"]
#     msgs = [
#         {"role":"system","content":system},
#         {"role":"user","content":(
#             f"Find a room on {t['date']} at {t['time']} for {t['duration_min']} minutes, "
#             f"{t['attendees']} attendees. Needs: {', '.join(t['needs']) if t['needs'] else 'none'}. "
#             f\"Accessible required: {t['accessible_required']}\""
#         )}
#     ]
#     r1 = client.chat.completions.create(model="gpt-5", messages=msgs, tools=tools, tool_choice="auto")
#     msgs.append(r1.choices[0].message)
#     for tc in r1.choices[0].message.tool_calls or []:
#         args = json.loads(tc.function.arguments)
#         out = get_rooms_and_availability(args["date"], args["time"], args["duration_min"])
#         for r in out["rooms"]:
#             r["booked_blocks_today"] = sum(1 for (d,_,_) in next(x for x in ROOMS if x["id"]==r["id"])["booked"] if d==args["date"])
#         msgs.append({"role":"tool","tool_call_id":tc.id,"name":tc.function.name,"content":json.dumps(out)})
#     r2 = client.chat.completions.create(model="gpt-5", messages=msgs)
#     out_text = r2.choices[0].message.content
#     choice = extract_choice(out_text)
#     exp = sample["expected_choice"]
#     jp = f"Task output:\n{out_text}\n\nTask expected answer:\nfinal_choice: {exp}\n\nReturn JSON: {{\"score\": <0..1>, \"reason\": \"<brief>\"}}"
#     j = client.chat.completions.create(model="gpt-5",
#                                        messages=[{"role":"system","content":"Be a strict grader of exact room choice."},
#                                                  {"role":"user","content":jp}])
#     return {"id": sample["id"], "choice": choice, "expected": exp, "judge": j.choices[0].message.content}


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_rooms_and_availability",
            "description": "Return meeting rooms with capacity, equipment, accessibility, distance, and booked time slots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM 24h local"},
                    "duration_min": {"type": "integer", "description": "Meeting duration minutes"},
                },
                "required": ["date", "time", "duration_min"],
            },
        },
    },
]

# Local tool database (there might be multiple plausible fits)
ROOMS: List[Room] = [
    {
        "id": "Orion",
        "capacity": 4,
        "equipment": ["tv", "whiteboard"],
        "accessible": True,
        "distance_m": 12,
        "booked": [("2025-10-13", "10:00", 60), ("2025-10-13", "15:00", 30)],
    },
    {
        "id": "Lyra",
        "capacity": 10,
        "equipment": ["projector", "whiteboard", "confphone"],
        "accessible": True,
        "distance_m": 30,
        "booked": [("2025-10-13", "09:30", 30), ("2025-10-13", "11:00", 60)],
    },
    {
        "id": "Vega",
        "capacity": 6,
        "equipment": ["tv"],
        "accessible": False,
        "distance_m": 22,
        "booked": [("2025-10-13", "14:00", 60)],
    },
    {
        "id": "Nova",
        "capacity": 12,
        "equipment": ["ledwall", "whiteboard", "confphone"],
        "accessible": True,
        "distance_m": 45,
        "booked": [],
    },
    {
        "id": "Quark",
        "capacity": 8,
        "equipment": ["projector", "whiteboard"],
        "accessible": False,
        "distance_m": 18,
        "booked": [("2025-10-13", "10:30", 30)],
    },
    # Two extra to create harder ties
    {
        "id": "Atlas",
        "capacity": 6,
        "equipment": ["projector", "whiteboard"],
        "accessible": True,
        "distance_m": 10,
        "booked": [("2025-10-13", "09:00", 30), ("2025-10-13", "13:30", 30)],
    },
    {
        "id": "Pulse",
        "capacity": 8,
        "equipment": ["tv", "whiteboard", "confphone"],
        "accessible": True,
        "distance_m": 8,
        "booked": [("2025-10-13", "16:30", 30)],
    },
]


def overlaps(start: str, dur: int, other_start: str, other_dur: int) -> bool:
    def tmin(t: str):
        return int(t[:2]) * 60 + int(t[3:])

    a0, a1 = tmin(start), tmin(start) + dur
    b0, b1 = tmin(other_start), tmin(other_start) + other_dur
    return max(a0, b0) < min(a1, b1)


def get_rooms_and_availability(date: str, time_str: str, duration_min: int) -> AvailableRooms:
    avail: List[RoomStatus] = []
    for r in ROOMS:
        free = all(
            not (b_date == date and overlaps(time_str, duration_min, b_time, b_dur))
            for (b_date, b_time, b_dur) in r["booked"]
        )
        item: RoomStatus = {
            **r,
            "free": free,
        }
        avail.append(item)
    return {"rooms": avail}
