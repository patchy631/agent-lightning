import json
import re
import time
import traceback
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


# @rollout
def room_selector(task: RoomSelectionTask):
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
        "Return No Room if no room is found that satisfies the constraints.\n"
        "Return strictly:\n"
        "final_choice: <ROOM_ID>\nreason: <one line stating the decisive criteria>\n"
    )

    print("=== Task ===")
    print(task)

    task_input = task["task_input"]

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
        model="gpt-5-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        reasoning_effort="low",
    )
    messages.append(resp.choices[0].message)

    for tc in resp.choices[0].message.tool_calls or []:
        if tc.function.name == "get_rooms_and_availability":
            args = json.loads(tc.function.arguments)
            try:
                tool_output = get_rooms_and_availability(args["date"], args["time"], args["duration_min"])
            except Exception as e:
                tool_output = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(tool_output),
                }
            )

    final = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        reasoning_effort="low",
    )
    answer_text = final.choices[0].message.content
    print("=== Model Answer ===\n", answer_text)

    # Judge exact choice against expected
    expected_choice = task["expected_choice"]

    judge_prompt = f"""Task output:
    {answer_text}

    Task expected answer:
    final_choice: {expected_choice}

    Score the match on a 0-1 scale. Return JSON: {{"score": <0..1>, "reason": "<brief>"}}
    """
    judge = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Be a strict grader of exact room choice."},
            {"role": "user", "content": judge_prompt},
        ],
    )
    print("=== Judge ===\n", judge.choices[0].message.content)


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


if __name__ == "__main__":
    for line in open("room_tasks_merged.jsonl"):
        task = json.loads(line)

        room_selector(task)

        # from agentlightning.tracer import AgentOpsTracer
        # from agentlightning.types import Span

        # tracer = AgentOpsTracer()
        # tracer.init()
        # tracer.init_worker(0)
        # with tracer.trace_context():
        #     room_selector(task)
        # spans = []
        # for span in tracer.get_last_trace():
        #     spans.append(Span.from_opentelemetry(span, "dummy", "dummy", 0))
        #     print(" Span name: ", span.name, "Span attributes:", span.attributes)
        # from agentlightning.adapter.messages import TraceMessagesAdapter

        # adapter = TraceMessagesAdapter()
        # messages = adapter.adapt(spans)
        # print(messages)
        # break
