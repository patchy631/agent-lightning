# Copyright (c) Microsoft. All rights reserved.

"""Helper module for APO example with room selection."""

from typing import List, Tuple, TypedDict


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
