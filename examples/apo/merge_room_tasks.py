import json

data: list[dict] = []

for line in open("room_tasks.jsonl").readlines():
    line_data = json.loads(line)
    line_data.pop("id")
    if line_data not in data:
        data.append(line_data)

for line in open("room_tasks_2.jsonl").readlines():
    line_data = json.loads(line)
    line_data.pop("id")
    if line_data not in data:
        data.append(line_data)

with open("room_tasks_merged.jsonl", "w") as f:
    for line in data:
        f.write(json.dumps(line) + "\n")
