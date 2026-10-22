"""Extract complete JSON objects from mixed multi-rank console output."""
import json


def log_dicts(text):
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        offset = 0
        while True:
            start = line.find('{', offset)
            if start < 0:
                break
            try:
                value, end = decoder.raw_decode(line, start)
            except json.JSONDecodeError:
                offset = start + 1
                continue
            offset = end
            if isinstance(value, dict):
                yield value
