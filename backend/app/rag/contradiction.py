from typing import List, Dict

def detect_contradiction(citations: List[Dict]) -> bool:
    if len(citations) < 2:
        return False

    texts = [c["quote"].lower() for c in citations[:4]]
    opposites = [
        ("must", "must not"),
        ("required", "not required"),
        ("allowed", "not allowed"),
        ("can", "cannot"),
    ]

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            for x, y in opposites:
                if (x in a and y in b) or (y in a and x in b):
                    return True
    return False