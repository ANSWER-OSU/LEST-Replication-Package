import json
from collections import Counter

INPUT_PATH = (
    "data/evaluation/"
    "self_harm_wildguard_result.json"
)

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


harmful_items = [

    item

    for item in data

    if item.get(
        "wildguard_prompt_harmfulness"
    ) == "harmful"
]


counter = Counter()

for item in harmful_items:

    chain = item[
        "mutation_chain"
    ]

    if len(chain) == 1:

        counter[
            chain[0]
        ] += 1

    else:

        counter[
            "mixed"
        ] += 1


print(
    "\ntotal harmful prompts:",
    len(harmful_items)
)

print(
    "\nmutation distribution:"
)

for name, count in sorted(
    counter.items()
):

    print(
        f"{name}: {count}"
    )
