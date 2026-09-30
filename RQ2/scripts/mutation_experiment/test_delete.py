from delete_mutation import delete_mutation

tests = [
    "The clever scientist quickly presented the surprising discovery.",
    "Explain the difference between DNA and RNA.",
    "Give me a vegetarian pasta recipe.",
    "What are the top attractions in Kyoto?",
]
for t in tests:
    print("ORIG:", t)
    for _ in range(5):
        print("   ->", delete_mutation(t))
    print()
