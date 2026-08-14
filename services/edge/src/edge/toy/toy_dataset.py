toy_dataset = [
"""def slugify(text):
    # 🔧 forge: begin
    result = text.lower().replace(" ", "-")
    return result
    # 🔧 forge: end — verified
""",

"""def add_numbers(a, b):
    # 🔧 forge: begin
    total = a + b
    return total
    # 🔧 forge: end — verified
""",

"""def read_lines(path):
    # 🔧 forge: begin
    with open(path) as f:
        lines = f.readlines()
    return lines
    # 🔧 forge: end — verified
""",

"""def is_even(n):
    # 🔧 forge: begin
    return n % 2 == 0
    # 🔧 forge: end — verified
""",

"""def count_words(text):
    # 🔧 forge: begin
    words = text.split()
    return len(words)
    # 🔧 forge: end — verified
""",

"""def clamp(value, low, high):
    # 🔧 forge: begin
    if value < low:
        return low
    if value > high:
        return high
    return value
    # 🔧 forge: end — verified
""",

"""def reverse_string(s):
    # 🔧 forge: begin
    return s[::-1]
    # 🔧 forge: end — verified
""",

"""def sum_list(items):
    # 🔧 forge: begin
    total = 0
    for item in items:
        total += item
    return total
    # 🔧 forge: end — verified
""",
]