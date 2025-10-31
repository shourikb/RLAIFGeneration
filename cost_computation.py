import json
import re

# adjust this to your actual file path
file_path = "sample5.jsonl"

total_input = 0
total_output = 0
lines = 0

with open(file_path, "r") as f:
    for line in f:
        data = json.loads(line)
        info = data.get("info", "")

        # find the token counts using regex
        input_match = re.search(r"Total Input Tokens:\s*(\d+)", info)
        output_match = re.search(r"Total Output Tokens:\s*(\d+)", info)

        if input_match:
            total_input += int(input_match.group(1))
        if output_match:
            total_output += int(output_match.group(1))
        
        lines += 1

# example prices (change as needed)
price_per_1M_input = 0.05  # dollars per 1K input tokens
price_per_1M_output = 0.08  # dollars per 1K output tokens

total_input_cost = (total_input / 1000000) * price_per_1M_input
total_output_cost = (total_output / 1000000) * price_per_1M_output

total_cost = (
    total_input_cost + total_output_cost
)

print(f"Total input tokens: {total_input / 1000000}M")
print(f"Total output tokens: {total_output / 1000000}M")
print(f"Average input tokens per line: {total_input / lines if lines else 0}")
print(f"Average output tokens per line: {total_output / lines if lines else 0}")
print(f"Input cost: ${total_input_cost:.4f}")
print(f"Output cost: ${total_output_cost:.4f}")
print(f"Estimated cost: ${total_cost:.4f}")
