import json
import re

file_path = 'data/raw/prompts_train.csv'

with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Define a pattern to match the required parts of the text
pattern = re.compile(r"Subject:: (.*?)\nBody:: (.*?)\n\nWhat should this email be categorized as\?\n\[/INST\] (.*?) <s>")

# Parsing the entire file content as a single string since entries span multiple lines
file_content = "".join(lines)  # Join all lines into a single string for multiline regex search

# Use findall instead of search to capture all matches in the file content
matches = pattern.findall(file_content)

# Transform matches into a list of dictionaries for easier handling
extracted_data = [
    {"Subject": match[0], "Body": match[1], "Category": match[2]}
    for match in matches
]
limited_extracted_data = extracted_data[:-100]

# Display the first few extracted items to verify the extraction
# Display the first few extracted items to verify the extraction
json_file_path = 'data/cleaned/extracted_email_data_1300.json'

# Write the extracted data to a JSON file
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(limited_extracted_data, json_file, indent=4)
