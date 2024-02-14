# First, we'll remove the 'Category' field from each entry in the extracted data
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
uncategorized_emails = [{key: value for key, value in item.items() if key != "Category"} for item in extracted_data]
limited_uncategorized_emails = uncategorized_emails[:100]
# Can modify [:100] to get quantity you want.
# Define the path for the new JSON file without categories
uncategorized_json_file_path = 'data/uncategorized_emails.json'

# Write the uncategorized data to a new JSON file
with open(uncategorized_json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(uncategorized_emails, json_file, indent=4)
