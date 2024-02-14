import json

# Your dataset
dataset = [
    {
        "Subject": "RE: NERC Statements on Impact of Security Threats on RTOs",
        "Body": "I agree with Joe. The IOUs will point to NERC as an objective third party on these issues.",
        "Category": "Company Business/Strategy"
    },
    {
        "Subject": "RE: NERC Meeting Today",
        "Body": "There was an all day meeting of the NERC/reliability legislation group today. I will provide a more detailed report but the group completed the process of reviewing the changes that some had suggested to shorten and streamline the NERC electric reliability organization legislation. Sarah and I asked a series of questions and made comments on our key issues and concerns. I want to give you a more complete report once I have gone back over the now final draft version. The timing being imposed by NERC is that they will circulate a clean version of the proposal tomorrow or Monday. They have asked for comments by next Thursday August 16th with an indication of whether each company/organization does or does not sign on to support it. They will then transmit the proposal and the endorsement letter to Congress and the Administration so they have it as Hill and Energy Dept. staff work on electricity drafting issues this month. I pointed out that EPSA is not due to meet internally with its members to discuss these issues until after the NERC deadline. That is not deterring NERC from moving forward with the above time frame.",
        "Category": "Company Business/Strategy"
    }
]

# Instruction to append before each entry
instruction = "You are reviewing an email using its subject and body. Based on the text in its subject and body, please categorize this email into one of the following categories:\n1. 'Company Business/Strategy'\n2. 'Purely Personal'\n3. 'Personal but in a professional context'\n4. 'Logistic Arrangements'\n5. 'Employment arrangements'\n6. 'Document editing/checking/collaboration'\n7. 'Empty message (due to missing attachment)'\n8. 'Empty message'\nPlease provide only one category (e.g., 'Purely Personal')."

# Preprocess dataset
for item in dataset:
    item["Subject"] = instruction + "\n\nSubject: " + item["Subject"]
    item["Body"] = "Body: " + item["Body"]

# Example of how the modified dataset looks
print(json.dumps(dataset[0], indent=4))
