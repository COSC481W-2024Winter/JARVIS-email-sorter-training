

# JARVIS Email Sorter Training

## Overview

The JARVIS Email Sorter Training Module is a crucial component of the [JARVIS](https://github.com/COSC481W-2024Winter/JARVIS) ecosystem, specifically designed for the development, training, and refinement of an email categorization algorithm. This module utilizes machine learning techniques to automate the process of sorting emails into predefined categories, thus enhancing user productivity and email management efficiency. The core development is in Python, leveraging various libraries and frameworks suited for machine learning and data processing tasks. 

The associated JARVIS issue is [PBI 1: Sort Emails](https://github.com/COSC481W-2024Winter/JARVIS/issues/11)

## Dataset

For the training and testing phases, this module will use the [Enron Labeled Emails with Subjects dataset](https://huggingface.co/datasets/neelblabla/enron_labeled_emails_with_subjects-llama2-7b_finetuning?row=0) available on Hugging Face. This dataset includes a comprehensive collection of real-world email communications, categorized for effective training of email sorting algorithms.

## Categories

The email sorter aims to categorize emails into the following predefined categories:

1. **Company Business/Strategy:** Emails related to the company's operations, strategic planning, and business decisions.
2. **Purely Personal:** Personal emails that do not relate to professional activities or contexts.
3. **Personal but in a Professional Context:** Personal communications that occur within a professional setting or relationship.
4. **Logistic Arrangements:** Emails concerning logistical planning, such as meetings, travel arrangements, or event planning.
5. **Employment Arrangements:** Communications related to employment, including hiring, termination, and job role discussions.
6. **Document Editing/Checking/Collaboration:** Emails focused on the collaborative editing, reviewing, or creation of documents.
7. **Empty Message (due to missing attachment):** Emails that reference attachments which are not present or accessible, leading to an empty or incomplete message.

## Development Objectives

- **Machine Learning Model Development:** The primary focus is to create an effective machine learning model capable of accurately sorting emails into the aforementioned categories. This involves preprocessing the dataset, selecting appropriate features, training models, and validating their performance.
- **Evaluation and Optimization:** Continuously evaluate the model's performance and optimize its accuracy and efficiency, aiming to meet and exceed baseline accuracy metrics.

## Timeline

- **Initial Setup and Data Preprocessing:** Begin with setting up the development environment and preprocessing the dataset for training.
- **Model Training and Evaluation:** Progress to training the initial models, followed by rigorous evaluation and optimization phases.
- **Deployment Strategy Assessment:** Depending on the model's computational requirements, assess whether to deploy the sorter on a remote server or locally on devices.
