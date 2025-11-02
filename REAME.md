Data Janitor — Automated Dataset Cleaning Service

This project provides a simple automated data-cleaning service. Trying to automate one step at a time in the data engineering lifecycle.
Users can upload a dataset, preview detected issues, preview cleaning actions, and then download a cleaned dataset and a cleaning report.

The application performs the following operations:

Load and inspect structured data files

Detect missing values, inconsistent types, duplicates, and numeric outliers

Suggest data type conversions (string → numeric, string → datetime)

Fill missing values using median, mean, mode, or constant values

IQR clipping

Drop duplicate records

Output a cleaned dataset

Generate a text-based cleaning report (downloadable as PDF)

This is version 1.0 of this implementation.

Supported File Types

CSV

JSON / JSONL / NDJSON

Excel (.xlsx, .xls)

Parquet

XML

HTML table input

REQUIREMENTS:

Install all the dependencies in the requirements.txt file