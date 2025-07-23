import json
import os
from openai import OpenAI
from google.cloud import vision
from google.oauth2 import service_account

# --- V10 Prompt Generation Functions (Comprehensive & Flexible) ---

def get_stage1_prompt(raw_text, doc_name, page_num):
    """
    MODIFIED: A comprehensive "master" prompt designed to handle a wide variety of bill formats,
    including complex summary tables and detailed single-site sections.
    """
    return f"""
    You are a world-class data extraction expert for a data science team. Your only goal is to extract historical consumption data from utility bills. You must meticulously ignore all costs, charges, and financial data.

    Analyze the text from page {page_num} of document "{doc_name}":
    ---
    {raw_text}
    ---

    Follow this comprehensive rule-based process:

    1.  **Identify Global Information (Applies to the whole document):**
        - Find the main `customer_name` (e.g., "Town of Two Hills", "Flagstaff County", "Hughes Petroleum Ltd."). If not available, skip it.
        - Find the primary `customer_identifier`. This is a persistent ID for the customer, often labeled "Customer ID", or similar (e.g., "C358437-2"). **Crucially, you must IGNORE temporary identifiers like "Invoice Number" or Account number since these are mostly related to the service provider.** 
        - Find a global `billing_period` if one is stated for the entire invoice (e.g., "THIS IS YOUR INVOICE FOR AUGUST 01, 2022 TO AUGUST 31, 2022" or something like that). This will be the default period.
        - any detail including site address, site id/number. make sure that one site id corresponds to one site number


    2.  **Scan for Data Blocks (A page can have multiple blocks):**
        - **Data Block Type A (Multi-Site Summary Table):** Look for a table listing multiple sites. Headers might include "Site ID", "Site Description", "Name", and a consumption column. make sure that one site id corresponds to one site number
        - **Data Block Type B (Single-Site Detail Section):** Look for a section focused on one location, often starting with "SERVICE SUPPLIED TO" or "Site Detail".

    3.  **Data Extraction Logic per Block:**
        - **For a Summary Table (Type A):**
            a. For EACH row, extract the `site_id`, `site_name` or `service_address`.
            b. Find the `consumption_value`. The column header could be "Total kWh", "Total Energy (kWh)", "Consumption", "Usage in m³", "GJ", or similar. Be flexible. If there is such a table, give it top priority, whatever is in the table should appear in the output
            c. Extract the `consumption_unit` from the header or data.
            d. Use the global `billing_period` found in Step 1 for every record in this table.
        - **For a Detail Section (Type B):**
            a. Extract the specific `service_address` for this site.
            b. Find a `site_id` if available within this section.
            c. Find the specific `billing_period` for this section (e.g., "Consumption Period From...To...", or a pair of dates near the meter readings). This specific period OVERRIDES the global one for this record.
            d. Find the `consumption_value` and `consumption_unit` from lines like "Usage in m³", "Amount of electric energy you used", or "Metered Energy".

    4.  **Final Output Structure:**
        - Create a JSON object containing the `customer_name` and `customer_identifier`.
        - Include a `consumption_records` list. Each object in this list is a unique record you found from any data block. A record MUST contain `site_id` (if available), `service_address` (or `site_name`), `billing_period`, `consumption_value`, and `consumption_unit`.
        - If a page contains no discernible consumption data (e.g., a cover page or a water bill with no usage), the `consumption_records` list should be empty.

    Example Output from a Summary Table:
    {{
      "customer_name": "Flagstaff County",
      "customer_identifier": "1001043",
      "consumption_records": [
        {{
          "site_id": "10015480522",
          "service_address": "Commercial PW",
          "billing_period": "2022-08-01 to 2022-08-31",
          "consumption_value": 670.81,
          "consumption_unit": "kWh"
        }}
      ]
    }}

    Example Output from a Detailed Gas Bill Page:
    {{
      "customer_name": "Wynyard, Town Of",
      "customer_identifier": "422 070 0000 3",
      "consumption_records": [
        {{
          "site_id": "4720700797",
          "service_address": "323 Bosworth St, Wynyard, SOA 4T0",
          "billing_period": "2023-02-09 to 2023-03-09",
          "consumption_value": 1052.728,
          "consumption_unit": "m³"
        }}
      ]
    }}
    """

def get_stage2_prompt(list_of_page_outputs, entity_filter):
    """
    MODIFIED: This prompt creates a cleanly labeled nested structure, using the more specific
    site_id for grouping.
    """
    filter_instruction = (
        f'The user has provided a filter: "{entity_filter}". Your final output must ONLY contain data for that entity.'
        if entity_filter
        else "The user has not provided a filter. Report on ALL unique entities found in the data."
    )

    # Flatten the data structure from Stage 1 into a simple list of all records
    all_consumption_records = []
    for page_data in list_of_page_outputs:
        customer_name = page_data.get("customer_name")
        customer_identifier = page_data.get("customer_identifier")
        for record in page_data.get("consumption_records", []):
            record['customer_name'] = customer_name
            record['customer_identifier'] = customer_identifier
            all_consumption_records.append(record)

    return f"""
    You are a data scientist's assistant. You will be given a list of JSON objects, where each object is a clean consumption record.

    Your task is to group these records into a final, clearly-labeled, nested JSON structure.

    1.  **Top Level:** The root of the JSON should be an object with a single key, "customers", which is a list of customer objects.
    2.  **Customer Level:** Each object in the "customers" list should have a `customer_name` and `customer_identifier` key, plus a `sites` list.
    3.  **Site Level:** Each object in the "sites" list should have a `site_id` (if available) and `service_address` (or `site_name`) key, plus a `data` list.
    4.  **Data Level:** The `data` list should contain all the consumption record objects for that site. The objects inside this list should ONLY contain `billing_period`, `consumption_value`, and `consumption_unit`. Do not repeat parent information.
    5.  **Grouping Logic:** Group records first by `customer_name`, then by `customer_identifier`, and finally by a unique combination of `site_id` and `service_address`.
    6.  **Sorting:** The final records in each `data` list should be sorted by billing period, oldest to newest.
    7.  **Apply Filter:** {filter_instruction}

    Here is the list of all consumption records:
    ---
    {json.dumps(all_consumption_records, indent=2)}
    ---

    Present the final, consolidated report as a nested JSON object, following the exact structure described above.
    """

# --- Helper Functions (No changes needed) ---

def extract_text_from_pages(file_path, google_credentials):
    vision_client = vision.ImageAnnotatorClient(credentials=google_credentials)
    _, file_extension = os.path.splitext(file_path)
    mime_type = ""
    if file_extension.lower() == '.pdf':
        mime_type = 'application/pdf'
    elif file_extension.lower() in ('.jpg', '.jpeg', '.png'):
        mime_type = 'image/jpeg'
    else:
        raise ValueError(f"Unsupported file type: {file_extension}.")
    with open(file_path, 'rb') as f:
        content = f.read()
    pages = []
    if mime_type == 'application/pdf':
        input_config = vision.InputConfig(mime_type=mime_type, content=content)
        features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)]
        requests = [vision.AnnotateFileRequest(input_config=input_config, features=features)]
        batch_response = vision_client.batch_annotate_files(requests=requests)
        for i, file_response in enumerate(batch_response.responses):
            for j, image_response in enumerate(file_response.responses):
                if image_response.error.message:
                    print(f"Warning: Error on PDF page {j+1}: {image_response.error.message}")
                    continue
                pages.append({"page_num": j + 1, "text": image_response.full_text_annotation.text})
    else:
        image = vision.Image(content=content)
        response = vision_client.document_text_detection(image=image)
        if response.error.message:
            raise Exception(f'Error from Vision API: {response.error.message}')
        pages.append({"page_num": 1, "text": image_response.full_text_annotation.text})
    return pages

def run_gpt_analysis(prompt, openai_api_key):
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        json_string = response.choices[0].message.content.strip()
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from OpenAI: {e}\\nResponse: {json_string}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during OpenAI call: {e}")
        return None

# --- NEW Refactored Core Logic ---

def run_stage1_for_file(filepath, google_credentials, openai_api_key):
    """
    NEW: Runs Stage 1 processing for a single file.
    """
    doc_name = os.path.basename(filepath)
    all_page_outputs = []
    pages = extract_text_from_pages(filepath, google_credentials)
    for page in pages:
        print(f"    - Analyzing page {page['page_num']} of {doc_name}...")
        prompt = get_stage1_prompt(page['text'], doc_name, page['page_num'])
        page_summary = run_gpt_analysis(prompt, openai_api_key)
        if page_summary and page_summary.get("consumption_records"):
            all_page_outputs.append(page_summary)
    return all_page_outputs

def run_stage2_aggregation(all_page_outputs, openai_api_key, entity_filter=None):
    """
    NEW: Runs Stage 2 aggregation on a collection of Stage 1 results.
    """
    if not all_page_outputs:
        raise Exception("No data was extracted to aggregate.")
    
    print(f"  - Stage 2: Aggregating data from {len(all_page_outputs)} page summaries...")
    final_prompt = get_stage2_prompt(all_page_outputs, entity_filter)
    final_report = run_gpt_analysis(final_prompt, openai_api_key)
    
    if not final_report:
        raise Exception("Stage 2 Aggregation failed to synthesize the final report.")
        
    return final_report
