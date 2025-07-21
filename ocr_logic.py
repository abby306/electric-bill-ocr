import json
import os
from openai import OpenAI
from google.cloud import vision
from google.oauth2 import service_account

# --- V5 Prompt Generation Functions (Balanced Approach) ---

def get_stage1_prompt(raw_text, doc_name, page_num):
    """
    MODIFIED: A balanced prompt that prioritizes finding a single total amount,
    but falls back to summing charge components if a total is not available.
    """
    return f"""
    You are a data extraction expert for utility bills. Your primary goal is to identify each distinct service location on the page and extract its billing information for a single period.

    Analyze the text from page {page_num} of document "{doc_name}":
    ---
    {raw_text}
    ---

    Follow these steps precisely for each service location you identify on the page:

    1.  **Identify Key Information:**
        - `customer_name`: The name of the customer (e.g., "Wynyard, Town Of"). This is mandatory.
        - `account_number`: The primary account or customer number.
        - `service_address`: The full address for the service location. This is mandatory.
        - `billing_period`: The date range for the bill. **Ensure the start date is before the end date.**

    2.  **Determine Total Cost (Use one of two methods):**
        - **Method A (Preferred):** Look for a clear, final "Total Amount Due", "Total Bill", or "Amount Due". If you find it, use this value for the `total_amount_due`.
        - **Method B (Fallback):** If and ONLY IF you cannot find a single total amount, then look for the following charge components: `gas_delivery_service_cost`, `gas_supply_cost`, and `federal_carbon_tax`. If found, list them in a `charge_breakdown` list and calculate their sum for the `total_amount_due`.

    3.  **Final Output Structure:**
        - Create a JSON object containing the `customer_name` and `account_number`.
        - Include a `site_bills` list. Each object in this list represents a unique service location from the page and must contain its `service_address`, `billing_period`, and either the `total_amount_due` (from Method A) or both the `total_amount_due` and `charge_breakdown` (from Method B).

    Example Output (Method A - Simple Bill):
    {{
      "customer_name": "ABC Corp",
      "account_number": "12345",
      "site_bills": [
        {{
          "service_address": "100 Main St, Anytown",
          "billing_period": "2023-01-01 to 2023-01-31",
          "total_amount_due": 550.75
        }}
      ]
    }}

    Example Output (Method B - Complex Bill):
    {{
      "customer_name": "Wynyard, Town Of",
      "account_number": "422 070 0000 3",
      "site_bills": [
        {{
          "service_address": "323 Bosworth St, Wynyard, SOA 4T0",
          "billing_period": "2023-02-09 to 2023-03-09",
          "charge_breakdown": {{
            "gas_delivery_service_cost": 147.20,
            "gas_supply_cost": 146.13,
            "federal_carbon_tax": 108.21
          }},
          "total_amount_due": 401.54
        }}
      ]
    }}
    """

def get_stage2_prompt(list_of_page_outputs, entity_filter):
    """
    Aggregates the structured data from Stage 1.
    This prompt is robust enough to handle both simple and complex records.
    """
    filter_instruction = (
        f'The user has provided a filter: "{entity_filter}". Your final output must ONLY contain data for that entity.'
        if entity_filter
        else "The user has not provided a filter. Report on ALL unique entities found in the data."
    )

    # Flatten the data structure from Stage 1 to create a simple list of all bills
    all_site_bills = []
    for page_data in list_of_page_outputs:
        customer_name = page_data.get("customer_name")
        account_number = page_data.get("account_number")
        for bill in page_data.get("site_bills", []):
            bill['customer_name'] = customer_name
            bill['account_number'] = account_number
            all_site_bills.append(bill)

    return f"""
    You are a senior financial analyst. You will be given a list of JSON objects, where each object is a complete billing record for a specific site and a single billing period. Some records may include a charge breakdown, others may not.

    Your task is to group these records into a final nested report.

    1.  **Primary Grouping:** Group all records by `customer_name`.
    2.  **Secondary Grouping:** Within each customer, group by `account_number`.
    3.  **Tertiary Grouping:** Within each account, group by `service_address`.
    4.  **Final Structure:** Each service address should have a `billing_records` list containing all its billing period objects. Preserve the `charge_breakdown` field only if it exists in the source record.
    5.  **Apply Filter:** {filter_instruction}

    Here is the list of all billing records:
    ---
    {json.dumps(all_site_bills, indent=2)}
    ---

    Present the final, consolidated report as a nested JSON object.
    """

# --- Helper and Orchestration Functions ---

def extract_text_from_pages(file_path, google_credentials):
    """
    Extracts text from each page of a file (PDF or image).
    Returns a list of dictionaries, e.g., [{"page_num": 1, "text": "..."}].
    """
    vision_client = vision.ImageAnnotatorClient(credentials=google_credentials)
    _, file_extension = os.path.splitext(file_path)
    mime_type = ""

    if file_extension.lower() == '.pdf':
        mime_type = 'application/pdf'
    elif file_extension.lower() in ('.jpg', '.jpeg', '.png'):
        mime_type = 'image/jpeg' # Treat PNG as JPEG for Vision API
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
        pages.append({"page_num": 1, "text": response.full_text_annotation.text})
        
    return pages

def run_gpt_analysis(prompt, openai_api_key):
    """Helper function to run a single GPT analysis and parse JSON."""
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

def process_documents_v3(filepaths, google_credentials, openai_api_key, entity_filter=None):
    """
    REVISED: The main V3 processing pipeline updated for the new prompt structure.
    """
    print("Starting V5 processing pipeline...")
    all_page_outputs = []

    # --- STAGE 1: Page-by-Page Analysis ---
    for fp in filepaths:
        doc_name = os.path.basename(fp)
        print(f"  - Stage 1: Processing document '{doc_name}'")
        try:
            pages = extract_text_from_pages(fp, google_credentials)
            for page in pages:
                print(f"    - Analyzing page {page['page_num']}...")
                prompt = get_stage1_prompt(page['text'], doc_name, page['page_num'])
                page_summary = run_gpt_analysis(prompt, openai_api_key)
                
                if page_summary and page_summary.get("site_bills"):
                    all_page_outputs.append(page_summary)

        except Exception as e:
            print(f"Error processing document {doc_name}: {e}")
            continue

    if not all_page_outputs:
        raise Exception("Stage 1 Analysis failed. No billing data could be extracted from any page.")

    # --- STAGE 2: Final Aggregation ---
    print(f"  - Stage 2: Aggregating data from {len(all_page_outputs)} pages...")
    final_prompt = get_stage2_prompt(all_page_outputs, entity_filter)
    final_report = run_gpt_analysis(final_prompt, openai_api_key)
    
    if not final_report:
        raise Exception("Stage 2 Aggregation failed. Could not synthesize the final report.")

    return final_report
