import os
import glob
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
from google.oauth2.service_account import Credentials
import gspread
import io
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def ocr_space_file(filename, overlay=False, api_key='helloworld', language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    return r.content.decode()

def compress_image_lossless(image_path, max_size_mb=1):
    """
    Compress image to under specified MB with minimal quality loss
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    with Image.open(image_path) as img:
        # Convert to RGB if necessary (for JPEG compatibility)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # If image is already small enough, return original
        img.save(image_path, format='JPEG', quality=95, optimize=True)
        if os.path.getsize(image_path) <= max_size_bytes:
            return image_path
        
        # Binary search for optimal quality
        min_quality, max_quality = 20, 95
        
        while min_quality <= max_quality:
            mid_quality = (min_quality + max_quality) // 2
            
            # Save with current quality to a buffer
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=mid_quality, optimize=True)
            
            if buffer.tell() <= max_size_bytes:
                # Size is acceptable, try higher quality
                min_quality = mid_quality + 1
                best_buffer = buffer
            else:
                # Size too large, try lower quality
                max_quality = mid_quality - 1
        
        # Save the best result
        with open(image_path, 'wb') as f:
            f.write(best_buffer.getvalue())
        
        return image_path

# Configure Google Sheets
def get_google_sheets_client():
    """Initialize Google Sheets client"""
    try:
        # Try to use service account credentials
        service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        if service_account_file and os.path.exists(service_account_file):
            creds = Credentials.from_service_account_file(
                service_account_file,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            return gspread.authorize(creds)
        else:
            # Fallback to service account JSON from environment variable
            service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
            if service_account_json:
                import json
                creds_info = json.loads(service_account_json)
                creds = Credentials.from_service_account_info(
                    creds_info,
                    scopes=['https://www.googleapis.com/auth/spreadsheets']
                )
                return gspread.authorize(creds)
            else:
                raise Exception("No Google service account credentials found")
    except Exception as e:
        print(f"Error initializing Google Sheets client: {str(e)}")
        return None

def insert_to_google_sheets(data):
    """Insert extracted data to Google Sheets"""
    try:
        client = get_google_sheets_client()
        if not client:
            return {"error": "Failed to connect to Google Sheets"}
        
        # Get spreadsheet (you can configure this in .env)
        spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")
        worksheet_name = os.getenv("GOOGLE_WORKSHEET_NAME")

        # print(spreadsheet_id, worksheet_name)  # Debugging output

        # print(f"Using spreadsheet ID: {spreadsheet_id}, Worksheet: {worksheet_name}", flash=True)
        if not spreadsheet_id:
            return {"error": "Google Spreadsheet ID not configured"}
        
        # Open the spreadsheet
        spreadsheet = client.open_by_key(spreadsheet_id)
        
        # Try to get the worksheet, create if it doesn't exist
        try:
            # print(spreadsheet_id, worksheet_name)  # Debugging output
            worksheet = spreadsheet.worksheet(worksheet_name)
            # print(f"Worksheet title: {worksheet.title}")  # Debugging output  
        except gspread.WorksheetNotFound:
            # Create new worksheet with headers
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)
            headers = [
                "Name", 
                "Gender",
                "Date of Birth", 
                "NID Number", 
                "Phone", 
                "Education",
                "Fathers Name", 
                "Mothers Name", 
                "Present Address", 
            ]
            worksheet.append_row(headers)
        
        # Prepare row data
        extracted_data = data.get('extracted_data', {})
        
        row_data = [
            extracted_data.get('name', ''),
            extracted_data.get('gender', ''),
            extracted_data.get('date_of_birth', ''),
            extracted_data.get('nid_number', '').replace(" ", "").replace("-", ""),
            parse_bd_mobile_number(extracted_data.get('phone', '')),
            extracted_data.get('education', ''),
            extracted_data.get('fathers_name', ''),
            extracted_data.get('mothers_name', ''),
            extracted_data.get('present_address', '')
        ]
        
       
        # Insert the row
        worksheet.append_row(row_data)
        
        return {
            "success": True,
            "message": "Data successfully inserted to Google Sheets",
            "spreadsheet_url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        }
        
    except Exception as e:
        return {"error": f"Failed to insert to Google Sheets: {str(e)}"}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(image_path):
    """Extract text from image using OCR"""
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Error extracting text from {image_path}: {str(e)}")

        return ""


def parse_bd_mobile_number(number: str) -> str:
    """
    Parse a Bangladeshi mobile number to extract the last 10 digits.
    Example:
        +88 01628439632 -> 1628439632
        01628439632 -> 1628439632
        1628439632 -> 1628439632
        invalid_number -> invalid_number
    """
    # Remove spaces and dashes
    number = number.replace(" ", "").replace("-", "")
    
    # Patterns to match valid Bangladeshi numbers
    patterns = [
        r'^\+8801[0-9]{9}$',  # +8801XXXXXXXXX
        r'^8801[0-9]{9}$',    # 8801XXXXXXXXX
        r'^01[0-9]{9}$',      # 01XXXXXXXXX
        r'^1[0-9]{9}$'        # 1XXXXXXXXX
    ]
    
    for pattern in patterns:
        if re.match(pattern, number):
            return number[-10:]  # Return last 10 digits

    # If no pattern matches, return the original
    return number

def parse_resume_fields_with_gemini(raw_text):
    """Parse resume fields using Gemini API"""
    prompt = f"""
    Extract the following fields from this resume/CV text and return them in a structured JSON format.
    If any field is not found, use empty string as the value.
    
    Required fields:
    - Name (Title Case)
    - Gender (Male / Female)
    - Date of Birth (DD/MM/YYYY format)
    - NID Number (National ID)
    - Phone / Contact Number
    - Educational Qualifications. Return just one word with "Honors" or "HSC" or "SSC". prioritize the latest education (Honors > HSC > SSC). add (Ongoing) inside parenthesis if the education is ongoing.
    - Fathers Name
    - Mothers Name
    - Present Address (present address or  mailing address)
    
    Resume Text:
    {raw_text}
    
    Please return the response in this exact JSON format:
    {{
        "name": "extracted name",
        "gender": "extracted gender",
        "date_of_birth": "extracted DOB",
        "nid_number": "extracted NID",
        "phone": "extracted phone",
        "education": "extracted education",
        "fathers_name": "extracted father's name",
        "mothers_name": "extracted mother's name",
        "present_address": "extracted Present Address",
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        # Try to parse as JSON
        try:
            # Clean the response text to extract JSON
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            parsed_data = json.loads(response_text)
            return parsed_data
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw text
            return {"raw_response": response.text}
    except Exception as e:
        return {"error": f"Error processing with Gemini: {str(e)}"}

import re




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    # Create a unique session folder
    session_id = str(uuid.uuid4())
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    combined_text = ""
    uploaded_files = []
    
    # print("Starting file processing...")  # Debugging output

    # Process each uploaded file
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid filename conflicts
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            
            filepath = os.path.join(session_folder, filename)
            file.save(filepath)
            
            # compressed_filepath = compress_image_lossless(filepath)


            # Extract text from the image
            # text = extract_text(filepath)
            ocr_result = ocr_space_file(filepath, overlay=False, api_key='K88849901888957', language='eng')
            ocr_data = json.loads(ocr_result)
            # print(ocr_data)
            parsed_text = ocr_data['ParsedResults'][0]['ParsedText']
            # print(parsed_text)
            text = parsed_text.strip()  
            combined_text += f"\n--- Content from {filename} ---\n{text}\n"
            uploaded_files.append(filename)
    
    if not combined_text.strip():
        return jsonify({'error': 'No text could be extracted from the images'}), 400
    
    # print("Text extraction complete. Parsing with Gemini...")  # Debugging output
    # Parse the combined text using Gemini
    parsed_fields = parse_resume_fields_with_gemini(combined_text)
    
    # Add metadata
    result = {
        'session_id': session_id,
        'uploaded_files': uploaded_files,
        'processed_at': datetime.now().isoformat(),
        'extracted_data': parsed_fields,
        'raw_text': combined_text
    }
    
    # Save result to file for reference
    result_file = os.path.join(session_folder, 'result.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    

    # print(result)
    # Insert data to Google Sheets
    sheets_result = insert_to_google_sheets(result)
    result['sheets_result'] = sheets_result

    # print("Google Sheets insertion result:", sheets_result)  # Debugging output
    
    return jsonify(result)

@app.route('/sheets-status/<session_id>')
def get_sheets_status(session_id):
    """Get the Google Sheets insertion status for a session"""
    try:
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        result_file = os.path.join(session_folder, 'result.json')
        
        if not os.path.exists(result_file):
            return jsonify({'error': 'Session not found'}), 404
        
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify({
            'sheets_result': data.get('sheets_result', {}),
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting sheets status: {str(e)}'}), 500

if __name__ == '__main__':
    # Make sure pytesseract is configured (adjust path if needed)
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
    app.run(debug=True, host='0.0.0.0', port=3000)