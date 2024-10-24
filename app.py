from flask import Flask, request, jsonify, render_template
import docx2txt
import PyPDF2
import os
import json
import groq
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Initialize Groq client
client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_text_from_cv(file):
    """
    Extracts text from a CV/resume file (PDF or DOCX) stored in memory.

    Args:
        file: FileStorage object from request.files

    Returns:
        The extracted text as a string, or None if an error occurs or the file type is not supported.
    """
    try:
        filename = secure_filename(file.filename)
        if filename.lower().endswith(".pdf"):
            # Read PDF directly from memory
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif filename.lower().endswith((".docx", ".doc")):
            # For DOCX, we need to save temporarily in memory
            text = docx2txt.process(io.BytesIO(file.read()))
            return text
        else:
            return "Unsupported file type. Please provide a PDF or DOCX file."
    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if not file:
                return "No file provided", 400

            # Extract text directly from the uploaded file
            resume_text = extract_text_from_cv(file)
            if resume_text.startswith("An error occurred") or resume_text.startswith("Unsupported file type"):
                return resume_text, 400

            # Create the completion request
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": (f"You are a resume parsing expert. I will provide you with the content of a resume: {resume_text}. \n"
                                    "Please extract the key information and organize it into a structured JSON format. Ensure to include the following fields:\n\n"
                                    "Name: Full name of the individual.\n"
                                    "Professional Summary: A brief overview of the individual's career and skills. (if available or leave as empty)\n"
                                    "Contact Information: Email, phone number, and address.\n"
                                    "Work Experience: A list of previous positions, including:\n"
                                    "Job Title\nCompany Name\nStart Date and End Date (if available or leave as empty)\n"
                                    "Location (if available or leave as empty)\nJob Responsibilities (if available or leave as empty)\n"
                                    "Education: A list of academic qualifications, including:\nDegree\nInstitution Name\nStart Date and End Date (if available or leave as empty)\n"
                                    "Location (if available or leave as empty)\nSkills: Key technical and non-technical skills. (if available or leave as empty)\n"
                                    "Certifications: Any certifications or courses completed. (if available or leave as empty)\n"
                                    "Projects: Notable projects, including:\nProject Title\nDescription (if available or leave as empty)\n"
                                    "Technologies Used (if available or leave as empty)\nLanguages: Languages spoken and proficiency level. (if available or leave as empty)\n"
                                    "Awards: Any notable awards or recognitions. (if available or leave as empty)\n"
                                    "Please ensure the JSON output is well-structured and all relevant details are captured.")
                    }
                ],
                temperature=1,
                max_tokens=2048,
                top_p=1,
                stream=False,
                response_format={"type": "json_object"},
                stop=None,
            )

            response = completion.choices[0].message.content.strip()
            try:
                resume_json = json.loads(response)
            except json.JSONDecodeError as e:
                return f"Error decoding JSON: {e}", 500

            return render_template('result.html', resume_json=resume_json)
        except Exception as e:
            return f"An error occurred: {e}", 500
    return render_template('upload.html')

@app.route('/result', methods=['GET'])
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
