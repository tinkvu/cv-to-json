# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
import docx2txt
import PyPDF2
import os
import json
from groq import Groq

# Add Flask to your dependencies
# pip install Flask docx2txt PyPDF2 groq

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def extract_text_from_cv(file_path):
    """
    Extracts text from a CV/resume file (PDF or DOCX).

    Args:
        file_path: The path to the CV/resume file.

    Returns:
        The extracted text as a string, or None if an error occurs or the file type is not supported.
    """
    try:
        if not os.path.exists(file_path):
            return "File not found."

        if file_path.lower().endswith(".pdf"):
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page].extract_text()
                return text
        elif file_path.lower().endswith((".docx", ".doc")):
            return docx2txt.process(file_path)
        else:
            return "Unsupported file type. Please provide a PDF or DOCX file."
    except Exception as e:
        return f"An error occurred: {e}"

# Initialize Groq client
client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])

@app.route('/')
def landing_page():
    return render_template('landing.html')  # Render landing page

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file:
                file_path = f"./uploads/{file.filename}"  # Save the file to a directory
                file.save(file_path)
                resumeText = extract_text_from_cv(file_path)

                # Create the completion request
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": (f"You are a resume parsing expert. I will provide you with the content of a resume: {resumeText}. \n"
                                        "Please extract the key information and organize it into a structured JSON format. Ensure to include the following fields:\n\n"
                                        "Name: Full name of the individual.\n"
                                        "Professional Summary: A brief overview of the individual’s career and skills. (if available or leave as empty)\n"
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

                return render_template('result.html', resume_json=resume_json)  # Render JSON output
        except Exception as e:
            return f"An error occurred: {e}", 500  # Return error message
    return render_template('upload.html')  # Render upload form

@app.route('/result', methods=['GET'])
def result():
    # This route can be used to render the result page if needed
    return render_template('result.html')  # Render result page

@app.route('/test')
def test():
    return "<h1>This is a test page!</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
