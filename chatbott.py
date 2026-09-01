import gradio as gr
from groq import Groq
from getpass import getpass
import pandas as pd
import os

# ==========================================
# GROQ API
# ==========================================

api_key = getpass("Enter your Groq API Key: ")

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-20b"


# ==========================================
# SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are a powerful general-purpose AI assistant.

You can help users with:

- General questions
- Mathematics and calculations
- Programming
- Python, Java, C, C++, SQL, HTML, CSS and JavaScript
- Debugging code
- Data analysis
- CSV and Excel analysis
- Academic questions
- Summarization
- Writing and rewriting
- Technical explanations
- Step-by-step problem solving

Be accurate, helpful and fast.

For calculations, show the calculation clearly.

For programming questions, provide clean working code.

For difficult concepts, explain them in simple language.

When analyzing data, identify useful patterns, statistics,
missing values and important observations.

Never invent information when the required information is unavailable.
"""


# ==========================================
# FILE ANALYSIS
# ==========================================

def analyze_file(file_path):

    if file_path is None:
        return ""

    try:

        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".csv":
            df = pd.read_csv(file_path)

        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)

        else:
            return "The uploaded file is not CSV or Excel."

        result = f"""
FILE INFORMATION

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column Names:
{list(df.columns)}

Data Types:
{df.dtypes.to_string()}

Missing Values:
{df.isnull().sum().to_string()}

Statistics:
{df.describe(include="all").to_string()}

First 10 Rows:
{df.head(10).to_string()}
"""

        return result

    except Exception as e:

        return f"Could not analyze file: {str(e)}"


# ==========================================
# CHATBOT FUNCTION
# ==========================================

def chatbot(message, history, file):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # ======================================
    # ADD CHAT HISTORY
    # ======================================

    for item in history:

        # New Gradio format
        if isinstance(item, dict):

            role = item.get("role")
            content = item.get("content")

            if role in ["user", "assistant"]:

                if isinstance(content, str):

                    messages.append({
                        "role": role,
                        "content": content
                    })

        # Old Gradio format
        elif isinstance(item, (list, tuple)):

            if len(item) >= 2:

                user_message = item[0]
                assistant_message = item[1]

                if user_message:

                    messages.append({
                        "role": "user",
                        "content": str(user_message)
                    })

                if assistant_message:

                    messages.append({
                        "role": "assistant",
                        "content": str(assistant_message)
                    })


    # ======================================
    # FILE ANALYSIS
    # ======================================

    if file is not None:

        file_data = analyze_file(file)

        messages.append({
            "role": "system",
            "content": f"""
The user uploaded a data file.

Here is the file information:

{file_data}

Use this information when answering questions about the file.
"""
        })


    # ======================================
    # USER MESSAGE
    # ======================================

    messages.append({
        "role": "user",
        "content": message
    })


    # ======================================
    # GROQ
    # ======================================

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=messages,

            temperature=0.3,

            max_tokens=4096

        )

        answer = response.choices[0].message.content

        return answer

    except Exception as e:

        return f"❌ Groq Error: {str(e)}"


# ==========================================
# UI
# ==========================================

file_upload = gr.File(
    label="📁 Upload CSV / Excel",
    file_types=[".csv", ".xlsx", ".xls"],
    type="filepath"
)


demo = gr.ChatInterface(

    fn=chatbot,

    additional_inputs=[file_upload],

    title="🤖 Universal AI Assistant",

    description="""
    ⚡ Powered by Groq

    💬 Chat | 🧮 Calculations | 💻 Coding | 📊 Data Analysis |
    📚 Study | ✍️ Writing | 🧠 Problem Solving
    """,

    textbox=gr.Textbox(
        placeholder="Ask me anything...",
        container=True
    )
)


# ==========================================
# LAUNCH
# ==========================================

demo.launch(share=True)
