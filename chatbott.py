import streamlit as st
from groq import Groq

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# =========================
# GROQ CLIENT
# =========================

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

MODEL = "openai/gpt-oss-20b"

# =========================
# AI SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
You are a highly capable general-purpose AI assistant.

You can help users with:

- General questions
- Mathematics
- Calculations
- Logical reasoning
- Python
- Java
- C
- C++
- JavaScript
- HTML
- CSS
- SQL
- Programming
- Code debugging
- Code explanation
- Data analysis
- Academic subjects
- Engineering subjects
- Writing
- Rewriting
- Summarization
- Translation
- Technical explanations
- Step-by-step problem solving
- Project ideas
- Project documentation

Rules:

1. Give clear and accurate answers.
2. For mathematical problems, show the calculation.
3. For programming problems, provide complete and clean code.
4. Explain code when useful.
5. For academic questions, explain in simple language.
6. For complex questions, give structured answers.
7. Do not invent facts.
8. If you don't know something, say so.
9. Keep simple answers concise.
10. Give detailed answers when the user asks for detail.
"""

# =========================
# SESSION MEMORY
# =========================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("🤖 AI Assistant")

    st.write("Powered by Groq")

    st.divider()

    st.subheader("Capabilities")

    st.write("💬 General Chat")
    st.write("🧮 Mathematics")
    st.write("💻 Programming")
    st.write("🐛 Debugging")
    st.write("📊 Data Analysis")
    st.write("📚 Education")
    st.write("✍️ Writing")
    st.write("📝 Summarization")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        st.session_state.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        st.rerun()

# =========================
# MAIN UI
# =========================

st.title("🤖 Universal AI Assistant")

st.caption(
    "Ask questions, solve problems, write code, calculate, learn and more."
)

# =========================
# DISPLAY PREVIOUS MESSAGES
# =========================

for message in st.session_state.messages:

    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =========================
# USER INPUT
# =========================

user_input = st.chat_input(
    "Ask me anything..."
)

# =========================
# PROCESS USER MESSAGE
# =========================

if user_input:

    # Display user message
    with st.chat_message("user"):

        st.markdown(user_input)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages,
                    temperature=0.3,
                    max_tokens=4096
                )

                answer = response.choices[0].message.content

                st.markdown(answer)

                # Save AI response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )
