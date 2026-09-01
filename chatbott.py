import streamlit as st
from groq import Groq

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Universal AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# ==============================
# API KEY
# ==============================

api_key = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-20b"


# ==============================
# SYSTEM PROMPT
# ==============================

SYSTEM_PROMPT = """
You are a powerful general-purpose AI assistant.

You can help with:

- General questions
- Mathematics and calculations
- Programming
- Python
- Java
- C
- C++
- SQL
- HTML
- CSS
- JavaScript
- Debugging
- Data analysis
- Academic questions
- Writing
- Summarization
- Technical explanations
- Step-by-step problem solving

Be helpful, accurate and clear.

For calculations, show the steps.

For programming questions, provide clean working code.

For difficult topics, explain them simply.
"""


# ==============================
# TITLE
# ==============================

st.title("🤖 Universal AI Assistant")

st.caption(
    "⚡ Powered by Groq | 💬 Chat | 🧮 Math | 💻 Coding | 📊 Analysis"
)


# ==============================
# CHAT MEMORY
# ==============================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


# ==============================
# DISPLAY CHAT
# ==============================

for message in st.session_state.messages:

    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==============================
# USER INPUT
# ==============================

user_input = st.chat_input(
    "Ask me anything..."
)


# ==============================
# CHAT RESPONSE
# ==============================

if user_input:

    # Display user message

    with st.chat_message("user"):

        st.markdown(user_input)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # Generate response

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

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(f"Error: {e}")


# ==============================
# SIDEBAR
# ==============================

with st.sidebar:

    st.header("⚙️ Settings")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        st.rerun()

    st.divider()

    st.write("### Capabilities")

    st.write("💬 General Chat")
    st.write("🧮 Mathematics")
    st.write("💻 Programming")
    st.write("🐛 Debugging")
    st.write("📚 Education")
    st.write("📊 Data Analysis")
    st.write("✍️ Writing")
    st.write("📝 Summarization")
