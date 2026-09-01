import streamlit as st
from groq import Groq

# ============================================================
# GPT-OSS 20B CHATBOT
# STREAMLIT + GROQ API
# ============================================================

MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """
You are a helpful, intelligent and friendly AI assistant.

Rules:
- Answer questions accurately.
- Explain difficult topics clearly.
- Use examples when useful.
- Do not make up information.
- Remember the conversation context.
- Be concise unless the user asks for detailed information.
"""

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GPT-OSS 20B Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 16px;
    color: #9ca3af;
    margin-bottom: 25px;
}

.chat-box {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🤖 GPT-OSS 20B Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">⚡ Powered by Groq</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_..."
    )

    st.caption(
        "Your API key is used only for this session."
    )

    st.divider()

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1
    )

    # --------------------------------------------------------
    # REASONING
    # --------------------------------------------------------

    reasoning = st.selectbox(
        "🧠 Reasoning Effort",
        options=[
            "low",
            "medium",
            "high"
        ],
        index=1
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL INFO
    # --------------------------------------------------------

    st.markdown("### 🧠 Model")

    st.code(
        "openai/gpt-oss-20b"
    )

    st.markdown("### ⚡ Provider")

    st.write("Groq API")

    st.markdown("### 💬 Type")

    st.write("Normal AI Chatbot")

    st.markdown("### 🧠 Memory")

    st.write("Conversation History")

    st.divider()

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

# ============================================================
# CHAT INPUT
# ============================================================

user_message = st.chat_input(
    "Type your message..."
)

# ============================================================
# PROCESS MESSAGE
# ============================================================

if user_message:

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if not api_key.strip():

        st.error(
            "⚠️ Please enter your Groq API key in the sidebar."
        )

        st.stop()

    # --------------------------------------------------------
    # ADD USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    # Display user message
    with st.chat_message("user"):

        st.markdown(
            user_message
        )

    # --------------------------------------------------------
    # CREATE GROQ CLIENT
    # --------------------------------------------------------

    try:

        client = Groq(
            api_key=api_key.strip()
        )

        # ----------------------------------------------------
        # BUILD MESSAGE HISTORY
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        for message in st.session_state.messages:

            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"]
                }
            )

        # ----------------------------------------------------
        # ASSISTANT RESPONSE
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            response_placeholder = st.empty()

            with st.spinner("Thinking..."):

                response = client.chat.completions.create(

                    model=MODEL,

                    messages=messages,

                    temperature=float(
                        temperature
                    ),

                    reasoning_effort=reasoning,

                    max_completion_tokens=4096
                )

                answer = (
                    response.choices[0]
                    .message
                    .content
                )

                if not answer:

                    answer = (
                        "I couldn't generate a response."
                    )

            response_placeholder.markdown(
                answer
            )

        # ----------------------------------------------------
        # SAVE ASSISTANT RESPONSE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

    except Exception as e:

        error_message = (
            "❌ Groq API Error\n\n"
            f"{str(e)}"
        )

        with st.chat_message("assistant"):

            st.error(
                error_message
            )

        # Remove the user message if request failed
        if st.session_state.messages:

            st.session_state.messages.pop()
