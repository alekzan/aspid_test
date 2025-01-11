import streamlit as st
import random
import uuid
import os

# Import your chatbot_graph code
from chatbot_graph import call_model

# Use the custom SQLite binary for Chroma
os.environ["SQLITE_BINARY"] = "./bin/sqlite3"


def main():
    st.title("Chatbot with Langgraph")

    # 0) Restart Conversation Button
    if st.button("Restart Conversation"):
        # Clear session state for a fresh chat
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []
        # Rerun the app so changes take effect
        st.rerun()

    # 1) Maintain a single thread_id for the entire session
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())

    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    # 2) Generate a random phone number once per session
    if "phone_number" not in st.session_state:
        st.session_state["phone_number"] = f"+52 555 {random.randint(1000, 9999)}"
    phone_number = st.session_state["phone_number"]

    # 3) Keep a "chat history" in Streamlit for display only
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Show existing chat messages (user & assistant) in the UI
    for message_dict in st.session_state["messages"]:
        with st.chat_message(message_dict["role"]):
            st.markdown(message_dict["content"])

    # 4) Capture user input via Streamlit's chat_input
    user_text = st.chat_input("Ask something to the chatbot:")
    if user_text:
        # Immediately display user's message
        user_msg = {"role": "user", "content": user_text}
        st.session_state["messages"].append(user_msg)
        with st.chat_message("user"):
            st.markdown(user_text)

        # 5) Call the model with only the last user message plus config
        response_text, message_type = call_model(user_text, phone_number, config)

        # 6) Display the assistant's response
        if response_text:
            bot_msg = {"role": "assistant", "content": response_text}
            st.session_state["messages"].append(bot_msg)
            with st.chat_message("assistant"):
                st.markdown(response_text)


if __name__ == "__main__":
    main()
