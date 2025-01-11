import streamlit as st
import random
import uuid

# Import your chatbot_graph code
from chatbot_graph import call_model

import os

os.environ["SQLITE_BINARY"] = "./bin/sqlite3"


def main():
    st.title("Chatbot with Langgraph")

    # 1) We store a single thread_id for the entire session,
    #    just like you do in Redis for WhatsApp
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())

    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    # 2) A phone number is also stable per session
    if "phone_number" not in st.session_state:
        st.session_state["phone_number"] = f"+52 555 {random.randint(1000, 9999)}"
    phone_number = st.session_state["phone_number"]

    # 3) We'll keep a "chat history" in Streamlit *only for display*,
    #    but we do NOT pass it to call_model.
    #    Because your code in chatbot_graph.py persists conversation in a DB checkpoint,
    #    we only need to pass the user's new message each time.
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Show the existing messages
    for message_dict in st.session_state["messages"]:
        with st.chat_message(message_dict["role"]):
            st.markdown(message_dict["content"])

    # 4) Chat input
    user_text = st.chat_input("Ask something to the chatbot:")
    if user_text:
        # Display user text immediately
        user_msg = {"role": "user", "content": user_text}
        st.session_state["messages"].append(user_msg)
        with st.chat_message("user"):
            st.markdown(user_text)

        # 5) EXACTLY like WhatsApp, pass a single string (the user’s last message)
        #    plus the phone number + config to call_model
        response_text, message_type = call_model(user_text, phone_number, config)

        # 6) Show the response
        if response_text:
            bot_msg = {"role": "assistant", "content": response_text}
            st.session_state["messages"].append(bot_msg)
            with st.chat_message("assistant"):
                st.markdown(response_text)


if __name__ == "__main__":
    main()
