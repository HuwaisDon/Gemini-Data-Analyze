import streamlit as st # <--- CRITICAL FIX: Ensure Streamlit is imported first
import pandas as pd
import os
import io
from google import genai
from google.genai import types

# --- API Key Setup ---
if "GEMINI_API_KEY" not in os.environ:
    st.error("Please set the GEMINI_API_KEY environment variable.")
    st.stop()
    
# Initialize the Gemini Client
client = genai.Client()

# --- Utility Function to Get DataFrame Info ---
def get_df_info(df):
    """Generates a text description of the DataFrame's structure."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    # Return the first 5 rows and the info
    return f"DataFrame head:\n{df.head().to_markdown()}\n\nDataFrame info (columns, dtypes):\n{buffer.getvalue()}"

# --- Custom Tool Definition ---
def python_tool(code: str) -> str:
    """
    Executes Python code in the current environment to analyze the 'df' DataFrame.
    Returns the result of the code execution.
    """
    # This ensures the 'df' variable is available inside the exec() call.
    # We use st.session_state.df to access the data.
    exec_globals = {'df': st.session_state.df, 'pd': pd}
    
    try:
        old_stdout = io.StringIO()
        import sys
        sys.stdout = old_stdout
        
        # Execute the code
        exec(code, exec_globals)
        
        sys.stdout = sys.__stdout__ # Restore stdout
        output = old_stdout.getvalue()
        
        # Clean up the output string to prevent massive returns
        if len(output) > 2000:
            return f"Execution successful, but output was too long ({len(output)} chars). First 2000 chars: {output[:2000]}"
            
        return output if output else "Code executed successfully. No direct print output returned."
    
    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"Execution Error: {e}"

# --- Streamlit App Setup ---
st.set_page_config(page_title="Gemini Data Analyzer (Multi-Turn)")
st.title("💬 Gemini Data Analyst Chatbot")

# 1. Initialize Chat History and DF in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None


# --- File Upload and Initial Data Load ---
uploaded_file = st.file_uploader("Upload a CSV or Excel file (Upload clears chat history)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
        # Store df in session state so the tool can access it
        st.session_state.df = df
        
        st.subheader("Data Preview:")
        st.dataframe(df.head())
        
        # Reset chat history when a new file is uploaded
        st.session_state.messages = [] 

        # Initialize the system instruction for the model
        # NEW and STRICTER system_instruction
        system_instruction = (
            "You are an expert data analyst. You have access to a pandas DataFrame named 'df'. "
            "The DataFrame structure is:\n"
            f"{get_df_info(df)}\n"
            "**ABSOLUTELY ALL PYTHON CODE MUST BE EXECUTED BY CALLING THE 'python_tool' function.** "
            "The argument to 'python_tool' must be a single string containing the valid, runnable Python code. "
            "Do not attempt to call any other functions like 'value_counts' directly. "
            "Always generate and run Python code to calculate the answer. "
            "Do not display the full DataFrame. Only provide the final, computed answer."
       )
        
        # Store the system instruction for the chat model to use on initialization
        st.session_state.system_instruction = system_instruction


    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.info("Please check if the file is correctly formatted (CSV/Excel) and not corrupted.")
        st.session_state.df = None


# --- Chat Interface and Logic ---

# 2. Display existing messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle new user input (prompt)
if st.session_state.df is not None:
    # Use st.chat_input for a persistent, conversational prompt box
    if prompt := st.chat_input("Ask your next question about the data..."):
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to history (role must be 'user' or 'model', not 'assistant')
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # --- Model Call with History ---
        with st.spinner("Asking Gemini to analyze the data..."):
            
            # --- CRITICAL FIX: MANUALLY CONSTRUCT CONTENT AND PART OBJECTS ---
            contents = []
            for msg in st.session_state.messages:
                # 1. Create the Part object
                part = types.Part(text=msg["content"])
                
                # The Gemini API uses 'model' for its responses, but Streamlit uses 'assistant'
                # We normalize the role for the API call here.
                api_role = 'model' if msg["role"] == 'assistant' else msg["role"]
                
                # 2. Create the Content object, including the correct role
                contents.append(
                    types.Content(role=api_role, parts=[part])
                )
            
            # The actual API call to initiate the multi-turn conversation
            # We send ALL previous messages (history) to the chat creator.
            chat_session = client.chats.create(
                model='gemini-2.5-flash',
                history=contents[:-1], # Pass ALL previous messages (excluding the current one)
                config=types.GenerateContentConfig(
                    system_instruction=st.session_state.system_instruction,
                    tools=[python_tool]
                )
            )

            # Get the model's response by sending the LATEST user message
            response = chat_session.send_message(prompt)

            # Display and store assistant's response
            with st.chat_message("assistant"):
                st.markdown(response.text)
            
            # Store the response using the Streamlit role 'assistant'
            st.session_state.messages.append({"role": "assistant", "content": response.text})