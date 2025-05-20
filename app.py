import streamlit as st
import whisper
import tempfile
import os
import hashlib

st.set_page_config(page_title="Whisper Transcriber", layout="centered")
st.title("🎙️ Audio Transcriber with OpenAI Whisper")

st.markdown("Upload one or more audio files and get the transcribed text using Whisper.")

model_size = st.selectbox("Choose Whisper model size", ["tiny", "base", "small", "medium", "large"])
uploaded_files = st.file_uploader("Upload audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True)

# Initialize transcription cache in session state
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = {}

def get_file_hash(file):
    # Compute a hash of the file contents to uniquely identify it
    return hashlib.sha256(file.read()).hexdigest()

if uploaded_files:
    with st.spinner("Loading Whisper model..."):
        model = whisper.load_model(model_size)

    for uploaded_file in uploaded_files:
        st.subheader(f"📁 {uploaded_file.name}")
        st.audio(uploaded_file)

        # Compute file hash to use as key
        uploaded_file.seek(0)  # make sure pointer is at start before hashing
        file_hash = get_file_hash(uploaded_file)
        uploaded_file.seek(0)  # reset pointer after reading for hash

        if file_hash not in st.session_state.transcriptions:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner(f"Transcribing {uploaded_file.name}..."):
                result = model.transcribe(tmp_path)
                st.session_state.transcriptions[file_hash] = result["text"]

            os.remove(tmp_path)

        transcript_text = st.session_state.transcriptions[file_hash]

        with st.expander("🔍 View Transcription"):
            st.text_area(label="Transcript", value=transcript_text, height=200)
            txt_filename = uploaded_file.name.rsplit('.', 1)[0] + "_transcript.txt"
            st.download_button(
                label="📥 Download Transcript",
                data=transcript_text,
                file_name=txt_filename,
                mime="text/plain"
            )
else:
    st.info("Please upload at least one audio file to transcribe.")