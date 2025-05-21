import streamlit as st
import whisper
import tempfile
import os
import hashlib

st.set_page_config(page_title="Whisper Transcriber", layout="centered")
st.title("🎙️ Audio Transcriber with OpenAI Whisper")

st.markdown("Upload one or more audio files and get the transcribed text using Whisper.")

# model_size = st.selectbox("Choose Whisper model size", ["tiny", "base", "small", "medium", "large"])
model_size = st.selectbox("Choose Whisper model size", ["tiny", "base"])
uploaded_files = st.file_uploader("Upload audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True)

# Initialize transcription cache in session state
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = {}

# Helper: generate SHA-256 from file content
def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

# Track current valid file hashes from this upload
current_file_hashes = set()
duplicate_files = []

if uploaded_files:
    with st.spinner("Loading Whisper model..."):
        model = whisper.load_model(model_size)

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        file_hash = get_file_hash(file_bytes)
        uploaded_file.seek(0)

        # If this file has already been seen *in this upload session*, skip
        if file_hash in current_file_hashes:
            duplicate_files.append(uploaded_file.name)
            continue
        current_file_hashes.add(file_hash)

        st.subheader(f"📁 {uploaded_file.name}")
        st.audio(uploaded_file)

        # If this file is not already in cache, transcribe it
        if file_hash not in st.session_state.transcriptions:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            with st.spinner(f"Transcribing {uploaded_file.name}..."):
                result = model.transcribe(tmp_path)
                st.session_state.transcriptions[file_hash] = result["text"]

            os.remove(tmp_path)

        transcript_text = st.session_state.transcriptions[file_hash]
        unique_key = f"{file_hash}_transcript"

        with st.expander("🔍 View Transcription"):
            st.text_area(label="Transcript", value=transcript_text, height=200, key=f"text_{unique_key}")
            txt_filename = uploaded_file.name.rsplit('.', 1)[0] + "_transcript.txt"
            st.download_button(
                label="📥 Download Transcript",
                data=transcript_text,
                file_name=txt_filename,
                mime="text/plain",
                key=f"download_{unique_key}"
            )

    # Prune cache: remove transcriptions no longer associated with uploaded files
    st.session_state.transcriptions = {
        k: v for k, v in st.session_state.transcriptions.items() if k in current_file_hashes
    }

    if duplicate_files:
        st.warning(f"⚠️ Skipped duplicate file(s): {', '.join(duplicate_files)}")

else:
    st.info("Please upload at least one audio file to transcribe.")
    # Also reset the cache if all files are removed
    st.session_state.transcriptions = {}
