import streamlit as st
import whisper
import tempfile
import os
import hashlib

from pyannote.audio import Pipeline
from datetime import timedelta
import torch

st.set_page_config(page_title="Whisper Transcriber with Diarization", layout="centered")
st.title("🎙️ Whisper Transcriber + Speaker Diarization")

st.markdown("Upload audio files to transcribe with OpenAI Whisper and detect speakers with Pyannote.")

model_size = st.selectbox("Choose Whisper model size", ["tiny", "base", "small", "medium", "large"])
uploaded_files = st.file_uploader("Upload audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True)

# Initialize cache
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = {}

def get_file_hash(file):
    return hashlib.sha256(file.read()).hexdigest()

if uploaded_files:
    with st.spinner("Loading Whisper model..."):
        model = whisper.load_model(model_size)

    with st.spinner("Loading Pyannote diarization model..."):
        hf_token = st.secrets["HUGGINGFACE_TOKEN"]
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=hf_token)

    for uploaded_file in uploaded_files:
        st.subheader(f"📁 {uploaded_file.name}")
        st.audio(uploaded_file)

        uploaded_file.seek(0)
        file_hash = get_file_hash(uploaded_file)
        uploaded_file.seek(0)

        if file_hash not in st.session_state.transcriptions:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner(f"Transcribing {uploaded_file.name}..."):
                result = model.transcribe(tmp_path)
                transcription = result["text"]

            with st.spinner(f"Diarizing {uploaded_file.name}..."):
                diarization = diarization_pipeline(tmp_path)

            os.remove(tmp_path)

            # Store results in session
            st.session_state.transcriptions[file_hash] = {
                "transcript": transcription,
                "diarization": diarization
            }

        data = st.session_state.transcriptions[file_hash]
        diarization = data["diarization"]
        transcript_text = data["transcript"]

        st.markdown("### 🗣️ Speaker Segments")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = str(timedelta(seconds=round(turn.start)))
            end = str(timedelta(seconds=round(turn.end)))
            st.markdown(f"**{speaker}**: {start} → {end}")

        st.markdown("### 📝 Full Transcript")
        with st.expander("🔍 View Transcript"):
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
