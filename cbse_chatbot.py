import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
import av  # PyAV is used internally by streamlit-webrtc for audio frames
import wave
import os
import pandas as pd
import difflib
import openai
from gtts import gTTS
from playsound import playsound

# Set page title
st.title("CBSE Chatbot – Voice Enabled QA")

# Instructions for the user
st.write("**Instructions:** You can ask a question by voice or text. For voice, click **Start** to record and **Stop** when done. If voice input is unavailable, please use the text box below.")

# Load the CBSE data from CSV
DATA_PATH = "data.csv"
knowledge_df = None
if os.path.exists(DATA_PATH):
    knowledge_df = pd.read_csv(DATA_PATH)
    # Ensure columns are string type for matching
    knowledge_df = knowledge_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Initialize OpenAI API (make sure to set your API key in an environment variable or st.secrets)
openai.api_key = st.secrets("OPENAI_API_KEY", "")  # or st.secrets["OPENAI_API_KEY"]

# Session state for audio buffer
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = bytearray()

# Set up the WebRTC streamer for audio input only
webrtc_ctx = webrtc_streamer(
    key="listen",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False}
)

# UI elements for text input
text_question = st.text_input("Type your question here (if not using voice):")

# Placeholder for status messages
status_placeholder = st.empty()

user_question = None  # will hold the final question in text form

# Handle voice recording and transcription
if webrtc_ctx.state.playing:
    # Recording in progress
    status_placeholder.info("🎤 **Listening...** Please speak and then press stop.")
    st.session_state.audio_bytes = bytearray()  # reset buffer at start of recording
    # We will capture audio frames in real-time
    first_frame_info = None
    # Continuously read frames until recording stops
    while webrtc_ctx.state.playing:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                # No frame received in 1 second, just try again
                continue
            for frame in audio_frames:
                # Convert AudioFrame to bytes and append to buffer
                frame_bytes = frame.to_ndarray().tobytes()
                st.session_state.audio_bytes.extend(frame_bytes)
                # Store audio format info from first frame (for WAV header)
                if not first_frame_info:
                    sample_rate = frame.sample_rate
                    sample_width = frame.format.bytes  # bytes per sample
                    channels = len(frame.layout.channels)
                    first_frame_info = True
        else:
            # audio_receiver not ready (should not happen while playing)
            continue
    # When the loop exits, recording has stopped
    status_placeholder.info("✅ **Recording stopped**. Processing your question...")
    
    # Save the recorded bytes as a WAV file for transcription
    wav_path = "temp.wav"
    wf = wave.open(wav_path, "wb")
    wf.setnchannels(channels if first_frame_info else 1)
    wf.setsampwidth(sample_width if first_frame_info else 2)
    wf.setframerate(sample_rate if first_frame_info else 16000)
    wf.writeframes(st.session_state.audio_bytes)
    wf.close()
    
    # Use OpenAI Whisper to transcribe the audio
    try:
        with open(wav_path, "rb") as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file)
        user_question = transcription["text"].strip()
    except Exception as e:
        st.error("Failed to transcribe audio. Please try again or use text input.")
        user_question = None

# If user typed a question, use that (voice takes precedence if both exist)
if user_question is None or user_question == "":
    if text_question:
        user_question = text_question.strip()

if user_question:
    # Display the question on the app
    st.markdown(f"**Question:** {user_question}")
    
    # Find relevant info from CSV data (if loaded)
    context_info = ""
    if knowledge_df is not None and "question" in knowledge_df.columns and "answer" in knowledge_df.columns:
        # Attempt to find the best match for the question in the data
        questions = knowledge_df["question"].astype(str).tolist()
        # Use difflib to find closest match
        match = difflib.get_close_matches(user_question.lower(), questions, n=1, cutoff=0.6)
        if match:
            match_idx = questions.index(match[0])
            context_answer = str(knowledge_df.loc[match_idx, "answer"])
            context_info = context_answer  # we will feed this to OpenAI
    elif knowledge_df is not None:
        # If data has no clear question/answer structure, we could search any text fields for keywords
        # (For simplicity, not implemented in detail here)
        pass

    # Prepare prompt for OpenAI (include context if available)
    openai_prompt = user_question
    if context_info:
        openai_prompt = f"According to the CBSE data, \"{context_info}\". Now answer the question: {user_question}"
    
    # Call OpenAI ChatCompletion (GPT-3.5) to get an answer
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant. Answer the question based on CBSE curriculum data."},
                {"role": "user", "content": openai_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        answer_text = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error("Failed to get answer from OpenAI. Please check your API key or try again.")
        answer_text = ""

    if answer_text:
        # Display the answer text
        st.markdown(f"**Answer:** {answer_text}")
        # Convert answer to speech using gTTS
        try:
            tts = gTTS(text=answer_text, lang='en')
            tts.save("response.mp3")
        except Exception as e:
            st.error("Failed to generate speech audio.")
        else:
            # Play the audio in the browser
            audio_file = open("response.mp3", "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
            audio_file.close()
            # Attempt to play sound on server (will work only if running locally)
            try:
                playsound("response.mp3")
            except Exception:
                # On Streamlit Cloud, this will likely fail or do nothing
                pass
