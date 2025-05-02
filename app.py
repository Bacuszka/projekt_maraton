import streamlit as st
import pandas as pd
import os
import json
import tempfile
import base64
from dotenv import load_dotenv
from pycaret.regression import load_model, predict_model
from openai import OpenAI
from langfuse import Langfuse
from audio_recorder_streamlit import audio_recorder
import requests
import joblib
import tempfile
from pycaret.regression import load_model as pycaret_load_model
import requests
import tempfile
import os
import streamlit as st

# 🔄 Wczytaj zmienne środowiskowe Langfuse
load_dotenv()

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

langfuse = Langfuse(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host=LANGFUSE_HOST
)

st.title("🏃‍♂️ Predykcja czasu maratonu na podstawie Twojej wypowiedzi")

# 🔑 Klucz OpenAI
api_key = st.text_input("🔑 Wprowadź swój klucz OpenAI API:", type="password")

if not api_key:
    st.warning("⚠️ Wprowadź klucz OpenAI, aby kontynuować.")
    st.stop()

client = OpenAI(api_key=api_key)
MODEL_FILENAME = "best_marathon_model"

MODEL_URL = "https://wiadro.fra1.cdn.digitaloceanspaces.com/best_marathon_model.pkl"

def load_model_from_spaces():
    try:
        # Pobierz plik modelu
        response = requests.get(MODEL_URL)
        response.raise_for_status()

        # Zapisz plik tymczasowo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            tmp_file.write(response.content)
            tmp_model_path = tmp_file.name

        # Wczytaj modelem z pycaret (NIE joblib)
        model = pycaret_load_model(tmp_model_path.replace(".pkl", ""))
        st.success("✅ Model został załadowany.")
        return model

    except Exception as e:
        st.error(f"❌ Błąd ładowania modelu z DigitalOcean: {e}")
        return None

def recognize_speech_from_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name

    try:
        with open(tmp_file_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                language="pl"
            )
        return transcript
    except Exception as e:
        st.error(f"❌ Błąd rozpoznawania mowy przez Whisper: {e}")
        return None

def extract_data_with_openai(text):
    prompt = f"""
Wypowiedź: "{text}"

Wyodrębnij dane w formacie JSON:
{{
  "płeć": "Mężczyzna" lub "Kobieta",
  "wiek": liczba całkowita,
  "tempo_5km": liczba zmiennoprzecinkowa (np. 6.8)
}}

Zwróć **tylko JSON**, bez żadnych komentarzy.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Jesteś pomocnym asystentem, który przetwarza wypowiedzi na dane wejściowe do modelu predykcji czasu maratonu."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        reply = response.choices[0].message.content.strip()
        st.code(reply, language="json")

        if reply.startswith("{") and reply.endswith("}"):
            return json.loads(reply)
        else:
            st.error("❌ Odpowiedź GPT nie była poprawnym JSON-em.")
            return None

    except Exception as e:
        st.error(f"❌ Błąd OpenAI: {e}")
        return None

st.markdown("""
🧾 **Instrukcja użytkowania:**

1. Kliknij przycisk nagrywania 🎙️ poniżej.
2. Wypowiedz dane w formacie: `mężczyzna, 33 lata, tempo 6,8`
3. Poczekaj na przetworzenie — model wygeneruje przewidywany czas maratonu.
""")

with st.spinner("🔄 Ładowanie modelu..."):
    model = load_model_from_spaces()
if model is None:
    st.stop()

audio_bytes = audio_recorder(pause_threshold=2.0)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    speech_input = recognize_speech_from_audio(audio_bytes)
    if speech_input:
        st.session_state.speech_input = speech_input

if 'speech_input' in st.session_state:
    st.markdown(f"🗣️ **Rozpoznana wypowiedź:** `{st.session_state.speech_input}`")
    extracted = extract_data_with_openai(st.session_state.speech_input)

    if extracted:
        try:
            gender = extracted.get("płeć")
            age = extracted.get("wiek")
            pace = extracted.get("tempo_5km")

            st.markdown("✏️ **Zweryfikuj lub popraw dane wejściowe:**")
            gender = st.radio("Płeć", ["Mężczyzna", "Kobieta"], index=0 if gender == "Mężczyzna" else 1)
            age = st.number_input("Wiek", min_value=1, value=age if isinstance(age, int) and age > 0 else 30, step=1)
            pace = st.number_input("Tempo na 5 km (min/km)", min_value=1.0, value=pace if isinstance(pace, float) and pace > 0 else 6.0, step=0.1)

            gender_value = 1 if gender.lower() == "mężczyzna" else 0
            five_km_time_min = pace * 5

            input_df = pd.DataFrame({
                'Płeć': [gender_value],
                'Wiek': [age],
                '5 km Czas': [five_km_time_min]
            })

            if st.button("🔮 Przewidź czas maratonu"):
                prediction = predict_model(model, data=input_df)

                if 'prediction_label' in prediction.columns:
                    predicted_time = abs(prediction.loc[0, 'prediction_label'])
                    hours = int(predicted_time // 60)
                    minutes = int(predicted_time % 60)

                    result_text = f"🏅 Przewidywany czas maratonu: **{hours}h {minutes}m**"
                    st.success(result_text)

                    if 'history' not in st.session_state:
                        st.session_state.history = []

                    record = {
                        "Płeć": gender,
                        "Wiek": age,
                        "Tempo": pace,
                        "Czas (h:m)": f"{hours}h {minutes}m"
                    }

                    st.session_state.history.append(record)

                    history_path = "history_maratonu.json"
                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)

                    trace = langfuse.trace(name="marathon_prediction")
                    trace.update(user_id="uzytkownik_streamlit")

                    trace.span(
                        name="rozpoznana_wypowiedz",
                        input="audio",
                        output=st.session_state.speech_input,
                        metadata={"etap": "rozpoznawanie_mowy"}
                    )

                    gpt_prompt = f"""
                    Wypowiedź: "{st.session_state.speech_input}"

                    Wyodrębnij dane w formacie JSON:
                    {{
                    "płeć": "Mężczyzna" lub "Kobieta",
                    "wiek": liczba całkowita,
                    "tempo_5km": liczba zmiennoprzecinkowa (np. 6.8)
                    }}

                    Zwróć **tylko JSON**, bez żadnych komentarzy.
                    """

                    trace.span(
                        name="prompt_do_gpt",
                        input=gpt_prompt,
                        output=json.dumps(extracted, ensure_ascii=False),
                        metadata={"etap": "openai_extract"}
                    )

                    trace.span(
                        name="dane_wejsciowe_model",
                        input=json.dumps({
                            "Płeć": gender_value,
                            "Wiek": age,
                            "5 km Czas": five_km_time_min
                        }, ensure_ascii=False),
                        metadata={"etap": "model_input"}
                    )

                    trace.span(
                        name="wynik_predykcji",
                        output=json.dumps(record, ensure_ascii=False),
                        metadata={"etap": "model_output"}
                    )
                else:
                    st.error("❌ Nie znaleziono kolumny 'prediction_label' w wyniku predykcji.")
        except Exception as e:
            st.error(f"❌ Błąd przetwarzania danych: {e}")
    else:
        st.error("❌ Nie udało się wyodrębnić danych z wypowiedzi.")

if 'history' in st.session_state and st.session_state.history:
    st.markdown("## 🕓 Historia predykcji")
    st.table(pd.DataFrame(st.session_state.history))