openai.api_key = " " # Replace with your actual API key


# Load Whisper model
model = whisper.load_model("tiny")  # You can use "base", "small", "medium", or "large"
print("✅ Whisper Model Loaded Successfully!")




class QueryContext:
    """
    A class to maintain context across multiple SQL queries.
    """
    def __init__(self):
        self.previous_query = None

    def update_context(self, new_query):
        """
        Updates the context with the new query.

        Args:
            new_query (str): The new SQL query.
        """
        self.previous_query = new_query

    def get_context(self):
        """
        Returns the current context (previous query).

        Returns:
            str: The previous SQL query, or None if no context exists.
        """
        return self.previous_query

# Initialize the context
query_context = QueryContext()

def get_voice_command(timeout=5, phrase_time_limit=10):
    """
    Captures audio from the microphone and transcribes it into text using Whisper.

    Args:
        timeout (int): Maximum time to wait for speech to start (in seconds).
        phrase_time_limit (int): Maximum duration of speech to capture (in seconds).

    Returns:
        str: Transcribed text, or None if no speech was detected.
    """
    recognizer = sr.Recognizer()

    with sr.Microphone(sample_rate=16000) as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎙️ Listening... Speak now!")

        try:
            # Capture audio with a phrase time limit
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("⏳ Timeout: No speech detected.")
            return None

        # Save recorded audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(audio.get_wav_data())
            temp_wav_path = temp_wav.name

        try:
            # Transcribe speech using Whisper
            result = model.transcribe(temp_wav_path, fp16=torch.cuda.is_available())
            transcribed_text = result['text'].strip()
            print(f"✅ Transcribed: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return None
        finally:
            # Clean up the temporary file
            os.remove(temp_wav_path)

def validate_sql(query):
    """
    Validates the syntax of an SQL query using sqlparse.

    Args:
        query (str): The SQL query to validate.

    Returns:
        bool: True if the query is valid, False otherwise.
    """
    try:
        parsed = sqlparse.parse(query)
        if not parsed:
            return False
        return True
    except Exception as e:
        print(f"❌ SQL validation failed: {e}")
        return False

def get_contextual_sql_query(transcribed_text):
    """
    Generates an SQL query based on the transcribed text and previous context.

    Args:
        transcribed_text (str): The text to convert into an SQL query.

    Returns:
        str: The generated SQL query, or None if an error occurs.
    """
    # Get the previous query from context
    previous_query = query_context.get_context()

    # Prepare the system and user messages
    messages = [
        {"role": "system", "content": "You're a data scientist helping with SQL queries. Respond only with the SQL query, no explanations."}
    ]

    # Add previous query as context if it exists
    if previous_query:
        messages.append({"role": "user", "content": f"Previous query: {previous_query}"})
        messages.append({"role": "user", "content": f"New request: {transcribed_text}"})
    else:
        messages.append({"role": "user", "content": transcribed_text})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        sql_query = response.choices[0].message.content.strip()

        # Validate the SQL query
        if validate_sql(sql_query):
            # Update the context with the new query
            query_context.update_context(sql_query)
            return sql_query
        else:
            print("⚠️ Generated SQL query is invalid.")
            return None
    except Exception as e:
        print(f"❌ SQL generation failed: {e}")
        return None

if __name__ == "__main__":
    while True:
        # Step 1: Get voice command
        transcribed_text = get_voice_command(timeout=5)
        print("📝 Transcribed Text:", transcribed_text)

        if transcribed_text:
            # Step 2: Generate SQL query with context
            sql_query = get_contextual_sql_query(transcribed_text)
            if sql_query:
                print("💾 Generated SQL Query:\n", sql_query)
            else:
                print("⚠️ Failed to generate SQL query.")
        else:
            print("⚠️ No voice input detected.")

        # Ask the user if they want to continue
        while True:
            continue_query = input("Do you want to continue? (y/n): ").strip().lower()
            if continue_query in ["y", "n"]:
                break
            else:
                print("⚠️ Please enter 'y' or 'n'.")

        if continue_query == "n":
            print("👋 Exiting...")
            break
