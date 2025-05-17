# Import Libraries
import google.generativeai as genai
from dotenv import load_dotenv
import os
from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from markupsafe import Markup
import markdown
from langchain_huggingface import HuggingFaceEmbeddings as Embeddings
from langchain_chroma import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
import azure.cognitiveservices.speech as speechsdk


class Maya:
    def __init__(self, maya, model="gemini-1.5-flash", temperature=0.7, existing_content=None):
        load_dotenv()
        self.key = os.getenv("GEMINI_KEY")
        genai.configure(api_key=self.key)
        self.model = model
        self.maya = maya
        self.temperature = temperature

        self.context = [
            {"role": "user", "parts": [{"text": maya}]},
            {"role": "model", "parts": [{"text": "Hi there! I'm Maya. What's on your mind today?"}]}
        ]

        self.context.extend(existing_content)
        

    def _initialize_retriever(self):
        embeddings = Embeddings(model_name="all-MiniLM-L6-v2")
        
        # Load existing Chroma DB
        physics_db = Chroma(persist_directory="Data/halliday_physics_db", embedding_function=embeddings)
        
        retriever = VectorStoreRetriever(
            vectorstore=physics_db,
            search_type='mmr',
            search_kwargs={'k': 3, 'lambda_mult': 0.5}  # lambda controls diversity vs similarity
            # 0.0 pure diversity, 1.0 pure similarity
        )

        return retriever

    def get_completion(self, prompt):

        if 'tutor' in self.maya.lower():
            try:
                relevant_docs = self._initialize_retriever()
                docs = relevant_docs.invoke(prompt)
                retrieval_context = '\n\nRetrieval Physics Concepts:\n'
                retrieval_context += '\n'.join(doc.page_content for doc in docs)
                enhanced_prompt = f"{retrieval_context}\n\nQuestion: {prompt}"
            except Exception as e:
                print(f"Retrieval Error: {e}")
                enhanced_prompt = prompt
        
        else:
            enhanced_prompt = prompt

        self.context.append({'role': 'user', 'parts': [{'text': enhanced_prompt}]})
        
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(
            self.context, generation_config = {'temperature': self.temperature})

        self.context.append({'role': 'model', 'parts': [{'text': response.text}]})
        return response.text
    

app = Flask(__name__, template_folder='Templates')

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(basedir, 'instance', 'database.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
app.secret_key = os.urandom(24)  # Use a random secret key for session management

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.Text, nullable=False)
    maya_response = db.Column(db.Text, nullable=False)


@app.route("/", methods=['GET', 'POST'])
def index():
    selected_value = session.get('selected_value', '1')
    
    if request.method == 'POST':
        # Forward to the message handler
        return send_message()
        
    return render_template("index.html", chat_history=ChatHistory.query.all(), selected_value=selected_value)


@app.route("/send_message", methods=['GET', 'POST'])
def send_message():
    if request.method == 'GET':
        # Redirect GET requests to the index page
        return redirect(url_for('index'))
        
    user_message = request.form.get('message', '')
    selected_value = request.form.get('chat-select', '1')
    session['selected_value'] = selected_value      # The type of session is a dictionary

    chat_log = ChatHistory.query.all()
    context = []

    for message in chat_log:
        context.append({'role': 'user', 'parts': [{'text': message.user_message}]})
        context.append({'role': 'model', 'parts': [{'text': message.maya_response}]})
    
    if selected_value == '2':
        file_path = os.path.join(os.path.dirname(__file__), 'Data', 'physics_tutor.txt')
        print('MAYA PHYSICS BOT')
    else:
        file_path = os.path.join(os.path.dirname(__file__), 'Data', 'maya.txt')
        print('MAYA FRIENDLY BOT')

    with open(file_path, 'r') as f:
        maya_instructions = f.read()

    maya = Maya(maya_instructions, existing_content=context)

    if user_message:
        try:
            maya_response = maya.get_completion(user_message)

            new_message = ChatHistory(user_message=user_message, maya_response=maya_response)
            db.session.add(new_message)
            db.session.commit()
        
        except Exception as e:
            print(f"Error: {e}")
            flash("Sorry, there was an error processing your request.")
    
    return redirect(url_for('index'))


@app.route("/speak", methods=['POST'])
def speak():
    try:
        # Load Azure Speech credentials from environment variables
        load_dotenv()
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        speech_region = os.getenv("AZURE_SPEECH_REGION")
        
        if not speech_key or not speech_region:
            return jsonify({"error": "Azure Speech credentials are not configured"}), 500
        
        # Get text to synthesize
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        # Configure speech synthesis
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        
        speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"  # Female Voice
        
        # Configure audio output format
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
        
        # Create a speech synthesizer without audio output (we'll get the audio data)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        
        # Synthesize text to speech
        result = synthesizer.speak_text_async(text).get()
        
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Return the audio data as a base64-encoded string
            import base64
            audio_data_base64 = base64.b64encode(result.audio_data).decode('utf-8')
            return jsonify({
                "audioData": audio_data_base64,
                "format": "audio/mp3"
            })
        else:
            error_details = result.cancellation_details
            return jsonify({
                "error": f"Speech synthesis failed: {error_details.reason}",
                "details": error_details.error_details if error_details else "No details"
            }), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.template_filter('markdown')
def markdown_filter(text):
    return Markup(markdown.markdown(text))
    

if __name__ == "__main__":

    with app.app_context():
        db.create_all()
        ChatHistory.query.delete()  # Clear the database for fresh start
        db.session.commit()

    app.run(host = "0.0.0.0", debug = True)
