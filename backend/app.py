from flask import Flask, request, jsonify
import pandas as pd
from difflib import get_close_matches
from flask_cors import CORS
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:5500"])

load_dotenv()

# Initialize dataframes as empty
index_df = pd.DataFrame()
constitution_df = pd.DataFrame()

# ---- LOAD YOUR DATA ----
try:
    # Get the current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    print(f"Looking for data in: {data_dir}")
    
    # Check if data directory exists
    if os.path.exists(data_dir):
        print(f"Files in data directory: {os.listdir(data_dir)}")
        
        # Load index.csv - it has different column names
        index_path = os.path.join(data_dir, "index.csv")
        if os.path.exists(index_path):
            index_df = pd.read_csv(index_path)
            # Your index.csv has: "Parts of the Indian Constitution", "Subject Mentioned in the Part", "Articles in Indian Constitution"
            print(f"Index columns: {index_df.columns.tolist()}")
        else:
            print("index.csv not found")
        
        # Load constitution.csv - it has "Articles" column
        constitution_path = os.path.join(data_dir, "Constitution Of India.csv")
        if os.path.exists(constitution_path):
            constitution_df = pd.read_csv(constitution_path)
            # Your constitution.csv has "Articles" column
            print(f"Constitution columns: {constitution_df.columns.tolist()}")
        else:
            print("Constitution Of India.csv not found")
            
        print("Data loading completed!")
        
    else:
        print("Data directory not found!")

except Exception as e:
    print(f"Error loading CSV files: {e}")

# ---- FIND RELEVANT CLAUSE FUNCTION ----
def find_relevant_clause(query):
    try:
        print(f"Searching for query: {query}")
        
        if index_df.empty or constitution_df.empty:
            return "Constitutional database not available - using general knowledge."
            
        # Since your CSV structures don't match perfectly, use general approach
        if not constitution_df.empty:
            # Get some sample constitutional content
            sample_content = constitution_df.iloc[:3]  # First 3 rows
            if "Articles" in constitution_df.columns:
                content_list = sample_content["Articles"].dropna().tolist()
            else:
                # Try first column if "Articles" doesn't exist
                content_list = sample_content.iloc[:, 0].dropna().tolist()
                
            context = "\n\n".join([str(content) for content in content_list])
            return f"Constitutional context:\n\n{context}"
        else:
            return "No constitutional data available."
            
    except Exception as e:
        print(f"Error in find_relevant_clause: {e}")
        return f"Error retrieving legal data: {str(e)}"

# ---- IMPROVED CHAT ROUTE ----
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200
    
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Please enter a valid question."})

    print(f"Received question: {user_input}")

    # Retrieve relevant part of constitution
    context = find_relevant_clause(user_input)

    # Initialize Groq client
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("GROQ_API_KEY not found in environment variables")
        return jsonify({"response": "Service configuration error. Please check API key."})

    try:
        # Generate response with Groq
        model = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0
        )
        
        prompt = f"""
        You are a legal assistant chatbot that specializes in the Constitution of India.
        
        User Question: {user_input}

        Constitutional Context Available:
        {context}

        Instructions:
        - Provide accurate information about Indian constitutional law
        - If specific articles are mentioned in the context, reference them
        - If the context is general, provide comprehensive information based on your knowledge
        - Always include a clear disclaimer that this is not professional legal advice
        - Suggest consulting a qualified lawyer for specific legal matters
        - Keep the response educational, clear, and helpful
        """
        
        print("Sending request to Groq API...")
        response = model.invoke(prompt)
        print("Successfully received response from Groq API")
        
        return jsonify({"response": response.content})
    
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return jsonify({"response": "I'm experiencing technical difficulties. Please try again in a moment."})

# Test route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "LegalBot Server is running", 
        "data_loaded": not index_df.empty,
        "constitution_loaded": not constitution_df.empty
    })

# ---- RUN SERVER ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting LegalBot server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)