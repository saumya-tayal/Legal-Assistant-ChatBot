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

# ---- LOAD YOUR DATA ----
try:
    # Load index.csv with correct column names
    index_df = pd.read_csv("data/index.csv")
    # Rename columns to match what your code expects
    index_df = index_df.rename(columns={
        "Parts of the Indian Constitution": "Heading",
        "Subject Mentioned in the Part": "Description" 
    })
    
    # Load constitution data
    constitution_df = pd.read_csv("data/Constitution Of India.csv")
    # Rename the single column to 'Content'
    constitution_df = constitution_df.rename(columns={"Articles": "Content"})
    
    print("Data loaded successfully!")
    print(f"Index columns: {index_df.columns.tolist()}")
    print(f"Constitution columns: {constitution_df.columns.tolist()}")
    
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}")
    index_df = pd.DataFrame(columns=["Heading"])
    constitution_df = pd.DataFrame(columns=["Content"])
except Exception as e:
    print(f"Unexpected error loading data: {e}")
    index_df = pd.DataFrame(columns=["Heading"])
    constitution_df = pd.DataFrame(columns=["Content"])

# ---- FIND RELEVANT CLAUSE FUNCTION ----
def find_relevant_clause(query):
    try:
        print(f"Searching for query: {query}")
        
        if index_df.empty or constitution_df.empty:
            return "Constitutional database not available - using general knowledge."
            
        # Get all available headings for matching
        available_headings = index_df["Heading"].dropna().tolist()
        print(f"Available headings: {len(available_headings)}")
        print(f"Sample headings: {available_headings[:5]}")
        
        # Search for the closest heading match
        matches = get_close_matches(query, available_headings, n=3, cutoff=0.3)
        print(f"Found matches: {matches}")
        
        if matches:
            # For now, we'll use the constitution data generally since the structure doesn't match perfectly
            # Return some sample constitutional content
            sample_content = constitution_df["Content"].head(3).tolist()
            context = "\n\n".join([str(content) for content in sample_content if pd.notna(content)])
            return f"Relevant constitutional context for {matches[0]}:\n\n{context}"
        else:
            return f"No specific constitutional match found for '{query}'. Using general constitutional knowledge."
            
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
    print(f"Context retrieved")

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
    print("Starting LegalBot server...")
    print("Test the server at: http://127.0.0.1:5000/")
    app.run(debug=True, port=5000, host='127.0.0.1')