# Import necessary libraries
from langchain_ollama.llms import OllamaLLM  # For using Ollama language models
from langchain_core.prompts import ChatPromptTemplate  # For creating prompt templates
from vector import retriever  # Import the retriever we created in vector.py

# Initialize the language model
print("Loading model...")
model = OllamaLLM(model="llama3.2")  # Using Llama 3.2 for generating responses
print("Model loaded successfully!")

# Define the prompt template that will be used for all questions
# This template structures how the AI will receive and process information
template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

# Create a prompt template object that will format our inputs
prompt = ChatPromptTemplate.from_template(template)

# Create a chain that combines the prompt and the model
# This chain will process prompts and generate responses
chain = prompt | model

print("System initialized, ready for questions...")

# Main conversation loop
while True:
    print("\n\n-------------------------------")
    try:
        # Get user input
        question = input("Ask your question (q to quit): ")
        print(f"\nYou asked: {question}")

        # Check if user wants to quit
        if question == "q":
            print("Exiting...")
            break

        # Use the retriever to find relevant restaurant reviews
        print("Searching for relevant documents...")
        docs = retriever.invoke(question)

        print(f"Found {len(docs)} relevant documents")

        # Extract the text content from the retrieved documents
        reviews = "\n".join([doc.page_content for doc in docs])

        # Generate a response using the language model
        print("Generating response...")
        result = chain.invoke({"reviews": reviews, "question": question})

        # Display the response
        print("\nResponse:")
        print(result)

    # Error handling to catch and display any issues
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        traceback.print_exc()