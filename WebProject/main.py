# Import the create_app function from the website package
from website import create_app

# Create the Flask application using the create_app function
app = create_app()

# Check if the script is being run directly
if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)