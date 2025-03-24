#!/bin/bash

# Step 1: Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv myenv

# Step 2: Activate the virtual environment
echo "Activating virtual environment..."
source myenv/bin/activate  # On Windows, use myenv\Scripts\activate

# Step 3: Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 4: Inform the user that the setup is complete
echo "Setup complete! To activate the environment, use:"
echo "source myenv/bin/activate  # On Windows: myenv\\Scripts\\activate"

