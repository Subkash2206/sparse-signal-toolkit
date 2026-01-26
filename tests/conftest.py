import sys
import os

# Add the src directory to the python path so we can import our modules
# This allows 'from dft import dft' to work even if we run pytest from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
