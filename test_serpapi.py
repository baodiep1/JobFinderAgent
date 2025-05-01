# Try different import approaches
try:
    from serpapi import GoogleSearch
    print("Successfully imported GoogleSearch from serpapi")
except ImportError as e:
    print(f"Error importing from serpapi: {e}")

try:
    import serpapi
    print(f"Successfully imported serpapi module: {serpapi}")
    print(f"Available attributes: {dir(serpapi)}")
except ImportError as e:
    print(f"Error importing serpapi module: {e}")

# Show installed packages
import subprocess
print("\nInstalled packages:")
result = subprocess.run(["pip", "list"], capture_output=True, text=True)
print(result.stdout)