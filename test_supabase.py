import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_KEY")

print(f"URL: {url}")
print(f"Key: {key[:10]}...")  # print first 10 chars for safety

if not url or not key:
    print("❌ Missing environment variables.")
    exit(1)

supabase = create_client(url, key)

try:
    # Try to read from a dummy table to test the connection
    result = supabase.table('config').select('*').limit(1).execute()
    print("✅ Connection successful.")
except Exception as e:
    print(f"❌ Connection failed: {e}")