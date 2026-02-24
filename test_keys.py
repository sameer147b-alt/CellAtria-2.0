import os
from dotenv import load_dotenv
from groq import Groq
from e2b_code_interpreter import Sandbox

# Force reload the .env file
load_dotenv(override=True)

def validate_keys():
    print("🔍 Starting API Key Validation...")
    
    # 1. Validate Groq
    try:
        # FIXED SPELLING: GROQ_API_KEY
        key = os.getenv("GROQ_API_KEY")
        if not key:
            print("❌ Groq API: FAILED - Key not found! Check .env spelling.")
        else:
            client = Groq(api_key=key)
            client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model="llama-3.3-70b-versatile",
            )
            print("✅ Groq API: SUCCESS")
    except Exception as e:
        print(f"❌ Groq API: FAILED - {e}")

    # 2. Validate E2B
    try:
        # The updated connection for the new E2B library
        sb = Sandbox.create(template=os.getenv("E2B_TEMPLATE_ID"))
        print(f"✅ E2B API: SUCCESS")
        sb.kill()
    except Exception as e:
        print(f"❌ E2B API: FAILED - {e}")

if __name__ == "__main__":
    validate_keys()