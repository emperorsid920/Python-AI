#!/usr/bin/env python3
import requests
import json
import os


def test_authentication():
    """Test API authentication"""
    print("🔐 Testing Authentication...")
    print("-" * 60)

    url = "https://api.dev.lisaagent.com/user/v2/login"

    payload = {
        "emailId": "test.inbound@autodept.biz",
        "password": "Post_Prod-6xC26^",
        "ipAddress": "127.0.0.1",
        "city": "Orlando",
        "region": "Florida",
        "country": "US",
        "coordinates": "28.5383,-81.3792",
        "providerOrg": "",
        "postal": "32801",
        "timezone": "America/New_York"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Try different possible token field names
        token = (data.get('token') or
                 data.get('jwtToken') or
                 data.get('accessToken') or
                 data.get('authToken'))

        if token:
            print("✅ SUCCESS! Token received:")
            print(f"   Token (first 50 chars): {token[:50]}...")
            print(f"   Full token length: {len(token)} characters")

            # Get the project root directory (go up one level from scripts/)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            data_dir = os.path.join(project_root, 'data')

            # Create data directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)

            # Save token to file
            token_file = os.path.join(data_dir, 'auth_token.txt')
            with open(token_file, 'w') as f:
                f.write(token)
            print(f"   ✓ Token saved to: {token_file}")

            return token
        else:
            print("❌ ERROR: No token in response")
            print(f"   Response keys: {list(data.keys())}")
            print(f"   Full response: {json.dumps(data, indent=2)}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON response")
        print(f"   Response text: {response.text}")
        return None


if __name__ == "__main__":
    token = test_authentication()

    if token:
        print("\n" + "=" * 60)
        print("✅ AUTHENTICATION TEST PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ AUTHENTICATION TEST FAILED")
        print("=" * 60)