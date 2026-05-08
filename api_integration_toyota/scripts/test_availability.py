#!/usr/bin/env python3
import requests
import json
import os
from datetime import datetime, timedelta


def load_token():
    """Load saved auth token"""
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    token_file = os.path.join(project_root, 'data', 'auth_token.txt')

    try:
        with open(token_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ ERROR: Token file not found at {token_file}")
        print("   Run test_auth.py first!")
        return None


def test_availability_api(token):
    """Test availability API with tomorrow's date"""
    print("\n📅 Testing Availability API...")
    print("-" * 60)

    # Calculate tomorrow's date
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")

    print(f"Requesting availability for: {tomorrow}, {day_after}")

    url = "https://api.dev.lisaagent.com/api/manual-appointment/inb-nlp/mb3-availability"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "enterpriseid": "75"
    }

    params = {
        "future-service-interaction-id": "1845765"
    }

    payload = {
        "fsiId": 1845765,
        "singleDateSelected": False,
        "dateList": True,
        "dateRange": [tomorrow, day_after],
        "selectedDateRange": [tomorrow, day_after],
        "allDayTag": True,
        "morningTag": False,
        "morningEarlyLateTag": "",
        "afternoonTag": False,
        "afternoonEarlyLateTag": "",
        "eveningTag": False,
        "eveningEarlyLateTag": "",
        "timeRangeSelectionCheckMark": False,
        "atAroundTag": False,
        "beforeTag": False,
        "afterTag": False,
        "betweenTag": False,
        "dealerCode": "01~01~645331991962816",
        "opCode": "00TOZ-OTH",
        "year": "2022",
        "model": "MODEL",
        "make": "MAKE",
        "defaultProcess": False,
        "timeOfTheDayCheckMark": False,
        "transportType": "D",
        "notes": "",
        "advisorId": None
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            params=params,
            json=payload,
            timeout=15
        )
        response.raise_for_status()

        data = response.json()

        # Get the project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.join(project_root, 'data')

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Save full response
        response_file = os.path.join(data_dir, 'sample_api_response.json')
        with open(response_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Full response saved to: {response_file}")

        # Analyze response
        if 'appointmentsMap' in data:
            print("\n✅ SUCCESS! Appointments data received:")
            print(f"   Number of dates: {len(data['appointmentsMap'])}")

            for date, slots in data['appointmentsMap'].items():
                available_count = sum(1 for s in slots if s.get('isAvailable') == True)
                total_count = len(slots)

                # Get day name from first available slot
                day_name = "Unknown"
                for slot in slots:
                    if slot.get('day'):
                        day_name = slot['day']
                        break

                print(f"\n   📅 {day_name}, {date}:")
                print(f"      Total slots: {total_count}")
                print(f"      Available: {available_count}")
                print(f"      Booked: {total_count - available_count}")

                # Show first 5 available slots
                available_slots = [s for s in slots if s.get('isAvailable') == True][:5]
                if available_slots:
                    times = [s['formattedTime'] for s in available_slots]
                    print(f"      First 5 available: {', '.join(times)}")

            return data
        else:
            print("❌ ERROR: No appointmentsMap in response")
            print(f"   Response keys: {list(data.keys())}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Status code: {e.response.status_code}")
            print(f"   Response: {e.response.text[:200]}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON response")
        print(f"   {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 AVAILABILITY API TEST")
    print("=" * 60)

    # Load token
    token = load_token()
    if not token:
        exit(1)

    print(f"✓ Token loaded (length: {len(token)} chars)")

    # Test availability
    result = test_availability_api(token)

    if result:
        print("\n" + "=" * 60)
        print("✅ AVAILABILITY API TEST PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ AVAILABILITY API TEST FAILED")
        print("=" * 60)