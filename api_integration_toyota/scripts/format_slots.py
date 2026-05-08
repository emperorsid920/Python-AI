#!/usr/bin/env python3
import json
import os


def format_time_for_customer(formatted_time, am_pm_flag):
    """
    Convert API time format to customer-friendly format

    Args:
        formatted_time: "06:00", "01:00", etc.
        am_pm_flag: 0 for AM, 1 for PM

    Returns:
        "6am", "1pm", etc.
    """
    hour, minute = formatted_time.split(':')
    hour = int(hour)

    # Handle midnight/noon edge cases
    if hour == 0:
        hour = 12

    suffix = 'am' if am_pm_flag == 0 else 'pm'

    # Only show minutes if not on the hour
    if minute == '00':
        return f"{hour}{suffix}"
    else:
        return f"{hour}:{minute}{suffix}"


def get_available_slots_for_date(appointments_map, date):
    """
    Get formatted available slots for a specific date

    Args:
        appointments_map: The appointmentsMap from API response
        date: "05/07/2026"

    Returns:
        dict with day_name and list of formatted times
    """
    slots = appointments_map.get(date, [])

    # Filter available slots only
    available = [s for s in slots if s.get('isAvailable') == True]

    if not available:
        return {
            'day_name': None,
            'date': date,
            'slots': [],
            'message': "fully booked"
        }

    # Get day name from first slot
    day_name = available[0].get('day', 'Unknown')

    # Format times
    formatted_times = []
    for slot in available:
        time_str = format_time_for_customer(
            slot['formattedTime'],
            slot['formattedTimeAm_Pm']
        )
        formatted_times.append(time_str)

    return {
        'day_name': day_name,
        'date': date,
        'slots': formatted_times,
        'total_available': len(formatted_times)
    }


def format_for_prompt(appointments_map, max_slots_shown=5):
    """
    Format API response for prompt/customer presentation

    Args:
        appointments_map: The appointmentsMap from API response
        max_slots_shown: How many slots to show per day

    Returns:
        Formatted string for customer
    """
    results = []

    for date in sorted(appointments_map.keys()):
        slot_info = get_available_slots_for_date(appointments_map, date)

        if slot_info['slots']:
            day_name = slot_info['day_name']
            # Show first N slots
            shown_slots = slot_info['slots'][:max_slots_shown]
            more_count = len(slot_info['slots']) - max_slots_shown

            slots_text = ', '.join(shown_slots)
            if more_count > 0:
                slots_text += f" (and {more_count} more)"

            results.append(f"{day_name}, {date}: {slots_text}")
        else:
            results.append(f"{date}: Fully booked")

    return '\n'.join(results)


def demo_formatting():
    """Demo the formatting with saved API response"""

    # Load saved response
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    response_file = os.path.join(project_root, 'data', 'sample_api_response.json')

    with open(response_file, 'r') as f:
        data = json.load(f)

    appointments_map = data.get('appointmentsMap', {})

    print("=" * 60)
    print("🎨 SLOT FORMATTING DEMO")
    print("=" * 60)

    # Show raw API format
    print("\n📥 RAW API FORMAT (first 3 available slots):")
    first_date = list(appointments_map.keys())[0]
    first_slots = appointments_map[first_date]
    available = [s for s in first_slots if s.get('isAvailable') == True][:3]

    for slot in available:
        print(
            f"   • {slot['formattedTime']} (AM/PM flag: {slot['formattedTimeAm_Pm']}) → {slot.get('day', 'N/A')} {slot.get('tod', 'N/A')}")

    # Show formatted version
    print("\n📤 CUSTOMER-FRIENDLY FORMAT:")
    for slot in available:
        formatted = format_time_for_customer(slot['formattedTime'], slot['formattedTimeAm_Pm'])
        print(f"   • {formatted}")

    # Show full date formatting
    print("\n💬 WHAT PROMPT WOULD SAY TO CUSTOMER:")
    print("-" * 60)
    customer_message = format_for_prompt(appointments_map, max_slots_shown=5)
    print(customer_message)
    print("-" * 60)

    # Test individual date lookup
    print("\n🔍 TESTING DATE LOOKUP:")
    for date in sorted(appointments_map.keys()):
        info = get_available_slots_for_date(appointments_map, date)
        print(f"\n   {date}:")
        print(f"   • Day: {info['day_name']}")
        print(f"   • Available slots: {info['total_available']}")
        print(f"   • First 5: {', '.join(info['slots'][:5])}")

    print("\n" + "=" * 60)
    print("✅ FORMATTING DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo_formatting()