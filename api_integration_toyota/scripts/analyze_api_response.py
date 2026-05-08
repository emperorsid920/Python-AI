#!/usr/bin/env python3
import json
import os


def analyze_response():
    """Analyze saved API response"""

    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    response_file = os.path.join(project_root, 'data', 'sample_api_response.json')

    print("\n🔍 Analyzing API Response...")
    print("=" * 60)

    with open(response_file, 'r') as f:
        data = json.load(f)

    # Top-level keys
    print("\n📋 TOP-LEVEL KEYS:")
    for key in data.keys():
        print(f"   • {key}")

    # Analyze appointmentsMap
    if 'appointmentsMap' in data:
        print("\n📅 APPOINTMENTS MAP STRUCTURE:")

        first_date = list(data['appointmentsMap'].keys())[0]
        first_slots = data['appointmentsMap'][first_date]

        print(f"   Dates returned: {len(data['appointmentsMap'])}")
        print(f"   Slots per day: {len(first_slots)}")
        print(f"\n   SAMPLE SLOT STRUCTURE (first available slot):")

        # Find first available slot
        sample_slot = None
        for slot in first_slots:
            if slot.get('isAvailable') == True:
                sample_slot = slot
                break

        if sample_slot:
            for key, value in sample_slot.items():
                print(f"      • {key}: {value} (type: {type(value).__name__})")

    # Time analysis
    print("\n⏰ TIME ANALYSIS:")
    all_times = set()
    for date, slots in data['appointmentsMap'].items():
        for slot in slots:
            if slot.get('isAvailable'):
                all_times.add(slot['formattedTime'])

    sorted_times = sorted(list(all_times))
    print(f"   Earliest slot: {sorted_times[0]}")
    print(f"   Latest slot: {sorted_times[-1]}")
    print(f"   Total unique times: {len(sorted_times)}")

    # Calculate intervals
    if len(sorted_times) > 1:
        # Convert to minutes
        def time_to_minutes(t):
            h, m = t.split(':')
            return int(h) * 60 + int(m)

        intervals = []
        for i in range(len(sorted_times) - 1):
            diff = time_to_minutes(sorted_times[i + 1]) - time_to_minutes(sorted_times[i])
            intervals.append(diff)

        unique_intervals = set(intervals)
        print(f"   Interval(s): {unique_intervals} minutes")

    # Day-of-week analysis
    print("\n📆 DAY-OF-WEEK LABELS:")
    days_seen = set()
    for date, slots in data['appointmentsMap'].items():
        for slot in slots:
            if slot.get('day'):
                days_seen.add(slot['day'])
                break
    print(f"   Days: {', '.join(sorted(days_seen))}")

    # Time-of-day analysis
    print("\n🌅 TIME-OF-DAY LABELS:")
    tod_seen = set()
    tod_by_time = {}
    for date, slots in data['appointmentsMap'].items():
        for slot in slots:
            if slot.get('tod'):
                tod_seen.add(slot['tod'])
                time = slot.get('formattedTime')
                tod = slot.get('tod')
                if time and tod:
                    if tod not in tod_by_time:
                        tod_by_time[tod] = []
                    tod_by_time[tod].append(time)

    print(f"   Periods: {', '.join(sorted(tod_seen))}")
    for tod, times in sorted(tod_by_time.items()):
        times = sorted(set(times))
        print(f"   • {tod}: {times[0]} to {times[-1]}")

    # Availability ratio
    print("\n📊 AVAILABILITY STATISTICS:")
    total_slots = 0
    available_slots = 0
    for date, slots in data['appointmentsMap'].items():
        total_slots += len(slots)
        available_slots += sum(1 for s in slots if s.get('isAvailable') == True)

    print(f"   Total slots checked: {total_slots}")
    print(f"   Available: {available_slots} ({available_slots / total_slots * 100:.1f}%)")
    print(f"   Booked: {total_slots - available_slots} ({(total_slots - available_slots) / total_slots * 100:.1f}%)")

    # Check for fields that might be missing
    print("\n🔎 FIELD CONSISTENCY CHECK:")
    fields_by_availability = {'available': {}, 'booked': {}}

    for date, slots in data['appointmentsMap'].items():
        for slot in slots:
            is_avail = slot.get('isAvailable') == True
            category = 'available' if is_avail else 'booked'

            for field in slot.keys():
                fields_by_availability[category][field] = fields_by_availability[category].get(field, 0) + 1

    print("\n   Fields in AVAILABLE slots:")
    for field in sorted(fields_by_availability['available'].keys()):
        count = fields_by_availability['available'][field]
        print(f"      • {field}: {count} occurrences")

    print("\n   Fields in BOOKED slots:")
    for field in sorted(fields_by_availability['booked'].keys()):
        count = fields_by_availability['booked'][field]
        print(f"      • {field}: {count} occurrences")

    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    analyze_response()