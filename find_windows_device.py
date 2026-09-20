"""
SmartQC — find Windows/Snapdragon X laptop devices on Qualcomm AI Hub.
 
Run:
    python find_windows_device.py
"""
 
import qai_hub as hub
 
 
def main():
    devices = hub.get_devices()
    print(f"Total devices: {len(devices)}\n")
 
    matches = []
    for d in devices:
        name_lower = d.name.lower()
        os_lower = str(d.os).lower()
        attrs = " ".join(d.attributes).lower()
        if (
            "x elite" in name_lower
            or "snapdragon x" in name_lower
            or "windows" in os_lower
            or "os:windows" in attrs
        ):
            matches.append(d)
 
    if not matches:
        print("No Windows/Snapdragon X devices found by filter. Showing all unique OS values instead:")
        oses = sorted(set(str(d.os) for d in devices))
        for o in oses:
            print(f"  - {o}")
        return
 
    print("Matching Windows/Snapdragon X devices:\n")
    for d in matches:
        print(f"{d.name}  |  os: {d.os}")
 
 
if __name__ == "__main__":
    main()
 