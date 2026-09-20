"""
SmartQC — list all devices available on Qualcomm AI Hub.
 
Run:
    python list_devices.py
"""
 
import qai_hub as hub
 
 
def main():
    devices = hub.get_devices()
    print(f"Total devices: {len(devices)}\n")
    for d in devices:
        print(f"{d.name}  |  os: {d.os}  |  attributes: {d.attributes}")
 
 
if __name__ == "__main__":
    main()