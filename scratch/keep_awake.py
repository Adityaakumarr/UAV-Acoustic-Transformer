import ctypes
import time
import sys

# Windows Constants to prevent sleep
# ES_CONTINUOUS = 0x80000000
# ES_SYSTEM_REQUIRED = 0x00000001
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def keep_system_awake():
    print("Preventing Windows from going to sleep...")
    # Tell Windows that the system is busy and should not sleep
    state = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    
    if state == 0:
        print("Warning: Could not set execution state.")
        sys.exit(1)
        
    print("System will now stay awake. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60) # Wake up every minute just in case
    except KeyboardInterrupt:
        print("Restoring normal sleep behavior...")
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

if __name__ == "__main__":
    keep_system_awake()
