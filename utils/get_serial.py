from pyorbbecsdk import Context

ctx = Context()
device_list = ctx.query_devices()

for i in range(device_list.get_count()):
    serial = device_list.get_device_serial_number_by_index(i)
    print(f"Device {i}: serial = {serial}")
