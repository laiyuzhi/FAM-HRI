import argparse

import aria.sdk as aria
import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    # 1. Create DeviceClient instance, setting the IP address if specified
    device_client = aria.DeviceClient()

    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    # 2. Connect to Aria: this relies on adb and assumes you have it installed on your path.
    # Use set_client_config to specify it manually otherwise or connect via IP address.
    device = device_client.connect()

    # 3. Retrieve streaming_manager
    streaming_manager = device.streaming_manager

    # 4. Use custom config for streaming: use profile12 (no audio) and use ephemeral certs

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    # Note: by default streaming uses Wifi
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    streaming_config.security_options.use_ephemeral_certs = True

    streaming_manager.streaming_config = streaming_config

    print(
        f"Starting streaming over {args.streaming_interface} using {args.profile_name}"
    )
    streaming_manager.start_streaming()
    # Duration of time: 600s
    time.sleep(600)
    print('time out stop streaming')
    streaming_manager.stop_streaming()
    device_client.disconnect(device)
if __name__ == "__main__":
    main()
