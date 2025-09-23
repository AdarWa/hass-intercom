## Usage

1. Start the host server:
   ```bash
   python3 server.py --host 127.0.0.1 --port 8765
   ```

2. Launch the embedded intercom client. By default it generates a tone as outbound audio and plays inbound audio through the system output:
   ```bash
   python3 intercom_client.py --host 127.0.0.1 --port 8765 --client-id intercom-1
   ```

   Optional flags:
   - `--source-file /path/to/file.wav` to stream audio from a WAV file instead of the tone generator.
   - `--mic` (optionally with `--mic-device`) to capture audio from a connected microphone.
   - `--speaker-device hw:1,0` to play inbound audio through a specific PortAudio output device.
   - `--silence` to transmit silence.
   - `--sink-file /tmp/intercom_inbound.wav` to persist inbound audio instead of playing it.
   - `--mute` to disable playback.

3. Launch a Home Assistant client. It requests bidirectional audio automatically and plays inbound audio while streaming the configured source back to the intercom:
   ```bash
   python3 ha_client.py --host 127.0.0.1 --port 8765 --client-id ha-client-1
   ```

   Useful options:
   - `--tone` to transmit a tone to the intercom.
   - `--source-file /path/to/file.wav` to send a WAV file loop.
   - `--mic` (optionally with `--mic-device`) to send live audio from the default microphone.
   - `--speaker-device hw:1,0` to direct playback to a specific PortAudio output device.
   - `--sink-file /tmp/ha_inbound.wav` to store audio instead of playing it.
   - `--no-auto-start` to connect without immediately requesting audio.

Both clients honour sample rate, channel count, and frame duration settings (`--sample-rate`, `--channels`, `--frame-ms`) and can write or discard audio when the system output is unavailable. Live microphone capture and selecting a non-default speaker device require the optional `sounddevice` dependency with a working PortAudio stack.
