# Home Assistant Intercom System

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)
![Asyncio](https://img.shields.io/badge/asyncio-event%20driven-44cc11)
![Docker Compose](https://img.shields.io/badge/docker-compose-blue)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support-yellow)](https://www.buymeacoffee.com/AdarWa)

Asyncio-based host, embedded intercom client, and Home Assistant sample client for building a centrally managed smart intercom experience.

## Architecture
- **Host server** (`server.py`): Authenticates clients, enforces routing rules, and forwards JSON messages between agents.
- **Embedded intercom client** (`intercom_client.py`): Represents the physical door station, publishes events, and executes commands from the host.
- **Home Assistant clients** (`ha_client.py`): Automations or dashboards that subscribe to intercom events and issue commands via the host.
- **Android background client**: Optional mobile companion that listens for Home Assistant events and opens the microphone automatically. Source available at [HassIntercomAndroidClient](https://github.com/AdarWa/HassIntercomAndroidClient).

The server is the single source of truth: Home Assistant clients never communicate with each other directly, and every audio frame or command flows through the host. Additional background is documented in `AGENTS.md`.

## Features
- Event-driven asyncio host with strict protocol validation and per-client session tracking.
- Bidirectional audio streams with PCM frames exchanged as newline-delimited JSON.
- Optional Docker workflow that launches the host and sample clients.
- Reusable protocol helpers and audio utilities for integrating custom hardware.
- Android companion service that blends intercom controls into Home Assistant automations.

## Requirements
- Python 3.12+
- PortAudio/SoundDevice dependencies (for Debian/Ubuntu: `sudo apt install libportaudio2 portaudio19-dev libsndfile1`)
- Working microphone and speakers when running the audio clients locally

## Local Setup
1. Create a virtual environment using `uv` and activate it.
   ```bash
   uv sync
   source .venv/bin/activate
   ```

## Running Components
### Host server
```bash
python -m server --host 0.0.0.0 --port 8765
```

### Embedded intercom client
```bash
python -m intercom_client --host 127.0.0.1 --port 8765 --client-id intercom-1 --mic
```
`--mic` is required because the sample implementation captures audio from the default microphone. Use `--sample-rate`, `--channels`, or `--frame-ms` to match your hardware if needed.

### Home Assistant sample client
```bash
python -m ha_client --host 127.0.0.1 --port 8765 --client-id ha-client-1 --mic
```
Pass `--no-auto-start` to connect without immediately requesting audio, or `--stream-id` to request a specific stream identifier.

### Android background client
The Android service client subscribes to Home Assistant events, keeps an intercom session alive in the background, and activates the device microphone automatically so the experience feels native inside Home Assistant. Install or build it from [HassIntercomAndroidClient](https://github.com/AdarWa/HassIntercomAndroidClient) and configure it to point at your running host.

## Docker Compose Quickstart
The repository ships with a compose file that can run the host and the sample clients:
```bash
docker compose --profile server up
```
Start additional profiles as needed:
```bash
# Launch the embedded intercom sample client
docker compose --profile intercom up

# Launch the Home Assistant sample client
docker compose --profile ha up
```
Use separate terminals when running multiple services simultaneously.

## Protocol Reference
The complete message flow—registration, commands, responses, events, and audio frames—is described in `PROTOCOL.md`. Home Assistant integrations can reuse `protocol_client.py` for handling JSON framing and registration.

## Audio Notes
The sample clients depend on the system microphone and speakers. If you experience echo or feedback during two-way audio, review `enable-echo-cancelling.sh`, which shows how to enable the PulseAudio echo cancellation module on Linux hosts (run it outside Docker and only if you understand the changes).

## Support This Project
If the intercom system is helping your setup, consider supporting future development.

<p>
  <a href="https://www.buymeacoffee.com/AdarWa" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="42">
  </a>
</p>
