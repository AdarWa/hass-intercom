# Agents

## System Overview
The intercom system is composed of a central host that coordinates traffic between the physical intercom hardware and any Home Assistant integrations. The host exposes a server interface for remote clients while also running an embedded client that represents the outdoor intercom endpoint.

## Agent Roles

### Host Server
- Runs the authoritative protocol server and maintains long-lived sessions with connected agents.
- Authenticates and routes messages between Home Assistant clients and the embedded intercom client.
- Enforces directional messaging rules so clients never talk to each other directly.

### Embedded Intercom Client
- Lives alongside the host server and represents the physical outdoor intercom hardware.
- Initiates a client session to the host server to publish events such as doorbell presses and to receive commands (e.g., open door, start audio stream).
- Handles translation between protocol messages and the hardware-specific control layer.

### Home Assistant Clients
- External clients that connect from Home Assistant automations or dashboards.
- Use the host server API to subscribe to intercom events and issue commands that should reach the outdoor intercom.
- Must never exchange messages with other Home Assistant clients; every interaction goes through the host server.

## Communication Flow
1. The embedded intercom client establishes and maintains a session with the host server.
2. Home Assistant clients connect to the host server when they need to send commands or subscribe to intercom events.
3. The host server relays all permitted messages between Home Assistant clients and the embedded intercom client, applying validation and authorization policies.
4. Responses from the intercom flow back through the host server to the originating Home Assistant client.

This separation ensures the host remains the single source of truth and prevents any lateral communication between Home Assistant clients.
