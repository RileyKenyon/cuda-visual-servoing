# Plan: Replace ScrcpySink with Custom Socket-Based scrcpy Client

**TL;DR:** Replace the current ScrcpySink wrapper (which delegates to the embedded scrcpy library) with a custom C++ client that directly manages POSIX sockets for video and control streams. Implement in three phases: (1) raw socket prove-out, (2) codec metadata & frame header parsing, (3) protocol message serialization. Output H.264 packets to existing V4L2 sink infrastructure (matching current behavior). The existing FFmpeg dependencies and V4L2 output pipeline remain unchanged.

## Implementation Steps

### 1. Prepare Socket Infrastructure
- Modify [src/sinks/include/scrcpy_sink.hpp](src/sinks/include/scrcpy_sink.hpp) to replace scrcpy library references with native socket management
- Add socket classes: `ControlSocket` and `VideoSocket` (POSIX-based)
- Define connection manager to establish both sockets in correct order (video first, then control)
- Add thread-safe message queues for video packets and control commands

### 2. Implement Phase 1: Raw Socket Connection (Prove-Out)
- [src/sinks/src/scrcpy_sink.cpp](src/sinks/src/scrcpy_sink.cpp): Implement socket creation, bind/listen on port 27183
- Accept connections from device (video socket first, then control)
- Read raw bytes from each socket in blocking loops on separate threads
- Output received bytes to file/console for validation (no parsing yet)
- Build and test to confirm sockets connect to device

### 3. Implement Phase 2: Codec Metadata & Frame Header Parsing
- After socket accepts connection, read first 12 bytes from video socket (codec metadata: codec_id, width, height)
- Parse frame headers (12-byte bitfield: config_packet flag, key_frame flag, 62-bit PTS, u32 packet_size)
- Extract bitfield using bit manipulation (handle byte order correctly)
- Buffer complete H.264 packets (header + payload) in frame queue
- Route packets to existing V4L2 sink infrastructure

### 4. Integrate with Existing V4L2 Output
- Use the v4l2_sink implementation to write parsed H.264 packets to `/dev/video0`

### 5. Implement Phase 3: Control Message Protocol
- Create control message serializer in `ControlMessage` class or struct
- Implement serialization for key types: `INJECT_KEYCODE`, `INJECT_TOUCH_EVENT`, `INJECT_SCROLL_EVENT` (reference: https://github.com/Genymobile/scrcpy/blob/master/app/tests/test_control_msg_serialize.c)
- Control message structure: type byte + variable payload (format depends on message type)
- Add queue-based interface in ScrcpySink to enqueue control commands for sending
- Send serialized messages to control socket in blocking write calls

### 6. Remove scrcpy Library Dependency from Sink
- Update [src/sinks/CMakeLists.txt](src/sinks/CMakeLists.txt): remove scrcpy library linking if no longer needed elsewhere
- Scrcpy server still runs on device (handled by ADB), only remove client wrapper dependency
- Preserve V4L2 and FFmpeg linkage for output pipeline

### 7. Update ScrcpySink Public API
- Keep initialization signature compatible (port, device selection)
- Add methods: `sendKeycode()`, `sendTouchEvent()`, `sendScrollEvent()` for control messages
- Maintain frame delivery callback or queue interface for video pipeline

## Verification Checklist

- **Phase 1:** Confirm sockets accept connections; observe raw bytes flowing through console/log output
- **Phase 2:** Parse codec metadata (print resolution), verify frame header fields (check PTS increments, detect key frames), confirm packet sizes match
- **Phase 3:** Inject test control messages (keypress, touch tap) via ScrcpySink API; confirm device responds (key presses appear on-screen, touch events register)
- **Integration:** H.264 stream outputs to V4L2 device at `/dev/video0` with correct resolution; Piano Tiles app can consume video as camera source
- **End-to-end:** Run Piano Tiles app, interact with device via control messages, confirm video + control both work

## Design Decisions

- **Raw socket approach:** Start minimal (raw bytes) before tackling frame header bitfield parsing and protocol details—reduces initial complexity
- **Video output unchanged:** Continue using existing V4L2 + FFmpeg pipeline rather than adding new decode path
- **Thread model:** Separate reader threads for video/control sockets with thread-safe queues; main sink thread dispatches to V4L2
- **No ADB setup:** Assume ADB forwarding/reverse tunnel already configured by caller (as done in current scrcpy_sink setup)

## Protocol Reference

### Socket Connection & Setup
```
ADB Configuration:
├─ Client opens listening socket on port 27183
├─ Sets up ADB tunnel (reverse or forward)
│  - Default (reverse): adb reverse localabstract:scrcpy_<SCID> tcp:27183
│  - Fallback (forward): adb forward tcp:27183 localabstract:scrcpy
└─ Device then connects to client

Socket Order (whichever is enabled):
1. Video socket   (if --no-video not set)
2. Audio socket   (if --no-audio not set)  
3. Control socket (if --no-control not set)
```

### Video Stream Format (H.264)

**Initial codec metadata (12 bytes):**
```
├─ u32: Codec ID (H264=0, H265=1, AV1=2)
├─ u32: Initial video width
└─ u32: Initial video height
```

**Each packet prefix (12-byte frame header):**
```
Byte layout (most significant bit first):
┌─────────────────────────────────────────────────────────────────┐
│ CK......│........│........│........│........│........│........│........│
├─────────┼─────────────────────────────────────────────────────────┤
│ ││       │ 62-bit PTS (Presentation Timestamp)                  │
│ │└────── Key frame flag (u1)                                    │
│ └─────── Config packet flag (u1)                                │
└─────────────────────────────────────────────────────────────────┘
Then: u32 packet_size
Then: raw H.264 packet data
```

### Control Message Structure (Client → Device)

From scrcpy specification:

**Control message types:**
- `SC_CONTROL_MSG_TYPE_INJECT_KEYCODE` - Keyboard input
- `SC_CONTROL_MSG_TYPE_INJECT_TEXT` - Text input  
- `SC_CONTROL_MSG_TYPE_INJECT_TOUCH_EVENT` - Touch events
- `SC_CONTROL_MSG_TYPE_INJECT_SCROLL_EVENT` - Scroll events
- `SC_CONTROL_MSG_TYPE_BACK_OR_SCREEN_ON` - Back button / screen on
- `SC_CONTROL_MSG_TYPE_EXPAND_NOTIFICATION_PANEL` - System controls
- `SC_CONTROL_MSG_TYPE_COLLAPSE_NOTIFICATION_PANEL`
- `SC_CONTROL_MSG_TYPE_GET_CLIPBOARD` - Clipboard operations
- `SC_CONTROL_MSG_TYPE_SET_CLIPBOARD`
- `SC_CONTROL_MSG_TYPE_SET_SCREEN_POWER_MODE`
- `SC_CONTROL_MSG_TYPE_ROTATE_DEVICE`
- `SC_CONTROL_MSG_TYPE_START_APP` - App launching

**Example - Keycode message format:**
```
[type(1) | action(1) | keycode(4) | repeat(4) | metastate(4)]
  u8       u8         u32         u32        u32

action: AKEY_EVENT_ACTION_UP=0, AKEY_EVENT_ACTION_DOWN=1
keycode: AKEYCODE_* (Android key codes)
metastate: bit flags for SHIFT, CTRL, ALT modifiers
```

**Example - Touch Event format:**
```
[type(1) | pointer_id(8) | x(2) | y(2) | size(2) | pressure(2)]
  u8       u64            u16    u16    u16      u16
```

## Files to Modify/Create

### Modify
- [src/sinks/include/scrcpy_sink.hpp](src/sinks/include/scrcpy_sink.hpp) - Header interface, socket management classes
- [src/sinks/src/scrcpy_sink.cpp](src/sinks/src/scrcpy_sink.cpp) - Implementation with socket handling and frame processing
- [src/sinks/CMakeLists.txt](src/sinks/CMakeLists.txt) - Build configuration (may reduce scrcpy library linking)

### Reference
- [extern/scrcpy/doc/develop.md](https://github.com/Genymobile/scrcpy/blob/master/doc/develop.md) - Scrcpy protocol specification
- [extern/scrcpy/app/tests/test_control_msg_serialize.c](https://github.com/Genymobile/scrcpy/blob/master/app/tests/test_control_msg_serialize.c) - Control message examples
- [extern/scrcpy/app/src/scrcpy.c](extern/scrcpy/app/src/scrcpy.c) - Reference client implementation

## Questions for Refinement

- Should phases 2 & 3 be implemented immediately after phase 1, or deployed separately?
- Is there existing infrastructure in the codebase for thread management and message queues (e.g., boost, custom)?
- Should control messages be sent strictly as-is, or wrapped in additional framing/headers?
- Any specific Android key codes or touch coordinate systems to prioritize for prototype?
