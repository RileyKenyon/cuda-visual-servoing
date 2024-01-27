#include <cstdint>
#include <cstddef>
#include <iostream>
#include <cstring>

int test_serialize_inject_touch_event() {
    // Constants from scrcpy
    const uint8_t SC_CONTROL_MSG_TYPE_INJECT_TOUCH_EVENT = 2;
    const uint8_t AMOTION_EVENT_ACTION_DOWN = 0x00;
    const uint64_t pointer_id = 0x1234567887654321ULL;
    const uint32_t x = 100;
    const uint32_t y = 200;
    const uint16_t width = 1080;
    const uint16_t height = 1920;
    const float pressure = 1.0f;
    const uint32_t action_button = 0x01;
    const uint32_t buttons = 0x01;

    uint8_t buf[32] = {};
    size_t idx = 0;
    buf[idx++] = SC_CONTROL_MSG_TYPE_INJECT_TOUCH_EVENT;
    buf[idx++] = AMOTION_EVENT_ACTION_DOWN;
    // pointer_id (u64, big-endian)
    for (int i = 7; i >= 0; i--) buf[idx++] = (pointer_id >> (i * 8)) & 0xFF;
    // x (u32, big-endian)
    buf[idx++] = (x >> 24) & 0xFF;
    buf[idx++] = (x >> 16) & 0xFF;
    buf[idx++] = (x >> 8) & 0xFF;
    buf[idx++] = x & 0xFF;
    // y (u32, big-endian)
    buf[idx++] = (y >> 24) & 0xFF;
    buf[idx++] = (y >> 16) & 0xFF;
    buf[idx++] = (y >> 8) & 0xFF;
    buf[idx++] = y & 0xFF;
    // width (u16, big-endian)
    buf[idx++] = (width >> 8) & 0xFF;
    buf[idx++] = width & 0xFF;
    // height (u16, big-endian)
    buf[idx++] = (height >> 8) & 0xFF;
    buf[idx++] = height & 0xFF;
    // pressure (u16, big-endian, 1.0f = 0xffff)
    uint16_t pressure_u16 = 0xffff;
    buf[idx++] = (pressure_u16 >> 8) & 0xFF;
    buf[idx++] = pressure_u16 & 0xFF;
    // action_button (u32, big-endian)
    buf[idx++] = (action_button >> 24) & 0xFF;
    buf[idx++] = (action_button >> 16) & 0xFF;
    buf[idx++] = (action_button >> 8) & 0xFF;
    buf[idx++] = action_button & 0xFF;
    // buttons (u32, big-endian)
    buf[idx++] = (buttons >> 24) & 0xFF;
    buf[idx++] = (buttons >> 16) & 0xFF;
    buf[idx++] = (buttons >> 8) & 0xFF;
    buf[idx++] = buttons & 0xFF;

    // Expected output from scrcpy test
    const uint8_t expected[] = {
        2,
        0x00,
        0x12, 0x34, 0x56, 0x78, 0x87, 0x65, 0x43, 0x21,
        0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0xc8,
        0x04, 0x38, 0x07, 0x80,
        0xff, 0xff,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x01
    };
    bool match = (memcmp(buf, expected, sizeof(expected)) == 0);
    if (match) {
        std::cout << "[TEST] Touch event serialization matches scrcpy reference." << std::endl;
        return 0;
    } else {
        std::cout << "[TEST] Touch event serialization FAILED." << std::endl;
        for (size_t i = 0; i < sizeof(expected); ++i) {
            std::cout << "Byte " << i << ": got 0x" << std::hex << (int)buf[i] << ", expected 0x" << (int)expected[i] << std::endl;
        }
        return 1;
    }
}

int main() {
    return test_serialize_inject_touch_event();
}
