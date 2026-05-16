/*
 * ESP32 CSI Motion Sensor Firmware
 *
 * How it works:
 *   A background FreeRTOS task connects to the Wi-Fi gateway (your phone hotspot)
 *   via TCP every 100 ms.  The gateway replies with TCP RST (port closed) — that
 *   reply packet is received by the ESP32 as a Wi-Fi data frame, which triggers
 *   the CSI callback.  No second device or manual pinging needed.
 *
 *   When a hand (or body) moves near the ESP32, it alters the multi-path
 *   reflections of the Wi-Fi signal.  The amplitude of each OFDM sub-carrier
 *   shifts.  The Python script on the PC tracks those shifts to detect motion.
 *
 * Serial output — one line per CSI event:
 *   CSI:<rssi>,<noise_floor>,<channel>,<len>,<byte0>,<byte1>,...
 *   bytes are int8_t pairs: [imaginary, real] per sub-carrier.
 *
 * Also prints:
 *   INFO:   startup messages
 *   STATUS: heartbeat every 5 s (uptime, packet count, RSSI)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// ============================================================
// CONFIGURATION
// ============================================================
#define WIFI_SSID           "Oneplus8"
#define WIFI_PASS           "aightt14"
#define FALLBACK_AP_SSID    "ESP32_CSI_TEST"
#define FALLBACK_AP_PASS    "12345678"

// How often to probe the gateway (milliseconds).
// 100 ms = ~10 CSI packets per second — enough for smooth motion detection.
// Lower = more packets but heavier CPU load on the gateway.
#define PROBE_INTERVAL_MS   100
// ============================================================

static volatile uint32_t g_csi_count   = 0;
static volatile bool     g_wifi_up     = false;

// ── CSI callback ─────────────────────────────────────────────────────────────
// Called by the Wi-Fi driver for every received 802.11 data frame.
// Runs in the Wi-Fi task — keep it fast, the buf pointer is only valid here.
static void csi_callback(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len == 0) return;
    g_csi_count++;

    Serial.print("CSI:");
    Serial.print(info->rx_ctrl.rssi);
    Serial.print(",");
    Serial.print(info->rx_ctrl.noise_floor);
    Serial.print(",");
    Serial.print(info->rx_ctrl.channel);
    Serial.print(",");
    Serial.print(info->len);

    for (int i = 0; i < info->len; i++) {
        Serial.print(",");
        Serial.print(static_cast<int8_t>(info->buf[i]));
    }
    Serial.println();
}

// ── Traffic generator task ───────────────────────────────────────────────────
// Runs as a separate FreeRTOS task so it never blocks the Arduino loop.
// Opens a TCP connection to the gateway port 80 every PROBE_INTERVAL_MS.
// The phone's TCP stack sends back a RST (port closed) almost instantly —
// that RST is a Wi-Fi data frame addressed to the ESP32, which fires CSI.
static void traffic_task(void *param) {
    WiFiClient client;
    client.setTimeout(80);   // give up if no reply within 80 ms

    while (true) {
        if (g_wifi_up) {
            IPAddress gw = WiFi.gatewayIP();
            client.connect(gw, 80);   // triggers gateway → RST → CSI
            client.stop();
        }
        vTaskDelay(pdMS_TO_TICKS(PROBE_INTERVAL_MS));
    }
}

// ── CSI API setup ─────────────────────────────────────────────────────────────
static bool enable_csi() {
    wifi_csi_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.lltf_en           = 1;  // legacy sub-carriers (52)
    cfg.htltf_en          = 1;  // HT sub-carriers (56)
    cfg.stbc_htltf2_en    = 1;  // STBC variant
    cfg.ltf_merge_en      = 1;  // average LLTF + HT-LTF when both present
    cfg.channel_filter_en = 0;  // raw per-sub-carrier (do not smooth)
    cfg.manu_scale        = 0;  // auto gain

    esp_err_t err;
    err = esp_wifi_set_csi_config(&cfg);
    if (err != ESP_OK) {
        Serial.print("ERROR: csi_config: ");
        Serial.println(esp_err_to_name(err));
        return false;
    }
    err = esp_wifi_set_csi_rx_cb(csi_callback, nullptr);
    if (err != ESP_OK) {
        Serial.print("ERROR: csi_rx_cb: ");
        Serial.println(esp_err_to_name(err));
        return false;
    }
    err = esp_wifi_set_csi(true);
    if (err != ESP_OK) {
        Serial.print("ERROR: csi_enable: ");
        Serial.println(esp_err_to_name(err));
        return false;
    }
    return true;
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(2000);

    Serial.println();
    Serial.println("INFO: === ESP32 CSI Motion Sensor ===");
    Serial.print("INFO: Connecting to ");
    Serial.println(WIFI_SSID);

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    for (int i = 0; i < 30 && WiFi.status() != WL_CONNECTED; i++) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        g_wifi_up = true;
        Serial.print("INFO: Connected! IP=");
        Serial.print(WiFi.localIP());
        Serial.print(" GW=");
        Serial.print(WiFi.gatewayIP());
        Serial.print(" CH=");
        Serial.print(WiFi.channel());
        Serial.print(" RSSI=");
        Serial.print(WiFi.RSSI());
        Serial.println("dBm");
        Serial.print("INFO: Self-probing gateway every ");
        Serial.print(PROBE_INTERVAL_MS);
        Serial.println("ms for CSI");
    } else {
        Serial.print("INFO: Could not connect to ");
        Serial.println(WIFI_SSID);
        Serial.print("INFO: Fallback AP started: ");
        Serial.println(FALLBACK_AP_SSID);
        WiFi.mode(WIFI_AP);
        WiFi.softAP(FALLBACK_AP_SSID, FALLBACK_AP_PASS);
        Serial.print("INFO: AP IP: ");
        Serial.println(WiFi.softAPIP());
        Serial.println("INFO: Connect a phone to that AP — its traffic becomes CSI.");
    }

    if (enable_csi()) {
        Serial.println("INFO: CSI enabled.");
        Serial.println("INFO: Format: CSI:<rssi>,<noise>,<ch>,<len>,<bytes...>");
        Serial.println("INFO: ============================================");
    }

    // Start the self-probing task (priority 1, 4 KB stack)
    xTaskCreate(traffic_task, "csi_probe", 4096, nullptr, 1, nullptr);
    Serial.println("INFO: Probe task running.");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
    static uint32_t last_status_ms = 0;
    uint32_t now = millis();

    if (now - last_status_ms >= 5000) {
        last_status_ms = now;
        Serial.print("STATUS: uptime=");
        Serial.print(now / 1000);
        Serial.print("s pkts=");
        Serial.print(g_csi_count);
        if (g_wifi_up) {
            Serial.print(" rssi=");
            Serial.print(WiFi.RSSI());
            Serial.print("dBm");
        }
        Serial.println();
    }

    delay(10);
}
