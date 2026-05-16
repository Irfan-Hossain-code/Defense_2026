/*
 * CSI Node Firmware — ESP32-WROOM
 * Used for LEFT (COM9) and RIGHT (COM8) nodes.
 *
 * NODE_ID, WIFI_SSID, WIFI_PASS, and PROBE_INTERVAL_MS are all
 * injected by platformio.ini build_flags so this one file works
 * for both boards without editing.
 *
 * Serial output format (one line per CSI event):
 *   CSI:<rssi>,<noise_floor>,<channel>,<len>,<byte0>,<byte1>,...
 *   Bytes are int8_t pairs: [imaginary, real] per sub-carrier.
 *
 * The traffic_task pings the Wi-Fi gateway every PROBE_INTERVAL_MS
 * milliseconds so the gateway replies — that reply triggers CSI.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifndef NODE_ID
#define NODE_ID "NODE"
#endif
#ifndef WIFI_SSID
#define WIFI_SSID "YOUR_SSID"
#endif
#ifndef WIFI_PASS
#define WIFI_PASS "YOUR_PASS"
#endif
#ifndef PROBE_INTERVAL_MS
#define PROBE_INTERVAL_MS 100
#endif

static volatile uint32_t g_csi_count = 0;
static volatile bool     g_wifi_up   = false;

// ── CSI callback ──────────────────────────────────────────────────────────────
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

// ── Gateway probe task ────────────────────────────────────────────────────────
// Connects to the gateway port 80 every PROBE_INTERVAL_MS.
// The gateway sends TCP RST (port closed) → ESP32 receives it → CSI fires.
static void traffic_task(void *param) {
    WiFiClient client;
    client.setTimeout(80);
    while (true) {
        if (g_wifi_up) {
            client.connect(WiFi.gatewayIP(), 80);
            client.stop();
        }
        vTaskDelay(pdMS_TO_TICKS(PROBE_INTERVAL_MS));
    }
}

// ── CSI enable ────────────────────────────────────────────────────────────────
static bool enable_csi() {
    wifi_csi_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.lltf_en           = 1;
    cfg.htltf_en          = 1;
    cfg.stbc_htltf2_en    = 1;
    cfg.ltf_merge_en      = 1;
    cfg.channel_filter_en = 0;
    cfg.manu_scale        = 0;

    if (esp_wifi_set_csi_config(&cfg) != ESP_OK) {
        Serial.println("ERROR: csi_config failed");
        return false;
    }
    if (esp_wifi_set_csi_rx_cb(csi_callback, nullptr) != ESP_OK) {
        Serial.println("ERROR: csi_rx_cb failed");
        return false;
    }
    if (esp_wifi_set_csi(true) != ESP_OK) {
        Serial.println("ERROR: csi_enable failed");
        return false;
    }
    return true;
}

void setup() {
    Serial.begin(115200);
    delay(2000);

    Serial.println();
    Serial.println("INFO: === CSI Node [" NODE_ID "] ===");
    Serial.print("INFO: Connecting to " WIFI_SSID " ...");

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
    } else {
        Serial.println("ERROR: Wi-Fi failed. Check SSID/PASS.");
        return;
    }

    if (enable_csi()) {
        Serial.println("INFO: CSI enabled.");
        Serial.println("INFO: Format: CSI:<rssi>,<noise>,<ch>,<len>,<bytes...>");
        Serial.println("INFO: Ready.");
    }

    xTaskCreate(traffic_task, "probe", 4096, nullptr, 1, nullptr);
}

void loop() {
    static uint32_t last_status_ms = 0;
    uint32_t now = millis();
    if (now - last_status_ms >= 5000) {
        last_status_ms = now;
        Serial.print("STATUS: node=" NODE_ID " uptime=");
        Serial.print(now / 1000);
        Serial.print("s pkts=");
        Serial.print(g_csi_count);
        Serial.print(" rssi=");
        Serial.print(WiFi.RSSI());
        Serial.println("dBm");
    }
    delay(10);
}
