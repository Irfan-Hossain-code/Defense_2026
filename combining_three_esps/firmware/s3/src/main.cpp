/*
 * CSI Node Firmware — ESP32-S3 (MIDDLE node, COM7)
 *
 * Identical logic to the WROOM firmware.
 * NODE_ID="MIDDLE" is injected by platformio.ini.
 *
 * After flashing, this board can be powered from a USB power bank
 * (battery).  It connects to Wi-Fi automatically on boot and starts
 * probing even without a laptop attached.  When no serial cable is
 * connected, the CSI output just goes nowhere — that is fine.
 *
 * When USB IS connected (COM7), run_all_three.py will also read
 * the MIDDLE node's CSI to improve zone accuracy.
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
#define NODE_ID "MIDDLE"
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

static volatile uint32_t g_csi_count   = 0;
static volatile bool     g_wifi_up     = false;
static bool              g_csi_enabled = false;

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

static bool enable_csi() {
    wifi_csi_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.lltf_en           = 1;
    cfg.htltf_en          = 1;
    cfg.stbc_htltf2_en    = 1;
    cfg.ltf_merge_en      = 1;
    cfg.channel_filter_en = 0;
    cfg.manu_scale        = 0;

    if (esp_wifi_set_csi_config(&cfg) != ESP_OK) return false;
    if (esp_wifi_set_csi_rx_cb(csi_callback, nullptr) != ESP_OK) return false;
    if (esp_wifi_set_csi(true) != ESP_OK) return false;
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
        Serial.println("ERROR: Wi-Fi failed.");
        // Still start the task — it will keep retrying when wifi comes up
    }

    g_csi_enabled = enable_csi();
    if (g_csi_enabled) {
        Serial.println("INFO: CSI enabled OK.");
    } else {
        Serial.println("ERROR: CSI enable FAILED — check API compatibility.");
    }

    xTaskCreate(traffic_task, "probe", 4096, nullptr, 1, nullptr);
}

void loop() {
    static uint32_t last_ms = 0;
    uint32_t now = millis();
    if (now - last_ms >= 5000) {
        last_ms = now;
        Serial.print("STATUS: node=" NODE_ID " csi=");
        Serial.print(g_csi_enabled ? "OK" : "FAILED");
        Serial.print(" uptime=");
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
