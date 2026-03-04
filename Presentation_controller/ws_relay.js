/*
 * =====================================================
 *  Gesture Recognition Wearable — Phase 04
 *  WebSocket Relay Server (Node.js)
 *
 *  The ESP8266 connects here as a WebSocket CLIENT.
 *  The browser (gesture_presenter.html) also connects
 *  here as a CLIENT. This server relays gesture
 *  payloads from the ESP8266 to all browser clients.
 *
 *  Why a relay? Browser WebSocket cannot act as a
 *  server, and ESP8266 is already busy with WiFi +
 *  inference. Running a tiny relay on your PC or
 *  Raspberry Pi keeps both sides simple.
 *
 *  Requirements:
 *    npm install ws
 *
 *  Usage:
 *    node ws_relay.js
 *    node ws_relay.js --port 8080
 *
 *  Then:
 *    ESP8266 gesture_output.h WS_HOST = this machine's IP
 *    gesture_presenter.html   Host    = this machine's IP
 * =====================================================
 */

const WebSocket = require('ws');
const http      = require('http');

const PORT = parseInt(process.argv[2] || process.env.PORT || '8080');

// ── HTTP server (health check endpoint) ─────────────
const httpServer = http.createServer((req, res) => {
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: 'ok',
      clients: { esp: espClients.size, browsers: browserClients.size },
      uptime: process.uptime().toFixed(1)
    }));
  } else {
    res.writeHead(404);
    res.end();
  }
});

// ── WebSocket server ─────────────────────────────────
const wss = new WebSocket.Server({ server: httpServer, path: '/gesture' });

const espClients     = new Set();
const browserClients = new Set();

// Identify client type by User-Agent or first message
// ESP8266 WebSocket library sends "ESP8266" as user-agent
function isESP(req) {
  const ua = req.headers['user-agent'] || '';
  return ua.includes('ESP') || ua.includes('esp') || ua === '';
}

wss.on('connection', (ws, req) => {
  const ip   = req.socket.remoteAddress;
  const type = isESP(req) ? 'ESP8266' : 'Browser';

  if (type === 'ESP8266') {
    espClients.add(ws);
    log(`[+] ESP8266 connected  (${ip})  total ESP: ${espClients.size}`);
  } else {
    browserClients.add(ws);
    log(`[+] Browser connected  (${ip})  total browsers: ${browserClients.size}`);
    // Send current connection status to new browser client
    ws.send(JSON.stringify({ type: 'status', esp_connected: espClients.size > 0 }));
  }

  ws.on('message', (raw) => {
    const msg = raw.toString();

    // Try to parse as gesture payload
    try {
      const data = JSON.parse(msg);

      if (data.gesture) {
        // Relay to all browser clients
        const outgoing = JSON.stringify(data);
        let relayed = 0;
        for (const browser of browserClients) {
          if (browser.readyState === WebSocket.OPEN) {
            browser.send(outgoing);
            relayed++;
          }
        }
        log(`[G] ${data.gesture} (${(data.confidence * 100).toFixed(0)}%) -> relayed to ${relayed} browser(s)`);
      }

    } catch (_) {
      // Not JSON — log raw
      log(`[?] Raw message from ${type}: ${msg.slice(0, 80)}`);
    }
  });

  ws.on('close', () => {
    if (espClients.has(ws)) {
      espClients.delete(ws);
      log(`[-] ESP8266 disconnected.  Remaining: ${espClients.size}`);
      // Notify browsers that ESP disconnected
      broadcast(browserClients, JSON.stringify({ type: 'status', esp_connected: false }));
    } else {
      browserClients.delete(ws);
      log(`[-] Browser disconnected.  Remaining: ${browserClients.size}`);
    }
  });

  ws.on('error', (err) => {
    log(`[!] Error on ${type} socket: ${err.message}`);
  });
});

// ── Helpers ──────────────────────────────────────────
function broadcast(clients, message) {
  for (const client of clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  }
}

function log(msg) {
  const ts = new Date().toISOString().slice(11, 19);
  console.log(`[${ts}] ${msg}`);
}

// ── Start ─────────────────────────────────────────────
httpServer.listen(PORT, '0.0.0.0', () => {
  log(`WebSocket relay server listening on ws://0.0.0.0:${PORT}/gesture`);
  log(`Health check: http://localhost:${PORT}/health`);
  log(`Waiting for ESP8266 and browser connections...`);
});

process.on('SIGINT', () => {
  log('Shutting down...');
  wss.close();
  httpServer.close();
  process.exit(0);
});
