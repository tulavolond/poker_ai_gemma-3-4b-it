const express = require('express');
const fs = require('fs');
const tls = require('tls');
const https = require('https');
const http = require('http');


// const fs = require('fs')
const cors = require('cors')
const httpss = require('https')
const CryptoJS = require('crypto-js')
const moment = require('moment');
const { createProxyMiddleware } = require('http-proxy-middleware');
// const HttpsProxyAgent = require('https-proxy-agent')
// const fsp = require('fs').promises;
const geoip = require('geoip-lite');
const UAParser = require('ua-parser-js');
const path = require('path');

// const scannerMiddleware = require('./scanner-middleware');
const Analytics = require('./analytics');

const scannerMiddleware = require('./scanner');

// Хранилище состояния IP: { ip: { count, windowStart, blockUntil } }
const ipStore = new Map();

const LIMIT = 68;          // максимум запросов за 1 секунду
const WINDOW_MS = 1000;    // окно подсчёта (1 сек)
const BLOCK_MS = 3000;     // время блокировки (3 сек)

// Middleware-защита
function antiDdos(req, res, next) {
    //   const ip = req.ip || req.connection.remoteAddress || req.headers['x-forwarded-for'].split(',')[0].trim();
    let ip =
        req.ip ||
        req.connection.remoteAddress ||
        (req.headers['x-forwarded-for']?.split(',')[0].trim()) ||
        '0.0.0.0'; // или любой другой адрес по умолчанию
    const now = Date.now();

    ip = ip.replace('::ffff:', '');

    let record = ipStore.get(ip);

    // Если запись отсутствует – создаём новую
    if (!record) {
        record = { count: 0, windowStart: now, blockUntil: 0 };
        ipStore.set(ip, record);
    }

    // 1️⃣ Проверяем, не находится ли IP в блокировке
    if (record.blockUntil > now) {
        const secondsLeft = Math.ceil((record.blockUntil - now) / 1000);
        return res.status(429).send({
            error: 'Too Many Requests',
            message: `IP temporarily blocked. Try again in ${secondsLeft} second(s).`,
        });
    }

    // 2️⃣ Сбрасываем счётчик, если окно вышло
    if (now - record.windowStart > WINDOW_MS) {
        record.count = 0;
        record.windowStart = now;
    }

    // 3️⃣ Увеличиваем счётчик
    record.count += 1;

    // 4️⃣ Если превысили лимит – ставим блокировку
    if (record.count > LIMIT) {
        record.blockUntil = now + BLOCK_MS; // блокируем на BLOCK_MS миллисекунд
        // ОPTIONAL: сбросить счётчик, чтобы не считать дальше пока блок активен
        record.count = 0;
        return res.status(429).send({
            error: 'Too Many Requests',
            message: `IP blocked for ${BLOCK_MS / 1000} second(s) due to excessive requests.`,
        });
    }

    // Всё ок – продолжаем обработку
    next();
}

// Инициализация аналитики
const analytics = new Analytics();
// await analytics.init(); // Дождитесь загрузки данных перед запуском сервера

const host = '127.0.0.1'
const port = 80
const portssh = 443


// 1. Сначала подготавливаем данные сертификатов
const certOptions = {
    't2020.ru': {
        key: fs.readFileSync('/etc/letsencrypt/live/t2020-new.ru/privkey.pem'),
        cert: fs.readFileSync('/etc/letsencrypt/live/t2020-new.ru/fullchain.pem')
    },
    'watermark-png.ru': {
        key: fs.readFileSync('/etc/letsencrypt/live/watermark-png.ru/privkey.pem'),
        cert: fs.readFileSync('/etc/letsencrypt/live/watermark-png.ru/fullchain.pem')
    },
    'direct.t2020.ru': {  // Добавляем новый домен
        key: fs.readFileSync('/etc/letsencrypt/live/t2020.ru_0001/privkey.pem'),
        cert: fs.readFileSync('/etc/letsencrypt/live/t2020.ru_0001/fullchain.pem')
    },
    'p2p.t2020.ru': {  // Добавляем новый домен
        key: fs.readFileSync('/etc/letsencrypt/live/t2020-new.ru/privkey.pem'),
        cert: fs.readFileSync('/etc/letsencrypt/live/t2020-new.ru/fullchain.pem')
    },
    'game.t2020.ru': {  // Добавляем новый домен
        key: fs.readFileSync('/etc/letsencrypt/live/t2020.ru-0002/privkey.pem'),
        cert: fs.readFileSync('/etc/letsencrypt/live/t2020.ru-0002/fullchain.pem')
    },
    'yd.t2020.ru': {  // Добавляем новый домен
        key: fs.readFileSync('/etc/letsencrypt/live/t2020-new.ru/privkey.pem'),
        cert: fs.readFileSync('/etc/letsencrypt/live/t2020-new.ru/fullchain.pem')
    },
    'ya.t2020.ru': {  // Добавляем новый домен
        key: fs.readFileSync('/etc/letsencrypt/live/t2020-new.ru/privkey.pem'),
        cert: fs.readFileSync('/etc/letsencrypt/live/t2020-new.ru/fullchain.pem')
    },
    'demo.t2020.ru': {  // Добавляем новый домен
        key: fs.readFileSync('/etc/letsencrypt/live/demo.t2020.ru/privkey.pem'),
        cert: fs.readFileSync('/etc/letsencrypt/live/demo.t2020.ru/fullchain.pem')
    },
    'poker.t2020.ru': {  // Добавляем новый домен
        key: fs.readFileSync('/etc/letsencrypt/live/t2020-new.ru/privkey.pem'),
        cert: fs.readFileSync('/etc/letsencrypt/live/t2020-new.ru/fullchain.pem')
    },
};

// Создаем контексты (важно сделать это до создания сервера)
const secureContexts = {};
for (const domain in certOptions) {
    secureContexts[domain] = tls.createSecureContext(certOptions[domain]);
}

// 2. Создаем приложение Express
const app = express();
app.use(cors({
    origin: true,               // отражать Origin запроса (или функцию-валидатор)
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true
}));
// Подключаем middleware глобально (или только к нужным роутам)
app.use(antiDdos);
app.use(scannerMiddleware);

// app.use(express.static(__dirname + '/public/'))
app.use('/t2020.ru', cors(), express.static('/public/'))
app.use('/okx', cors(), express.static('t2020.ru'))
// app.use('/nextjs', cors(), express.static('t2020.ru'))
// app.use('/api/data', express.static('/api/data'))
// app.use('/api/advanced_metrics', express.static('/api/advanced_metrics'))
app.use('/chart', cors(), express.static('chart'))
app.use('/pool', express.static('pool'))
// Старая трейдинг-версия
app.use('/trading', express.static('trading'))

app.use('/wss_82', createProxyMiddleware({
    target: 'http://192.168.1.100:8082', // Порт вашего php server.php
    changeOrigin: true,
    ws: true, // Включает поддержку веб-сокетов
    // logLevel: 'debug', // Поможет увидеть логи проксирования в консоли Node.js
    proxyTimeout: 10 * 60 * 1000, // 10 минут ожидания ответа
    timeout: 10 * 60 * 1000,      // 10 минут удержания сокета
}));

app.use('/wss', createProxyMiddleware({
    target: 'http://192.168.1.100:8081', // Порт вашего php server.php
    changeOrigin: true,
    ws: true, // Включает поддержку веб-сокетов
    // logLevel: 'debug', // Поможет увидеть логи проксирования в консоли Node.js
    proxyTimeout: 10 * 60 * 1000, // 10 минут ожидания ответа
    timeout: 10 * 60 * 1000,      // 10 минут удержания сокета
}));


app.get('/stats', async (req, res, next) => {
    // Проверяем, что запрос пришел именно на нужный домен (если на сервере крутится несколько сайтов)
    if (req.hostname !== 'poker.t2020.ru') {
        return next(); // Пропускаем запрос дальше, если домен другой
    }

    try {
        const response = await fetch('http://192.168.1.100:8899/stats');

        if (!response.ok) {
            throw new Error(`Ошибка внутреннего сервера: ${response.status}`);
        }

        const data = await response.json();
        res.json(data);

    } catch (error) {
        console.error('Не удалось получить статистику:', error.message);
        res.status(500).json({ error: 'Ошибка при получении данных' });
    }
});

app.get('/deep_poker/stats', async (req, res, next) => {
    // Проверяем, что запрос пришел именно на нужный домен (если на сервере крутится несколько сайтов)
    if (req.hostname !== 'poker.t2020.ru') {
        return next(); // Пропускаем запрос дальше, если домен другой
    }

    try {
        const response = await fetch('http://192.168.1.100:8899/stats');

        if (!response.ok) {
            throw new Error(`Ошибка внутреннего сервера: ${response.status}`);
        }

        const data = await response.json();
        res.json(data);

    } catch (error) {
        console.error('Не удалось получить статистику:', error.message);
        res.status(500).json({ error: 'Ошибка при получении данных' });
    }
});

app.get('/api/ai/stats', async (req, res, next) => {
    // Проверяем, что запрос пришел именно на нужный домен (если на сервере крутится несколько сайтов)
    if (req.hostname !== 'poker.t2020.ru') {
        return next(); // Пропускаем запрос дальше, если домен другой
    }

    try {
        const response = await fetch('http://192.168.1.100:8899/stats');

        if (!response.ok) {
            throw new Error(`Ошибка внутреннего сервера: ${response.status}`);
        }

        const data = await response.json();
        res.json(data);

    } catch (error) {
        console.error('Не удалось получить статистику:', error.message);
        res.status(500).json({ error: 'Ошибка при получении данных' });
    }
});


// app.all перехватит и POST, и PUT, и любые другие методы от браузера
app.all('/api/ai', express.json(), async (req, res, next) => {
  if (req.hostname !== 'poker.t2020.ru') {
    return next();
  }

  try {
    // На сам AI-сервер всегда отправляем строго POST, который он поддерживает
    // const aiResponse = await fetch('http://5.164.29.62:45726', {
    const aiResponse = await fetch('http://192.168.1.100:8899', {
      method: req.method,
      headers: {
        'Content-Type': 'application/json; charset=utf-8'
      },
      body: JSON.stringify(req.body)
    });

    if (!aiResponse.ok) {
      const errText = await aiResponse.text();
      throw new Error(`AI сервер вернул ${aiResponse.status}: ${errText}`);
    }

    const data = await aiResponse.json();
    res.json(data);

  } catch (error) {
    console.error('Ошибка проксирования к AI:', error.message);
    res.status(500).json({ error: 'Не удалось связаться с AI-сервером' });
  }
});



// Выносим функцию отдельно для надежности
function fixHeaders(proxyReq, req, res) {
    // Получаем IP клиента
    let clientIp = req.headers['x-forwarded-for'] || req.socket.remoteAddress;

    // Если IP содержит префикс ::ffff:, убираем его, оставляя чистый IPv4
    if (clientIp && clientIp.includes('::ffff:')) {
        clientIp = clientIp.replace('::ffff:', '');
    }

    if (clientIp) {
        proxyReq.setHeader('X-Forwarded-For', clientIp);
        proxyReq.setHeader('X-Real-IP', clientIp);
        // console.log(`👉referer: ${clientIp} 👈`);
    }

    if (req.headers.referer) {
        proxyReq.setHeader('Referer', req.headers.referer);
    }
}


const wss_82 = createProxyMiddleware({
    target: 'http://192.168.1.100:8082', // Порт вашего php server.php
    changeOrigin: true,
    ws: true, // Включает поддержку веб-сокетов
    // logLevel: 'debug', // Поможет увидеть логи проксирования в консоли Node.js
    proxyTimeout: 10 * 60 * 1000, // 10 минут ожидания ответа
    timeout: 10 * 60 * 1000,      // 10 минут удержания сокета
    xfwd: true, // Пытается добавить заголовки автоматически
    cookieDomainRewrite: "pip2pip.ru", // Позволяет кукам правильно прокидываться обратно клиенту
    on: {
        proxyReq: fixHeaders
    }
})

const wss = createProxyMiddleware({
    target: 'http://192.168.1.100:8081', // Порт вашего php server.php
    changeOrigin: true,
    ws: true, // Включает поддержку веб-сокетов
    // logLevel: 'debug', // Поможет увидеть логи проксирования в консоли Node.js
    proxyTimeout: 10 * 60 * 1000, // 10 минут ожидания ответа
    timeout: 10 * 60 * 1000,      // 10 минут удержания сокета
    xfwd: true, // Пытается добавить заголовки автоматически
    cookieDomainRewrite: "p2p.t2020.ru", // Позволяет кукам правильно прокидываться обратно клиенту
    on: {
        proxyReq: fixHeaders
    }
})

// Маршрутизация HTTP-запросов
app.use('/wss_82', wss_82);
app.use('/wss', wss);

const apiProxyWatermark = createProxyMiddleware({
    target: 'http://192.168.1.97:5000',
    changeOrigin: true, // Меняет Origin хоста на целевой URL
    ws: false, // Поддержка веб-сокетов, если нужна
    secure: false
    // options: watermark_png,
});

const apiProxyGame = createProxyMiddleware({
    target: 'http://192.168.1.165:5000',
    changeOrigin: true, // Меняет Origin хоста на целевой URL
    ws: false, // Поддержка веб-сокетов, если нужна
    secure: false
});

const apiProxyPoker = createProxyMiddleware({
    target: 'http://192.168.1.100:8000',
    changeOrigin: true, // Меняет Origin хоста на целевой URL
    ws: false, // Поддержка веб-сокетов, если нужна
    secure: false,
});

const phpDirectProxy = createProxyMiddleware({
    // target: 'http://192.168.1.100:8086',
    target: 'http://127.0.0.1:8888',
    changeOrigin: true,
    ws: false,  // Обязательно для WebSocket
    secure: false,
    // xfwd: true,
    // on: {
    //     proxyReq: fixHeaders
    // }
});

const apiProxyP2p = createProxyMiddleware({
    target: 'http://192.168.1.100:8085',
    changeOrigin: true,
    secure: false,
    ws: false, // 🔥 ЭТО ОБЯЗАТЕЛЬНО! Включает проксирование WebSocket
    // xfwd: true,
    // on: {
    //     proxyReq: fixHeaders
    // },
});

// Конфигурация прокси (ws лучше включить, если используете серверные события)
const nextjsProxy = createProxyMiddleware({
    target: 'http://192.168.1.97:3000',
    changeOrigin: true,
    ws: false, // Включаем поддержку веб-сокетов на всякий случай
});

// Конфигурация прокси (ws лучше включить, если используете серверные события)
const nextjsProxyYD = createProxyMiddleware({
    target: 'http://192.168.1.104:3003',
    changeOrigin: true,
    ws: false, // Включаем поддержку веб-сокетов на всякий случай
    on: {
        proxyReq: fixHeaders
    }
});

function fixHeadersDemo(proxyReq, req) {
    proxyReq.setHeader('Host', 'demo.t2020.ru');
    proxyReq.setHeader('X-Forwarded-Host', 'demo.t2020.ru');
    proxyReq.setHeader('X-Forwarded-Proto', 'https');
    // X-Forwarded-For обычно уже ставится через xfwd: true
}

function fixHeadersPoker(proxyReq, req) {
    proxyReq.setHeader('Host', 'poker.t2020.ru');
    proxyReq.setHeader('X-Forwarded-Host', 'poker.t2020.ru');
    proxyReq.setHeader('X-Forwarded-Proto', 'https');
    // X-Forwarded-For обычно уже ставится через xfwd: true
}

const nextjsProxyDemo_WS = createProxyMiddleware({
    target: 'http://192.168.1.100:3005',
    changeOrigin: true,
    ws: true, // Включаем поддержку веб-сокетов на всякий случай
    xfwd: true,
    on: {
        proxyReq: fixHeadersDemo
    }
});
const nextjsProxyDemo = createProxyMiddleware({
    target: 'http://192.168.1.100:3005',
    changeOrigin: true,
    ws: false, // Включаем поддержку веб-сокетов на всякий случай
    xfwd: true,
    on: {
        proxyReq: fixHeadersDemo
    }
});
const nextjsProxyYD_WS = createProxyMiddleware({
    target: 'http://192.168.1.104:3003',
    changeOrigin: true,
    ws: true, // Включаем поддержку веб-сокетов на всякий случай
    on: {
        proxyReq: (proxyReq, req, res) => {
            // Force убирает потенциальные конфликты со сжатием, если Express использует compression()
            // req.headers['accept-encoding'] = 'identity';
            // Полностью удаляем заголовок сжатия, чтобы Next.js отдавал чистый JS/CSS
            proxyReq.removeHeader('accept-encoding');
        }
    }
});

// Конфигурация прокси (ws лучше включить, если используете серверные события)
const chat_t2020_ru = createProxyMiddleware({
    target: 'http://192.168.1.100:3100',
    changeOrigin: true,
    ws: false, // Включает поддержку веб-сокетов
    // proxyTimeout: 10 * 60 * 1000, // 10 минут ожидания ответа
    // timeout: 10 * 60 * 1000,      // 10 минут удержания сокета
});

const chat_t2020_ru_ws = createProxyMiddleware({
    target: 'http://192.168.1.100:3101',
    changeOrigin: true,
    secure: false,
    ws: true,
    on: {
        proxyReq: (proxyReq, req, res) => {
            // Force убирает потенциальные конфликты со сжатием, если Express использует compression()
            // req.headers['accept-encoding'] = 'identity';
            // Полностью удаляем заголовок сжатия, чтобы Next.js отдавал чистый JS/CSS
            proxyReq.removeHeader('accept-encoding');
        }
    }
});

// 1. Маршрут для чата (chat.t2020.ru)
app.use((req, res, next) => {
    if (!req.headers.host || !req.headers.host.startsWith('chat.t2020.ru')) {
        return next(); // Пропускаем, если это не чат
    }

    const isNextStaticOrApi =
        req.originalUrl.includes('/socket.io') ||
        // req.originalUrl.includes('nextjs') ||
        // req.originalUrl.startsWith('/_next') ||
        req.originalUrl.startsWith('/_next/hmr')
    // req.originalUrl.startsWith('/login') ||
    // req.originalUrl.startsWith('/api');

    if (isNextStaticOrApi) {
        console.log(`[Chat] Маршрутизация статики/API на nextjsProxy_chat`);
        return chat_t2020_ru_ws(req, res, next); // Важно: return!
    } else {
        return chat_t2020_ru(req, res, next);
    }
});


// 2. Middleware для редиректа в Express
app.use((req, res, next) => {
    if (req.headers.host === 'yd.t2020.ru') {
        return res.redirect(301, `https://ya.t2020.ru${req.url}`);
    }
    next();
});

// 4. Запускаем HTTP сервер
http.createServer(app).listen(port, () => {
    console.log(`Server listens http://${host}:${port}`);
});

// 5. Запускаем HTTPS сервер с использованием подготовленных данных
const server = https.createServer({
    // Дефолтные ключи
    key: certOptions['t2020.ru'].key,
    cert: certOptions['t2020.ru'].cert,

    SNICallback: (domain, cb) => {
        // console.log(`Domain: ${domain}`);
        const cleanDomain = domain.replace(/^www\./, '');
        // console.log(`cleanDomain: ${cleanDomain.replace('5.164.29.62', 't2020.ru')}`);
        const context = secureContexts[cleanDomain];
        if (context) {
            cb(null, context);
        } else {
            // Если домен не найден, отдаем дефолтный контекст t2020.ru
            cb(null, secureContexts['t2020.ru']);
        }
    }
}, app).listen(portssh, () => {
    console.log(`Server listens https://${host}:${portssh}`);
});

// Защита от зацикливания WebSocket
const wsConnectionTracker = {
    connections: new Map(), // key: host+url, value: { count, timer }

    check(host, url) {
        const key = `${host}${url}`;
        const now = Date.now();

        if (!this.connections.has(key)) {
            this.connections.set(key, { count: 0, timer: null });
        }

        const data = this.connections.get(key);
        data.count++;

        // Сбрасываем через 3 секунды
        clearTimeout(data.timer);
        data.timer = setTimeout(() => {
            data.count = 0;
            console.log(`[WS] Counter reset for ${key}`);
        }, 3000);

        return data.count;
    }
};

// Перенаправление WebSocket Handshake (событие upgrade)
server.on('upgrade', (req, socket, head) => {
    const host = req.headers.host || '';
    const url = req.url || '';

    // Проверяем на зацикливание
    const count = wsConnectionTracker.check(host, url);

    if (count > 5) {
        console.log(`[WS] ✅ ЗАЦИКЛИВАНИЕ обнаружено! ${host}${url} ([${count}] попыток)`);
        socket.destroy();

        // Перезапускаем сервер если зацикливание
        setTimeout(() => {
            process.exit(1);
        }, 100);
        return;
    }

    if (host.startsWith('t2020.ru') && (url.startsWith('/wss_82/') || url.startsWith('/api'))) {
        // console.log(`[WS /wss_82/] Перенаправление upgrade ${host}${url}`);
        return wss_82.upgrade(req, socket, head);

    } else if (host.startsWith('t2020.ru') && (url.startsWith('/wss/') || url.startsWith('/api'))) {
        // console.log(`[WS /wss/] Перенаправление upgrade ${host}${url}`);
        return wss.upgrade(req, socket, head);

    } else if (host.startsWith('chat.t2020.ru') &&
        (url.startsWith('/_next/hmr') ||
            url.startsWith('/socket.io') ||
            url.startsWith('/api'))) {
        // console.log(`[WS chat] Перенаправление upgrade ${host}${url}`);
        return chat_t2020_ru_ws.upgrade(req, socket, head);

    } else if ((host.startsWith('yd.t2020.ru') || (host.startsWith('ya.t2020.ru'))) &&
        (url.startsWith('/_next/hmr') ||
            url.startsWith('/socket.io') ||
            url.startsWith('/_next/static') ||
            url.startsWith('/api'))) {
        // console.log(`[YD] Перенаправление upgrade ${host}${url}`);
        return nextjsProxyYD_WS.upgrade(req, socket, head);

    } else if (host.startsWith('demo.t2020.ru') &&
        (url.startsWith('/_next/hmr') ||
            url.startsWith('/socket.io') ||
            url.startsWith('/_next/static') ||
            url.startsWith('/api'))) {
        // console.log(`[Demo] Перенаправление upgrade ${host}${url}`);
        return nextjsProxyDemo_WS.upgrade(req, socket, head);

    } else {
        // console.log(`[WS] Отклонено соединение (Host: ${host}, URL: ${url})`);
        socket.destroy();
        next();
    }
});


// Middleware трекинга
app.use((req, res, next) => {
    const path = req.path;

    if (analytics.userSessions.has(analytics.ip)) { return next(); }

    // 1. Пропускаем админку, API и служебные сборки Next.js
    if (path.startsWith('/admin') ||
        path.startsWith('/api/analytics') ||
        path.startsWith('/_next')
    ) {
        return next();
    }

    // 2. Игнорируем служебные запросы данных Next.js (RSC / Data Fetching)
    if ((req.query && req.query._rsc) || req.headers['x-nextjs-data'] || req.get('RSC')) {
        return next();
    }

    // 3. Игнорируем запросы к статическим файлам и ассетам
    const staticExtensions = ['.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.woff', '.woff2'];
    if (staticExtensions.some(ext => req.originalUrl.split('?')[0].endsWith(ext))) {
        return next();
    }

    // Передаем управление в трекер (без использования опасного `this.ip`)
    const cleanedIp = req.ip.replace('::ffff:', '');
    analytics.ip = cleanedIp;
    // if (!this.userSessions.has(ip)) {
    const reset = '\x1b[0m';
    const darkGray = '\x1b[90m'; // Для меток
    const lightGray = '\x1b[37m'; // Для значений
    const brightBlue = '\x1b[94m';
    // Получаем значение рефера. Если его нет, по умолчанию ставим '-'.
    const refererValue = req.headers.referer || req.referer || '-';

    // Выбираем цвет: если значение '-', используем серый (darkGray), иначе — серо-синий (brightBlue).
    // Обратите внимание: в вашем коде для серого используется darkGray, а для синего — brightBlue.
    const refererColor = refererValue === '-' ? darkGray : brightBlue;
    let decoded = '';
    let decodedUrl = '';
    try {
        decoded = decodeURIComponent(req.originalUrl.replace(/%(?![0-9a-fA-F]{2})/g, '%25'));
        decodedUrl = decodeURIComponent(req.url.replace(/%(?![0-9a-fA-F]{2})/g, '%25'));
    } catch (e) {
        // console.error("Ошибка декодирования URI:", url);
        decoded = ''; // или ''; в зависимости от вашей логики
        decodedUrl = '';
    }
    // Собираем итоговую строку, подставляя выбранный цвет.
    const baseLog = `${darkGray}[ip]:${reset} ${lightGray}${cleanedIp}${reset} ` +
        `${darkGray}[referer]:${reset} ${refererColor}${refererValue}${reset} ` +
        `${darkGray}[url]:${reset} ${lightGray}${req.hostname || decoded}${decodedUrl}${reset}`;
    console.log(baseLog);
    trackVisit(req);
    // }
    next();
});


// MIDDLEWARE: Проверяем домен и применяем прокси

app.use((req, res, next) => {
    const host = req.headers.host;
    const urlPath = req.originalUrl;

    // ИСПРАВЛЕНО: Если ХОТЯ БЫ ОДИН из них не строка, идем дальше, 
    // чтобы не вызвать ошибку .startsWith() или .includes()
    if (typeof host !== 'string' || typeof urlPath !== 'string') {
        return next();
    }

    // Теперь мы на 100% уверены, что host и urlPath — это строки
    const isNextjsRequest = host.startsWith('t2020.ru') && (
        urlPath.includes('nextjs') ||
        urlPath.startsWith('/_next') ||
        urlPath.startsWith('/crypto')
    );

    if (isNextjsRequest) {
        return nextjsProxy(req, res, next);
    }

    // Дальше используем переменную 'host' вместо 'req.headers.host' (это быстрее и чище)
    if (host.startsWith('game.t2020.ru')) {
        return apiProxyGame(req, res, next);
    }
    else if (host.startsWith('poker.t2020.ru') || host.startsWith('www.poker.t2020.ru')) {
        console.log('poker');
        return apiProxyPoker(req, res, next);
    }
    else if (host.startsWith('watermark-png.ru') || host.startsWith('www.watermark-png.ru')) {
        return apiProxyWatermark(req, res, next);
    }
    else if (host.startsWith('direct.t2020.ru') || host.startsWith('www.direct.t2020.ru')) {
        return phpDirectProxy(req, res, next);
    }
    else if (
        host.startsWith('yd.t2020.ru') || host.startsWith('www.yd.t2020.ru') ||
        host.startsWith('ya.t2020.ru') || host.startsWith('www.ya.t2020.ru')
    ) {
        return nextjsProxyYD(req, res, next);
    }
    else if (host.startsWith('demo.t2020.ru')) { // ИСПРАВЛЕНО: убрал дублирование
        return nextjsProxyDemo(req, res, next);
    }
    else if (host.startsWith('p2p.t2020.ru')) {
        return apiProxyP2p(req, res, next);
    }
    else {
        next();
    }
});


function trackVisit(req) {
    try {
        const cleanedIp = req.ip.replace('::ffff:', '');

        // 1. Проверка на дубликат
        // if (processedIps.has(cleanedIp)) {
        //     // console.log('[trackVisit] Duplicate IP ignored:', cleanedIp);
        //     return; // Выходим, если IP уже видели
        // }

        // Помечаем IP как обработанный
        // processedIps.add(cleanedIp);

        // Чтобы память не росла бесконечно, можно ограничить размер Set
        // или периодически его очищать (например, раз в сутки).

        const parser = new UAParser();
        parser.setUA(req.get('User-Agent') || '');

        const geo = geoip.lookup(cleanedIp);
        const referer = req.get('Referer') || req.headers.referer;
        const browser = parser.getBrowser();
        const country = geo?.country || 'Unknown';

        let finalQuery = { ...req.query };
        if (!finalQuery.utm_source && referer) {
            try {
                const refererUrl = new URL(referer);
                const refererParams = Object.fromEntries(refererUrl.searchParams.entries());
                if (refererParams.utm_source) {
                    finalQuery = { ...finalQuery, ...refererParams };
                }
            } catch (e) { }
        }

        const source = detectSource(referer, finalQuery);

        let decoded = '';
        try {
            decoded = decodeURIComponent(req.originalUrl.replace(/%(?![0-9a-fA-F]{2})/g, '%25'));
        } catch (e) {
            // console.error("Ошибка декодирования URI:", url);
            decoded = ''; // или ''; в зависимости от вашей логики
        }

        const visitData = {
            ip: cleanedIp,
            userAgent: req.get('User-Agent'),
            browser: browser || {},
            os: parser.getOS() || {},
            device: parser.getDevice() || {},
            deviceType: parser.getDevice().type || 'desktop',
            country: country,
            city: geo?.city || 'Unknown',
            referer: decodeURIComponent(referer) || null,
            url: (req.headers.host || req.hostname) + decoded,
            source,
            utm: {
                source: finalQuery.utm_source || null,
                medium: finalQuery.utm_medium || null,
                campaign: finalQuery.utm_campaign || null,
                term: finalQuery.utm_term || null,
                content: finalQuery.utm_content || null
            }
        };

        // Исправленная проверка переменных
        // console.log('[ip]: ', visitData.ip, '[referer]: ', referer || '', '[url]: ', visitData.url, JSON.stringify(visitData.utm));
        // Красивый вывод 
        const brightBlack = '\x1b[90m'; // Темно-серый (часто используется для комментариев)
        const brightRed = '\x1b[91m';
        const brightGreen = '\x1b[92m';
        const brightYellow = '\x1b[93m';
        const brightBlue = '\x1b[94m';
        const brightMagenta = '\x1b[95m';
        const brightCyan = '\x1b[96m';
        const brightWhite = '\x1b[97m';
        // Функция для проверки, есть ли в объекте хотя бы одно не-null значение 
        const reset = '\x1b[0m';
        const darkGray = '\x1b[90m'; // Для меток
        const lightGray = '\x1b[37m'; // Для значений

        if (browser && browser.name && country !== 'Unknown' && country === 'RU') {
            // console.log('[addVisit push]: ', JSON.stringify(visitData));
            analytics.addVisit(visitData);
        }

    } catch (error) {
        console.error('Ошибка трекинга:', error);
    }
}

function detectSource(referer, query) {
    // 1. Проверяем UTM (они уже могут быть вытащены как из URL, так и из Referer)
    if (query && query.utm_source) {
        return `UTM: ${query.utm_source}`;
    }

    if (!referer) return '-';

    const sources = {
        'google': 'Google',
        'alice': 'Alice',
        'yandex': 'Yandex',
        'bing': 'Bing',
        'facebook': 'Facebook',
        'vk.com': 'VK',
        'instagram': 'Instagram',
        'twitter': 'Twitter',
        'linkedin': 'LinkedIn',
        't.me': 'Telegram',
        'ok.ru': 'Odnoklassniki'
    };

    // 2. Ищем известные домены в referer
    for (const [domain, source] of Object.entries(sources)) {
        if (referer.includes(domain)) return source;
    }

    // 3. Если это обычный внешний сайт, пишем его hostname
    try {
        const url = new URL(referer);
        return `${url.hostname}`;
    } catch {
        return '-';
    }
}

// Роуты
app.get('/admin', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'dashboard.html'));
});
app.get('/chart.js', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'chart.js'));
});

// API для получения данных
app.get('/api/analytics/overview', (req, res) => {
    res.json(analytics.getOverview());
});

app.get('/api/analytics/sources', (req, res) => {
    res.json(analytics.getSources());
});

app.get('/api/analytics/countries', (req, res) => {
    res.json(analytics.getCountries());
});

app.get('/api/analytics/countries1', (req, res) => {
    res.json(analytics.getCountries1());
});

app.get('/api/analytics/devices', (req, res) => {
    res.json(analytics.getDevices());
});

app.get('/api/analytics/browsers', (req, res) => {
    res.json(analytics.getBrowsers());
});

app.get('/api/analytics/recent', (req, res) => {
    const limit = parseInt(req.query.limit) || 10;
    res.json(analytics.getRecentVisits(limit));
});

app.get('/api/analytics/daily', (req, res) => {
    const days = parseInt(req.query.days) || 30;
    res.json(analytics.getDailyStats(days));
});

// API для детальной информации об онлайн
app.get('/api/analytics/online-details', (req, res) => {
    const activeSessions = analytics.getOnlineDetails();

    res.json({
        onlineCount: activeSessions.length,
        activeSessions: activeSessions,
        calculationMethod: 'Уникальные пользователи с активностью за последние 5 минут',
        lastUpdate: new Date().toLocaleTimeString('ru-RU')
    });
});

// API для ручного сохранения
app.post('/api/analytics/save', async (req, res) => {
    const result = await analytics.manualSave();
    res.json(result);
});

// API для очистки данных
app.get('/api/analytics/clear', async (req, res) => {
    analytics.visits = [];
    analytics.stats.totalVisits = 0;
    analytics.stats.uniqueVisitors.clear();
    analytics.userSessions.clear();
    await analytics.saveData();
    console.log('analytics Данные очищены');
    res.json({ message: 'Данные очищены' });
});







// Middleware для обработки JSON
app.use(express.json())

app.post('/mexc', (req, res) => {
    // Получение данных из тела запроса
    const reqData = req.body

    function getYear() {
        let startyear = new Date(new Date().getTime()).getFullYear() + '.'
        startyear += String(Number(new Date(new Date().getTime()).getMonth()) + 1).length == 1 ?
            '0' + (String(Number(new Date(new Date().getTime()).getMonth()) + 1)) :
            (String(Number(new Date(new Date().getTime()).getMonth()) + 1))
        startyear += String(Number(new Date(new Date().getTime()).getDay()) - 1).length == 1 ?
            ('.0' + String(Number(new Date(new Date().getTime()).getDay()) - 1)) :
            String(Number(new Date(new Date().getTime()).getDay()) - 1)
        return startyear
    }
    function setNul(a) {
        const l = a.length == 1 ? '0' : ''
        return l + a
    }
    function getTm() {
        let d = new Date(new Date().getTime()).getHours().toString()
        let start = d + ':'
        start += setNul(new Date(new Date().getTime()).getMinutes().toString())
        start += ':'
        start += setNul(new Date(new Date().getTime()).getSeconds().toString())
        return start
    }
    console.log(`Полученные данные: ${getYear()} ${getTm()}`)

    // console.log('Полученные данные:', reqData)
    // reqData = reqData.toString().replaceAll("\'", "")
    // console.log('Convert:', reqData.symbol)

    // Отправка ответа
    // res.json({ message: 'Данные получены', data: reqData })
    // res.send(`${req.body.userName} - ${req.body.userAge}`)
    // res.writeHead(200, { 'Content-Type': 'text/plain' })
    // res.end('Данные получены')
    res.status(200).json({ message: 'ok' })
    res.end()

    function sendPostRequest(request_path, params) {
        params.timestamp = Date.now().toString()
        timestamp = params.timestamp
        const apiKey = 'mx0vgl2InmIgHsVG9l'//this.apiKey
        const apiSecret = '7bfbb040edd644328a17276ad8e135cc'//this.apiKey
        const stringifyKeyValuePair = ([key, value]) => {
            let valueString;
            if (typeof value === 'object') {
                valueString = JSON.stringify(value);
            } else {
                valueString = value;
            }
            return `${key}=${encodeURIComponent(valueString)}`;
        };
        const buildQueryString = params => {
            if (!params) return '';
            return Object.entries(params)
                .map(stringifyKeyValuePair)
                .join('&');
        };
        let queryString = buildQueryString({ ...params })
        queryString += '&signature=' + CryptoJS.enc.Hex.stringify(CryptoJS.HmacSHA256(queryString, apiSecret))
        const headers = {
            'Content-Type': 'application/json', // POST requests need this header
            'ApiKey': apiKey,
            'Request-Time': timestamp,
        }
        const options = {// /api/v3/order/mx0vgl2InmIgHsVG9l1752394624139{"symbol":"IMXUSDT","price":0.5055,"quantity":5,"side":"BUY","type":"LIMIT","timestamp":"1752394624139"}'
            hostname: 'api.mexc.com',
            path: request_path + '?' + queryString,
            method: 'POST',
            data: params,
            headers: headers
        }
        const req = httpss.request(options, (res) => {
            let data = ''
            res.on('data', (chunk) => {
                data += chunk
            })
            res.on('end', () => {
                console.log(data)
            })
        })
        if (params) {
            req.write(JSON.stringify(params))
        }
        req.end()
    }
    /**************************POST************************************************ */
    async function GetContractInformation(params) {//
        const getRequestPath = '/api/v3/order'
        let res = await sendPostRequest(getRequestPath, params)
        return await new Promise((resolve) => {
            resolve(res)
        })
    }
    const promise = new Promise((resolve) => resolve(GetContractInformation(
        reqData
        // {
        // symbol: 'IMXUSDT',
        // price: 0.5055,
        // quantity: 5,
        // side: 'BUY',
        // type: 'LIMIT',
        // }
    ))).then((result) => {
        if (result != undefined) {
            console.log('start')
            console.log(result)
            console.log('end')
        }
        // console.log(result.data)
        return result
    })
    // console.log('promise :>> ', promise)
})


// Вариант 3: Указание root (применимо, если вы хотите работать с относительными путями)

// app.get('/download', (req, res) => {
//     if (req.ip.replace('::ffff:', '') !== '192.168.1.1') {
//         return res.status(403).json({ message: 'Access denied' });
//     }
//     res.sendFile('trading_signals.txt', { root: __dirname });
// });
app.get('/download', (req, res) => {
    // Проверка IP адреса
    if (req.ip.replace('::ffff:', '') !== '192.168.1.1') {
        return res.status(403).json({ message: 'Access denied' });
    }

    // ЕСЛИ ЭТО HEAD-ЗАПРОС: отправляем только статус и заголовки, не читая файл
    if (req.method === 'HEAD') {
        return res.status(200).end();
    }

    // Для обычного GET-запроса отдаем файл
    return res.sendFile('trading_signals.txt', { root: __dirname }, (err) => {
        if (err && !res.headersSent) {
            // Дополнительная защита: если файл не найден или ошибка передачи
            return res.status(500).json({ message: 'File transfer error' });
        }
    });
});

app.get('/robots.txt', (req, res) => {
    res.sendFile('robots.txt', { root: __dirname });
});
app.get('/sitemap.xml', (req, res) => {
    res.sendFile('sitemap.xml', { root: __dirname });
});
app.get('/llms.txt', (req, res) => {
    res.sendFile('llms.txt', { root: __dirname });
});
app.get('/llms-full.txt', (req, res) => {
    res.sendFile('llms-full.txt', { root: __dirname });
});
app.get('/getpost/oneBot001.json', (req, res) => {
    res.sendFile('oneBot001.json', { root: __dirname + '/getpost/' });
});

app.use('/socket.io', express.static('t2020.ru'))
// app.use('/nextjs', express.static('nextjs'))
app.use('/.well-known/acme-challenge/', express.static('t2020.ru/.well-known/acme-challenge/'))
// app.use(express.static(__dirname))

// app.get('/', (req, res) => res.render('/public/index.html'))


app.use(express.static('public'));
app.use(express.urlencoded({ extended: true }));

// Вспомогательные функции парсинга
function parseTradingSignals(fileContent) {
    const signals = [];
    const lines = fileContent.trim().split('\n');
    const combinedLines = [];
    let currentLine = '';

    for (const line of lines) {
        const trimmedLine = line.trim();
        if (!trimmedLine) continue;

        if (trimmedLine.startsWith('google_table_bot')) {
            if (currentLine) combinedLines.push(currentLine);
            currentLine = trimmedLine;
        } else {
            currentLine += ' ' + trimmedLine;
        }
    }
    if (currentLine) combinedLines.push(currentLine);

    for (let lineNum = 0; lineNum < combinedLines.length; lineNum++) {
        try {
            const line = combinedLines[lineNum];
            let cleanLine = line.replace('google_table_bot, ', '').trim();
            const timestampEnd = cleanLine.indexOf(']');
            if (timestampEnd === -1) continue;

            const timestampPart = cleanLine.substring(0, timestampEnd + 1).trim();
            let restOfLine = cleanLine.substring(timestampEnd + 1).trim();
            const timestampClean = timestampPart.slice(1, -1);

            const fixedLine = timestampPart + ', ' + restOfLine;
            const parts = fixedLine.split(', ');

            const params = {};
            for (const part of parts) {
                if (part.includes(':')) {
                    const [key, ...valueParts] = part.split(':');
                    const value = valueParts.join(':').trim();
                    params[key.trim()] = value;
                }
            }

            const safeFloat = (value, defaultValue = 0.0) => {
                try {
                    let cleanValue = String(value).trim();
                    if (cleanValue.endsWith('%')) cleanValue = cleanValue.slice(0, -1);
                    return parseFloat(cleanValue) || defaultValue;
                } catch {
                    return defaultValue;
                }
            };

            const winTradesStr = params.win_trades || '0 (0%)';
            const winTradesMatch = winTradesStr.match(/(\d+)\s*\(([\d.-]+)%\)/);

            const signal = {
                line_number: lineNum + 1,
                timestamp: timestampClean,
                symbol: params.symbol || '',
                signal_type: params.signal_type || '',
                timeframe: params.timeframe || '',
                real: safeFloat(params.real),
                pnl: safeFloat(params.pnl),
                win_trades: winTradesMatch ? parseInt(winTradesMatch[1]) : 0,
                win_rate: winTradesMatch ? parseFloat(winTradesMatch[2]) : 0.0,
                profit_factor: safeFloat(params.profit_factor),
                en0: safeFloat(params.en0),
                en1: safeFloat(params.en1),
                tp0: safeFloat(params.tp0),
                tp1: safeFloat(params.tp1),
                tp2: safeFloat(params.tp2),
                tp3: safeFloat(params.tp3),
                stop_loss: safeFloat(params.stop_loss),
            };
            signals.push(signal);
        } catch (error) {
            console.log(`Error parsing line ${lineNum + 1}:`, error.message);
            continue;
        }
    }
    return signals;
}

function calculateRealPnl(signals) {
    try {
        if (!signals || signals.length === 0) {
            console.log('⚠️ Нет сигналов для расчета PnL');
            return [];
        }

        console.log('🔧 Начало расчета PnL, сигналов:', signals.length);

        // Сортируем по времени
        const sortedSignals = signals
            .filter(s => s.timestamp && s.timestamp.trim() !== '')
            .map(s => {
                try {
                    return {
                        ...s,
                        timestamp: moment(s.timestamp, 'DD.MM.YYYY HH:mm').toDate()
                    };
                } catch (error) {
                    console.log('⚠️ Ошибка парсинга времени:', s.timestamp, error.message);
                    return null;
                }
            })
            .filter(s => s !== null)
            .sort((a, b) => a.timestamp - b.timestamp);

        console.log(`✅ Отсортировано сигналов: ${sortedSignals.length}`);

        const signalTypes = {};
        sortedSignals.forEach(s => {
            signalTypes[s.signal_type] = (signalTypes[s.signal_type] || 0) + 1;
        });
        console.log('📊 Типы сигналов:', signalTypes);

        const openPositions = {};
        const trades = [];
        const capitalPerPair = 100.0;

        for (const signal of sortedSignals) {
            const { symbol, timeframe, real, signal_type, en0, timestamp } = signal;

            if (!symbol || !timeframe) {
                console.log('⚠️ Пропускаем сигнал без символа или таймфрейма:', signal);
                continue;
            }

            const positionKey = `${symbol}_${timeframe}`;

            if (['long', 'short'].includes(signal_type)) {
                if (!openPositions[positionKey]) {
                    openPositions[positionKey] = {
                        symbol,
                        timeframe,
                        type: signal_type,
                        entry_price: en0,
                        entry_time: timestamp,
                        capital: capitalPerPair
                    };
                    // console.log(`📈 Открыта позиция: ${positionKey} ${signal_type}`);
                }
            } else if (['exit_long', 'exit_short'].includes(signal_type)) {
                const position = openPositions[positionKey];
                if (position) {
                    if ((position.type === 'long' && signal_type === 'exit_long') ||
                        (position.type === 'short' && signal_type === 'exit_short')) {

                        const actualExitPrice = real;
                        let pnlPercent, pnlUsdt;

                        if (position.type === 'long') {
                            pnlPercent = ((actualExitPrice - position.entry_price) / position.entry_price) * 100;
                            pnlUsdt = (pnlPercent / 100) * position.capital;
                        } else {
                            pnlPercent = ((position.entry_price - actualExitPrice) / position.entry_price) * 100;
                            pnlUsdt = (pnlPercent / 100) * position.capital;
                        }

                        const trade = {
                            symbol,
                            timeframe,
                            position_type: position.type,
                            entry_time: position.entry_time,
                            exit_time: timestamp,
                            entry_price: position.entry_price,
                            exit_price: actualExitPrice,
                            pnl_percent: pnlPercent,
                            pnl_usdt: pnlUsdt,
                            capital_used: position.capital
                        };
                        trades.push(trade);
                        // console.log(`📉 Закрыта позиция: ${positionKey}, PnL: ${pnlUsdt.toFixed(2)}`);

                        delete openPositions[positionKey];
                    }
                }
            }
        }

        console.log(`✅ Найдено завершенных сделок: ${trades.length}`);
        return trades;

    } catch (error) {
        console.error('💥 Ошибка в calculateRealPnl:', error);
        console.error('Stack trace:', error.stack);
        return [];
    }
}

function calculateAdvancedMetrics(trades) {
    if (!trades || trades.length === 0) return {};

    const metrics = {};

    try {
        // 1. Топ символов по прибыльности
        const symbolPnl = {};
        trades.forEach(trade => {
            if (!symbolPnl[trade.symbol]) {
                symbolPnl[trade.symbol] = {
                    pnl_usdt: 0,
                    pnl_percent: 0,
                    trades_count: 0
                };
            }
            symbolPnl[trade.symbol].pnl_usdt += trade.pnl_usdt;
            symbolPnl[trade.symbol].pnl_percent += trade.pnl_percent;
            symbolPnl[trade.symbol].trades_count++;
        });

        // Вычисляем средние значения
        Object.keys(symbolPnl).forEach(symbol => {
            symbolPnl[symbol].pnl_percent /= symbolPnl[symbol].trades_count;
        });

        // Сортируем по PnL
        const sortedSymbols = Object.entries(symbolPnl)
            .sort(([, a], [, b]) => b.pnl_usdt - a.pnl_usdt)
            .slice(0, 10)
            .reduce((acc, [symbol, data]) => {
                acc[symbol] = {
                    pnl_usdt: Math.round(data.pnl_usdt * 100) / 100,
                    pnl_percent: Math.round(data.pnl_percent * 100) / 100,
                    trades_count: data.trades_count
                };
                return acc;
            }, {});

        metrics.top_symbols = sortedSymbols;

        // 2. Статистика по стратегиям
        const strategyStats = {};
        ['long', 'short'].forEach(type => {
            const typeTrades = trades.filter(t => t.position_type === type);
            if (typeTrades.length > 0) {
                strategyStats[type] = {
                    sum: Math.round(typeTrades.reduce((sum, t) => sum + t.pnl_usdt, 0) * 100) / 100,
                    mean: Math.round(typeTrades.reduce((sum, t) => sum + t.pnl_usdt, 0) / typeTrades.length * 100) / 100,
                    count: typeTrades.length,
                    pnl_percent_mean: Math.round(typeTrades.reduce((sum, t) => sum + t.pnl_percent, 0) / typeTrades.length * 100) / 100
                };
            }
        });
        metrics.strategy_stats = { pnl_usdt: strategyStats };

        // 3. Статистика по таймфреймам
        const timeframeStats = {};
        const timeframes = [...new Set(trades.map(t => t.timeframe))];
        timeframes.forEach(tf => {
            const tfTrades = trades.filter(t => t.timeframe === tf);
            if (tfTrades.length > 0) {
                timeframeStats[tf] = {
                    sum: Math.round(tfTrades.reduce((sum, t) => sum + t.pnl_usdt, 0) * 100) / 100,
                    mean: Math.round(tfTrades.reduce((sum, t) => sum + t.pnl_usdt, 0) / tfTrades.length * 100) / 100,
                    count: tfTrades.length,
                    pnl_percent_mean: Math.round(tfTrades.reduce((sum, t) => sum + t.pnl_percent, 0) / tfTrades.length * 100) / 100
                };
            }
        });
        metrics.timeframe_stats = { pnl_usdt: timeframeStats };

        // 4. Win rates
        const winRates = {};
        Object.keys(symbolPnl).forEach(symbol => {
            const symbolTrades = trades.filter(t => t.symbol === symbol);
            const wins = symbolTrades.filter(t => t.pnl_usdt > 0).length;
            winRates[symbol] = Math.round((wins / symbolTrades.length) * 100 * 100) / 100;
        });
        metrics.win_rates = winRates;

        // 5. Общая статистика
        const totalTrades = trades.length;
        const winningTrades = trades.filter(t => t.pnl_usdt > 0).length;
        const losingTrades = trades.filter(t => t.pnl_usdt < 0).length;
        const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

        const profitTrades = trades.filter(t => t.pnl_usdt > 0);
        const lossTrades = trades.filter(t => t.pnl_usdt < 0);

        const totalProfit = profitTrades.reduce((sum, t) => sum + t.pnl_usdt, 0);
        const totalLoss = Math.abs(lossTrades.reduce((sum, t) => sum + t.pnl_usdt, 0));

        const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : Infinity;

        metrics.advanced_stats = {
            total_trades: totalTrades,
            winning_trades: winningTrades,
            losing_trades: losingTrades,
            win_rate: Math.round(winRate * 100) / 100,
            total_pnl_usdt: Math.round(trades.reduce((sum, t) => sum + t.pnl_usdt, 0) * 100) / 100,
            avg_pnl_percent: Math.round(trades.reduce((sum, t) => sum + t.pnl_percent, 0) / totalTrades * 100) / 100,
            best_trade: Math.round(Math.max(...trades.map(t => t.pnl_usdt)) * 100) / 100,
            worst_trade: Math.round(Math.min(...trades.map(t => t.pnl_usdt)) * 100) / 100,
            profit_factor: profitFactor !== Infinity ? Math.round(profitFactor * 100) / 100 : '∞'
        };

    } catch (error) {
        console.log('Ошибка в calculate_advanced_metrics:', error);
        metrics.top_symbols = {};
        metrics.strategy_stats = { pnl_usdt: {} };
        metrics.timeframe_stats = { pnl_usdt: {} };
        metrics.win_rates = {};
        metrics.advanced_stats = {};
    }

    return metrics;
}

function calculatePnLChartData(trades) {
    if (!trades || trades.length === 0) return [];

    // Сортируем сделки по времени закрытия
    const sortedTrades = [...trades].sort((a, b) => new Date(a.exit_time) - new Date(b.exit_time));

    const pnlData = [];
    let cumulativePnl = 0;

    sortedTrades.forEach(trade => {
        cumulativePnl += trade.pnl_usdt;
        pnlData.push({
            time: moment(trade.exit_time).format('YYYY-MM-DD HH:mm'),
            pnl: Math.round(trade.pnl_usdt * 100) / 100,
            cumulative_pnl: Math.round(cumulativePnl * 100) / 100,
            symbol: trade.symbol,
            timeframe: trade.timeframe,
            position_type: trade.position_type
        });
    });

    return pnlData;
}


// Маршруты
app.get('/api/data', (req, res) => {

    const https = require('https');
    const fs = require('fs');

    const fileUrl = 'https://t2020.ru/download'; // Замените на нужный URL
    const localFilePath = 'downloaded_file.txt'; // Путь для сохранения файла

    const file = fs.createWriteStream(localFilePath);

    function getLocalFileSize(filePath) {
        try {
            const stats = fs.statSync(filePath);
            return stats.size; // Размер в байтах
        } catch (error) {
            // Если файл не существует или другая ошибка
            console.error(`Ошибка чтения локального файла: ${error.message}`);
            return null;
        }
    }

    async function getUrlFileSize(url) {
        try {
            // Отправляем HEAD запрос, чтобы не скачивать файл целиком
            const response = await fetch(url, {
                method: 'HEAD'
            });

            if (!response.ok) {
                throw new Error(`Ошибка сервера: ${response.status}`);
            }

            const contentLength = response.headers.get('content-length');

            if (contentLength) {
                return parseInt(contentLength, 10); // Размер в байтах (число)
            } else {
                console.warn('Заголовок Content-Length не найден в ответе сервера.');
                return null;
            }

        } catch (error) {
            console.error(`Ошибка при получении размера удаленного файла: ${error.message}`);
            return null;
        }
    }
    const getLocal = getLocalFileSize('trading_signals.txt')
    getUrlFileSize('https://t2020.ru/download').then(e => {
        if (getLocal != e) {

            https.get(fileUrl, (response) => {
                response.pipe(file); // Перенаправляет поток ответа в файл

                file.on('finish', () => {
                    file.close(); // Закрывает поток записи после завершения
                    console.log('Файл успешно загружен!');
                    fs.renameSync(localFilePath, 'trading_signals.txt');
                    console.log(`✅ Временный файл ${localFilePath} успешно заменен новым downloaded_file.txt Обновление завершено.`);
                });

            }).on('error', (err) => {
                try {
                    fs.unlink(localFilePath); // Удаляет файл, если произошла ошибка
                } catch { }
                console.error('Ошибка при загрузке файла:', err.message);
                console.error(`Удаляем файл, если произошла ошибка`);
            });
        }
    })



    try {
        console.log('🔍 Начало обработки /api/data', __dirname);

        // const filePath = path.join(__dirname, '/trading_signals.txt');
        const filePath = __dirname + '/trading_signals.txt';
        console.log('📁 Проверяем файл:', filePath);

        if (!fs.existsSync(filePath)) {
            console.log('❌ Файл не найден');
            return res.json({
                stats: null,
                capital_data: [],
                table_data: [],
                pnl_chart_data: [],
                symbols: [],
                timeframes: [],
                advanced_metrics: {}
            });
        }

        console.log('✅ Файл найден, читаем содержимое...');
        const fileContent = fs.readFileSync(filePath, 'utf8');
        console.log('📄 Размер файла:', fileContent.length, 'символов');

        const signals = parseTradingSignals(fileContent);
        console.log('📊 Распарсено сигналов:', signals.length);

        if (signals.length === 0) {
            console.log('❌ Нет сигналов после парсинга');
            return res.json({
                stats: null,
                capital_data: [],
                table_data: [],
                pnl_chart_data: [],
                symbols: [],
                timeframes: [],
                advanced_metrics: {}
            });
        }

        console.log('🧮 Вычисляем сделки...');
        const trades = calculateRealPnl(signals);
        console.log('💰 Найдено сделок:', trades.length);

        if (trades.length === 0) {
            console.log('❌ Нет завершенных сделок');
            return res.json({
                stats: null,
                capital_data: [],
                table_data: [],
                pnl_chart_data: [],
                symbols: [],
                timeframes: [],
                advanced_metrics: {}
            });
        }

        // Основная статистика
        console.log('📈 Вычисляем статистику...');
        const totalTrades = trades.length;
        const winningTrades = trades.filter(t => t.pnl_usdt > 0).length;
        const losingTrades = trades.filter(t => t.pnl_usdt < 0).length;
        const totalPnlUsdt = trades.reduce((sum, t) => sum + t.pnl_usdt, 0);
        const avgPnlPercent = trades.reduce((sum, t) => sum + t.pnl_percent, 0) / totalTrades;

        const uniquePairs = [...new Set(trades.map(t => `${t.symbol}_${t.timeframe}`))];
        const totalCapital = uniquePairs.length * 100;
        const finalCapital = totalCapital + totalPnlUsdt;
        const roi = (totalPnlUsdt / totalCapital) * 100;

        // Расширенная аналитика
        console.log('📊 Вычисляем расширенную аналитику...');
        const advancedMetrics = calculateAdvancedMetrics(trades);

        // Данные для графика капитала
        console.log('📉 Формируем данные для графика капитала...');
        const sortedTrades = [...trades].sort((a, b) => new Date(a.exit_time) - new Date(b.exit_time));
        const capitalData = [];
        let currentCapital = totalCapital;

        sortedTrades.forEach(trade => {
            currentCapital += trade.pnl_usdt;
            capitalData.push({
                time: moment(trade.exit_time).format('YYYY-MM-DD HH:mm'),
                capital: Math.round(currentCapital * 100) / 100,
                symbol: trade.symbol,
                timeframe: trade.timeframe
            });
        });

        // Данные для PnL графиков
        console.log('💰 Формируем данные для PnL графиков...');
        const pnlChartData = calculatePnLChartData(trades);

        // Данные для таблицы
        console.log('📋 Формируем табличные данные...');
        const tableData = trades.map(trade => ({
            symbol: trade.symbol,
            timeframe: trade.timeframe,
            position_type: trade.position_type,
            entry_time: moment(trade.entry_time).format('YYYY-MM-DD HH:mm'),
            exit_time: moment(trade.exit_time).format('YYYY-MM-DD HH:mm'),
            entry_price: Math.round(trade.entry_price * 100000) / 100000,
            exit_price: Math.round(trade.exit_price * 100000) / 100000,
            pnl_percent: Math.round(trade.pnl_percent * 100) / 100,
            pnl_usdt: Math.round(trade.pnl_usdt * 100) / 100
        }));

        console.log('✅ Все данные подготовлены:', {
            trades: trades.length,
            capital_data: capitalData.length,
            pnl_chart_data: pnlChartData.length,
            table_data: tableData.length
        });

        res.json({
            stats: {
                total_trades: totalTrades,
                winning_trades: winningTrades,
                losing_trades: losingTrades,
                win_rate: Math.round((winningTrades / totalTrades) * 100 * 100) / 100,
                total_pnl_usdt: Math.round(totalPnlUsdt * 100) / 100,
                avg_pnl_percent: Math.round(avgPnlPercent * 100) / 100,
                total_capital: totalCapital,
                final_capital: Math.round(finalCapital * 100) / 100,
                roi: Math.round(roi * 100) / 100,
                symbols: [...new Set(trades.map(t => t.symbol))].sort(),
                timeframes: [...new Set(trades.map(t => t.timeframe))].sort()
            },
            capital_data: capitalData,
            table_data: tableData,
            pnl_chart_data: pnlChartData,
            symbols: [...new Set(trades.map(t => t.symbol))].sort(),
            timeframes: [...new Set(trades.map(t => t.timeframe))].sort(),
            advanced_metrics: advancedMetrics
        });

    } catch (error) {
        console.error('💥 КРИТИЧЕСКАЯ ОШИБКА в /api/data:', error);
        console.error('Stack trace:', error.stack);
        res.status(500).json({
            error: 'Internal server error',
            message: error.message,
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
        });
    }
});

app.get('/api/advanced_metrics', (req, res) => {
    try {
        const filePath = path.join(__dirname, 'trading_signals.txt');
        if (!fs.existsSync(filePath)) {
            return res.json({});
        }

        const fileContent = fs.readFileSync(filePath, 'utf8');
        const signals = parseTradingSignals(fileContent);
        const trades = calculateRealPnl(signals);

        if (trades.length === 0) {
            return res.json({});
        }

        const metrics = calculateAdvancedMetrics(trades);
        res.json(metrics);
    } catch (error) {
        console.error('Error getting advanced metrics:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});


// ── 404 ──

app.use(function (req, res, next) {
    if (req.url.includes('/chart/fetchAPI')) {
        try {
            let parameters = url.parse(req.url, true).query
            for (const keys in parameters) {
                if (keys == 'endpoint' || keys == 'en0') {
                    continue
                }
                else {
                    parameters['endpoint'] += '&' + keys + '=' + parameters[keys]
                    if (keys == 'interval') {
                        ssymbol = parameters['endpoint'].split('?')[1].split('=')[1].split('&')[0]
                        interval = parameters[keys]
                    }
                }
            }

            request({
                method: 'GET',
                url: parameters['endpoint'],
                resolveWithFullResponse: true
            })
                .then((r1) => {
                    if (!res.headersSent) {              // ← защита
                        res.send(JSON.parse(r1.body))
                    }
                })
                .catch((err) => {
                    console.log(err)
                    if (!res.headersSent) {
                        res.status(502).json({ error: 'proxy_failed' })
                    }
                })
            return;   // ← ⚠️ ВАЖНО: не вызываем next() после запуска асинхронного запроса
        }
        catch (e) {
            console.log(e)
            if (!res.headersSent) {
                return res.status(500).json({ error: 'internal' })
            }
            return
        }
    }
    next();   // ← next() только если не обработали запрос
}, cors())

app.use((req, res, next) => {
    const wantsJson =
        req.path.startsWith('/api') ||
        (req.headers.accept || '').includes('application/json');

    // лог (у вас уже есть [404]-логгер — можно объединить)
    const line = `[404] ${req.method} ${req.originalUrl} ip=${req.ip} ua="${req.headers['user-agent'] || '-'}"`;
    console.log(new Date().toISOString(), line);
    if (typeof writeLog === 'function') writeLog(line);

    if (wantsJson) {
        return res.status(404).json({ error: 'not_found', path: req.path });
    }
    res.status(404).sendFile(path.join(__dirname, 'public', '404.html'));
});

// ── 500 (обязательно с 4 аргументами) ──
app.use((err, req, res, next) => {
    console.error('[500]', req.method, req.originalUrl, err);
    if (res.headersSent) return next(err);
    res.status(500).send('Internal Server Error');
});

