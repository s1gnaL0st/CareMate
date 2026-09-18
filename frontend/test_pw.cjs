const { chromium } = require('playwright');
const fs = require('fs');

function launchOptions() {
    const configured = process.env.PLAYWRIGHT_EXECUTABLE_PATH;
    const systemChrome = process.platform === 'win32'
        ? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        : '/usr/bin/google-chrome';
    const executablePath = configured || (fs.existsSync(systemChrome) ? systemChrome : undefined);
    return executablePath ? { headless: true, executablePath } : { headless: true };
}

(async () => {
    try {
        const baseUrl = process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:5173';
        console.log('Launching browser...');
        const browser = await chromium.launch(launchOptions());
        console.log('Creating page...');
        const page = await browser.newPage();
        console.log(`Navigating to ${baseUrl}...`);
        await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
        console.log('Page title:', await page.title());
        await browser.close();
        console.log('Success!');
    } catch (error) {
        console.error('Test failed:', error);
        process.exitCode = 1;
    }
})();
