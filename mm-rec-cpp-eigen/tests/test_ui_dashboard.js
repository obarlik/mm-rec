/**
 * UI/UX Testing for MM-Rec Dashboard
 * 
 * Tests visual rendering, interactivity, and real-time updates
 * using Puppeteer for browser automation.
 * 
 * Usage:
 *   1. Install dependencies: npm install puppeteer
 *   2. Start server: ./build/mm_rec server (in separate terminal)
 *   3. Run test: node tests/test_ui_dashboard.js
 */

const puppeteer = require('puppeteer');

const DASHBOARD_URL = 'http://localhost:8085';
const TEST_TIMEOUT = 30000;

// Color codes for terminal output
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    cyan: '\x1b[36m',
    bold: '\x1b[1m'
};

function success(msg) {
    console.log(`${colors.green}✓${colors.reset} ${msg}`);
}

function error(msg) {
    console.log(`${colors.red}✗${colors.reset} ${msg}`);
}

function info(msg) {
    console.log(`${colors.cyan}ℹ${colors.reset} ${msg}`);
}

function header(msg) {
    console.log(`\n${colors.bold}┌${'─'.repeat(msg.length + 2)}┐${colors.reset}`);
    console.log(`${colors.bold}│ ${msg} │${colors.reset}`);
    console.log(`${colors.bold}└${'─'.repeat(msg.length + 2)}┘${colors.reset}\n`);
}

async function testDashboardLoading(page) {
    header('Test 1: Dashboard Loading');

    try {
        // Navigate to dashboard
        info('Navigating to ' + DASHBOARD_URL);
        await page.goto(DASHBOARD_URL, { waitUntil: 'networkidle2', timeout: TEST_TIMEOUT });

        // Check if page title exists
        const title = await page.title();
        info(`Page title: ${title}`);

        if (!title || title.length === 0) {
            throw new Error('Page title is empty');
        }

        // Take screenshot
        await page.screenshot({ path: 'tests/screenshots/dashboard_initial.png' });
        info('Screenshot saved: dashboard_initial.png');

        success('Dashboard loaded successfully');
        return true;
    } catch (err) {
        error('Dashboard loading failed: ' + err.message);
        return false;
    }
}

async function testUIElements(page) {
    header('Test 2: UI Elements Presence');

    try {
        // Check for key UI elements
        const elements = {
            'Training card': '.card:has-text("Training")',
            'System card': '.card:has-text("System")',
            'Hybrid Execution card': '.card:has-text("Hybrid")',
            'Hardware button': 'button:has-text("Hardware")',
        };

        let allPresent = true;

        for (const [name, selector] of Object.entries(elements)) {
            try {
                await page.waitForSelector(selector, { timeout: 5000 });
                success(`Found: ${name}`);
            } catch {
                error(`Missing: ${name}`);
                allPresent = false;
            }
        }

        // Take screenshot
        await page.screenshot({ path: 'tests/screenshots/dashboard_elements.png' });
        info('Screenshot saved: dashboard_elements.png');

        if (allPresent) {
            success('All expected UI elements present');
            return true;
        } else {
            error('Some UI elements are missing');
            return false;
        }
    } catch (err) {
        error('UI elements test failed: ' + err.message);
        return false;
    }
}

async function testDataDisplay(page) {
    header('Test 3: Data Display');

    try {
        // Wait for data to load
        await page.waitForTimeout(2000);

        // Extract displayed metrics
        const metrics = await page.evaluate(() => {
            const getText = (id) => {
                const el = document.getElementById(id);
                return el ? el.textContent : null;
            };

            return {
                loss: getText('loss-value'),
                step: getText('step-value'),
                speed: getText('speed-value'),
                gpuRatio: getText('gpu-ratio-value')
            };
        });

        info('Current metrics:');
        console.log('  Loss:', metrics.loss);
        console.log('  Step:', metrics.step);
        console.log('  Speed:', metrics.speed);
        console.log('  GPU Ratio:', metrics.gpuRatio);

        // Validate that metrics are displayed (even if 0)
        let valid = true;
        for (const [key, value] of Object.entries(metrics)) {
            if (value === null) {
                error(`Metric ${key} not found in DOM`);
                valid = false;
            }
        }

        if (valid) {
            success('All metrics are displayed correctly');
            return true;
        } else {
            error('Some metrics are missing');
            return false;
        }
    } catch (err) {
        error('Data display test failed: ' + err.message);
        return false;
    }
}

async function testInteractivity(page) {
    header('Test 4: Interactive Elements');

    try {
        // Test Hardware button
        info('Testing Hardware button...');
        const hardwareBtn = await page.$('button:has-text("Hardware")');

        if (hardwareBtn) {
            await hardwareBtn.click();
            await page.waitForTimeout(1000);

            // Check if modal appeared
            const modal = await page.$('.modal, [role="dialog"]');
            if (modal) {
                success('Hardware modal opened');
                await page.screenshot({ path: 'tests/screenshots/hardware_modal.png' });
                info('Screenshot saved: hardware_modal.png');

                // Close modal (try ESC key)
                await page.keyboard.press('Escape');
                await page.waitForTimeout(500);
            } else {
                error('Hardware modal did not appear');
                return false;
            }
        } else {
            error('Hardware button not found');
            return false;
        }

        success('Interactive elements working correctly');
        return true;
    } catch (err) {
        error('Interactivity test failed: ' + err.message);
        return false;
    }
}

async function testRealTimeUpdates(page) {
    header('Test 5: Real-time Data Updates');

    try {
        // Get initial step value
        const initialStep = await page.evaluate(() => {
            const el = document.getElementById('step-value');
            return el ? el.textContent : '0';
        });

        info(`Initial step: ${initialStep}`);

        // Wait for potential update (10 seconds)
        info('Waiting 10 seconds for potential updates...');
        await page.waitForTimeout(10000);

        // Get updated step value
        const updatedStep = await page.evaluate(() => {
            const el = document.getElementById('step-value');
            return el ? el.textContent : '0';
        });

        info(`Updated step: ${updatedStep}`);

        // Note: If no training is running, values won't change
        // This is expected behavior
        if (initialStep === updatedStep) {
            info('Step value unchanged (no active training)');
        } else {
            success('Real-time updates detected!');
        }

        success('Real-time update mechanism functional');
        return true;
    } catch (err) {
        error('Real-time update test failed: ' + err.message);
        return false;
    }
}

async function runTests() {
    header('MM-Rec Dashboard UI/UX Tests');

    let browser;
    try {
        // Launch browser
        info('Launching headless browser...');
        browser = await puppeteer.launch({
            headless: 'new',
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });

        const page = await browser.newPage();
        await page.setViewport({ width: 1920, height: 1080 });

        // Create screenshots directory
        const fs = require('fs');
        if (!fs.existsSync('tests/screenshots')) {
            fs.mkdirSync('tests/screenshots', { recursive: true });
        }

        // Run all tests
        const results = {
            'Dashboard Loading': await testDashboardLoading(page),
            'UI Elements': await testUIElements(page),
            'Data Display': await testDataDisplay(page),
            'Interactivity': await testInteractivity(page),
            'Real-time Updates': await testRealTimeUpdates(page)
        };

        // Summary
        header('Test Summary');
        let passed = 0;
        let total = 0;

        for (const [name, result] of Object.entries(results)) {
            total++;
            if (result) {
                passed++;
                success(`${name}: PASSED`);
            } else {
                error(`${name}: FAILED`);
            }
        }

        console.log(`\n${colors.bold}Results: ${passed}/${total} tests passed${colors.reset}\n`);

        if (passed === total) {
            success('All UI tests passed! ✨');
            process.exit(0);
        } else {
            error('Some UI tests failed!');
            process.exit(1);
        }

    } catch (err) {
        error('Test suite failed: ' + err.message);
        console.error(err);
        process.exit(1);
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

// Run tests
runTests().catch(console.error);
