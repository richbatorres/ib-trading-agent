module.exports = {
    apps: [
        {
            name: "ib-trading-agent",
            script: "agent.py",
            interpreter: "python3",
            cwd: __dirname,
            autorestart: true,
            max_restarts: 5,
            min_uptime: "120s",
            restart_delay: 10000,
            env: {
                PYTHONUNBUFFERED: "1",
            },
        },
        {
            name: "dashboard-server",
            script: "scripts/dashboard_server.py",
            interpreter: "./venv/Scripts/python.exe",
            cwd: __dirname,
            autorestart: true,
            max_restarts: 3,
            min_uptime: "10s",
            restart_delay: 5000,
            args: "8888",
            env: {
                PYTHONUNBUFFERED: "1",
            },
        },
    ],
};
