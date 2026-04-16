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
    ],
};
