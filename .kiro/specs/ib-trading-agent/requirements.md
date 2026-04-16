# Requirements Document

## Introduction

This document specifies the requirements for an autonomous AI trading agent that connects to the Interactive Brokers (IB) platform via the TWS API. The agent automatically monitors real-time market data, applies multiple trading strategies (momentum, mean reversion, trend following), enforces strict risk management rules, integrates Polymarket sentiment data as a secondary signal, and operates continuously as a background service with crash recovery. The agent starts on an IB Paper Trading account and supports simple switching to a live account via a single configuration variable.

## Glossary

- **Agent**: The autonomous trading software system that monitors markets, makes trading decisions, and executes trades on Interactive Brokers.
- **IB_API**: The Interactive Brokers TWS API or IB Gateway API used for market data streaming and order execution.
- **Risk_Manager**: The subsystem responsible for enforcing all risk limits (portfolio loss, position size, stop-loss, cash buffer).
- **Strategy_Engine**: The subsystem that evaluates trading signals from multiple strategies (momentum, mean reversion, trend following) and produces trade recommendations.
- **Market_Data_Service**: The subsystem that streams and processes real-time market data from IB_API.
- **Polymarket_Service**: The subsystem that fetches and scores sentiment data from the Polymarket prediction market API.
- **State_Store**: The local database (SQLite or PostgreSQL) used for persisting agent state, trades, and analytical decisions.
- **Report_Generator**: The subsystem that produces daily HTML email reports summarizing portfolio performance and trading activity.
- **CLI_Interface**: The command-line interface for starting, stopping, querying status, and generating reports.
- **Watchlist**: A curated list of stock symbols the agent monitors for potential trading opportunities.
- **Trailing_Stop_Loss**: A stop-loss order that adjusts upward as the stock price increases, protecting unrealized gains.
- **Hard_Stop**: The portfolio-level loss threshold (default 20%) that triggers automatic closure of all positions and halts trading.
- **Cash_Buffer**: The minimum percentage of portfolio value (default 10%) that must remain in cash at all times.
- **EMA_Crossover**: A trend-following signal generated when a short-period Exponential Moving Average crosses a long-period EMA.
- **Paper_Trading**: IB's simulated trading environment used for testing without risking real capital.
- **Regular_Market_Hours**: NYSE/NASDAQ trading hours from 9:30 to 16:00 Eastern Time.
- **LLM_Service**: The subsystem that manages all interactions with the Anthropic LLM API (claude-sonnet-4-6) for sentiment interpretation, report generation, and unusual market condition analysis.
- **Deterministic_Layer**: The core trading logic layer implemented in pure Python/NumPy that operates without any LLM API calls, handling indicator calculations, signal generation, risk management, order execution, state persistence, and crash recovery.
- **LLM_Layer**: The optional layer that invokes the LLM API exclusively for complex, infrequent tasks such as Polymarket sentiment interpretation, daily report generation, and unusual market condition analysis.
- **Indicator_Cache**: The caching mechanism that stores calculated indicator values and incrementally updates them on each new price tick, avoiding full recalculation from scratch.
- **Dirty_Flag**: A mechanism that marks indicator inputs as changed, triggering recalculation only when input data has been modified.
- **Test_Suite**: The complete set of unit, integration, performance, and backtesting tests that can run locally without an active IB connection.

## Requirements

### Requirement 1: IB API Connection Management

**User Story:** As a trader, I want the agent to establish and maintain a reliable connection to Interactive Brokers, so that it can stream market data and execute trades.

#### Acceptance Criteria

1. WHEN the Agent starts, THE Agent SHALL connect to IB_API using the ib_insync library with the host, port, and account ID specified in the .env configuration file.
2. WHILE connected to IB_API, THE Agent SHALL maintain the connection using the ib_insync keep-alive mechanism and reconnect automatically within 30 seconds if the connection drops.
3. IF the IB_API connection fails after 5 consecutive reconnection attempts, THEN THE Agent SHALL log an ERROR, send an alert email to the configured email address, and enter a waiting state that retries connection every 60 seconds.
4. WHEN the ENVIRONMENT variable is set to "paper", THE Agent SHALL connect exclusively to the IB Paper Trading account and reject any configuration pointing to a live account.
5. WHEN the ENVIRONMENT variable is set to "live", THE Agent SHALL connect to the live IB account specified by IB_ACCOUNT_ID.

### Requirement 2: Real-Time Market Data Streaming

**User Story:** As a trader, I want the agent to continuously receive real-time market data, so that trading decisions are based on current prices and volumes.

#### Acceptance Criteria

1. WHILE connected to IB_API during Regular_Market_Hours, THE Market_Data_Service SHALL stream real-time price and volume data for all stocks on the Watchlist using IB_API streaming subscriptions.
2. WHEN new market data arrives for a Watchlist stock, THE Market_Data_Service SHALL publish the data to the Strategy_Engine within 100 milliseconds of receipt.
3. THE Market_Data_Service SHALL use asynchronous I/O (async/await) for all data processing to avoid blocking the event loop.
4. WHILE outside Regular_Market_Hours, THE Market_Data_Service SHALL continue receiving available data for watchlist preparation but SHALL NOT trigger trade execution.

### Requirement 3: Market Hours Awareness

**User Story:** As a trader, I want the agent to respect exchange trading hours, so that trades are only executed during regular market sessions.

#### Acceptance Criteria

1. THE Agent SHALL determine Regular_Market_Hours for NYSE and NASDAQ exchanges (9:30–16:00 Eastern Time) using exchange calendar data rather than hardcoded timezone offsets.
2. WHILE outside Regular_Market_Hours, THE Agent SHALL suppress all trade execution signals and log each suppressed signal at INFO level.
3. WHEN Regular_Market_Hours begin, THE Agent SHALL enable trade execution and log the market open event at INFO level.
4. WHEN Regular_Market_Hours end, THE Agent SHALL disable trade execution, log the market close event, and trigger the daily report generation process.

### Requirement 4: Momentum Trading Strategy

**User Story:** As a trader, I want the agent to detect short-term momentum signals, so that it can enter positions on stocks with strong directional movement.

#### Acceptance Criteria

1. WHEN new price data arrives for a Watchlist stock, THE Strategy_Engine SHALL calculate the Relative Strength Index (RSI) using a 14-period window.
2. WHEN new price data arrives for a Watchlist stock, THE Strategy_Engine SHALL calculate the MACD indicator using standard parameters (12, 26, 9).
3. WHEN the RSI crosses above 30 from below and the MACD histogram turns positive, THE Strategy_Engine SHALL generate a BUY momentum signal for that stock.
4. WHEN the RSI crosses below 70 from above and the MACD histogram turns negative, THE Strategy_Engine SHALL generate a SELL momentum signal for that stock.

### Requirement 5: Mean Reversion Strategy

**User Story:** As a trader, I want the agent to detect stocks that have deviated excessively from their average price, so that it can profit from price corrections.

#### Acceptance Criteria

1. WHEN new price data arrives for a Watchlist stock, THE Strategy_Engine SHALL calculate Bollinger Bands using a 20-period simple moving average and 2 standard deviations.
2. WHEN a stock price crosses below the lower Bollinger Band, THE Strategy_Engine SHALL generate a BUY mean-reversion signal for that stock.
3. WHEN a stock price crosses above the upper Bollinger Band, THE Strategy_Engine SHALL generate a SELL mean-reversion signal for that stock.

### Requirement 6: Trend Following Strategy

**User Story:** As a trader, I want the agent to follow established trends using EMA crossovers, so that it can ride sustained price movements.

#### Acceptance Criteria

1. WHEN new price data arrives for a Watchlist stock, THE Strategy_Engine SHALL calculate the 9-period and 21-period Exponential Moving Averages.
2. WHEN the 9-period EMA crosses above the 21-period EMA, THE Strategy_Engine SHALL generate a BUY trend-following signal for that stock.
3. WHEN the 9-period EMA crosses below the 21-period EMA, THE Strategy_Engine SHALL generate a SELL trend-following signal for that stock.

### Requirement 7: Volume Confirmation

**User Story:** As a trader, I want every trade signal confirmed by volume analysis, so that the agent avoids false signals in low-volume conditions.

#### Acceptance Criteria

1. THE Strategy_Engine SHALL reject any BUY or SELL signal for a stock whose current trading volume is below 1.5 times its 20-day average volume.
2. THE Strategy_Engine SHALL include only stocks with an average daily trading volume greater than 500,000 shares on the Watchlist.
3. WHEN a signal is rejected due to insufficient volume, THE Strategy_Engine SHALL log the rejection at INFO level with the stock symbol, signal type, current volume, and required volume threshold.

### Requirement 8: Earnings Report Avoidance

**User Story:** As a trader, I want the agent to avoid trading around earnings reports, so that it is not exposed to excessive volatility from earnings surprises.

#### Acceptance Criteria

1. THE Strategy_Engine SHALL maintain an earnings calendar for all Watchlist stocks using available IB_API fundamental data or a supplementary data source.
2. WHILE a Watchlist stock is within 2 trading days before or 1 trading day after its scheduled earnings report date, THE Strategy_Engine SHALL suppress all trade signals for that stock.
3. WHEN a trade signal is suppressed due to an upcoming earnings report, THE Strategy_Engine SHALL log the suppression at INFO level with the stock symbol and earnings date.

### Requirement 9: Portfolio Hard Stop

**User Story:** As a trader, I want the agent to automatically halt trading when portfolio losses exceed a threshold, so that catastrophic losses are prevented.

#### Acceptance Criteria

1. THE Risk_Manager SHALL calculate the current portfolio loss percentage relative to the initial portfolio value after every trade execution and at least once per minute during Regular_Market_Hours.
2. WHEN the portfolio loss percentage reaches or exceeds the MAX_PORTFOLIO_LOSS_PCT threshold (default 20%), THE Risk_Manager SHALL immediately close all open positions by submitting market sell orders via IB_API.
3. WHEN the Hard_Stop is triggered, THE Risk_Manager SHALL disable all trade execution, log the event at ERROR level, and send an alert email to the configured email address.
4. WHILE the Hard_Stop is active, THE Agent SHALL reject all new trade signals until the Agent is manually restarted by the user.

### Requirement 10: Per-Position Stop-Loss

**User Story:** As a trader, I want automatic stop-loss orders on every position, so that individual stock losses are capped.

#### Acceptance Criteria

1. WHEN a new position is opened, THE Risk_Manager SHALL immediately submit a stop-loss order to IB_API at a price equal to the entry price minus STOP_LOSS_PCT (default 5%) of the entry price.
2. WHEN a stop-loss order is triggered and executed by IB_API, THE Risk_Manager SHALL log the stop-loss execution at WARNING level with the stock symbol, entry price, exit price, and realized loss.
3. IF a stop-loss order submission fails, THEN THE Risk_Manager SHALL retry the submission up to 3 times and, if still unsuccessful, close the position with a market order.

### Requirement 11: Trailing Stop-Loss

**User Story:** As a trader, I want trailing stop-loss orders to protect unrealized gains, so that winning positions lock in profits as prices rise.

#### Acceptance Criteria

1. WHEN a position's unrealized gain exceeds 3%, THE Risk_Manager SHALL convert the fixed stop-loss order to a Trailing_Stop_Loss order with a trail amount equal to STOP_LOSS_PCT (default 5%) of the current price.
2. WHILE a Trailing_Stop_Loss is active, THE Risk_Manager SHALL verify that IB_API is adjusting the stop price upward as the stock price increases.
3. WHEN a Trailing_Stop_Loss order is triggered and executed, THE Risk_Manager SHALL log the execution at INFO level with the stock symbol, peak price, exit price, and realized gain.

### Requirement 12: Position Size and Cash Buffer Limits

**User Story:** As a trader, I want the agent to enforce diversification and cash reserve rules, so that the portfolio is not overexposed to any single stock.

#### Acceptance Criteria

1. WHEN a new trade is proposed, THE Risk_Manager SHALL reject the trade if the proposed position value would exceed MAX_POSITION_SIZE_PCT (default 25%) of the total portfolio value.
2. WHEN a new trade is proposed, THE Risk_Manager SHALL reject the trade if executing the trade would reduce the cash balance below CASH_BUFFER_PCT (default 10%) of the total portfolio value.
3. WHEN a trade is rejected due to a risk limit violation, THE Risk_Manager SHALL log the rejection at WARNING level with the stock symbol, proposed trade value, and the specific limit that was violated.
4. THE Risk_Manager SHALL evaluate risk limits in the following order: Hard_Stop check, Cash_Buffer check, position size check, stop-loss order readiness.

### Requirement 13: Polymarket Sentiment Integration

**User Story:** As a trader, I want the agent to incorporate prediction market sentiment into trading decisions, so that macroeconomic and political risk signals enhance trade quality.

#### Acceptance Criteria

1. THE Polymarket_Service SHALL fetch data from the Polymarket API every 15 minutes for prediction markets related to macroeconomic events, political risks, and Federal Reserve decisions.
2. THE Polymarket_Service SHALL compute a sentiment score between -1.0 (strongly bearish) and +1.0 (strongly bullish) based on the aggregated Polymarket data.
3. WHEN the Strategy_Engine evaluates a trade signal, THE Strategy_Engine SHALL incorporate the Polymarket sentiment score as a secondary weighting factor that adjusts signal confidence but does not independently trigger a trade.
4. IF the Polymarket API is unavailable, THEN THE Polymarket_Service SHALL use the last successfully fetched sentiment score and log a WARNING with the timestamp of the last successful fetch.

### Requirement 14: State Persistence

**User Story:** As a trader, I want the agent to persist its state to a database, so that no data is lost on shutdown or crash.

#### Acceptance Criteria

1. WHEN a trade is executed, THE State_Store SHALL persist the trade record (symbol, direction, entry price, quantity, timestamp, stop-loss price) to the database within 1 second of execution confirmation.
2. WHEN the Strategy_Engine completes an analysis cycle, THE State_Store SHALL persist the current Watchlist, active signals, and indicator values to the database.
3. THE State_Store SHALL use SQLite as the default database engine, configurable to PostgreSQL via the .env file for server deployments.
4. THE State_Store SHALL persist the current portfolio snapshot (positions, cash balance, total value, P&L) after every trade and at least once every 5 minutes during Regular_Market_Hours.

### Requirement 15: Graceful Shutdown

**User Story:** As a trader, I want the agent to shut down cleanly when stopped, so that all state is preserved and IB connections are properly closed.

#### Acceptance Criteria

1. WHEN the Agent receives a SIGINT or SIGTERM signal, THE Agent SHALL initiate a graceful shutdown sequence that completes within 30 seconds.
2. WHEN the graceful shutdown sequence begins, THE Agent SHALL cancel all pending orders on IB_API, persist the current state to State_Store, and disconnect from IB_API in that order.
3. WHEN the graceful shutdown sequence completes, THE Agent SHALL log the shutdown event at INFO level and exit with code 0.
4. IF the graceful shutdown sequence exceeds 30 seconds, THEN THE Agent SHALL force-close all connections, log an ERROR, and exit with code 1.

### Requirement 16: Crash Recovery

**User Story:** As a trader, I want the agent to recover automatically after a crash, so that it resumes operation without manual intervention.

#### Acceptance Criteria

1. WHEN the Agent starts and detects a previous state in State_Store, THE Agent SHALL load the last persisted portfolio snapshot, open positions, and pending orders.
2. WHEN recovering from a crash, THE Agent SHALL fetch market data for the period the Agent was offline and reconcile the persisted state with the current IB account state (positions, fills, order statuses).
3. WHEN state reconciliation reveals discrepancies between State_Store and IB account state, THE Agent SHALL update State_Store to match the IB account state and log each discrepancy at WARNING level.
4. WHEN crash recovery completes, THE Agent SHALL resume normal operation and log the recovery event at INFO level with the duration of the offline period.

### Requirement 17: Watchdog and Auto-Restart

**User Story:** As a trader, I want the agent to automatically restart after a crash, so that trading is not interrupted by unexpected failures.

#### Acceptance Criteria

1. THE Agent SHALL provide a systemd service unit file (Linux), a PM2 configuration file, and documentation for running as a persistent background service on Windows, Mac, and Linux.
2. WHEN the Agent process exits unexpectedly (non-zero exit code), THE watchdog mechanism SHALL restart the Agent within 10 seconds.
3. IF the Agent crashes more than 5 times within 10 minutes, THEN THE watchdog mechanism SHALL stop restart attempts, log an ERROR, and send an alert email to the configured email address.

### Requirement 18: Daily Email Report

**User Story:** As a trader, I want a daily email report after market close, so that I can review the agent's performance and current portfolio state.

#### Acceptance Criteria

1. WHEN the time reaches 18:00 Eastern Time on a trading day, THE Report_Generator SHALL generate and send an HTML email report to the configured EMAIL_ADDRESS.
2. THE Report_Generator SHALL include in the report: total portfolio value, daily change in absolute and percentage terms, a list of all trades executed that day (symbol, direction, entry price, exit price, P&L), top 3 winning and top 3 losing trades, all current open positions with unrealized P&L, and the current Polymarket sentiment summary.
3. WHEN the cumulative portfolio loss exceeds 10% of the initial portfolio value, THE Report_Generator SHALL include a WARNING banner in the report.
4. WHEN the cumulative portfolio loss exceeds 20% of the initial portfolio value, THE Report_Generator SHALL include a CRITICAL banner in the report.
5. THE Report_Generator SHALL format the email as clean, readable HTML with a consistent visual layout.

### Requirement 19: Configuration Management

**User Story:** As a trader, I want all configuration in a single .env file, so that I can easily adjust parameters without modifying code.

#### Acceptance Criteria

1. THE Agent SHALL read all configuration parameters from a single .env file at startup, including: IB_ACCOUNT_ID, IB_HOST, IB_PORT, ENVIRONMENT, EMAIL_ADDRESS, EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, EMAIL_SMTP_USER, EMAIL_SMTP_PASSWORD, MAX_PORTFOLIO_LOSS_PCT, MAX_POSITION_SIZE_PCT, STOP_LOSS_PCT, CASH_BUFFER_PCT, LLM_MODEL, and MAX_LLM_CALLS_PER_DAY.
2. THE Agent SHALL apply default values for risk and LLM parameters when not specified in the .env file: MAX_PORTFOLIO_LOSS_PCT=20, MAX_POSITION_SIZE_PCT=25, STOP_LOSS_PCT=5, CASH_BUFFER_PCT=10, LLM_MODEL=claude-sonnet-4-6, MAX_LLM_CALLS_PER_DAY=10.
3. WHEN a required configuration parameter (IB_ACCOUNT_ID, IB_HOST, IB_PORT, ENVIRONMENT) is missing from the .env file, THE Agent SHALL log an ERROR with the missing parameter name and exit with a non-zero exit code.
4. WHEN the ENVIRONMENT parameter contains a value other than "paper" or "live", THE Agent SHALL log an ERROR and exit with a non-zero exit code.

### Requirement 20: CLI Interface

**User Story:** As a trader, I want a simple command-line interface, so that I can control the agent and check its status easily.

#### Acceptance Criteria

1. WHEN the user executes `python agent.py start`, THE CLI_Interface SHALL start the Agent as a background process and log the startup event at INFO level.
2. WHEN the user executes `python agent.py stop`, THE CLI_Interface SHALL send a graceful shutdown signal to the running Agent process.
3. WHEN the user executes `python agent.py status`, THE CLI_Interface SHALL display the current portfolio value, cash balance, open positions with unrealized P&L, and the Agent's operational state (running, stopped, halted by Hard_Stop).
4. WHEN the user executes `python agent.py report`, THE CLI_Interface SHALL generate and send the daily email report immediately regardless of the current time.
5. IF the user executes `python agent.py` with an unrecognized command, THEN THE CLI_Interface SHALL display a usage message listing all valid commands and exit with a non-zero exit code.

### Requirement 21: Logging

**User Story:** As a trader, I want detailed logging of all agent decisions and trades, so that I can audit and debug the agent's behavior.

#### Acceptance Criteria

1. THE Agent SHALL log all messages to both a rotating log file and the console simultaneously using Python's standard logging module.
2. THE Agent SHALL use log level INFO for normal operational events (trades executed, signals generated, market open/close), WARNING for risk-related events (stop-loss triggered, risk limit approached, Polymarket API unavailable), and ERROR for failures (connection lost, crash recovery, Hard_Stop triggered).
3. THE Agent SHALL retain log files for a minimum of 30 days using a time-based rotating file handler.
4. THE Agent SHALL include a timestamp, log level, module name, and descriptive message in every log entry.
5. WHEN the LLM_Service makes an API call, THE Agent SHALL log the call at INFO level with the purpose of the call, the model used, the number of input tokens, the number of output tokens, and the total token count.

### Requirement 22: Cross-Platform Compatibility

**User Story:** As a trader, I want the agent to run on any operating system without code changes, so that I can deploy it on my local machine or a remote server.

#### Acceptance Criteria

1. THE Agent SHALL execute without modification on Windows 10+, macOS 12+, and Linux (Ubuntu 20.04+) with Python 3.10 or later installed.
2. THE Agent SHALL use only cross-platform Python libraries and avoid OS-specific system calls.
3. THE Agent SHALL use relative file paths for all local resources (database, logs, configuration) so that the installation directory is portable.

### Requirement 23: LLM Hybrid Architecture

**User Story:** As a trader, I want the agent to use a hybrid architecture where the LLM API is not in the critical hot path of every trading decision, so that operational costs remain low and core trading logic operates without external API dependencies.

#### Acceptance Criteria

1. THE Deterministic_Layer SHALL implement all core trading logic (indicator calculations, signal generation, risk management, order execution, state persistence, crash recovery) as pure Python/NumPy code without any LLM API calls.
2. THE Deterministic_Layer SHALL operate fully autonomously without requiring any LLM_Service availability.
3. THE LLM_Service SHALL use the Anthropic Python SDK to invoke the model specified by the LLM_MODEL configuration variable (default: claude-sonnet-4-6).
4. THE LLM_Service SHALL invoke the LLM API exclusively for Polymarket sentiment interpretation and news analysis (maximum 4 calls per day), daily email report generation (1 call per day), and interpretation of unusual market conditions that do not match known patterns (as needed).
5. WHEN the number of LLM API calls in a calendar day reaches MAX_LLM_CALLS_PER_DAY (default: 10), THE LLM_Service SHALL skip all subsequent LLM steps for the remainder of that day and log a WARNING with the current call count.
6. WHEN an LLM API call completes, THE LLM_Service SHALL log the call at INFO level with the purpose, model name, input token count, output token count, and total token count.
7. IF the LLM API is unavailable or returns an error, THEN THE LLM_Service SHALL log the error at WARNING level and the Deterministic_Layer SHALL continue operating without interruption.

### Requirement 24: Performance Optimization

**User Story:** As a trader, I want the agent to execute trading decisions with minimal latency and efficient resource usage, so that market opportunities are captured quickly and the system runs smoothly on modest hardware.

#### Acceptance Criteria

1. THE Agent SHALL implement all I/O operations (IB API calls, Polymarket fetches, database operations, email sending) using async/await and SHALL NOT use blocking calls in the main event loop.
2. THE Agent SHALL use asyncio.gather() for parallel execution of independent I/O operations.
3. THE Strategy_Engine SHALL implement all technical indicator calculations (RSI, MACD, EMA, Bollinger Bands) using NumPy vectorized operations and SHALL NOT use Python for-loops for operations on price arrays.
4. THE Strategy_Engine SHALL cache calculated indicator values and incrementally update them on each new price tick rather than recalculating from scratch.
5. THE Strategy_Engine SHALL use collections.deque with fixed size for rolling price and volume windows.
6. THE Strategy_Engine SHALL implement a Dirty_Flag mechanism so that each indicator recalculates only when its input data has changed.
7. THE Agent SHALL measure and log the time elapsed from market signal receipt to order submission, with a target latency of less than 100 milliseconds.
8. THE Agent SHALL generate a short performance report in the logs once per week summarizing signal-to-order latency statistics.
9. THE Agent SHALL maintain a modular architecture so that critical calculation components can be replaced with Cython extensions without changes to the rest of the codebase.

### Requirement 25: Local Test Suite

**User Story:** As a developer, I want a complete test suite that runs locally without an active IB connection, so that I can verify correctness before every deployment.

#### Acceptance Criteria

1. WHEN the user executes `python agent.py test`, THE CLI_Interface SHALL run all tests (unit, integration, performance, backtesting) using pytest.
2. WHEN the user executes `python agent.py test --unit`, THE CLI_Interface SHALL run only unit tests.
3. WHEN the user executes `python agent.py test --integ`, THE CLI_Interface SHALL run only integration tests.
4. WHEN the user executes `python agent.py test --perf`, THE CLI_Interface SHALL run only performance tests.
5. THE Test_Suite SHALL include unit tests in test/test_indicators.py covering: test_rsi_overbought, test_rsi_oversold, test_macd_crossover, test_ema_crossover, test_bollinger_bands, test_stop_loss_trigger, test_portfolio_hard_stop, test_position_size_limit, and test_cash_buffer.
6. THE Test_Suite SHALL include integration tests in test/test_agent.py covering: test_buy_signal_flow, test_sell_signal_flow, test_recovery_flow, test_market_closed_no_trades, test_email_report_generation, and test_polymarket_signal_integration.
7. THE Test_Suite SHALL include performance tests in test/test_performance.py covering: test_indicator_calculation_speed (less than 10ms for 1000 candles), test_signal_to_order_latency (less than 100ms end-to-end), and test_numpy_vs_loop_benchmark (NumPy at least 10x faster than Python loop).
8. THE Test_Suite SHALL include backtesting smoke tests in test/test_backtest.py covering: test_strategy_not_bankrupt (30 days historical data with loss not exceeding 20%) and test_strategy_makes_trades (strategy generates at least some signals on test data).
9. THE Test_Suite SHALL use pytest as the test framework, and every test SHALL be deterministic using fixed random seeds and mocks for all external API calls.
