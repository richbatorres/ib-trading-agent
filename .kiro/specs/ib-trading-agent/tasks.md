# Implementation Plan: IB Trading Agent

## Overview

Build an autonomous AI trading agent for Interactive Brokers from scratch. The implementation follows an incremental approach: project scaffolding and configuration first, then core data models and database, followed by each component in dependency order (ConnectionManager → MarketDataService → StrategyEngine → RiskManager → OrderExecutor), then supporting services (MarketHoursService, PolymarketService, StateManager, ReportGenerator, LLMService, IndicatorCache), CLI interface with test commands, and finally integration wiring with deployment configuration, performance optimization, and comprehensive test suite. Each task builds on previous tasks so there is no orphaned code.

## Tasks

- [x] 1. Project scaffolding and configuration
  - [x] 1.1 Create project directory structure and dependency files
    - Create the directory layout: `src/`, `src/models/`, `src/services/`, `src/strategies/`, `tests/`, `tests/unit/`, `tests/property/`, `tests/integration/`, `tests/performance/`, `tests/backtest/`, `test/`, `data/`, `logs/`
    - Create `pyproject.toml` (or `requirements.txt`) with dependencies: `ib_insync`, `apscheduler`, `aiohttp`, `python-dotenv`, `aiosqlite`, `hypothesis`, `pytest`, `pytest-asyncio`, `anthropic`, `numpy`
    - Create a `.env.example` file with all configuration variables documented (including LLM_MODEL and MAX_LLM_CALLS_PER_DAY)
    - Create a minimal `README.md` with setup instructions
    - _Requirements: 19.1, 22.1, 22.2, 22.3, 23.3, 24.3_

  - [x] 1.2 Implement AgentConfig dataclass and .env loading
    - Create `src/config.py` with the `AgentConfig` dataclass
    - Implement `AgentConfig.from_env()` to load from `.env` using `python-dotenv`
    - Validate required fields (IB_ACCOUNT_ID, IB_HOST, IB_PORT, ENVIRONMENT) — exit with non-zero code if missing
    - Validate ENVIRONMENT is "paper" or "live" — exit with non-zero code otherwise
    - Apply defaults for optional risk parameters: MAX_PORTFOLIO_LOSS_PCT=20, MAX_POSITION_SIZE_PCT=25, STOP_LOSS_PCT=5, CASH_BUFFER_PCT=10
    - Apply defaults for LLM parameters: LLM_MODEL=claude-sonnet-4-6, MAX_LLM_CALLS_PER_DAY=10
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 1.4, 23.3_

  - [ ]* 1.3 Write property test for configuration validation (Property 16)
    - **Property 16: Configuration validation**
    - Use Hypothesis to generate random `.env` file content
    - Verify defaults are applied for missing optional parameters (including LLM_MODEL and MAX_LLM_CALLS_PER_DAY)
    - Verify exit with non-zero code for missing required parameters
    - Verify exit with non-zero code for invalid ENVIRONMENT values
    - **Validates: Requirements 1.4, 19.2, 19.3, 19.4**

  - [ ]* 1.4 Write unit tests for AgentConfig
    - Test loading a valid `.env` with all fields
    - Test missing required fields produce errors
    - Test default values are applied correctly
    - Test invalid ENVIRONMENT value is rejected
    - _Requirements: 19.1, 19.2, 19.3, 19.4_

- [x] 2. Data models and database schema
  - [x] 2.1 Define domain object dataclasses
    - Create `src/models/domain.py` with: `TradeSignal`, `ApprovedTrade`, `TradeRecord`, `PortfolioSnapshot`, `Discrepancy`, `AnalysisState`, `AgentState`
    - All dataclasses must match the design document field definitions exactly
    - _Requirements: 14.1, 14.2, 14.4_

  - [x] 2.2 Implement StateManager with SQLite schema
    - Create `src/services/state_manager.py` with the `StateManager` class
    - Implement `initialize()` to create all tables (trades, portfolio_snapshots, analysis_state, earnings_calendar, agent_state, llm_calls) with WAL mode enabled
    - Implement `persist_trade()`, `persist_portfolio_snapshot()`, `persist_analysis_state()`
    - Implement `load_last_state()` for crash recovery
    - Implement `reconcile_with_ib()` stub (IB interaction wired later)
    - Use `aiosqlite` for async database access
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 16.1_

  - [ ]* 2.3 Write unit tests for StateManager
    - Test table creation and WAL mode
    - Test persist and load round-trip for trades, snapshots, analysis state
    - Test `load_last_state()` returns None on empty database
    - Test `load_last_state()` returns most recent state
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [x] 3. Logging setup
  - [x] 3.1 Implement logging configuration
    - Create `src/logging_config.py` with a `setup_logging()` function
    - Configure dual output: rotating file handler (30-day retention) + console handler
    - Use Python's standard `logging` module with `TimedRotatingFileHandler`
    - Format: timestamp, log level, module name, descriptive message
    - Log files stored in `logs/` directory with relative paths
    - Ensure logging infrastructure supports LLM token count logging (purpose, model, input/output/total tokens per call)
    - _Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 22.3_

- [x] 4. Checkpoint — Verify foundation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Technical indicators and StrategyEngine
  - [x] 5.1 Implement technical indicator calculations
    - Create `src/strategies/indicators.py` with pure functions: `calculate_rsi()`, `calculate_macd()`, `calculate_bollinger_bands()`, `calculate_ema()`
    - ALL indicator calculations MUST use NumPy vectorized operations — Python for-loops on price arrays are strictly forbidden
    - RSI: Wilder's smoothing method, 14-period default, output in [0, 100], implemented with NumPy array operations
    - MACD: standard (12, 26, 9) parameters, returns (macd_line, signal_line, histogram), EMA via np.convolve() or pandas ewm()
    - Bollinger Bands: 20-period SMA, 2 std devs, returns (upper, middle, lower) with upper > middle > lower, using np.mean() and np.std()
    - EMA: output bounded between min and max of input prices, using np.convolve() or pandas ewm()
    - _Requirements: 4.1, 4.2, 5.1, 6.1, 24.3_

  - [ ]* 5.2 Write property test for technical indicator bounds (Property 1)
    - **Property 1: Technical indicator bounds**
    - Use Hypothesis to generate random float price sequences of sufficient length
    - Verify RSI ∈ [0, 100] for all valid inputs
    - Verify Bollinger Bands: upper > middle > lower, middle = SMA
    - Verify EMA ∈ [min(prices), max(prices)]
    - **Validates: Requirements 4.1, 4.2, 5.1, 6.1**

  - [x] 5.3 Implement StrategyEngine signal generation
    - Create `src/services/strategy_engine.py` with the `StrategyEngine` class
    - Implement `process_tick()` as the main entry point
    - Implement momentum strategy: RSI crosses above 30 + MACD histogram positive → BUY; RSI crosses below 70 + MACD histogram negative → SELL
    - Implement mean reversion strategy: price below lower BB → BUY; price above upper BB → SELL
    - Implement trend following strategy: 9-EMA crosses above 21-EMA → BUY; 9-EMA crosses below 21-EMA → SELL
    - Implement volume confirmation filter: reject signals if current_volume < 1.5 × avg_20day_volume
    - Implement earnings blackout check: suppress signals within 2 days before / 1 day after earnings
    - Implement Polymarket sentiment weighting: adjust confidence but never trigger trade alone
    - Implement multi-strategy agreement: increase confidence when strategies agree
    - _Requirements: 4.3, 4.4, 5.2, 5.3, 6.2, 6.3, 7.1, 7.2, 8.2, 13.3_

  - [ ]* 5.4 Write property test for momentum signal correctness (Property 3)
    - **Property 3: Momentum signal correctness**
    - Use Hypothesis to generate price sequences with RSI/MACD crossover conditions
    - Verify BUY signal when RSI crosses above 30 and MACD histogram turns positive
    - Verify SELL signal when RSI crosses below 70 and MACD histogram turns negative
    - **Validates: Requirements 4.3, 4.4**

  - [ ]* 5.5 Write property test for mean reversion signal correctness (Property 4)
    - **Property 4: Mean reversion signal correctness**
    - Use Hypothesis to generate price sequences with Bollinger Band crossover conditions
    - Verify BUY signal when price crosses below lower BB
    - Verify SELL signal when price crosses above upper BB
    - **Validates: Requirements 5.2, 5.3**

  - [ ]* 5.6 Write property test for trend following signal correctness (Property 5)
    - **Property 5: Trend following signal correctness**
    - Use Hypothesis to generate price sequences with EMA crossover conditions
    - Verify BUY signal when 9-EMA crosses above 21-EMA
    - Verify SELL signal when 9-EMA crosses below 21-EMA
    - **Validates: Requirements 6.2, 6.3**

  - [ ]* 5.7 Write property test for volume confirmation filter (Property 6)
    - **Property 6: Volume confirmation filter**
    - Use Hypothesis to generate random (current_volume, avg_volume) pairs
    - Verify signal rejected when current_volume < 1.5 × avg_volume
    - **Validates: Requirements 7.1**

  - [ ]* 5.8 Write property test for watchlist volume filter (Property 7)
    - **Property 7: Watchlist volume filter**
    - Use Hypothesis to generate random stock data with varying average daily volumes
    - Verify only stocks with avg daily volume > 500,000 are included
    - **Validates: Requirements 7.2**

  - [ ]* 5.9 Write property test for earnings blackout suppression (Property 8)
    - **Property 8: Earnings blackout suppression**
    - Use Hypothesis to generate random (current_date, earnings_date) pairs
    - Verify signals suppressed when current_date is within 2 days before or 1 day after earnings
    - **Validates: Requirements 8.2**

  - [ ]* 5.10 Write unit tests for StrategyEngine
    - Test each strategy with known price sequences that should produce specific signals
    - Test volume rejection logging includes symbol, signal type, current volume, threshold
    - Test earnings blackout logging includes symbol and earnings date
    - Test Polymarket sentiment adjusts confidence but does not trigger trades alone
    - _Requirements: 4.3, 4.4, 5.2, 5.3, 6.2, 6.3, 7.1, 7.3, 8.2, 8.3, 13.3_

- [x] 6. Checkpoint — Verify indicators and strategy engine
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. RiskManager
  - [x] 7.1 Implement RiskManager core logic
    - Create `src/services/risk_manager.py` with the `RiskManager` class
    - Implement `evaluate_signal()` with risk checks in strict order: hard stop → cash buffer → position size → stop-loss readiness
    - Implement `check_portfolio_loss()` — called every minute during market hours
    - Implement `trigger_hard_stop()` — close all positions via market orders, disable trading, log ERROR, send alert email
    - Implement `place_stop_loss()` — stop-loss at entry_price × (1 − STOP_LOSS_PCT/100), retry up to 3 times, fallback to market order
    - Implement `upgrade_to_trailing_stop()` — convert to trailing stop when unrealized gain > 3%
    - Implement `calculate_position_size()` — max shares within position size and cash buffer constraints
    - Log rejections at WARNING level with symbol, proposed value, and specific violated limit
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3, 11.1, 11.2, 11.3, 12.1, 12.2, 12.3, 12.4_

  - [ ]* 7.2 Write property test for hard stop activation and enforcement (Property 9)
    - **Property 9: Hard stop activation and enforcement**
    - Use Hypothesis to generate random portfolio states with varying loss percentages
    - Verify hard stop activates when loss ≥ MAX_PORTFOLIO_LOSS_PCT
    - Verify all incoming signals rejected while hard stop is active
    - **Validates: Requirements 9.2, 9.4**

  - [ ]* 7.3 Write property test for stop-loss price calculation (Property 10)
    - **Property 10: Stop-loss price calculation**
    - Use Hypothesis to generate random (entry_price, STOP_LOSS_PCT) pairs
    - Verify stop-loss price = entry_price × (1 − STOP_LOSS_PCT / 100)
    - **Validates: Requirements 10.1**

  - [ ]* 7.4 Write property test for trailing stop conversion threshold (Property 11)
    - **Property 11: Trailing stop conversion threshold**
    - Use Hypothesis to generate random (entry_price, current_price) pairs
    - Verify conversion to trailing stop when (current − entry) / entry > 0.03
    - **Validates: Requirements 11.1**

  - [ ]* 7.5 Write property test for risk limit enforcement (Property 12)
    - **Property 12: Risk limit enforcement**
    - Use Hypothesis to generate random portfolio states and trade proposals
    - Verify rejection when position value > MAX_POSITION_SIZE_PCT of portfolio
    - Verify rejection when trade would reduce cash below CASH_BUFFER_PCT
    - **Validates: Requirements 12.1, 12.2**

  - [ ]* 7.6 Write property test for risk check evaluation order (Property 13)
    - **Property 13: Risk check evaluation order**
    - Use Hypothesis to generate trades that violate multiple risk limits simultaneously
    - Verify rejection reason corresponds to first violated limit in order: hard stop → cash buffer → position size → stop-loss readiness
    - **Validates: Requirements 12.4**

  - [ ]* 7.7 Write unit tests for RiskManager
    - Test each risk check individually with specific scenarios
    - Test rejection logging includes correct details
    - Test stop-loss retry logic (3 retries then market order fallback)
    - Test trailing stop conversion at exactly 3% boundary
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3, 11.1, 12.1, 12.2, 12.3, 12.4_

- [x] 8. Checkpoint — Verify risk management
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. ConnectionManager and MarketDataService
  - [x] 9.1 Implement ConnectionManager
    - Create `src/services/connection_manager.py` with the `ConnectionManager` class
    - Implement `connect()` using ib_insync with config host/port/clientId
    - Validate environment: paper mode uses port 7497/4002, live mode uses port 7496/4001
    - Implement `disconnect()` for graceful disconnection
    - Implement `_on_disconnected()` with auto-reconnect: up to 5 attempts at 30s intervals, then alert email and 60s retry
    - Wire `IB.disconnectedEvent`, `IB.connectedEvent`, `IB.errorEvent` handlers
    - Implement `is_connected()` status check
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 9.2 Implement MarketDataService
    - Create `src/services/market_data_service.py` with the `MarketDataService` class
    - Implement `subscribe_all()` — qualify contracts and subscribe with `reqMktData()` using genericTickList='233,165'
    - Implement `unsubscribe_all()` — cancel all market data subscriptions
    - Implement `on_pending_tickers()` callback for `IB.pendingTickersEvent` — extract price/volume, forward to StrategyEngine within 100ms
    - Implement `get_price_history()` and `get_volume_history()` using rolling deque per symbol
    - Use async I/O throughout, never block the event loop
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 9.3 Write unit tests for ConnectionManager and MarketDataService
    - Test connection with paper vs live port selection
    - Test reconnection logic with mocked IB events
    - Test 5-failure alert email trigger
    - Test market data subscription and tick forwarding
    - Test price/volume history rolling window
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3_

- [x] 10. OrderExecutor
  - [x] 10.1 Implement OrderExecutor
    - Create `src/services/order_executor.py` with the `OrderExecutor` class
    - Implement `execute_trade()` — place market order via IB, on fill immediately place stop-loss, persist trade record
    - Implement `_on_exec_details()` callback for `IB.execDetailsEvent` — record fill details, trigger stop-loss placement
    - Implement `_on_order_status()` callback for `IB.orderStatusEvent` — log status changes
    - Implement `cancel_all_pending()` — cancel all open orders (used in graceful shutdown and hard stop)
    - _Requirements: 10.1, 10.2, 10.3, 14.1_

  - [ ]* 10.2 Write unit tests for OrderExecutor
    - Test order placement with mocked IB
    - Test fill callback triggers stop-loss placement
    - Test trade record persistence on fill
    - Test cancel_all_pending cancels all open orders
    - _Requirements: 10.1, 10.2, 14.1_

- [x] 11. MarketHoursService
  - [x] 11.1 Implement MarketHoursService
    - Create `src/services/market_hours_service.py` with the `MarketHoursService` class
    - Implement `update_schedule()` — fetch trading schedule from IB using `reqContractDetails()` for a reference contract (SPY), parse `liquidHours`
    - Implement `is_market_open()` — returns True during regular market hours (9:30–16:00 ET)
    - Implement `next_market_open()` and `next_market_close()` — return datetime of next open/close
    - Use exchange calendar data, not hardcoded timezone offsets
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 11.2 Write property test for no trades outside market hours (Property 2)
    - **Property 2: No trades outside market hours**
    - Use Hypothesis to generate random trade signals with off-hours timestamps
    - Verify all signals are suppressed when market is closed
    - **Validates: Requirements 2.4, 3.2**

  - [ ]* 11.3 Write unit tests for MarketHoursService
    - Test market open/close detection at boundary times
    - Test next_market_open/close calculations
    - Test holiday handling
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 12. PolymarketService
  - [x] 12.1 Implement PolymarketService
    - Create `src/services/polymarket_service.py` with the `PolymarketService` class
    - Implement `fetch_markets()` — GET requests to Polymarket Gamma API (`https://gamma-api.polymarket.com/markets`) filtered by relevant tags (economics, politics, fed, inflation, recession, interest-rates)
    - Implement `compute_sentiment()` — aggregate market probabilities into score from −1.0 to +1.0, weighted by volume and recency
    - Implement `update()` — called every 15 minutes by scheduler; on API failure, keep last score and log WARNING
    - Expose `sentiment_score` and `last_fetch_time` properties
    - Use `aiohttp.ClientSession` for async HTTP requests
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [ ]* 12.2 Write property test for sentiment score bounds and independence (Property 14)
    - **Property 14: Sentiment score bounds and independence**
    - Use Hypothesis to generate random Polymarket market data
    - Verify computed sentiment score ∈ [−1.0, +1.0]
    - Verify no trade triggered by sentiment score alone (without accompanying strategy signal)
    - **Validates: Requirements 13.2, 13.3**

  - [ ]* 12.3 Write unit tests for PolymarketService
    - Test sentiment computation with known market data
    - Test API failure fallback to last score
    - Test 15-minute update scheduling
    - _Requirements: 13.1, 13.2, 13.4_

- [x] 13. Checkpoint — Verify all services
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. ReportGenerator
  - [x] 14.1 Implement ReportGenerator
    - Create `src/services/report_generator.py` with the `ReportGenerator` class
    - Implement `generate_report()` — build HTML report with: total portfolio value, daily P&L (absolute + percentage), trade list (symbol, direction, entry/exit price, P&L), top 3 winners/losers, open positions with unrealized P&L, Polymarket sentiment summary
    - Implement `send_report()` — send via SMTP using `smtplib` with config EMAIL_SMTP_* parameters; on failure, retry once after 60s and store HTML locally
    - Implement `_render_warning_banner()` — WARNING banner at >10% loss, CRITICAL banner at >20% loss
    - Format as clean, readable HTML email
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_

  - [ ]* 14.2 Write unit tests for ReportGenerator
    - Test HTML report contains all required sections
    - Test WARNING banner appears at >10% loss
    - Test CRITICAL banner appears at >20% loss
    - Test SMTP failure fallback stores HTML locally
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_

- [x] 15. State reconciliation and crash recovery
  - [x] 15.1 Implement crash recovery flow in StateManager
    - Complete the `reconcile_with_ib()` implementation — compare persisted state with live IB account state (positions, fills, order statuses)
    - Update local state to match IB on discrepancies, log each at WARNING level
    - Implement recovery sequence: load last state → connect to IB → fetch current IB state → reconcile → fetch missed market data → resume
    - Log recovery event at INFO level with offline duration
    - _Requirements: 16.1, 16.2, 16.3, 16.4_

  - [ ]* 15.2 Write property test for state reconciliation convergence (Property 15)
    - **Property 15: State reconciliation convergence**
    - Use Hypothesis to generate random (persisted_local_state, current_ib_state) pairs
    - Verify after reconciliation, local state matches IB state for all positions, fills, and order statuses
    - **Validates: Requirements 16.2, 16.3**

  - [ ]* 15.3 Write unit tests for crash recovery
    - Test recovery with no previous state (fresh start)
    - Test recovery with existing state and no discrepancies
    - Test recovery with discrepancies — verify local updated to match IB
    - Test offline duration logging
    - _Requirements: 16.1, 16.2, 16.3, 16.4_

- [x] 16. Graceful shutdown
  - [x] 16.1 Implement graceful shutdown handler
    - Create signal handlers for SIGINT and SIGTERM in the main agent module
    - Implement shutdown sequence: cancel all pending orders → persist current state → disconnect from IB
    - Enforce 30-second timeout — force-close and exit with code 1 if exceeded
    - Log shutdown event at INFO level, exit with code 0 on success
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

  - [ ]* 16.2 Write unit tests for graceful shutdown
    - Test shutdown sequence order of operations
    - Test 30-second timeout enforcement
    - Test exit codes (0 for success, 1 for timeout)
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

- [x] 17. CLI Interface
  - [x] 17.1 Implement CLIInterface and agent.py entry point
    - Create `agent.py` as the main entry point with argparse
    - Implement `start` command — start agent as background process, log startup at INFO
    - Implement `stop` command — send SIGTERM to running agent process for graceful shutdown
    - Implement `status` command — display portfolio value, cash balance, open positions with unrealized P&L, agent state (running/stopped/halted)
    - Implement `report` command — generate and send daily report immediately
    - Display usage message for unrecognized commands, exit with non-zero code
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5_

  - [ ]* 17.2 Write unit tests for CLIInterface
    - Test each command produces expected behavior
    - Test unrecognized command shows usage and exits non-zero
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5_

- [x] 18. Checkpoint — Verify all components individually
  - Ensure all tests pass, ask the user if questions arise.

- [x] 19. Agent orchestration and wiring
  - [x] 19.1 Implement main Agent class and event loop wiring
    - Create `src/agent.py` with the main `TradingAgent` class
    - Wire the full event-driven pipeline: ConnectionManager → MarketDataService → StrategyEngine (with IndicatorCache) → RiskManager → OrderExecutor
    - Initialize all components with proper dependency injection, including LLMService and IndicatorCache
    - Set up APScheduler's `AsyncIOScheduler` on the ib_insync event loop for: Polymarket updates (every 15 min), portfolio loss checks (every 1 min during market hours), portfolio snapshots (every 5 min during market hours), daily report at 18:00 ET, weekly performance report
    - Wire IB event callbacks: `pendingTickersEvent` → MarketDataService, `execDetailsEvent` → OrderExecutor, `orderStatusEvent` → OrderExecutor, `disconnectedEvent` → ConnectionManager
    - Implement startup sequence: load config → setup logging → initialize StateManager → initialize LLMService → connect to IB → check for crash recovery → subscribe to market data → start scheduler
    - Implement market open/close event handling: enable/disable trade execution, trigger report generation on close
    - Wire LLMService into PolymarketService (for sentiment interpretation) and ReportGenerator (for report content generation)
    - _Requirements: 1.1, 2.1, 2.3, 3.3, 3.4, 4.1, 13.1, 14.4, 18.1, 23.1, 23.2, 23.4, 24.8_

  - [x] 19.2 Wire watchlist management with volume filtering
    - Implement watchlist construction: filter candidate stocks by average daily volume > 500,000 shares
    - Implement earnings calendar integration: fetch and cache earnings dates for watchlist stocks
    - Wire watchlist to MarketDataService for subscription management
    - _Requirements: 7.2, 8.1_

- [x] 20. Deployment configuration
  - [x] 20.1 Create deployment and service configuration files
    - Create `systemd/trading-agent.service` — systemd unit file for Linux with auto-restart on failure, 5-restart limit in 10 minutes
    - Create `ecosystem.config.js` — PM2 configuration file with auto-restart and crash limit
    - Add deployment documentation for Windows (Task Scheduler / NSSM), Mac (launchd), and Linux (systemd)
    - Implement watchdog crash limit: stop restarts after 5 crashes in 10 minutes, send alert email
    - _Requirements: 17.1, 17.2, 17.3_

- [x] 21. Final checkpoint — Full integration verification
  - Ensure all tests pass, ask the user if questions arise.

- [x] 22. LLMService implementation
  - [x] 22.1 Implement LLMService
    - Create `src/services/llm_service.py` with the `LLMService` class
    - Initialize `anthropic.AsyncAnthropic` client using ANTHROPIC_API_KEY from environment
    - Implement `_call_llm()` internal method: check daily limit (MAX_LLM_CALLS_PER_DAY), make API call, log token usage (purpose, model, input/output/total tokens) at INFO level
    - Implement daily counter reset at midnight (new calendar day)
    - When daily limit reached: skip LLM step, log WARNING with current call count, return empty/fallback response
    - On API error: log WARNING, return empty/fallback response — deterministic layer continues unaffected
    - Implement `interpret_sentiment()` for Polymarket sentiment interpretation (max 4x daily)
    - Implement `generate_report_content()` for daily email report narrative (1x daily)
    - Implement `interpret_unusual_conditions()` for unusual market condition analysis (as needed)
    - Persist each LLM call to `llm_calls` table in StateManager
    - _Requirements: 23.1, 23.2, 23.3, 23.4, 23.5, 23.6, 23.7, 21.5_

  - [ ]* 22.2 Write property test for LLM daily call limit (Property 17)
    - **Property 17: LLM daily call limit enforcement**
    - Use Hypothesis to generate random sequences of LLM call requests within a day
    - Verify calls are capped at MAX_LLM_CALLS_PER_DAY
    - Verify subsequent calls are skipped with WARNING logged
    - Verify counter resets on new calendar day
    - **Validates: Requirements 23.5, 23.6**

  - [ ]* 22.3 Write property test for deterministic layer independence (Property 18)
    - **Property 18: Deterministic layer independence from LLM**
    - Use Hypothesis to generate random LLM failure states (API error, rate limit, network failure)
    - Verify indicator calculations, signal generation, risk management, order execution, and state persistence continue without interruption
    - **Validates: Requirements 23.1, 23.2, 23.7**

  - [ ]* 22.4 Write unit tests for LLMService
    - Test daily call limit enforcement with mocked Anthropic client
    - Test token logging includes purpose, model, input/output/total tokens
    - Test API error handling — graceful degradation, WARNING logged
    - Test daily counter reset at midnight
    - Test `interpret_sentiment()`, `generate_report_content()`, `interpret_unusual_conditions()` with mocked responses
    - _Requirements: 23.3, 23.4, 23.5, 23.6, 23.7_

- [x] 23. Indicator caching and performance optimization
  - [x] 23.1 Implement IndicatorCache with dirty flag mechanism
    - Create `src/strategies/indicator_cache.py` with the `IndicatorCache` class
    - Implement rolling price and volume windows using `collections.deque` with fixed size
    - Implement `update_price()` and `update_volume()` — append to deque, mark dependent indicators as dirty
    - Implement `get_indicator()` — return cached value if not dirty, None if dirty
    - Implement `set_indicator()` — store value and clear dirty flag
    - Implement `is_dirty()` — check if indicator needs recalculation
    - Implement `get_prices()` and `get_volumes()` — return deque contents as NumPy arrays for vectorized calculation
    - _Requirements: 24.4, 24.5, 24.6_

  - [x] 23.2 Integrate IndicatorCache into StrategyEngine
    - Update `StrategyEngine` to accept `IndicatorCache` in constructor
    - Update `process_tick()` to check cache before calculating indicators
    - Only recalculate indicators when dirty flag is set
    - Store calculated values back in cache after computation
    - _Requirements: 24.4, 24.6_

  - [x] 23.3 Add signal-to-order latency profiling
    - Instrument the pipeline from `MarketDataService.on_pending_tickers()` through to `OrderExecutor.execute_trade()` with timing measurements
    - Log signal-to-order latency at DEBUG level for every signal, at WARNING level if > 100ms
    - Implement weekly performance report: collect latency statistics and log a summary once per week via APScheduler
    - _Requirements: 24.7, 24.8_

  - [ ]* 23.4 Write property test for NumPy indicator equivalence (Property 19)
    - **Property 19: NumPy indicator equivalence**
    - Use Hypothesis to generate random price sequences of sufficient length
    - Implement reference iterative (for-loop) versions of RSI, MACD, EMA, Bollinger Bands
    - Verify NumPy vectorized results match reference within floating-point tolerance (1e-10)
    - **Validates: Requirements 24.3**

  - [ ]* 23.5 Write property test for indicator cache consistency (Property 20)
    - **Property 20: Indicator cache consistency**
    - Use Hypothesis to generate random sequences of price updates
    - Verify cached value (when not dirty) equals fresh calculation from current window
    - Verify dirty flag is set after price update for all dependent indicators
    - **Validates: Requirements 24.4, 24.6**

  - [ ]* 23.6 Write unit tests for IndicatorCache
    - Test dirty flag set on price update
    - Test cache hit returns stored value when not dirty
    - Test cache miss returns None when dirty
    - Test deque fixed size — oldest values evicted
    - Test NumPy array conversion from deque
    - _Requirements: 24.4, 24.5, 24.6_

- [x] 24. CLI test command implementation
  - [x] 24.1 Add test subcommand to CLI
    - Update `agent.py` argparse to support `python agent.py test` command
    - Implement `test` command: run all tests via pytest
    - Implement `test --unit` flag: run only unit tests (tests/unit/ and test/test_indicators.py)
    - Implement `test --integ` flag: run only integration tests (tests/integration/ and test/test_agent.py)
    - Implement `test --perf` flag: run only performance tests (tests/performance/ and test/test_performance.py)
    - All tests must be runnable without active IB connection (use mocks)
    - _Requirements: 25.1, 25.2, 25.3, 25.4_

- [x] 25. Local test suite — Specific test files
  - [x] 25.1 Implement unit tests in test/test_indicators.py
    - Create `test/test_indicators.py` with specific test functions:
    - `test_rsi_overbought()`: verify RSI > 70 for known overbought price sequence
    - `test_rsi_oversold()`: verify RSI < 30 for known oversold price sequence
    - `test_macd_crossover()`: verify MACD crossover detection with known data
    - `test_ema_crossover()`: verify 9/21 EMA crossover detection with known data
    - `test_bollinger_bands()`: verify BB calculation with known data
    - `test_stop_loss_trigger()`: verify stop-loss triggers at correct price
    - `test_portfolio_hard_stop()`: verify hard stop activates at 20% loss
    - `test_position_size_limit()`: verify position size limit enforcement at 25%
    - `test_cash_buffer()`: verify cash buffer enforcement at 10%
    - All tests must use fixed seeds and be deterministic
    - _Requirements: 25.5_

  - [x] 25.2 Implement integration tests in test/test_agent.py
    - Create `test/test_agent.py` with specific test functions:
    - `test_buy_signal_flow()`: end-to-end buy signal from market data to order (mocked IB)
    - `test_sell_signal_flow()`: end-to-end sell signal from market data to order (mocked IB)
    - `test_recovery_flow()`: crash recovery loads state and reconciles (mocked IB)
    - `test_market_closed_no_trades()`: verify no trades executed outside market hours
    - `test_email_report_generation()`: verify report generation produces valid HTML (mocked SMTP)
    - `test_polymarket_signal_integration()`: verify Polymarket sentiment affects signal confidence (mocked API)
    - All tests must use mocks for external API calls
    - _Requirements: 25.6_

  - [x] 25.3 Implement performance tests in test/test_performance.py
    - Create `test/test_performance.py` with specific test functions:
    - `test_indicator_calculation_speed()`: verify all indicators calculate in < 10ms for 1000 candles
    - `test_signal_to_order_latency()`: verify end-to-end signal processing in < 100ms (mocked IB)
    - `test_numpy_vs_loop_benchmark()`: verify NumPy vectorized implementation is at least 10x faster than equivalent Python for-loop
    - All tests must use fixed seeds and be deterministic
    - _Requirements: 25.7_

  - [x] 25.4 Implement backtesting smoke tests in test/test_backtest.py
    - Create `test/test_backtest.py` with specific test functions:
    - `test_strategy_not_bankrupt()`: run strategy on 30 days of historical test data, verify portfolio loss does not exceed 20%
    - `test_strategy_makes_trades()`: run strategy on 30 days of historical test data, verify at least some trade signals are generated
    - Use deterministic historical data (fixed dataset or fixed seed for generation)
    - All external API calls must be mocked
    - _Requirements: 25.8, 25.9_

- [x] 26. Checkpoint — Verify all new components
  - Ensure all new tests (LLMService, IndicatorCache, performance, backtesting, CLI test) pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at reasonable intervals
- Property tests validate the 20 correctness properties defined in the design document using Hypothesis
- Unit tests validate specific examples, edge cases, and integration points
- Performance tests validate latency targets and NumPy vectorization speedup
- Backtesting smoke tests validate strategy viability on historical data
- The implementation language is Python 3.10+ as specified in the design document
- All file paths use relative paths for cross-platform compatibility (Windows/Mac/Linux)
- All technical indicator calculations must use NumPy vectorized operations (no Python for-loops on price arrays)
- LLM API calls are limited to MAX_LLM_CALLS_PER_DAY (default 10) and logged with token counts for cost tracking
