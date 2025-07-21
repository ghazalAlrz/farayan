def visualize_performance_metrics(self, metrics):
    """Visualize performance metrics"""
    fig = plt.figure(figsize=(20, 8))

    # Prepare data for plotting
    models = ["Black-Scholes", "BOPM"]
    option_types = ["Calls", "Puts"]

    # MAE comparison
    ax1 = plt.subplot(1, 3, 1)
    x = np.arange(len(option_types))
    width = 0.35

    mae_bs = [
        metrics["Calls"]["Black-Scholes"]["MAE"],
        metrics["Puts"]["Black-Scholes"]["MAE"],
    ]
    mae_bopm = [metrics["Calls"]["BOPM"]["MAE"], metrics["Puts"]["BOPM"]["MAE"]]

    bars1 = ax1.bar(
        x - width / 2, mae_bs, width, label="Black-Scholes", alpha=0.8, color="#ff7f0e"
    )
    bars2 = ax1.bar(
        x + width / 2, mae_bopm, width, label="BOPM", alpha=0.8, color="#2ca02c"
    )

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"${height:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    for bar in bars2:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"${height:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax1.set_title(
        "Mean Absolute Error (MAE)\nAverage absolute difference between model and market prices",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel("MAE ($)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(option_types)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # RMSE comparison
    ax2 = plt.subplot(1, 3, 2)
    rmse_  # Options Pricing Analysis: BOPM vs Black-Scholes vs Market Data


# Analyzing Apple (AAPL) options data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class OptionsAnalyzer:
    def __init__(self, symbol="AAPL"):
        self.symbol = symbol
        self.stock_data = None
        self.options_data = None
        self.current_price = None
        self.risk_free_rate = 0.05  # 5% risk-free rate

    def fetch_market_data(self):
        """Fetch stock data and options data from Yahoo Finance"""
        print(f"Fetching market data for {self.symbol}...")

        # Fetch stock data
        stock = yf.Ticker(self.symbol)
        self.stock_data = stock.history(period="1y")
        self.current_price = self.stock_data["Close"].iloc[-1]

        # Get options expiration dates
        exp_dates = stock.options

        # Use the first available expiration date
        if exp_dates:
            exp_date = exp_dates[0]
            options_chain = stock.option_chain(exp_date)
            self.calls_data = options_chain.calls
            self.puts_data = options_chain.puts
            self.expiration_date = exp_date

        print(f"Current stock price: ${self.current_price:.2f}")
        print(f"Options expiration date: {self.expiration_date}")

    def visualize_stock_data(self):
        """Create visualizations of stock data"""
        fig = plt.figure(figsize=(16, 14))

        # Calculate statistics
        returns = self.stock_data["Close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        avg_volume = self.stock_data["Volume"].mean()
        price_change_1y = (
            (self.stock_data["Close"].iloc[-1] / self.stock_data["Close"].iloc[0]) - 1
        ) * 100

        # Stock price over time
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(
            self.stock_data.index,
            self.stock_data["Close"],
            linewidth=2,
            color="#1f77b4",
        )
        ax1.set_title(
            f"{self.symbol} Stock Price Over Last Year\n"
            + f"Current: ${self.stock_data['Close'].iloc[-1]:.2f} | "
            + f"1Y Change: {price_change_1y:+.1f}%",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price ($)")
        ax1.grid(True, alpha=0.3)
        # Add price range annotation
        min_price = self.stock_data["Close"].min()
        max_price = self.stock_data["Close"].max()
        ax1.text(
            0.02,
            0.98,
            f"Range: ${min_price:.2f} - ${max_price:.2f}",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Volume
        ax2 = plt.subplot(2, 2, 2)
        ax2.bar(
            self.stock_data.index, self.stock_data["Volume"], alpha=0.7, color="#ff7f0e"
        )
        ax2.set_title(
            f"{self.symbol} Trading Volume\n"
            + f"Average Daily Volume: {avg_volume / 1e6:.1f}M shares",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volume (shares)")
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x / 1e6:.0f}M"))

        # Daily returns distribution
        ax3 = plt.subplot(2, 2, 3)
        n, bins, patches = ax3.hist(
            returns, bins=50, alpha=0.7, edgecolor="black", color="#2ca02c"
        )
        ax3.set_title(
            f"Daily Returns Distribution\n"
            + f"Mean: {returns.mean() * 100:.3f}% | Std: {returns.std() * 100:.3f}% | "
            + f"Annualized Vol: {volatility * 100:.1f}%",
            fontsize=12,
            fontweight="bold",
        )
        ax3.set_xlabel("Daily Return")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)
        # Add normal distribution overlay
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = (
            (
                (1 / (sigma * np.sqrt(2 * np.pi)))
                * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            )
            * len(returns)
            * (bins[1] - bins[0])
        )
        ax3.plot(x, normal_dist, "r-", linewidth=2, label="Normal Distribution")
        ax3.legend()

        # Moving averages
        self.stock_data["MA20"] = self.stock_data["Close"].rolling(window=20).mean()
        self.stock_data["MA50"] = self.stock_data["Close"].rolling(window=50).mean()

        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(
            self.stock_data.index,
            self.stock_data["Close"],
            label="Close Price",
            linewidth=2,
        )
        ax4.plot(
            self.stock_data.index, self.stock_data["MA20"], label="20-day MA", alpha=0.8
        )
        ax4.plot(
            self.stock_data.index, self.stock_data["MA50"], label="50-day MA", alpha=0.8
        )

        # Calculate MA signals
        current_vs_ma20 = (
            (self.current_price / self.stock_data["MA20"].iloc[-1]) - 1
        ) * 100
        current_vs_ma50 = (
            (self.current_price / self.stock_data["MA50"].iloc[-1]) - 1
        ) * 100

        ax4.set_title(
            f"{self.symbol} Price with Moving Averages\n"
            + f"vs 20-day MA: {current_vs_ma20:+.1f}% | vs 50-day MA: {current_vs_ma50:+.1f}%",
            fontsize=12,
            fontweight="bold",
        )
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Price ($)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("stock_data_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Display calculated parameters
        print(f"\n📊 CALCULATED MODEL PARAMETERS")
        print(f"{'=' * 50}")
        print(f"Annual Volatility (σ):     {volatility:.4f} ({volatility * 100:.2f}%)")
        print(f"Current Stock Price (S₀):  ${self.current_price:.2f}")
        print(
            f"Risk-free Rate (r):        {self.risk_free_rate:.3f} ({self.risk_free_rate * 100:.1f}%)"
        )
        print(f"Average Daily Return:      {returns.mean() * 100:.3f}%")
        print(f"Daily Volatility:          {returns.std() * 100:.3f}%")
        print(
            f"Sharpe Ratio (approx):     {(returns.mean() * 252 - self.risk_free_rate) / volatility:.3f}"
        )

        return volatility

    def calculate_time_to_expiry(self):
        """Calculate time to expiry in years"""
        exp_date = datetime.strptime(self.expiration_date, "%Y-%m-%d")
        today = datetime.now()
        time_to_expiry = (exp_date - today).days / 365.0
        return max(time_to_expiry, 0.01)  # Ensure positive value

    def black_scholes_call(self, S, K, T, r, sigma):
        """Black-Scholes formula for call options"""
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    def black_scholes_put(self, S, K, T, r, sigma):
        """Black-Scholes formula for put options"""
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price

    def binomial_option_pricing(self, S, K, T, r, sigma, n_steps, option_type="call"):
        """Binomial Option Pricing Model (BOPM)"""
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

        # Initialize asset prices at maturity
        asset_prices = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            asset_prices[i] = S * (u ** (n_steps - i)) * (d**i)

        # Initialize option values at maturity
        option_values = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            if option_type == "call":
                option_values[i] = max(0, asset_prices[i] - K)
            else:  # put
                option_values[i] = max(0, K - asset_prices[i])

        # Backward induction
        for step in range(n_steps - 1, -1, -1):
            for i in range(step + 1):
                option_values[i] = np.exp(-r * dt) * (
                    p * option_values[i] + (1 - p) * option_values[i + 1]
                )

        return option_values[0]

    def compare_models(self, volatility):
        """Compare BOPM, Black-Scholes, and market prices"""
        T = self.calculate_time_to_expiry()

        # Select options near the money
        calls_atm = self.calls_data[
            (self.calls_data["strike"] >= self.current_price * 0.9)
            & (self.calls_data["strike"] <= self.current_price * 1.1)
        ].copy()

        puts_atm = self.puts_data[
            (self.puts_data["strike"] >= self.current_price * 0.9)
            & (self.puts_data["strike"] <= self.current_price * 1.1)
        ].copy()

        print(f"\n📋 MODEL COMPARISON SETUP")
        print(f"{'=' * 50}")
        print(f"Selected {len(calls_atm)} call options and {len(puts_atm)} put options")
        print(
            f"Strike price range: ${self.current_price * 0.9:.2f} - ${self.current_price * 1.1:.2f}"
        )
        print(f"Time to expiration: {T * 365:.0f} days ({T:.4f} years)")
        print(f"Options expiry date: {self.expiration_date}")
        print(f"Using volatility: {volatility:.4f} ({volatility * 100:.2f}%)")

        # Calculate model prices for calls
        calls_atm["BS_Price"] = calls_atm["strike"].apply(
            lambda K: self.black_scholes_call(
                self.current_price, K, T, self.risk_free_rate, volatility
            )
        )
        calls_atm["BOPM_Price"] = calls_atm["strike"].apply(
            lambda K: self.binomial_option_pricing(
                self.current_price, K, T, self.risk_free_rate, volatility, 100, "call"
            )
        )

        # Calculate model prices for puts
        puts_atm["BS_Price"] = puts_atm["strike"].apply(
            lambda K: self.black_scholes_put(
                self.current_price, K, T, self.risk_free_rate, volatility
            )
        )
        puts_atm["BOPM_Price"] = puts_atm["strike"].apply(
            lambda K: self.binomial_option_pricing(
                self.current_price, K, T, self.risk_free_rate, volatility, 100, "put"
            )
        )

        return calls_atm, puts_atm

    def visualize_model_comparison(self, calls_atm, puts_atm):
        """Create comparison visualizations"""
        fig = plt.figure(figsize=(18, 14))

        # Calculate overall statistics
        T = self.calculate_time_to_expiry()
        total_call_volume = (
            calls_atm["volume"].sum() if "volume" in calls_atm.columns else 0
        )
        total_put_volume = (
            puts_atm["volume"].sum() if "volume" in puts_atm.columns else 0
        )

        # Call options comparison
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(
            calls_atm["strike"],
            calls_atm["lastPrice"],
            "o-",
            label="Market Price",
            linewidth=3,
            markersize=8,
            color="#1f77b4",
        )
        ax1.plot(
            calls_atm["strike"],
            calls_atm["BS_Price"],
            "s-",
            label="Black-Scholes",
            linewidth=2,
            markersize=6,
            color="#ff7f0e",
        )
        ax1.plot(
            calls_atm["strike"],
            calls_atm["BOPM_Price"],
            "^-",
            label="BOPM (100 steps)",
            linewidth=2,
            markersize=6,
            color="#2ca02c",
        )

        # Add moneyness information
        atm_strike = calls_atm.iloc[
            (calls_atm["strike"] - self.current_price).abs().argsort()[:1]
        ]
        ax1.axvline(
            x=self.current_price,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Current Stock Price",
        )

        ax1.set_title(
            f"Call Options: Model vs Market Prices\n"
            + f"Expiry: {self.expiration_date} ({T * 365:.0f} days) | ATM Strike: ~${atm_strike['strike'].iloc[0]:.0f}",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_xlabel("Strike Price ($)")
        ax1.set_ylabel("Option Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Put options comparison
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(
            puts_atm["strike"],
            puts_atm["lastPrice"],
            "o-",
            label="Market Price",
            linewidth=3,
            markersize=8,
            color="#1f77b4",
        )
        ax2.plot(
            puts_atm["strike"],
            puts_atm["BS_Price"],
            "s-",
            label="Black-Scholes",
            linewidth=2,
            markersize=6,
            color="#ff7f0e",
        )
        ax2.plot(
            puts_atm["strike"],
            puts_atm["BOPM_Price"],
            "^-",
            label="BOPM (100 steps)",
            linewidth=2,
            markersize=6,
            color="#2ca02c",
        )

        ax2.axvline(
            x=self.current_price,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Current Stock Price",
        )
        ax2.set_title(
            f"Put Options: Model vs Market Prices\n"
            + f"Expiry: {self.expiration_date} ({T * 365:.0f} days) | ATM Strike: ~${atm_strike['strike'].iloc[0]:.0f}",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlabel("Strike Price ($)")
        ax2.set_ylabel("Option Price ($)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Error analysis for calls
        calls_bs_error = calls_atm["lastPrice"] - calls_atm["BS_Price"]
        calls_bopm_error = calls_atm["lastPrice"] - calls_atm["BOPM_Price"]

        ax3 = plt.subplot(2, 2, 3)
        x = range(len(calls_atm))
        width = 0.35
        bars1 = ax3.bar(
            [i - width / 2 for i in x],
            calls_bs_error,
            width,
            label="Black-Scholes Error",
            alpha=0.8,
            color="#ff7f0e",
        )
        bars2 = ax3.bar(
            [i + width / 2 for i in x],
            calls_bopm_error,
            width,
            label="BOPM Error",
            alpha=0.8,
            color="#2ca02c",
        )

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 if height > 0 else height - 0.03,
                f"{height:.2f}",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=8,
            )

        mean_bs_error = calls_bs_error.mean()
        mean_bopm_error = calls_bopm_error.mean()

        ax3.set_title(
            f"Call Options: Pricing Errors (Market - Model)\n"
            + f"Mean BS Error: ${mean_bs_error:.3f} | Mean BOPM Error: ${mean_bopm_error:.3f}\n"
            + f"Note: Positive = Model Underprices, Negative = Model Overprices",
            fontsize=11,
            fontweight="bold",
        )
        ax3.set_xlabel("Option Index (sorted by strike)")
        ax3.set_ylabel("Pricing Error ($)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1)

        # Set x-axis labels to show strike prices
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"${s:.0f}" for s in calls_atm["strike"]], rotation=45)

        # Error analysis for puts
        puts_bs_error = puts_atm["lastPrice"] - puts_atm["BS_Price"]
        puts_bopm_error = puts_atm["lastPrice"] - puts_atm["BOPM_Price"]

        ax4 = plt.subplot(2, 2, 4)
        x = range(len(puts_atm))
        bars3 = ax4.bar(
            [i - width / 2 for i in x],
            puts_bs_error,
            width,
            label="Black-Scholes Error",
            alpha=0.8,
            color="#ff7f0e",
        )
        bars4 = ax4.bar(
            [i + width / 2 for i in x],
            puts_bopm_error,
            width,
            label="BOPM Error",
            alpha=0.8,
            color="#2ca02c",
        )

        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 if height > 0 else height - 0.03,
                f"{height:.2f}",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=8,
            )

        mean_bs_error_puts = puts_bs_error.mean()
        mean_bopm_error_puts = puts_bopm_error.mean()

        ax4.set_title(
            f"Put Options: Pricing Errors (Market - Model)\n"
            + f"Mean BS Error: ${mean_bs_error_puts:.3f} | Mean BOPM Error: ${mean_bopm_error_puts:.3f}\n"
            + f"Note: Positive = Model Underprices, Negative = Model Overprices",
            fontsize=11,
            fontweight="bold",
        )
        ax4.set_xlabel("Option Index (sorted by strike)")
        ax4.set_ylabel("Pricing Error ($)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1)

        # Set x-axis labels to show strike prices
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"${s:.0f}" for s in puts_atm["strike"]], rotation=45)

        plt.tight_layout()
        plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

        print(f"\n📊 PRICING ERROR ANALYSIS")
        print(f"{'=' * 50}")
        print(f"CALL OPTIONS:")
        print(f"  Black-Scholes mean error: ${mean_bs_error:+.3f}")
        print(f"  BOPM mean error:          ${mean_bopm_error:+.3f}")
        print(f"PUT OPTIONS:")
        print(f"  Black-Scholes mean error: ${mean_bs_error_puts:+.3f}")
        print(f"  BOPM mean error:          ${mean_bopm_error_puts:+.3f}")
        print(
            f"\nNote: Errors represent single-day snapshot on {datetime.now().strftime('%Y-%m-%d')}"
        )
        print(f"Positive errors = Model underprices vs market")

    def calculate_performance_metrics(self, calls_atm, puts_atm):
        """Calculate performance metrics for model evaluation"""
        # Calculate errors
        calls_bs_error = calls_atm["lastPrice"] - calls_atm["BS_Price"]
        calls_bopm_error = calls_atm["lastPrice"] - calls_atm["BOPM_Price"]
        puts_bs_error = puts_atm["lastPrice"] - puts_atm["BS_Price"]
        puts_bopm_error = puts_atm["lastPrice"] - puts_atm["BOPM_Price"]

        # Calculate metrics
        metrics = {
            "Calls": {
                "Black-Scholes": {
                    "MAE": np.mean(np.abs(calls_bs_error)),
                    "RMSE": np.sqrt(np.mean(calls_bs_error**2)),
                    "MAPE": np.mean(np.abs(calls_bs_error / calls_atm["lastPrice"]))
                    * 100,
                },
                "BOPM": {
                    "MAE": np.mean(np.abs(calls_bopm_error)),
                    "RMSE": np.sqrt(np.mean(calls_bopm_error**2)),
                    "MAPE": np.mean(np.abs(calls_bopm_error / calls_atm["lastPrice"]))
                    * 100,
                },
            },
            "Puts": {
                "Black-Scholes": {
                    "MAE": np.mean(np.abs(puts_bs_error)),
                    "RMSE": np.sqrt(np.mean(puts_bs_error**2)),
                    "MAPE": np.mean(np.abs(puts_bs_error / puts_atm["lastPrice"]))
                    * 100,
                },
                "BOPM": {
                    "MAE": np.mean(np.abs(puts_bopm_error)),
                    "RMSE": np.sqrt(np.mean(puts_bopm_error**2)),
                    "MAPE": np.mean(np.abs(puts_bopm_error / puts_atm["lastPrice"]))
                    * 100,
                },
            },
        }

        return metrics

    def visualize_performance_metrics(self, metrics):
        """Visualize performance metrics"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Prepare data for plotting
        models = ["Black-Scholes", "BOPM"]
        option_types = ["Calls", "Puts"]

        # MAE comparison
        mae_data = []
        for opt_type in option_types:
            for model in models:
                mae_data.append(metrics[opt_type][model]["MAE"])

        x = np.arange(len(option_types))
        width = 0.35

        ax1.bar(
            x - width / 2,
            [
                metrics["Calls"]["Black-Scholes"]["MAE"],
                metrics["Puts"]["Black-Scholes"]["MAE"],
            ],
            width,
            label="Black-Scholes",
            alpha=0.8,
        )
        ax1.bar(
            x + width / 2,
            [metrics["Calls"]["BOPM"]["MAE"], metrics["Puts"]["BOPM"]["MAE"]],
            width,
            label="BOPM",
            alpha=0.8,
        )
        ax1.set_title("Mean Absolute Error (MAE)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("MAE ($)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(option_types)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # RMSE comparison
        ax2.bar(
            x - width / 2,
            [
                metrics["Calls"]["Black-Scholes"]["RMSE"],
                metrics["Puts"]["Black-Scholes"]["RMSE"],
            ],
            width,
            label="Black-Scholes",
            alpha=0.8,
        )
        ax2.bar(
            x + width / 2,
            [metrics["Calls"]["BOPM"]["RMSE"], metrics["Puts"]["BOPM"]["RMSE"]],
            width,
            label="BOPM",
            alpha=0.8,
        )
        ax2.set_title("Root Mean Square Error (RMSE)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("RMSE ($)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(option_types)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # MAPE comparison
        ax3.bar(
            x - width / 2,
            [
                metrics["Calls"]["Black-Scholes"]["MAPE"],
                metrics["Puts"]["Black-Scholes"]["MAPE"],
            ],
            width,
            label="Black-Scholes",
            alpha=0.8,
        )
        ax3.bar(
            x + width / 2,
            [metrics["Calls"]["BOPM"]["MAPE"], metrics["Puts"]["BOPM"]["MAPE"]],
            width,
            label="BOPM",
            alpha=0.8,
        )
        ax3.set_title(
            "Mean Absolute Percentage Error (MAPE)", fontsize=14, fontweight="bold"
        )
        ax3.set_ylabel("MAPE (%)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(option_types)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("performance_metrics.png", dpi=300, bbox_inches="tight")
        plt.show()

    def create_summary_table(self, calls_atm, puts_atm, metrics):
        """Create and visualize summary table"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("tight")
        ax.axis("off")

        # Create summary data
        summary_data = []

        # Add header
        summary_data.append(
            [
                "Metric",
                "Calls - Black-Scholes",
                "Calls - BOPM",
                "Puts - Black-Scholes",
                "Puts - BOPM",
            ]
        )

        # Add metrics
        summary_data.append(
            [
                "MAE ($)",
                f"{metrics['Calls']['Black-Scholes']['MAE']:.3f}",
                f"{metrics['Calls']['BOPM']['MAE']:.3f}",
                f"{metrics['Puts']['Black-Scholes']['MAE']:.3f}",
                f"{metrics['Puts']['BOPM']['MAE']:.3f}",
            ]
        )

        summary_data.append(
            [
                "RMSE ($)",
                f"{metrics['Calls']['Black-Scholes']['RMSE']:.3f}",
                f"{metrics['Calls']['BOPM']['RMSE']:.3f}",
                f"{metrics['Puts']['Black-Scholes']['RMSE']:.3f}",
                f"{metrics['Puts']['BOPM']['RMSE']:.3f}",
            ]
        )

        summary_data.append(
            [
                "MAPE (%)",
                f"{metrics['Calls']['Black-Scholes']['MAPE']:.2f}",
                f"{metrics['Calls']['BOPM']['MAPE']:.2f}",
                f"{metrics['Puts']['Black-Scholes']['MAPE']:.2f}",
                f"{metrics['Puts']['BOPM']['MAPE']:.2f}",
            ]
        )

        # Additional statistics
        summary_data.append(["", "", "", "", ""])
        summary_data.append(["Additional Statistics", "", "", "", ""])
        summary_data.append(
            ["Current Stock Price ($)", f"{self.current_price:.2f}", "", "", ""]
        )
        summary_data.append(
            [
                "Time to Expiration (days)",
                f"{self.calculate_time_to_expiry() * 365:.0f}",
                "",
                "",
                "",
            ]
        )
        summary_data.append(
            ["Risk-free Rate (%)", f"{self.risk_free_rate * 100:.1f}", "", "", ""]
        )
        summary_data.append(
            ["Number of Call Options Analyzed", f"{len(calls_atm)}", "", "", ""]
        )
        summary_data.append(
            ["Number of Put Options Analyzed", f"{len(puts_atm)}", "", "", ""]
        )

        # Create table
        table = ax.table(cellText=summary_data, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)

        # Style the table
        for i in range(len(summary_data)):
            for j in range(len(summary_data[i])):
                cell = table[i, j]
                if i == 0:  # Header row
                    cell.set_facecolor("#4472C4")
                    cell.set_text_props(weight="bold", color="white")
                elif i == 5:  # Additional statistics header
                    cell.set_facecolor("#D5E4F7")
                    cell.set_text_props(weight="bold")
                elif summary_data[i][0] == "":  # Empty row
                    cell.set_facecolor("#F8F9FA")
                else:
                    cell.set_facecolor("#F8F9FA")

        plt.title(
            "Options Pricing Model Performance Summary",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.savefig("summary_table.png", dpi=300, bbox_inches="tight")
        plt.show()


# Main execution
def main():
    print("=" * 60)
    print("OPTIONS PRICING ANALYSIS: BOPM vs BLACK-SCHOLES vs MARKET")
    print("=" * 60)

    # Initialize analyzer
    analyzer = OptionsAnalyzer("AAPL")

    # Step 1: Fetch market data
    print("\n1. FETCHING MARKET DATA")
    print("-" * 30)
    analyzer.fetch_market_data()

    # Step 2: Visualize stock data and calculate volatility
    print("\n2. ANALYZING STOCK DATA")
    print("-" * 30)
    volatility = analyzer.visualize_stock_data()

    # Step 3: Compare models
    print("\n3. COMPARING PRICING MODELS")
    print("-" * 30)
    calls_atm, puts_atm = analyzer.compare_models(volatility)

    # Step 4: Visualize comparisons
    print("\n4. VISUALIZING MODEL COMPARISONS")
    print("-" * 30)
    analyzer.visualize_model_comparison(calls_atm, puts_atm)

    # Step 5: Calculate performance metrics
    print("\n5. CALCULATING PERFORMANCE METRICS")
    print("-" * 30)
    metrics = analyzer.calculate_performance_metrics(calls_atm, puts_atm)

    # Step 6: Visualize performance metrics
    print("\n6. PERFORMANCE METRICS VISUALIZATION")
    print("-" * 30)
    analyzer.visualize_performance_metrics(metrics)

    # Step 7: Create summary table
    print("\n7. CREATING SUMMARY TABLE")
    print("-" * 30)
    analyzer.create_summary_table(calls_atm, puts_atm, metrics)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Generated images for your PDF report:")
    print("- stock_data_analysis.png")
    print("- model_comparison.png")
    print("- performance_metrics.png")
    print("- summary_table.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
