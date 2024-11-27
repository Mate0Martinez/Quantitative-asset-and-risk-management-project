import streamlit as st #ignore the warning, if streamlit is installed correctly it will work
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.express as px
import Portfolio_classes as pc
import threading
from apscheduler.schedulers.background import BackgroundScheduler
import multiprocessing
import time
from TradeBot import bot_class as tb
import logging


# Set page configuration
st.set_page_config(layout="wide", page_title="Portfolio Dashboard")

####################################### CUSTOM CSS #######################################
st.markdown(
    """
    <style>
    /* Adjust top margin for the main panel */
    .main > div {
        padding-top: 1rem; /* Add minimal padding for slight spacing */
    }

    /* Adjust the sidebar content slightly lower */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem; /* Add minimal padding for slight spacing */
    }

    /* Common Styles */
    .container {
        background-color: #262730; /* Dark gray background */
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5); /* Optional shadow */
        color: white;
        position: relative;
        padding: 10px;
        margin: 10px; /* Space between elements */
    }

    /* Statistic Container */
    .statistic-container {
        height: 100px; /* Fixed height for statistics */
    }

    .stat-title {
        position: absolute;
        top: 5px;
        left: 10px;
        font-size: 12px;
        font-weight: bold;
        color: #ffffffaa; /* Slightly lighter color for the title */
    }

    .stat-value {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: bold;
        height: 100%;
    }
    """,
    unsafe_allow_html=True,
)

####################################### FUNCTIONS #######################################
@st.cache_data
def load_data(markets, sectors):
    ## Import the .csv files
    commodities_prices = pd.read_csv('Data/commodity_prices.csv', index_col=0)
    emerging_prices = pd.read_csv('Data/emerging_prices.csv', index_col=0)
    emerging_sectors = pd.read_csv('Data/emerging_sectors.csv', index_col=0)
    euro_prices = pd.read_csv('Data/euro_prices.csv', index_col=0)
    euro_sectors = pd.read_csv('Data/euro_sectors.csv', index_col=0)
    sp_prices = pd.read_csv('Data/sp_prices.csv', index_col=0)
    sp_sectors = pd.read_csv('Data/sp_sectors.csv', index_col=0)

    ## Initialize the dataframes
    data = pd.DataFrame()
    sectors_data = pd.DataFrame()

    ## Depending on the markets selected, we add the corresponding returns
    for market in markets:
        if market == 'US':
            data = pd.concat([data, sp_prices], axis=1)
            sectors_data = pd.concat([sectors_data, sp_sectors], axis=0)
        elif market == 'EU':
            data = pd.concat([data, euro_prices], axis=1)
            sectors_data = pd.concat([sectors_data, euro_sectors], axis=0)
        elif market == 'EM':
            data = pd.concat([data, emerging_prices], axis=1)
            sectors_data = pd.concat([sectors_data, emerging_sectors], axis=0)
    
    ## Depending on the sectors selected, we filter out the data
    tickers = sectors_data[sectors_data['TRBC BUSI SEC NAME'].isin(sectors)].NAME
    data = data.iloc[:,data.columns.isin(tickers)]

    sectors_data = sectors_data[sectors_data['TRBC BUSI SEC NAME'].isin(sectors)].set_index('NAME')[['TRBC BUSI SEC NAME']]

    if len(data) == 0:
        st.warning('No data available for the selected markets and sectors.')
    else:
        return data, sectors_data

def plot_efficient_frontier_without_risky(shortyes, risk_aversion):
    prices, sectors_data = load_data(markets, sectors)
    efficient_frontier = pc.EfficientFrontier(prices, short=shortyes)
    frontier = efficient_frontier.get_efficient_frontier()
    gammas = np.linspace(-5, 5, 500)
    gamma_zero_index = np.argmin(np.abs(gammas))
    gamma_index = np.argmin(np.abs(gammas - risk_aversion))  # Closest gamma to risk_aversion
    sharpe_MV, mu_MV, vol_MV = efficient_frontier.metrics_MV(gamma=risk_aversion)

    #### to find the optimal weights according to our data
    returns = prices.pct_change().dropna()
    mu = returns.mean().values * 252  # Annualized expected returns
    vol = returns.std().values * np.sqrt(252)  # Annualized volatilities
    correl_matrix = returns.corr().values  # Correlation matrix
    covmat = vol.reshape(1, -1) * correl_matrix * vol.reshape(-1, 1)  # Covariance matrix
    n = mu.shape[0]
    x0 = np.ones(n) / n

    weights_mv = pd.DataFrame(np.round(efficient_frontier.efficient_frontier(n, x0, covmat, mu, risk_aversion)[2]*100,2), index=prices.columns, columns=['Holding (%)'])
    weights_mv = weights_mv.merge(sectors_data, left_index=True, right_index=True)[['TRBC BUSI SEC NAME', 'Holding (%)']].rename(columns={'TRBC BUSI SEC NAME': 'Sector'})

    # Prepare data for plotly
    x = [f[1] for f in frontier]  # Volatility
    y = [f[0] for f in frontier]  # Return

    # Create Plotly figure
    fig = go.Figure()

    # Add Efficient Frontier Line
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Efficient Frontier'))

    # Add Minimum Variance Portfolio
    fig.add_trace(go.Scatter(
        x=[frontier[gamma_zero_index][1]],
        y=[frontier[gamma_zero_index][0]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='MVP'
    ))

    # Add User Portfolio
    fig.add_trace(go.Scatter(
        x=[frontier[gamma_index][1]],
        y=[frontier[gamma_index][0]],
        mode='markers',
        marker=dict(color='green', size=10, symbol='star'),
        name='Your Portfolio'
    ))

    fig.update_layout(
        title_text=f"Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Return",
        paper_bgcolor="#262730",
        plot_bgcolor="#262730",
        font=dict(color="white"),
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h")  # Legend at bottom
    )

    return fig, sharpe_MV, mu_MV, vol_MV, weights_mv


def plot_efficient_frontier_with_risky(risk_free_rate, shortyes, risk_aversion):
    prices, sectors_data = load_data(markets, sectors)
    efficient_frontier = pc.EfficientFrontier(prices, risk_free_rate=risk_free_rate, short=shortyes)
    frontier = efficient_frontier.get_efficient_frontier()
    gammas = np.linspace(-5, 5, 500)
    gamma_zero_index = np.argmin(np.abs(gammas))
    gamma_index = np.argmin(np.abs(gammas - risk_aversion))
    sharpe_MV, mu_MV, vol_MV = efficient_frontier.metrics_MV(gamma=risk_aversion)

    # Prepare data for plotly
    x_frontier = [f[1] for f in frontier[0]]  # Volatility of Efficient Frontier
    y_frontier = [f[0] for f in frontier[0]]  # Return of Efficient Frontier

    x_cml = [f[1] for f in frontier[1]]  # Volatility of Capital Market Line
    y_cml = [f[0] for f in frontier[1]]  # Return of Capital Market Line

    #### to find the optimal weights according to our data
    returns = prices.pct_change().dropna()
    mu = returns.mean().values * 252  # Annualized expected returns
    vol = returns.std().values * np.sqrt(252)  # Annualized volatilities
    correl_matrix = returns.corr().values  # Correlation matrix
    covmat = vol.reshape(1, -1) * correl_matrix * vol.reshape(-1, 1)  # Covariance matrix
    n = mu.shape[0]
    x0 = np.ones(n) / n

    weights_mv = pd.DataFrame(np.round(efficient_frontier.efficient_frontier(n, x0, covmat, mu, risk_aversion)[2]*100,2), index=prices.columns, columns=['Holding (%)'])
    weights_mv = weights_mv.merge(sectors_data, left_index=True, right_index=True)[['TRBC BUSI SEC NAME', 'Holding (%)']].rename(columns={'TRBC BUSI SEC NAME': 'Sector'})

    # Create Plotly figure
    fig = go.Figure()

    # Add Efficient Frontier Line
    fig.add_trace(go.Scatter(x=x_frontier, y=y_frontier, mode='lines', name='Efficient Frontier'))

    # Add Capital Market Line
    fig.add_trace(go.Scatter(x=x_cml, y=y_cml, mode='lines', name='Capital Market Line', line=dict(dash='dash')))

    # Add Minimum Variance Portfolio
    fig.add_trace(go.Scatter(
        x=[frontier[0][gamma_zero_index][1]],
        y=[frontier[0][gamma_zero_index][0]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Minimum Variance Portfolio'
    ))

    # Add Tangency Portfolio
    fig.add_trace(go.Scatter(
        x=[frontier[4]],
        y=[frontier[3]],
        mode='markers',
        marker=dict(color='orange', size=10, symbol='star'),
        name='Tangency Portfolio'
    ))

    # Add User Portfolio
    fig.add_trace(go.Scatter(
        x=[frontier[0][gamma_index][1]],
        y=[frontier[0][gamma_index][0]],
        mode='markers',
        marker=dict(color='green', size=10, symbol='star'),
        name='Your Portfolio'
    ))

    fig.update_layout(
        title_text=f"Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Return",
        paper_bgcolor="#262730",
        plot_bgcolor="#262730",
        font=dict(color="white"),
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h")  # Legend at bottom
    )

    return fig, sharpe_MV, mu_MV, vol_MV, weights_mv

def ERC(markets, sectors):
    # Unpack the tuple returned by load_data
    prices, sectors_data = load_data(markets, sectors)
    
    # Pass the `prices` DataFrame to the Portfolio class
    portfolio = pc.Portfolio(prices)
    weights_erc = portfolio.ERC()

    # Compute performance
    perf = pc.get_performance(prices, weights_erc)
    perf.index = pd.to_datetime(perf.index)

    # Create and format the plotly plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf.index,
        y=perf.values,
        mode='lines',
        name='Portfolio Performance',
        line=dict(color='royalblue', width=2)
    ))

    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Performance",
        paper_bgcolor="#262730",
        plot_bgcolor="#262730",
        font=dict(color="white"),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='white',
            ticks='outside',
            tickformat="%b %Y"
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='white',
            ticks='outside'
        ),
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h")  # Legend at bottom
    )

    st.session_state.erc_plot = fig  # Save the plot to session state

    # Compute Metrics
    prtfl_return = np.dot(portfolio.mu, weights_erc)
    prtfl_vol = np.sqrt(weights_erc @ portfolio.covmat @ weights_erc)
    risk_free_rate = portfolio.risk_free_rate if portfolio.risk_free_rate is not None else 0
    sharpe_ratio = (prtfl_return - risk_free_rate) / prtfl_vol

    # Create a DataFrame to display the weights
    weights_erc = pd.DataFrame(weights_erc, index=prices.columns, columns=["Holding (%)"])
    
    # Store results in session_state for reuse
    st.session_state.erc_results = {
        'weights_erc': weights_erc,
        'return_erc': prtfl_return,
        'vol_erc': prtfl_vol,
        'sharpe_erc': sharpe_ratio,
        'perf': perf,
    }

    return weights_erc, prtfl_return, prtfl_vol, sharpe_ratio

def MDP(markets, sectors):
    prices, sectors_data = load_data(markets, sectors)
    portfolio = pc.Portfolio(prices)
    weights_mdp = portfolio.MDP()
    perf = pc.get_performance(prices, weights_mdp)
    perf.index = pd.to_datetime(perf.index)

    # Create and format the plotly plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf.index,
        y=perf.values,
        mode='lines',
        name='Portfolio Performance',
        line=dict(color='royalblue', width=2)
    ))

    fig.update_layout(
        title="Portfolio Performance (MDP)",
        xaxis_title="Date",
        yaxis_title="Performance",
        paper_bgcolor="#262730",
        plot_bgcolor="#262730",
        font=dict(color="white"),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='white',
            ticks='outside',
            tickformat="%b %Y"
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='white',
            ticks='outside'
        ),
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h")  # Legend at bottom
    )

    st.session_state.mdp_plot = fig  # Save the plot to session state

    # Metrics
    prtfl_return = np.dot(portfolio.mu, weights_mdp)
    prtfl_vol = np.sqrt(weights_mdp @ portfolio.covmat @ weights_mdp)
    risk_free_rate = portfolio.risk_free_rate if portfolio.risk_free_rate is not None else 0
    sharpe_ratio = (prtfl_return - risk_free_rate) / prtfl_vol

    # Create a DataFrame to display the weights
    weights_mdp = pd.DataFrame(weights_mdp, index=prices.columns, columns=["Holding (%)"])

    # Store results in session_state for reuse
    st.session_state.mdp_results = {
        'weights_mdp': weights_mdp,
        'return_mdp': prtfl_return,
        'vol_mdp': prtfl_vol,
        'sharpe_mdp': sharpe_ratio,
        'perf': perf
    }

    return weights_mdp, prtfl_return, prtfl_vol, sharpe_ratio

def EW(markets, sectors):
    prices, sectors_data = load_data(markets, sectors)
    portfolio = pc.Portfolio(prices)
    weights_eq = portfolio.EW()
    perf = pc.get_performance(prices, weights_eq)
    perf.index = pd.to_datetime(perf.index)

    # Create and format the plotly plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf.index,
        y=perf.values,
        mode='lines',
        name='Portfolio Performance',
        line=dict(color='royalblue', width=2)
    ))

    fig.update_layout(
        title="Portfolio Performance (Equal Weighting)",
        xaxis_title="Date",
        yaxis_title="Performance",
        paper_bgcolor="#262730",
        plot_bgcolor="#262730",
        font=dict(color="white"),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='white',
            ticks='outside',
            tickformat="%b %Y"
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='white',
            ticks='outside'
        ),
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h")  # Legend at bottom
    )

    st.session_state.eq_plot = fig  # Save the plot to session state

    # Metrics
    prtfl_return = np.dot(portfolio.mu, weights_eq)
    prtfl_vol = np.sqrt(weights_eq @ portfolio.covmat @ weights_eq)
    risk_free_rate = portfolio.risk_free_rate if portfolio.risk_free_rate is not None else 0
    sharpe_ratio = (prtfl_return - risk_free_rate) / prtfl_vol

    # Create a DataFrame to display the weights
    weights_eq = pd.DataFrame(weights_eq, index=prices.columns, columns=["Holding (%)"])

    # Store results in session_state for reuse
    st.session_state.eq_results = {
        'weights_eq': weights_eq,
        'return_eq': prtfl_return,
        'vol_eq': prtfl_vol,
        'sharpe_eq': sharpe_ratio,
        'perf': perf
    }

    return weights_eq, prtfl_return, prtfl_vol, sharpe_ratio



def create_pie_chart(df, title="Portfolio Holdings Distribution"):
    # Filter out rows with zero holdings
    df_filtered = df[df["Holding (%)"] > 0]

    # Use the DataFrame's index as labels if there is no "Company" column
    labels = df_filtered.index if "Company" not in df_filtered.columns else df_filtered["Company"]

    # Create a pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=df_filtered["Holding (%)"],
                hoverinfo="label+percent",
                textinfo="percent",
                textfont=dict(size=14),
                hole=0.5,  # Makes the pie chart a donut shape and smaller
            )
        ]
    )

    # Update layout for styling
    fig.update_layout(
        title_text=title,
        paper_bgcolor="#262730",
        font=dict(color="white"),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="v",  # Horizontal legend
        ),
    )

    return fig


####################################### SIDEBAR #######################################
with st.sidebar:
    optimization_method = st.selectbox(
    'Select portfolio', 
    ['Select an option', 'Mean variance', 'Equal Risk Contribution', 'Most Diversified', 'Black Litterman', 'Equally weighted']
)

    if (optimization_method != 'Black Litterman') & (optimization_method != 'Select an option'):
        markets = st.multiselect('Select markets', ['US', 'EU', 'EM'])
        sectors = st.multiselect('Select sectors', ['Holding Companies', 'Utilities', 'Industrial & Commercial Services', 'Banking & Investment Services', 'Healthcare Services & Equipment',
                                                    'Chemicals', 'Consumer Goods Conglomerates', 'Technology Equipment', 'Software & IT Services', 'Real Estate','Energy - Fossil Fuels',
                                                    'Industrial Goods', 'Applied Resources', 'Mineral Resources', 'Cyclical Consumer Products', 'Transportation', 'Retailers'])
    
    if optimization_method == 'Mean variance':
        shortyes = st.checkbox('Short-selling?', value=False)
        riskyes = st.checkbox('Risk-free rate?', value=False)
        if riskyes:
            risk_free_rate = st.number_input('Risk-free rate', value=0.01, step=0.01)
        risk_aversion = float(st.select_slider('How much risk you want to take?', options=[f"{i:.2f}" for i in np.linspace(0, 1, 10+1)]))
        
        if st.button("Generate"):
            if optimization_method == 'Mean variance':
                if riskyes:
                    fig1, sharpe_stat, mu_stat, vol_stat, weights_mv = plot_efficient_frontier_with_risky(risk_free_rate, shortyes,risk_aversion)
                else:
                    fig1, sharpe_stat, mu_stat, vol_stat, weights_mv = plot_efficient_frontier_without_risky(shortyes,risk_aversion)
                
                fig2 = create_pie_chart(weights_mv, title="Portfolio Holdings Distribution")
                
                st.session_state["fig1"] = fig1
                st.session_state["fig2"] = fig2
                st.session_state["sharpe_stat"] = np.round(sharpe_stat,2)
                st.session_state["mu_stat"] = f'{np.round(mu_stat * 100,2)}%'
                st.session_state["vol_stat"] = f'{np.round(vol_stat * 100,2)}%'

    elif optimization_method == 'Equal Risk Contribution':
        if st.button("Generate"):
            weights_erc, mu_stat, vol_stat, sharpe_stat = ERC(markets, sectors)
            fig1 = st.session_state.erc_plot
            fig2 = create_pie_chart(weights_erc, title="Portfolio Holdings Distribution")
            
            # Save results to session state
            st.session_state["fig1"] = fig1
            st.session_state["fig2"] = fig2
            st.session_state["mu_stat"] = f'{np.round(mu_stat * 100, 2)}%'
            st.session_state["vol_stat"] = f'{np.round(vol_stat * 100, 2)}%'
            st.session_state["sharpe_stat"] = f'{np.round(sharpe_stat, 2)}'

    elif optimization_method == 'Most Diversified':
        if st.button("Generate"):
            weights_mdp, mu_stat, vol_stat, sharpe_stat = MDP(markets, sectors)
            fig1 = st.session_state.mdp_plot
            fig2 = create_pie_chart(weights_mdp, title="Portfolio Holdings Distribution")
            
            # Save results to session state
            st.session_state["fig1"] = fig1
            st.session_state["fig2"] = fig2
            st.session_state["mu_stat"] = f'{np.round(mu_stat * 100, 2)}%'
            st.session_state["vol_stat"] = f'{np.round(vol_stat * 100, 2)}%'
            st.session_state["sharpe_stat"] = f'{np.round(sharpe_stat, 2)}'
    
    elif optimization_method == 'Equally weighted':
        if st.button("Generate"):
            weights_eq, mu_stat, vol_stat, sharpe_stat = EW(markets, sectors)
            fig1 = st.session_state.eq_plot
            fig2 = create_pie_chart(weights_eq, title="Portfolio Holdings Distribution")
            
            # Save results to session state
            st.session_state["fig1"] = fig1
            st.session_state["fig2"] = fig2
            st.session_state["mu_stat"] = f'{np.round(mu_stat * 100, 2)}%'
            st.session_state["vol_stat"] = f'{np.round(vol_stat * 100, 2)}%'
            st.session_state["sharpe_stat"] = f'{np.round(sharpe_stat, 2)}'
                

####################################### MAIN #######################################
st.title("Portfolio Summary")

if "fig1" in st.session_state and "fig2" in st.session_state:
    # Top Statistics with Restored Backgrounds and Spacing
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="container statistic-container">
                <div class="stat-title">Return</div>
                <div class="stat-value">{st.session_state.get("mu_stat", "0.00%")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="container statistic-container">
                <div class="stat-title">Volatility</div>
                <div class="stat-value">{st.session_state.get("vol_stat", "0.00%")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="container statistic-container">
                <div class="stat-title">Sharpe Ratio</div>
                <div class="stat-value">{st.session_state.get("sharpe_stat", "0.00%")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="container statistic-container">
                <div class="stat-title">Max Drawdown</div>
                <div class="stat-value">0.00%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Bottom Plots with Proper Spacing and Titles in Top-Left
    col5, col6 = st.columns(2)
    with col5:  # Efficient Frontier Plot
        st.plotly_chart(st.session_state["fig1"], use_container_width=True)

    with col6:  # Portfolio Holdings Distribution
        st.plotly_chart(st.session_state["fig2"], use_container_width=True)
else:
    st.info("Press 'Generate' to display the portfolio summary and plots.")