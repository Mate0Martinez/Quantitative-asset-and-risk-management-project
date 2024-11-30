import streamlit as st #ignore the warning, if streamlit is installed correctly it will work
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.express as px
import Portfolio_classes as pc
from apscheduler.schedulers.background import BackgroundScheduler
import multiprocessing
import time
from TradeBot import bot_class as tb

######################################## STATE MANAGEMENT ########################################
if 'BL' not in st.session_state:
    st.session_state.BL = None
bot_process = None
if 'bot_active' not in st.session_state or 'bot_active' == None:
    bot_process = None
    st.session_state.bot_active = None
else:
    bot_process = st.session_state.bot_active


####################################### CUSTOM CSS #######################################
# Set page configuration
st.set_page_config(layout="wide", page_title="Lunae Capital", page_icon="Data/logo_rond.png")
st.logo("Data/logo.png",icon_image="Data/logo_rond.png")
st.html("""
  <style>
    [alt=Logo] {
      height: 9rem;
    }
  </style>
        """)
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
    sharpe_MV, mu_MV, vol_MV, weights_mv = efficient_frontier.metrics_pf(gamma=risk_aversion)

    weights_mv = pd.DataFrame(weights_mv, index=prices.columns, columns=['Holding (%)'])
   
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
        x=[vol_MV],
        y=[mu_MV],
        mode='markers',
        marker=dict(color='green', size=10, symbol='star'),
        name='Your Portfolio'
    ))

    fig.update_layout(
        title_text=f"Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Return",
        xaxis_tickformat=".2%",
        yaxis_tickformat=".2%",
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
    sharpe_pf, mu_pf, vol_pf, weights_pf = efficient_frontier.metrics_pf(gamma=risk_aversion)

    # Prepare data for plotly
    x_frontier = [f[1] for f in frontier[0]]  # Volatility of Efficient Frontier
    y_frontier = [f[0] for f in frontier[0]]  # Return of Efficient Frontier

    x_cml = [f[1] for f in frontier[1]]  # Volatility of Capital Market Line
    y_cml = [f[0] for f in frontier[1]]  # Return of Capital Market Line


    weights_pf = pd.DataFrame(weights_pf, index=np.append(prices.columns.values, 'Risk Free Rate'), columns=['Holding (%)'])
    #weights_pf = weights_pf.merge(sectors_data, left_index=True, right_index=True)[['TRBC BUSI SEC NAME', 'Holding (%)']].rename(columns={'TRBC BUSI SEC NAME': 'Sector'})

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
        x=[vol_pf],
        y=[mu_pf],
        mode='markers',
        marker=dict(color='green', size=10, symbol='star'),
        name='Your Portfolio'
    ))

    fig.update_layout(
        title_text=f"Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Return",
        xaxis_tickformat=".2%",
        yaxis_tickformat=".2%",
        xaxis_range=[-0.01, frontier[4]+0.1],
        yaxis_range=[-0.1, frontier[3]+0.1],
        paper_bgcolor="#262730",
        plot_bgcolor="#262730",
        font=dict(color="white"),
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h")  # Legend at bottom
    )

    return fig, sharpe_pf, mu_pf, vol_pf, weights_pf

def ERC(markets, sectors):
    # Unpack the tuple returned by load_data
    prices, sectors_data = load_data(markets, sectors)
    
    # Pass the `prices` DataFrame to the Portfolio class
    portfolio = pc.Portfolio(prices)
    weights_erc = portfolio.ERC()
    RRC = portfolio.RRC(weights_erc,portfolio.covmat)*100
    ARC = portfolio.ARC(weights_erc,portfolio.covmat)*100
    #put the RRC and ARC in one dataframe
    RC = pd.DataFrame([RRC,ARC],index=['Relative Risk Contribution','Absolute Risk Contribution'],columns=prices.columns)
    RC = RC.applymap(lambda x: round(x,2))
    RC = RC.applymap(lambda x: str(x)+'%')


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
    max_drawdown = pc.max_drawdown(perf)
    # Create a DataFrame to display the weights
    weights_erc = pd.DataFrame(weights_erc, index=prices.columns, columns=["Holding (%)"])
    
    # Store results in session_state for reuse
    st.session_state.erc_results = {
        'weights_erc': weights_erc,
        'return_erc': prtfl_return,
        'vol_erc': prtfl_vol,
        'sharpe_erc': sharpe_ratio,
        'perf': perf,
        'max_drawdown': max_drawdown,
        'RC': RC
    }

    return weights_erc, prtfl_return, prtfl_vol, sharpe_ratio, max_drawdown, RC

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
    max_drawdown = pc.max_drawdown(perf)
    # Create a DataFrame to display the weights
    weights_mdp = pd.DataFrame(weights_mdp, index=prices.columns, columns=["Holding (%)"])

    # Store results in session_state for reuse
    st.session_state.mdp_results = {
        'weights_mdp': weights_mdp,
        'return_mdp': prtfl_return,
        'vol_mdp': prtfl_vol,
        'sharpe_mdp': sharpe_ratio,
        'perf': perf,
        'max_drawdown': max_drawdown
    }

    return weights_mdp, prtfl_return, prtfl_vol, sharpe_ratio, max_drawdown

def BL(markets,sectors,risk_free_rate=None):
    prices, sectors_data = load_data(markets, sectors)
    if st.session_state.BL is None:
        st.write('No views added, if you want to see a regular mean variance portfolio go to the mean variance portfolio page.')
       
    else:
        black_litterman = pc.BlackLitterman(prices,risk_free_rate=risk_free_rate)
        P = np.zeros((len(st.session_state.BL),len(prices.columns)))#+1 for the risk free asset
        Q = np.zeros(len(st.session_state.BL))
        omega = np.zeros((len(st.session_state.BL),len(st.session_state.BL)))
        for i in range(len(st.session_state.BL)):
            P[i,prices.columns.get_loc(st.session_state.BL['Asset'].iloc[i])] = 1
            Q[i] = 0.1 if st.session_state.BL['View'].iloc[i] == 'Bullish' else -0.1
            omega[i, i] = 0.01 if st.session_state.BL['Confidence level'].iloc[i] == 'Certain' else 0.05 if st.session_state.BL['Confidence level'].iloc[i] == 'Moderate' else 0.1
        black_litterman.add_views(P, Q, omega)
        opt_tau = black_litterman.optimal_tau()
        weights_bl = black_litterman.BL(tau=0.1)


    perf = pc.get_performance(prices,weights_bl)
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
        title="Portfolio Performance (Black Litterman)",
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

    st.session_state.bl_plot = fig  # Save the plot to session state

    #calculate max drawdown
    max_drawdown = pc.max_drawdown(perf)
    #Metrics
    prtfl_return = np.dot(black_litterman.mu, weights_bl)
    prtfl_vol = np.sqrt(weights_bl @ black_litterman.covmat @ weights_bl)
    risk_free_rate = black_litterman.risk_free_rate if black_litterman.risk_free_rate is not None else 0
    sharpe_ratio = (prtfl_return - risk_free_rate) / prtfl_vol

    weights_bl = pd.DataFrame(weights_bl, index=prices.columns, columns=["Holding (%)"])
    # Store results in session_state for reuse
    st.session_state.bl_results = {
        'weights_bl': weights_bl,
        'return_bl': prtfl_return,
        'vol_bl': prtfl_vol,
        'sharpe_bl': sharpe_ratio,
        'perf': perf,
        'max_drawdown': max_drawdown
    }

    return weights_bl, prtfl_return, prtfl_vol, sharpe_ratio, max_drawdown

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
    max_drawdown = pc.max_drawdown(perf)
    # Create a DataFrame to display the weights
    weights_eq = pd.DataFrame(weights_eq, index=prices.columns, columns=["Holding (%)"])

    # Store results in session_state for reuse
    st.session_state.eq_results = {
        'weights_eq': weights_eq,
        'return_eq': prtfl_return,
        'vol_eq': prtfl_vol,
        'sharpe_eq': sharpe_ratio,
        'perf': perf,
        'max_drawdown': max_drawdown
    }

    return weights_eq, prtfl_return, prtfl_vol, sharpe_ratio, max_drawdown



def create_pie_chart(df, title="Portfolio Holdings Distribution"):
    # Filter out rows with zero holdings
    df_filtered = df[df["Holding (%)"] > 0.00001]

    # Use the DataFrame's index as labels if there is no "Company" column
    labels = df_filtered.index if "Company" not in df_filtered.columns else df_filtered["Company"]

    # Create a pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=df_filtered["Holding (%)"],
                hoverinfo="label+percent",
                textinfo="percent" if len(labels) < 20 else "none",
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

def create_bar_chart(df, title="Portfolio Holdings Distribution"):
    df = df[abs(df["Holding (%)"]) > 0.001]
    # Use the DataFrame's index as labels if there is no "Company" column
    labels = df.index if "Company" not in df.columns else df["Company"]
    colors = ['forestgreen' if percent > 0 else 'firebrick' for percent in df["Holding (%)"]]

    # Create a pie chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=df["Holding (%)"],
                hoverinfo="x + y",
                marker=dict(color=colors)
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
        xaxis=dict(
            tickvals=labels,
            ticktext=[f"{label}: {percent*100:.2f}%" for label, percent in zip(labels, df["Holding (%)"])]
        ),

    )

    return fig
####################################### SIDEBAR #######################################
with st.sidebar:
    optimization_method = st.selectbox(
    'Select portfolio', 
    ['Select an option', 'Mean variance', 'Equal Risk Contribution', 'Most Diversified', 'Black Litterman', 'Equally weighted']
)

    if (optimization_method == 'Select an option'):
        # presenting our website
        st.markdown("""
    <div style="text-align: justify;">
        Welcome to our Portfolio Optimization Dashboard! This platform was developed by a team of five students at HEC Lausanne in the MScF program. It allows you to optimize your portfolio using different methods. You can select the method you want to use in the sidebar. You can also select the markets and sectors you want to include in your portfolio.
    </div>
    """,
    unsafe_allow_html=True)
        
        # Add names and LinkedIn hyperlinks at the bottom of the sidebar
        st.markdown("""
            <div style="text-align: center; margin-top: 50px;">
                <p style="font-size: 18px; font-weight: bold;">Developed by:</p>
                <p style="font-size: 16px;">
                    <a href="https://www.linkedin.com/in/mateo-martinez-428224257/" target="_blank" style="text-decoration: none; color: #0072b1;">Mateo Martinez</a><br>
                    <a href="https://www.linkedin.com/in/dorentin-morina/" target="_blank" style="text-decoration: none; color: #0072b1;">Dorentin Morina</a><br>
                    <a href="https://www.linkedin.com/in/shpetim-tafili-b38149264/" target="_blank" style="text-decoration: none; color: #0072b1;">Shpetim Tafili</a><br>
                    <a href="https://www.linkedin.com/in/wzed/" target="_blank" style="text-decoration: none; color: #0072b1;">Wassim Zeddoug</a><br>
                    <a href="https://www.linkedin.com/in/jeremy-bourqui-27b006273/" target="_blank" style="text-decoration: none; color: #0072b1;">Jeremy Bourqui</a>
                </p>
            </div>
            """, unsafe_allow_html=True)

    if (optimization_method != 'Select an option'):
        container = st.container()
        all_checked = st.checkbox("Select all")
 
        if all_checked:
            markets = st.multiselect('Select markets', ['US', 'EU', 'EM'], ['US', 'EU', 'EM'])
            sectors = st.multiselect('Select sectors', ['Holding Companies', 'Utilities', 'Industrial & Commercial Services', 'Banking & Investment Services', 'Healthcare Services & Equipment',
                                                    'Chemicals', 'Consumer Goods Conglomerates', 'Technology Equipment', 'Software & IT Services', 'Real Estate','Energy - Fossil Fuels',
                                                    'Industrial Goods', 'Applied Resources', 'Mineral Resources', 'Cyclical Consumer Products', 'Transportation', 'Retailers'], ['Holding Companies', 'Utilities', 'Industrial & Commercial Services', 'Banking & Investment Services', 'Healthcare Services & Equipment',
                                                    'Chemicals', 'Consumer Goods Conglomerates', 'Technology Equipment', 'Software & IT Services', 'Real Estate','Energy - Fossil Fuels',
                                                    'Industrial Goods', 'Applied Resources', 'Mineral Resources', 'Cyclical Consumer Products', 'Transportation', 'Retailers'])
    
        else:
            markets = st.multiselect('Select markets', ['US', 'EU', 'EM'])
            sectors = st.multiselect('Select sectors', ['Holding Companies', 'Utilities', 'Industrial & Commercial Services', 'Banking & Investment Services', 'Healthcare Services & Equipment',
                                                    'Chemicals', 'Consumer Goods Conglomerates', 'Technology Equipment', 'Software & IT Services', 'Real Estate','Energy - Fossil Fuels',
                                                    'Industrial Goods', 'Applied Resources', 'Mineral Resources', 'Cyclical Consumer Products', 'Transportation', 'Retailers'])
    
       
        

    if optimization_method == 'Mean variance':
        shortyes = st.checkbox('Short-selling?', value=False)
        if shortyes:
            st.markdown(
            """
            <div style="color: red; font-size: 12px; margin-top: -10px;">
                Warning: allowing short selling makes your investment more aggressive.
            </div>
            """,
            unsafe_allow_html=True
        )
        riskyes = st.checkbox('Risk-free rate?', value=False)
        if riskyes:
            risk_free_rate = st.number_input('Risk-free rate (in %)', value=1.00, step=0.01)
            risk_free_rate = risk_free_rate / 100
            st.write("You can use for example the [10 year US Treasury rate](https://www.cnbc.com/quotes/US10Y) as a risk-free rate.")
        risk_aversion = st.select_slider('How much risk you want to take?', options=["Minimum Risk", "Conservative", "Balanced", "Aggressive"])
        risk_aversion = 0 if risk_aversion == "Minimum Risk" else 0.1 if risk_aversion == "Conservative" else 0.2 if risk_aversion == "Balanced" else 0.4
        if st.button("Generate"):
            if optimization_method == 'Mean variance':
                if riskyes:
                    fig1, sharpe_stat, mu_stat, vol_stat, weights_mv = plot_efficient_frontier_with_risky(risk_free_rate, shortyes,risk_aversion)
                else:
                    fig1, sharpe_stat, mu_stat, vol_stat, weights_mv = plot_efficient_frontier_without_risky(shortyes,risk_aversion)
                
                fig2 = create_pie_chart(weights_mv, title="Portfolio Holdings Distribution")
                fig3 = create_bar_chart(weights_mv, title="Portfolio Holdings Distribution")
                st.session_state["fig1"] = fig1
                st.session_state["fig2"] = fig2
                st.session_state["fig3"] = fig3
                st.session_state["sharpe_stat"] = np.round(sharpe_stat,2)
                st.session_state["mu_stat"] = f'{np.round(mu_stat * 100,2)}%'
                st.session_state["vol_stat"] = f'{np.round(vol_stat * 100,2)}%'

    elif optimization_method == 'Equal Risk Contribution':
        if st.button("Generate"):
            weights_erc, mu_stat, vol_stat, sharpe_stat, max_drawdown,RC = ERC(markets, sectors)

            fig1 = st.session_state.erc_plot
            fig2 = create_pie_chart(weights_erc, title="Portfolio Holdings Distribution")
            fig3 = create_bar_chart(weights_erc, title="Portfolio Holdings Distribution")
            # Save results to session state
            st.session_state["fig1"] = fig1
            st.session_state["fig2"] = fig2
            st.session_state["fig3"] = fig3
            st.session_state["mu_stat"] = f'{np.round(mu_stat * 100, 2)}%'
            st.session_state["vol_stat"] = f'{np.round(vol_stat * 100, 2)}%'
            st.session_state["sharpe_stat"] = f'{np.round(sharpe_stat, 2)}'
            st.session_state["max_drawdown"] = f'{np.round(max_drawdown*100, 2)}%'
            st.session_state["RC"] = RC



    elif optimization_method == 'Most Diversified':
        if st.button("Generate"):
            weights_mdp, mu_stat, vol_stat, sharpe_stat,max_drawdown = MDP(markets, sectors)
            fig1 = st.session_state.mdp_plot
            fig2 = create_pie_chart(weights_mdp, title="Portfolio Holdings Distribution")
            fig3 = create_bar_chart(weights_mdp, title="Portfolio Holdings Distribution")
            # Save results to session state
            st.session_state["fig1"] = fig1
            st.session_state["fig2"] = fig2
            st.session_state["fig3"] = fig3
            st.session_state["mu_stat"] = f'{np.round(mu_stat * 100, 2)}%'
            st.session_state["vol_stat"] = f'{np.round(vol_stat * 100, 2)}%'
            st.session_state["sharpe_stat"] = f'{np.round(sharpe_stat, 2)}'
            st.session_state["max_drawdown"] = f'{np.round(max_drawdown*100, 2)}%'
    
    elif optimization_method == 'Equally weighted':
        if st.button("Generate"):
            weights_eq, mu_stat, vol_stat, sharpe_stat,max_drawdown = EW(markets, sectors)
            fig1 = st.session_state.eq_plot
            fig2 = create_pie_chart(weights_eq, title="Portfolio Holdings Distribution")
            fig3 = create_bar_chart(weights_eq, title="Portfolio Holdings Distribution")
            # Save results to session state
            st.session_state["fig1"] = fig1
            st.session_state["fig2"] = fig2
            st.session_state["fig3"] = fig3
            st.session_state["mu_stat"] = f'{np.round(mu_stat * 100, 2)}%'
            st.session_state["vol_stat"] = f'{np.round(vol_stat * 100, 2)}%'
            st.session_state["sharpe_stat"] = f'{np.round(sharpe_stat, 2)}'
            st.session_state["max_drawdown"] = f'{np.round(max_drawdown*100, 2)}%'
                
    elif optimization_method == 'Black Litterman':
        #need to load the assets to be able to select them
        if (markets == []) or (sectors == []):
            price, sectors = load_data(['US', 'EU', 'EM'], ['Holding Companies', 'Utilities', 'Industrial & Commercial Services', 'Banking & Investment Services', 'Healthcare Services & Equipment',
                                                'Chemicals', 'Consumer Goods Conglomerates', 'Technology Equipment', 'Software & IT Services', 'Real Estate','Energy - Fossil Fuels',
                                                'Industrial Goods', 'Applied Resources', 'Mineral Resources', 'Cyclical Consumer Products', 'Transportation', 'Retailers'])
            name_assets = price.columns
        else:
            price, sectors_data = load_data(markets, sectors)
            name_assets = price.columns
        select_asset = st.selectbox('Select option', (name_assets))
        views = st.selectbox('Views', ['Bullish', 'Bearish'])
        uncertainty = st.selectbox('How confident are you of your view?', ['Certain', 'Moderate', 'Uncertain'])
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('Add view'):
                if st.session_state.BL is None:
                    st.session_state.BL = pd.DataFrame(columns=['Asset', 'View'])
                    st.session_state.BL = pd.DataFrame({'Asset': [select_asset], 'View': [views], 'Confidence level': [uncertainty]})
                else:
                    st.session_state.BL = pd.concat([st.session_state.BL, pd.DataFrame({'Asset': [select_asset], 'View': [views], 'Confidence level': [uncertainty]})])
                st.write('View added')
        with col2:
            if st.button('Delete view'):
                st.session_state.BL = st.session_state.BL[st.session_state.BL['Asset'] != select_asset]
                st.write('View deleted')
        with col3:
            if st.button('Delete all views'):
                st.session_state.BL = None
                st.write('Views deleted')
        st.write(st.session_state.BL)
        if st.button("Generate"):
            if st.session_state.BL is None:
                st.warning('No views added, if you want to see a normal mean variance portfolio go to the mean variance portfolio page')
            
            else:
                if not all(st.session_state.BL['Asset'].isin(price.columns)):
                    st.warning('Some assets in the views are not in the selected markets and sectors.')
                else:
                    weights_bl, mu_stat, vol_stat, sharpe_stat,max_drawdown = BL(markets, sectors, risk_free_rate=0.03)
                    fig1 = st.session_state.bl_plot
                    fig2 = create_pie_chart(weights_bl, title="Portfolio Holdings Distribution")
                    fig3 = create_bar_chart(weights_bl, title="Portfolio Holdings Distribution")
                    # Save results to session state
                    st.session_state["fig1"] = fig1
                    st.session_state["fig2"] = fig2
                    st.session_state["fig3"] = fig3
                    st.session_state["mu_stat"] = f'{np.round(mu_stat * 100, 2)}%'
                    st.session_state["vol_stat"] = f'{np.round(vol_stat * 100, 2)}%'
                    st.session_state["sharpe_stat"] = f'{np.round(sharpe_stat, 2)}'
                    st.session_state["max_drawdown"] = f'{np.round(max_drawdown*100, 2)}%'

####################################### MAIN #######################################
st.title("Portfolio Summary")

if "fig1" in st.session_state and "fig2" in st.session_state and "fig3" in st.session_state:
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
        if optimization_method == "Mean variance":

            st.markdown(
                f"""
                <div class="container statistic-container">
                    <div class="stat-title"></div>
                    <div class="stat-value">Mean variance</div>
                </div>
                """,   
                unsafe_allow_html=True,
            )
        else:
            
            st.markdown(
                f"""
                <div class="container statistic-container">
                    <div class="stat-title">Max Drawdown</div>
                    <div class="stat-value">{st.session_state.get("max_drawdown", "0.00%")}</div>
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
    
    st.plotly_chart(st.session_state["fig3"], use_container_width=True)
    if optimization_method == 'Equal Risk Contribution':
        st.write(st.session_state["RC"])
else:
    st.info("Press 'Generate' to display the portfolio summary and plots.")



################################# BOT #################################

# Function to start the bot
def start_bot():
    bot_process = st.session_state.bot_active
    if bot_process is None or not bot_process.is_alive():
        bot_process = multiprocessing.Process(target=run_bot)
        
        bot_process.start()
        st.success("Trading bot started successfully.")
        st.session_state.bot_active = bot_process
    else:
        st.warning("Trading bot is already running.")
       
# Function to stop the bot
def stop_bot():
    bot_process = st.session_state.bot_active
    if bot_process and bot_process.is_alive():
        bot_process.terminate()
        bot_process.join()
        bot_process = None
        st.success("Trading bot stopped successfully.")
        st.session_state.bot_active = None
       
    else:
        st.warning("No trading bot is currently running.")
     

# Function to run the bot
def run_bot():
    scheduler = BackgroundScheduler()
    bot = tb.TradingBot()  # Initialize the bot
    scheduler.add_job(
    bot.trading_MAV,
    'cron',
    day_of_week='mon-fri',
    hour='0-23',
    minute='*', # * is every minute, can change it easily to 15 for example 1,16,31,46
    #start_date='2024-01-12 12:00:00',
    #europe paris timezone
    timezone='Europe/Paris'
    )
    scheduler.start()  # Start the scheduler

    scheduler.print_jobs()

    # Keep the scheduler running
    try:
        while True:
            time.sleep(1)         
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()

st.title("Trading Bot Control Panel")

# Start button
if st.button("Start Trading Bot"):
    start_bot()

# Stop button
if st.button("Stop Trading Bot"):
    stop_bot()

# Bot status

if bot_process and bot_process.is_alive():
    st.info("Trading bot is currently running.")
else:
    st.info("Trading bot is not running.")

with open('trade_log_MAV.csv', 'r') as f:
            lines = f.readlines()
            if len(lines) > 0:
                #put the first and last line in the dataframe
                df = pd.DataFrame([x.split(',') for x in lines[::len(lines)-1]], columns=lines[0].split(','))
                #df drop first row
                df = df.drop(0)
                st.write(df)




########################################################################

st.markdown(
    """
    <style>
    /* Add some space between the plots and the footer */
    .footer {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 15%;
        bottom: 0;
        width: 100%;
        background-color: #100E22;
        color: #fff;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    </style>
    <div class="footer">
        The information provided by our website is for educational purposes only and does not constitute financial advice or investment recommendations.
    </div>
""", unsafe_allow_html=True)
