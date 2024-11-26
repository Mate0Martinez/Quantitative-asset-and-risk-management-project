'''
don't forget to install streamlit using pip install streamlit inside the terminal
to check if it is installed use streamlit --version in the terminal
to run the code use streamlit run Streamlit.py in the terminal
if you get an error about urllib3, use:
pip uninstall urllib3
pip install urllib3==1.26.7

unfortunately, Streamlit recognizes every comment written with commas as a string doc and adds it to the page
so the comments are gonna have to be written with # instead of commas for the rest of the code

the way streamlit works is that each action reruns the whole script.
this is why the functions are declared outside of what is rendered on the page in the main part
'''
import streamlit as st #ignore the warning, if streamlit is installed correctly it will work
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import Portfolio_classes as pc
import threading
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_jobstore('memory')  # Ensure jobs are stored in memory

#STATE INITIALIZATION
#state is used to store the data that is gonna be used in the app
#this is done to keep some sensitive data during changes in the page
if 'BL' not in st.session_state:
    st.session_state.BL = None

if 'bot_active' not in st.session_state:
    st.session_state.bot_active = False

#DECLARATION OF FUNCTIONS
#functions used when a button is clicked
#the idea is that when a button is clicked the script will run the function this allows to run the function only when the button is clicked
#else the whole thing runs when the page is loaded
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

    if len(data) == 0:
        st.warning('No data available for the selected markets and sectors.')
    else:
        return data

   
def plot_efficient_frontier_without_risky(shortyes,risk_aversion):
    prices = load_data(markets, sectors)
    efficient_frontier = pc.EfficientFrontier(prices, short = shortyes)
    frontier = efficient_frontier.get_efficient_frontier()
    gammas = np.linspace(-5, 5, 500)
    gamma_zero_index = np.argmin(np.abs(gammas))
    gamma_index = np.argmin(np.abs(gammas - risk_aversion)) #à verifier
    sharpe_MV, mu_MV, vol_MV = efficient_frontier.metrics_MV(gamma=risk_aversion)
    fig, ax = plt.subplots()
    ax.plot([f[1] for f in frontier], [f[0] for f in frontier], '-', label="Efficient Frontier")
    ax.plot(frontier[gamma_zero_index][1], frontier[gamma_zero_index][0], color='r', marker='D', label='Minimum Variance Portfolio')
    ax.plot(frontier[gamma_index][1], frontier[gamma_index][0], color='g', marker='*', label='Your portfolio', markersize=10)
    ax.set_title(f'Efficient Frontier - Short-selling: {shortyes}')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.legend()
    if shortyes is False:
        ax.set_ylim(-0.5, 1)
        ax.set_xlim(0, 0.8)
    else:
        ax.set_ylim(-4, 9)
        ax.set_xlim(0, 2.25)

    # Add text annotations for Sharpe, mu, and vol
    ax.text(0.6, 0.7, 
            f"Sharpe: {sharpe_MV:.2f}\nμ: {mu_MV:.2%}\nσ: {vol_MV:.2%}", 
            fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    st.pyplot(fig)
   

def plot_efficient_frontier_with_risky(risk_free_rate, shortyes,risk_aversion):
    prices = load_data(markets, sectors)
    efficient_frontier = pc.EfficientFrontier(prices, risk_free_rate=risk_free_rate,short=shortyes)
    frontier = efficient_frontier.get_efficient_frontier()
    fig, ax = plt.subplots()
    gammas = np.linspace(-5, 5, 500)
    gamma_zero_index = np.argmin(np.abs(gammas))
    gamma_index = np.argmin(np.abs(gammas - risk_aversion))
    sharpe_MV, mu_MV, vol_MV = efficient_frontier.metrics_MV(gamma=risk_aversion)
    ax.plot([f[1] for f in frontier[0]], [f[0] for f in frontier[0]], '-', label="Efficient Frontier")
    ax.plot(frontier[0][gamma_zero_index][1], frontier[0][gamma_zero_index][0], color='r', marker='D', label='Minimum Variance Portfolio')
    ax.plot([f[1] for f in frontier[1]], [f[0] for f in frontier[1]], '--', label="Capital Market Line")
    ax.plot(frontier[0][gamma_index][1], frontier[0][gamma_index][0], color='g', marker='*', label='Your portfolio', markersize=10)
    ax.plot(frontier[4], frontier[3], color='orange', marker='*', label='Tangency Portfolio', markersize=10)
    ax.set_title(f'Efficient Frontier - Short-selling: {shortyes} - Risk-free rate: {risk_free_rate}')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.legend()
    if shortyes is False:
        ax.set_ylim(-0.5, 1)
        ax.set_xlim(0, 0.8)
    else:
        ax.set_ylim(-4, 9)
        ax.set_xlim(0, 2.25)

    # Add text annotations for Sharpe, mu, and vol
    ax.text(0.6,0.7, # changer la position
            f"Sharpe: {sharpe_MV:.2f}\nμ: {mu_MV:.2%}\nσ: {vol_MV:.2%}", 
            fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    st.pyplot(fig)

def ERC(markets, sectors):
    prices = load_data(markets, sectors)
    portfolio = pc.Portfolio(prices)
    weights_erc = portfolio.ERC()
    perf = pc.get_performance(prices,weights_erc)
    perf.index = pd.to_datetime(perf.index)
    fig, ax = plt.subplots()
    ax.plot(perf)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    

    st.session_state.erc_plot = fig  # Save the plot to session state

    # Metrics
    prtfl_return = np.dot(portfolio.mu, weights_erc)
    prtfl_vol = np.sqrt(weights_erc @ portfolio.covmat @ weights_erc)
    risk_free_rate = portfolio.risk_free_rate if portfolio.risk_free_rate is not None else 0
    sharpe_ratio = (prtfl_return - risk_free_rate) / prtfl_vol

    # Store results in session_state for reuse
    st.session_state.erc_results = {
        'weights_erc': weights_erc,
        'return_erc': prtfl_return,
        'vol_erc': prtfl_vol,
        'sharpe_erc': sharpe_ratio,
        'perf': perf
    }

    return weights_erc, prtfl_return, prtfl_vol, sharpe_ratio

def MDP(markets, sectors):
    prices = load_data(markets, sectors)
    portfolio = pc.Portfolio(prices)
    weights_mdp = portfolio.MDP()
    perf = pc.get_performance(prices,weights_mdp)
    perf.index = pd.to_datetime(perf.index)
    fig, ax = plt.subplots()
    ax.plot(perf)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    st.session_state.mdp_plot = fig  # Save the plot to session state

    # Metrics
    prtfl_return = np.dot(portfolio.mu, weights_mdp)
    prtfl_vol = np.sqrt(weights_mdp @ portfolio.covmat @ weights_mdp)
    risk_free_rate = portfolio.risk_free_rate if portfolio.risk_free_rate is not None else 0
    sharpe_ratio = (prtfl_return - risk_free_rate) / prtfl_vol

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
    prices = load_data(markets, sectors)
    portfolio = pc.Portfolio(prices)
    weights_eq = portfolio.EW()
    perf = pc.get_performance(prices,weights_eq)
    perf.index = pd.to_datetime(perf.index)
    fig, ax = plt.subplots()
    ax.plot(perf)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    st.session_state.eq_plot = fig  # Save the plot to session state

    #Metrics
    prtfl_return = np.dot(portfolio.mu, weights_eq)
    prtfl_vol = np.sqrt(weights_eq @ portfolio.covmat @ weights_eq)
    risk_free_rate = portfolio.risk_free_rate if portfolio.risk_free_rate is not None else 0
    sharpe_ratio = (prtfl_return - risk_free_rate) / prtfl_vol

    # Store results in session_state for reuse
    st.session_state.eq_results = {
        'weights_eq': weights_eq,
        'return_eq': prtfl_return,
        'vol_eq': prtfl_vol,
        'sharpe_eq': sharpe_ratio,
        'perf': perf
    }

    return weights_eq, prtfl_return, prtfl_vol, sharpe_ratio

def BL(markets,sectors,risk_free_rate=None):
    prices = load_data(markets, sectors)
    if st.session_state.BL is None:
        st.write('No views added, if you want to see a normal mean variance portfolio go to the mean variance portfolio page')
        #black_litterman = pc.BlackLitterman(prices,risk_free_rate=risk_free_rate)
        #mv = pc.EfficientFrontier(prices,risk_free_rate=risk_free_rate,short=True)
        #st.write(1/black_litterman.implied_phi)
        #weights_bl = mv.efficient_frontier(mv.n+1,mv.x0_mod,mv.covmat_mod,mv.mu_mod,1/black_litterman.implied_phi)[2]
    else:
        black_litterman = pc.BlackLitterman(prices,risk_free_rate=risk_free_rate)
        P = np.zeros((len(st.session_state.BL),len(prices.columns)+1))#+1 for the risk free asset
        Q = np.zeros(len(st.session_state.BL))
        omega = np.zeros((len(st.session_state.BL),len(st.session_state.BL)))
        for i in range(len(st.session_state.BL)):
            P[i,prices.columns.get_loc(st.session_state.BL['Asset'].iloc[i])] = 1
            Q[i] = 0.1 if st.session_state.BL['View'].iloc[i] == 'Bullish' else -0.1
            omega[i, i] = 0.01 if st.session_state.BL['Confidence level'].iloc[i] == 'Certain' else 0.05 if st.session_state.BL['Confidence level'].iloc[i] == 'Moderate' else 0.1
        black_litterman.add_views(P, Q, omega)
        opt_tau = black_litterman.optimal_tau()
        weights_bl = black_litterman.BL(tau=opt_tau)
        #add the risk free asset to the prices
    prices['Risk Free Asset'] = risk_free_rate


    perf = pc.get_performance(prices,weights_bl)
    perf.index = pd.to_datetime(perf.index)
    fig, ax = plt.subplots()
    ax.plot(perf)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    st.session_state.bl_plot = fig  # Save the plot to session state

    #Metrics
    if risk_free_rate is not None:
        prtfl_return = np.dot(black_litterman.mu_mod, weights_bl)
        prtfl_vol = np.sqrt(weights_bl @ black_litterman.covmat_mod @ weights_bl)
        risk_free_rate = black_litterman.risk_free_rate if black_litterman.risk_free_rate is not None else 0
        sharpe_ratio = (prtfl_return - risk_free_rate) / prtfl_vol
    else:
        prtfl_return = np.dot(black_litterman.mu, weights_bl)
        prtfl_vol = np.sqrt(weights_bl @ black_litterman.covmat @ weights_bl)
        risk_free_rate = black_litterman.risk_free_rate if black_litterman.risk_free_rate is not None else 0
        sharpe_ratio = (prtfl_return - risk_free_rate) / prtfl_vol

    # Store results in session_state for reuse
    st.session_state.bl_results = {
        'weights_bl': weights_bl,
        'return_bl': prtfl_return,
        'vol_bl': prtfl_vol,
        'sharpe_bl': sharpe_ratio,
        'perf': perf
    }

    return weights_bl, prtfl_return, prtfl_vol, sharpe_ratio


####### Bot Control Functions
def start_hft_bot():
    import logging
    logging.basicConfig(level=logging.INFO)  # Avoid scheduler warnings
    if not scheduler.running:  # Prevent multiple scheduler starts
        scheduler.start()
        bot_thread = threading.Thread(target=scheduler.start)
        bot_thread.daemon = True  # Ensure thread closes when Streamlit stops
        bot_thread.start()

def stop_hft_bot():
    if scheduler.running:  # Safely stop the scheduler
        scheduler.shutdown(wait=False)  # Ensure the scheduler stops immediately
    st.session_state.bot_active = False
    st.sidebar.info("Bot has been stopped.")
#######

#MAIN PART OF THE SITE

# Title of the app
st.title('Quantitative Asset and Risk Management Project')

# Sidebar for user input
st.sidebar.header('Choice of portfolio')

# User input fields
portfolio_choice = st.sidebar.selectbox('Select portfolio', ['Mean variance', 'Equal Risk Contribution', 'Most Diversified', 'Black Litterman','Equally weighted'])

#changes the page in function of which portfolio is selected in the sidebar
if portfolio_choice == 'Mean variance':
    st.write('Mean variance portfolio')
    markets = st.multiselect('Select markets', ['US', 'EU', 'EM'])
    sectors = st.multiselect('Select sectors', ['Holding Companies', 'Utilities', 'Industrial & Commercial Services', 'Banking & Investment Services', 'Healthcare Services & Equipment',
                                                'Chemicals', 'Consumer Goods Conglomerates', 'Technology Equipment', 'Software & IT Services', 'Real Estate','Energy - Fossil Fuels',
                                                'Industrial Goods', 'Applied Resources', 'Mineral Resources', 'Cyclical Consumer Products', 'Transportation', 'Retailers'])
    shortyes = st.checkbox('Short-selling?', value=False)
    riskyes = st.checkbox('Risk-free rate?', value=False)
    if riskyes:
        risk_free_rate = st.number_input('Risk-free rate', value=0.01, step=0.01)
    risk_aversion = float(st.select_slider('How much risk you want to take?', options=[f"{i:.2f}" for i in np.linspace(0, 1, 11)]))
    
    if st.button('Plot Efficient Frontier'):

        if riskyes: #plot the efficient frontier without the risk free asset

            plot_efficient_frontier_with_risky(risk_free_rate, shortyes,risk_aversion)
        else: #plot the efficient frontier with the risk free asset
            plot_efficient_frontier_without_risky(shortyes,risk_aversion)

   
elif portfolio_choice == 'Equal Risk Contribution':
    st.write('Equal Risk Contribution Portfolio')
    markets = st.multiselect('Select markets', ['US', 'EU', 'EM'])
    sectors = st.multiselect('Select sectors', ['Holding Companies', 'Utilities', 'Industrial & Commercial Services', 
                                                'Banking & Investment Services', 'Healthcare Services & Equipment',
                                                'Chemicals', 'Consumer Goods Conglomerates', 'Technology Equipment', 
                                                'Software & IT Services', 'Real Estate', 'Energy - Fossil Fuels',
                                                'Industrial Goods', 'Applied Resources', 'Mineral Resources', 
                                                'Cyclical Consumer Products', 'Transportation', 'Retailers'])

    # Add a button to trigger computation
    if st.button('Compute Equal Risk Contribution Portfolio'):
        if 'erc_results' not in st.session_state or markets != st.session_state.markets or sectors != st.session_state.sectors:
            # If ERC results are not already computed, call ERC to calculate and store the results
            weights_erc, return_erc, vol_erc, sharpe_erc = ERC(markets, sectors)
            st.session_state.sectors = sectors
            st.session_state.markets = markets
        else:
            # Use cached results
            weights_erc = st.session_state.erc_results['weights_erc']
            return_erc = st.session_state.erc_results['return_erc']
            vol_erc = st.session_state.erc_results['vol_erc']
            sharpe_erc = st.session_state.erc_results['sharpe_erc']

        if 'erc_plot' in st.session_state:  # Retain the plot if it exists
            st.pyplot(st.session_state.erc_plot)

        st.write(weights_erc)
        st.write(f'Returns of Portfolio: {return_erc}')
        st.write(f'Volatility of Portfolio: {vol_erc}')
        st.write(f'Sharpe Ratio of Portfolio: {sharpe_erc}')
    else:
        st.write('Click the button above to compute the Equal Risk Contribution portfolio.')
        st.session_state.sectors = None
        st.session_state.markets = None


elif portfolio_choice == 'Most Diversified':
    st.write('Most Diversified Portfolio')
    markets = st.multiselect('Select markets', ['US', 'EU', 'EM'])
    sectors = st.multiselect('Select sectors', ['Holding Companies', 'Utilities', 'Industrial & Commercial Services', 
                                                'Banking & Investment Services', 'Healthcare Services & Equipment',
                                                'Chemicals', 'Consumer Goods Conglomerates', 'Technology Equipment', 
                                                'Software & IT Services', 'Real Estate', 'Energy - Fossil Fuels',
                                                'Industrial Goods', 'Applied Resources', 'Mineral Resources', 
                                                'Cyclical Consumer Products', 'Transportation', 'Retailers'])
    
    # Add a button to trigger computation
    if st.button('Compute Most Diversified Portfolio'):
        if 'mdp_results' not in st.session_state or markets != st.session_state.markets or sectors != st.session_state.sectors:
            # If MDP results are not already computed, call MDP to calculate and store the results
            weights_mdp, return_mdp, vol_mdp, sharpe_mdp = MDP(markets, sectors)
            st.session_state.sectors = sectors
            st.session_state.markets = markets
        else:
            # Use cached results
            weights_mdp = st.session_state.mdp_results['weights_mdp']
            return_mdp = st.session_state.mdp_results['return_mdp']
            vol_mdp = st.session_state.mdp_results['vol_mdp']
            sharpe_mdp = st.session_state.mdp_results['sharpe_mdp']

        if 'mdp_plot' in st.session_state:  # Retain the plot if it exists
            st.pyplot(st.session_state.mdp_plot)

        st.write(weights_mdp)
        st.write(f'Returns of Portfolio: {return_mdp}')
        st.write(f'Volatility of Portfolio: {vol_mdp}')
        st.write(f'Sharpe Ratio of Portfolio: {sharpe_mdp}')
    else:
        st.write('Click the button above to compute the Most Diversified portfolio.')
        st.session_state.sectors = None
        st.session_state.markets = None

elif portfolio_choice == 'Black Litterman':
    st.write('Black Litterman portfolio')
    st.write('Be careful as changing the markets and sectors will reset the views.')
    markets = st.multiselect('Select markets', ['US', 'EU', 'EM'])
    sectors = st.multiselect('Select sectors', ['Holding Companies', 'Utilities', 'Industrial & Commercial Services', 'Banking & Investment Services', 'Healthcare Services & Equipment',
                                                'Chemicals', 'Consumer Goods Conglomerates', 'Technology Equipment', 'Software & IT Services', 'Real Estate','Energy - Fossil Fuels',
                                                'Industrial Goods', 'Applied Resources', 'Mineral Resources', 'Cyclical Consumer Products', 'Transportation', 'Retailers'])
    if (markets == []) or (sectors == []):
        price = load_data(['US', 'EU', 'EM'], ['Holding Companies', 'Utilities', 'Industrial & Commercial Services', 'Banking & Investment Services', 'Healthcare Services & Equipment',
                                                'Chemicals', 'Consumer Goods Conglomerates', 'Technology Equipment', 'Software & IT Services', 'Real Estate','Energy - Fossil Fuels',
                                                'Industrial Goods', 'Applied Resources', 'Mineral Resources', 'Cyclical Consumer Products', 'Transportation', 'Retailers'])
        name_assets = price.columns
    else:
        price = load_data(markets, sectors)
        name_assets = price.columns
    select_asset = st.selectbox('Select option', (name_assets))
    views = st.selectbox('Views', ['Bullish', 'Bearish'])
    uncertainty = st.selectbox('How confident are you of your view?', ['Certain', 'Moderate', 'Uncertain'])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button('Add view'):
            if st.session_state.BL is None:
                st.session_state.BL = pd.DataFrame(columns=['Asset', 'View'])
                st.session_state.BL = pd.DataFrame({'Asset': [select_asset], 'View': [views], 'Confidence level': [uncertainty]})
            else:
                st.session_state.BL = pd.concat([st.session_state.BL, pd.DataFrame({'Asset': [select_asset], 'View': [views], 'Confidence level': [uncertainty]})])
            st.write('View added')
            st.write(st.session_state.BL)
    with col2:
        if st.button('Show views'):
            st.write(st.session_state.BL)
    with col3:
        if st.button('Delete view'):
            st.session_state.BL = st.session_state.BL[st.session_state.BL['Asset'] != select_asset]
            st.write('View deleted')
            st.write(st.session_state.BL)
    with col4:
        if st.button('Delete all views'):
            st.session_state.BL = None
            st.write('Views deleted')
    if st.button('Compute Black Litterman Portfolio'):
        #check if the views are in the markets and sectors selected
        if st.session_state.BL is None:
            st.warning('No views added, if you want to see a normal mean variance portfolio go to the mean variance portfolio page')
            #weights_bl, return_bl, vol_bl, sharpe_mdp = BL(markets, sectors, risk_free_rate=0.03)
            #st.pyplot(st.session_state.bl_plot)
            #st.write(weights_bl)
            #st.write(f'Returns of Portfolio: {return_bl}')
            #st.write(f'Volatility of Portfolio: {vol_bl}')
            #st.write(f'Sharpe Ratio of Portfolio: {sharpe_mdp}')

        else:
            if not all(st.session_state.BL['Asset'].isin(price.columns)):
                st.warning('Some assets in the views are not in the selected markets and sectors.')
            else:
                weights_bl, return_bl, vol_bl, sharpe_mdp = BL(markets, sectors, risk_free_rate=0.03)
                st.pyplot(st.session_state.bl_plot)
                st.write(weights_bl)
                st.write(f'Returns of Portfolio: {return_bl}')
                st.write(f'Volatility of Portfolio: {vol_bl}')
                st.write(f'Sharpe Ratio of Portfolio: {sharpe_mdp}')
                    
    else:
        st.write('Click the button above to compute the Black Litterman portfolio.')
        


elif portfolio_choice == 'Equally weighted':
    st.write('Equally Weighted Portfolio')
    markets = st.multiselect('Select markets', ['US', 'EU', 'EM'])
    sectors = st.multiselect('Select sectors', ['Holding Companies', 'Utilities', 'Industrial & Commercial Services', 
                                                'Banking & Investment Services', 'Healthcare Services & Equipment',
                                                'Chemicals', 'Consumer Goods Conglomerates', 'Technology Equipment', 
                                                'Software & IT Services', 'Real Estate', 'Energy - Fossil Fuels',
                                                'Industrial Goods', 'Applied Resources', 'Mineral Resources', 
                                                'Cyclical Consumer Products', 'Transportation', 'Retailers'])
    
    # Add a button to trigger computation
    if st.button('Compute Equally Weighted Portfolio'):
        if 'eq_results' not in st.session_state or markets != st.session_state.markets or sectors != st.session_state.sectors:
            # If eq results are not already computed, call EW to calculate and store the results
            weights_eq, return_eq, vol_eq, sharpe_eq = EW(markets, sectors)
            st.session_state.sectors = sectors
            st.session_state.markets = markets
        else:
            # Use cached results
            weights_eq = st.session_state.eq_results['weights_eq']
            return_eq = st.session_state.eq_results['return_eq']
            vol_eq = st.session_state.eq_results['vol_eq']
            sharpe_eq = st.session_state.eq_results['sharpe_eq']

        if 'eq_plot' in st.session_state:  # Retain the plot if it exists
            st.pyplot(st.session_state.eq_plot)

        st.write(weights_eq)
        st.write(f'Returns of Portfolio: {return_eq}')
        st.write(f'Volatility of Portfolio: {vol_eq}')
        st.write(f'Sharpe Ratio of Portfolio: {sharpe_eq}')
    else:
        st.write('Click the button above to compute the equally weighted portfolio.')
        st.session_state.sectors = None
        st.session_state.markets = None



###### Bot Control in Sidebar
st.sidebar.title("Trading Bot Control")
if st.sidebar.checkbox("Activate Trading Bot", value=False):
    if not st.session_state.bot_active:
        st.sidebar.warning("Starting the bot...")
        start_hft_bot()
        st.session_state.bot_active = True
        st.sidebar.success("Bot is running!")
    else:
        st.sidebar.info("Bot is already active.")
else:
    if st.session_state.bot_active:
        stop_hft_bot()
        st.sidebar.info("Bot has been stopped.")
######

if st.session_state.bot_active:
    st.success("The bot is currently running.")
else:
    st.warning("The bot is not active.")


# Placeholder for future functionality
st.write('This is a placeholder for future functionality.')



