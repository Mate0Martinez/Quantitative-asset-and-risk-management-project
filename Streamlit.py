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
import Portfolio_classes as pc
#from TradeBot.Trading_bot import scheduler
#import threading


#DECLARATION OF FUNCTIONS
#functions used when a button is clicked
#the idea is that when a button is clicked the script will run the function this allows to run the function only when the button is clicked
#else the whole thing runs when the page is loaded
@st.cache_data
def load_data():
    prices = pd.read_excel('SP500 for Code.xlsx', sheet_name='SP50 2015', index_col=0, parse_dates=True)
    prices = prices.loc['2019-12-31':'2024-10-15']
    return prices


def plot_efficient_frontier_without_risky(shortyes):
    prices = load_data()
    efficient_frontier = pc.EfficientFrontier(prices, short = shortyes)
    frontier = efficient_frontier.get_efficient_frontier()
    gammas = np.linspace(-5, 5, 500)
    gamma_zero_index = np.argmin(np.abs(gammas))
    fig, ax = plt.subplots()
    ax.plot([f[1] for f in frontier], [f[0] for f in frontier], '-', label="Efficient Frontier")
    ax.plot(frontier[gamma_zero_index][1], frontier[gamma_zero_index][0], color='r', marker='D', label='Minimum Variance Portfolio')
    ax.set_title(f'Efficient Frontier - Short-selling: {shortyes}')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    if shortyes is False:
        ax.set_ylim(-0.5, 1)
        ax.set_xlim(0, 0.8)
    else:
        ax.set_ylim(-4, 9)
        ax.set_xlim(0, 2.25)
    st.pyplot(fig)
   

def plot_efficient_frontier_with_risky(risk_free_rate, shortyes):
    prices = load_data()
    efficient_frontier = pc.EfficientFrontier(prices, risk_free_rate=risk_free_rate,short=shortyes)
    frontier = efficient_frontier.get_efficient_frontier()
    fig, ax = plt.subplots()
    gammas = np.linspace(-5, 5, 500)
    gamma_zero_index = np.argmin(np.abs(gammas))
    ax.plot([f[1] for f in frontier[0]], [f[0] for f in frontier[0]], '-', label="Efficient Frontier")
    ax.plot(frontier[0][gamma_zero_index][1], frontier[0][gamma_zero_index][0], color='r', marker='D', label='Minimum Variance Portfolio')
    ax.plot([f[1] for f in frontier[1]], [f[0] for f in frontier[1]], '--', label="Capital Market Line")
    ax.plot(frontier[3], frontier[4], color='g', marker='*', label='Tangency Portfolio', markersize=10)
    ax.set_title(f'Efficient Frontier - Short-selling: {shortyes} - Risk-free rate: {risk_free_rate}')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    if shortyes is False:
        ax.set_ylim(-0.5, 1)
        ax.set_xlim(0, 0.8)
    else:
        ax.set_ylim(-4, 9)
        ax.set_xlim(0, 2.25)
    st.pyplot(fig)

def ERC():
    prices = load_data()
    portfolio = pc.Portfolio(prices)
    weights_erc = portfolio.ERC()
    perf = pc.get_performance(prices,weights_erc)
    fig, ax = plt.subplots()
    ax.plot(perf)
    st.pyplot(fig)
    return weights_erc

def MDP():
    prices = load_data()
    portfolio = pc.Portfolio(prices)
    weights_mdp = portfolio.MDP()
    perf = pc.get_performance(prices,weights_mdp)
    fig, ax = plt.subplots()
    ax.plot(perf)
    st.pyplot(fig)
    return weights_mdp

def EW():
    prices = load_data()
    portfolio = pc.Portfolio(prices)
    weights_eq = portfolio.EW()
    perf = pc.get_performance(prices,weights_eq)
    fig, ax = plt.subplots()
    ax.plot(perf)
    st.pyplot(fig)
    return weights_eq

def BL():
    prices = load_data()
    portfolio = pc.Portfolio(prices)
    weights_bl = "Not ready yet"
    return weights_bl


##########################
# Bot control, à modifier j'ai fais dla merde
##########################

#def start_hft_bot(): #  this function avoids to block the streamlit app from running
#    import logging
#    logging.basicConfig(level=logging.INFO)  # To avoid scheduler warnings
#    bot_thread = threading.Thread(target=scheduler.start)
#    bot_thread.daemon = True  # Ensures the thread will close when the Streamlit app stops
#    bot_thread.start()

##########################
# Bot control, à modifier j'ai fais dla merde
##########################


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
    shortyes = st.checkbox('Short-selling?', value=False)
    riskyes = st.checkbox('Risk-free rate?', value=False)
    if riskyes:
        risk_free_rate = st.number_input('Risk-free rate', value=0.01, step=0.01)
        #risk_free_rate = st.select_slider('Risk-free rate', options=(i for i in np.linspace(0, 0.1, 11)))
    if st.button('Plot Efficient Frontier'):

        if riskyes: #plot the efficient frontier without the risk free asset

            plot_efficient_frontier_with_risky(risk_free_rate, shortyes)
        else: #plot the efficient frontier with the risk free asset
            plot_efficient_frontier_without_risky(shortyes)
   
elif portfolio_choice == 'Equal Risk Contribution':
    st.write('Equal Risk Contribution portfolio')
    weights_erc = ERC()
    st.write(weights_erc)
elif portfolio_choice == 'Most Diversified':
    st.write('Most Diversified portfolio')
    weights_mdp = MDP()
    st.write(weights_mdp)
elif portfolio_choice == 'Black Litterman':
    st.write('Black Litterman portfolio')
    

elif portfolio_choice == 'Equally weighted':
    st.write('Equally weighted portfolio')
    weights_eq = EW()
    st.write(weights_eq)


# Placeholder for future functionality
st.write('This is a placeholder for future functionality.')


##########################
# Bot control, à modifier j'ai fais dla merde
##########################
#'''If bot does not work, don't forget to install dependecies: pip install -r TradeBot/requirements.txt'''
#'''st.sidebar.title("Trading Bot Control")
#if st.sidebar.checkbox("Activate Trading Bot"):
#    st.sidebar.success("Trading Bot is Active")

#   if st.sidebar.button("Start Bot"):
#        st.sidebar.warning("Starting the bot...")
#        start_hft_bot()
#        st.sidebar.success("Bot is running!")
#else:
#    st.sidebar.info("Trading Bot is Inactive")'''
##########################
# Bot control, à modifier j'ai fais dla merde
##########################

