# function is a logarithm of the quotient of the last last 2 timesteps (one hour apart)
import numpy as np



def log_reward_function(history):
        history['portfolio_valuation']=history['portfolio_valuation'].astype(float)
        return np.log(history["portfolio_valuation"].values[ -1] / history["portfolio_valuation"].values[ -2])

# function is the difference between the last 2 time steps
def diff_reward_function(history):
        return history["portfolio_valuation"].values[ -1] - history["portfolio_valuation"].values[ -2]

# An total growth of the portfolio since first timestep
def cumulative_reward_function(history):
        history['portfolio_valuation']=history['portfolio_valuation'].astype(float)
        return float(np.diff([history["portfolio_valuation"].values[ -1] , history["portfolio_valuation"].values[ 0]])/len(history["portfolio_valuation"]))

# Shape Ratio reward function
def sharpe_reward_function(history):
        history['portfolio_valuation']=history['portfolio_valuation'].astype(float)
        return float(history["portfolio_valuation"].values[ -1] /(np.std(history["portfolio_valuation"])+1e-7))

# Final evaluation (total portfolio increase/decrease)
def final_evalulation(history):
        history['portfolio_valuation']=history['portfolio_valuation'].astype(float)
        return (history["portfolio_valuation"].values[ -1] / history["portfolio_valuation"].values[ 0]) - 1

def n_trades(history):
        history['position']=history['position'].astype(float)
        return sum(np.abs(np.diff(history['position'])))
