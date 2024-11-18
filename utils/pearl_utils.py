# Import all the Pearl related methods
import numpy as np
from tqdm import tqdm

from Pearl.pearl.action_representation_modules.one_hot_action_representation_module import \
    OneHotActionTensorRepresentationModule
from Pearl.pearl.history_summarization_modules.lstm_history_summarization_module import \
    LSTMHistorySummarizationModule
from Pearl.pearl.neural_networks.sequential_decision_making.actor_networks import \
    VanillaContinuousActorNetwork
from Pearl.pearl.neural_networks.sequential_decision_making.q_value_networks import \
    VanillaQValueNetwork
from Pearl.pearl.pearl_agent import PearlAgent
from Pearl.pearl.policy_learners.exploration_modules.common.normal_distribution_exploration import \
    NormalDistributionExploration
from Pearl.pearl.policy_learners.sequential_decision_making.deep_q_learning import \
    DeepQLearning
from Pearl.pearl.policy_learners.sequential_decision_making.deep_sarsa import \
    DeepSARSA
from Pearl.pearl.policy_learners.sequential_decision_making.double_dqn import \
    DoubleDQN
from Pearl.pearl.policy_learners.sequential_decision_making.td3 import TD3
from Pearl.pearl.replay_buffers import BasicReplayBuffer
from Pearl.pearl.utils.functional_utils.train_and_eval.online_learning import \
    online_learning


from Pearl.pearl.utils.instantiations.environments.gym_environment import \
    GymEnvironment
import optuna
from utils. utils import make_hidden_dims
import pickle
from utils.reward_functions import log_reward_function,cumulative_reward_function,sharpe_reward_function

def create_dqn_model(

        observation_space_dim=25, 
        action_space_dim=2,
        hidden_dims=[64, 64], 
        training_rounds=20,
        learning_rate = 0.001,
        discount_factor = 0.99,
        batch_size = 128,
        target_update_freq = 10,
        soft_update_tau = 0.75,  # a value of 1 indicates no soft updates
        is_conservative = False,
        conservative_alpha = False,
        replay_buffer_size = 10_000,
        lstm=True,
        history_length=168,
        **kwargs):
    
    agent = PearlAgent(
        policy_learner=DeepQLearning(
        state_dim=observation_space_dim,
        action_space=action_space_dim, 
        hidden_dims=hidden_dims,
        training_rounds=training_rounds,
        learning_rate = learning_rate,
        discount_factor = discount_factor,
        batch_size = batch_size,
        target_update_freq = target_update_freq,
        soft_update_tau = soft_update_tau,  # a value of 1 indicates no soft updates
        is_conservative = is_conservative,
        conservative_alpha = conservative_alpha,
        action_representation_module=OneHotActionTensorRepresentationModule(
            max_number_actions=action_space_dim
            ),

        ),
        replay_buffer=BasicReplayBuffer(replay_buffer_size),
        history_summarization_module=LSTMHistorySummarizationModule(
                                            observation_dim=observation_space_dim,
                                            action_dim=action_space_dim,
                                            hidden_dim=observation_space_dim,
                                            history_length=history_length,
                                            ) if lstm else None,
        
        )
    return agent

def create_ddqn_model(

        observation_space_dim=25, 
        action_space_dim=2,
        hidden_dims=[64, 64], 
        training_rounds=20,
        learning_rate = 0.001,
        discount_factor = 0.99,
        batch_size = 128,
        target_update_freq = 10,
        soft_update_tau = 0.75,  # a value of 1 indicates no soft updates
        is_conservative = False,
        conservative_alpha = False,
        replay_buffer_size = 10_000,
        history_length=168,
        lstm=True,
        **kwargs):


        
    agent = PearlAgent(
        policy_learner=DoubleDQN(
        state_dim=observation_space_dim,
        action_space=action_space_dim, 
        hidden_dims=hidden_dims,
        training_rounds=training_rounds,
        learning_rate = learning_rate,
        discount_factor = discount_factor,
        batch_size = batch_size,
        target_update_freq = target_update_freq,
        soft_update_tau = soft_update_tau,  # a value of 1 indicates no soft updates
        is_conservative = is_conservative,
        conservative_alpha = conservative_alpha,
        action_representation_module=OneHotActionTensorRepresentationModule(
            max_number_actions=action_space_dim
        ),
    ),
    replay_buffer=BasicReplayBuffer(replay_buffer_size),
    history_summarization_module=LSTMHistorySummarizationModule(
                                        observation_dim=observation_space_dim,
                                        action_dim=action_space_dim,
                                        hidden_dim=observation_space_dim,
                                        history_length=history_length,
                                        ) if lstm else None,
    )
    return agent


def train_pearl_model(agent, env, n_epochs=10,learn_after_episode=True,learning_steps=20):

    env=GymEnvironment(env)
    bar=tqdm(range(n_epochs))
    info = online_learning(
        agent=agent,
        env=env,
        number_of_episodes=n_epochs if learn_after_episode==True else None,
        number_of_steps=168*n_epochs if learn_after_episode==False else None,
        learn_after_episode=learn_after_episode,  
        print_every_x_steps=1,
        print_every_x_episodes=1,   # print returns after every 10 episdoes

        learn_every_k_steps= learning_steps if learn_after_episode==False else None, # print returns after every 10 steps
        record_period=169,
            # instead of updating after every environment interaction, Q networks are updates at the end of each episode
        seed=0
    )


    return agent

def test_pearl_model( agent,env,n_samples=100):

    env = GymEnvironment(env)
    profits=[]
    n_trades=[]
    for i in range(n_samples):
        observation, action_space = env.reset()
        agent.reset(observation, action_space)
        done = False
        while not done:
            action = agent.act(exploit=True)
            action_result = env.step(action)
            agent.observe(action_result)
            # no agent.learn() while testing
            done = action_result.done
        
        # Get back the original test environment (get it out of its Pearl wrapper)
        test_env=env.env.unwrapped
        profits.append(test_env.historical_info['portfolio_valuation',-1])
        n_trades.append(sum(np.abs(np.diff(test_env.historical_info['position']))))

    return np.mean(profits),np.mean(n_trades)

def load_agent_from_study(study_path,study_name,observation_space_dim=30,action_space_dim=2,version=1):
    study = optuna.create_study(study_name=study_name,
                            directions=["maximize", "maximize"],
                            storage=study_path,
                            load_if_exists=True,
                            )
    best_trials=study.best_trials
    best_trail=best_trials[version]
    best_params=best_trail.params
    
    algo=best_params.pop('algorithm')

    learning_params={'learn_after_episode':best_params.pop('learn_after_episode'),
                        'learning_steps':best_params.pop('learning_steps'),
                        'n_epochs':best_params.pop('n_epochs'),
                        'reward_function':best_params.pop('reward_function')
                        }
    best_params['hidden_dims']=make_hidden_dims(n_layers=best_params.pop('n_layers'),n_units=best_params.pop('n_units'))
    best_params['lstm']=best_params.pop('use_lstm') if 'use_lstm' in best_params else best_params.pop('lstm')
    best_params['observation_space_dim']=observation_space_dim
    best_params['action_space_dim']=action_space_dim
    if algo=='dqn':
        agent=create_dqn_model(**best_params)
    elif algo=='ddqn':
        agent=create_ddqn_model(**best_params)
    
    return agent,learning_params

def load_agent_weights(agent,weight_path):
    weights=pickle.load(open(weight_path,'rb'))
    agent.policy_learner.load_state_dict(weights)
    return agent

def train_production_agent(agent,learning_params,train_env,test_env,save_path):
    reward_functions=[log_reward_function,cumulative_reward_function,sharpe_reward_function]

    reward_func=reward_functions[learning_params.pop('reward_function')]
    train_env.reward_function=reward_func
    test_env.reward_function=reward_func
    agent=train_pearl_model(agent,train_env,**learning_params)

    profit,n_trades=test_pearl_model(agent,test_env)
    print(f"Testing Return AVG Profit: {profit}, AVG Number of Trades: {n_trades}")
    agent=train_pearl_model(agent,test_env,**learning_params)
    pickle.dump(agent.policy_learner.state_dict(),open(save_path,'wb'))
    return agent,profit,n_trades