import numpy as np
import os
from stable_baselines.results_plotter import load_results, ts2xy
import tensorflow as tf

# best_mean_reward0, best_mean_reward1, n_steps0, n_steps1 = -np.inf, -np.inf, 0, 1
variables = {
    "best_mean_reward0": -np.inf,
    "best_mean_reward1": -np.inf,
    "n_steps0": 0,
    "n_steps1": 0
}

# Create log dir
log_dir = "tmp/"
results_dir = "logs/"
benchmark_dir = "benchmarks/"

os.makedirs(log_dir, exist_ok=True)

def tfSummary(tag, val):
    """ Scalar Value Tensorflow Summary
    """
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])

def getBestRewardCallback(args):
    if not os.path.isdir(log_dir + args.prefix):
        os.makedirs(log_dir + args.prefix, exist_ok=True)

    def bestRewardCallback(_locals, _globals, model=None):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global variables
        model_id = _locals['self'].model_id
        
        # Print stats every 1000 calls
        if args.alg == 'ppo2':
            divider = 2
        elif args.alg == 'sac':
            divider = 1
        
        best_mean_reward = variables[("best_mean_reward{}".format(model_id))]
        n_steps = variables[("n_steps{}".format(model_id))]
        if (n_steps + 1) % divider == 0 and (n_steps + 1) / divider > 1:
        

        # if True:
            # Evaluate policy training performance
            # x, y = ts2xy(load_results(log_dir+args.prefix), 'timesteps')
            r = []
            
            with open((log_dir+args.prefix+(('/log/log.model{}.csv').format(model_id))), 'r') as file_handler:
                lines = file_handler.readlines()

                for idx, line in enumerate(reversed(lines)):       

                    if '{' in line:
                        break
                    elif idx <100:

                        r.append(float(line.split(',')[0]))
                    else:
                        break

            total_r = sum(r)

            if total_r > best_mean_reward:
                best_mean_reward = total_r
                

                print('Saving new best model', best_mean_reward, model_id)
                if model is not None:
                    model.save(log_dir + args.prefix + '/' + str(n_steps) +('_best_model{}.pkl').format(model_id))
                else:
                    _locals['self'].save(log_dir + args.prefix + '/' + str(n_steps) +('_best_model{}.pkl').format(model_id))






            # if len(x) > 0:
            #     mean_reward = np.mean(y[-100:])
            #     # print(mean_reward)
            #     # print(x[-1], 'timesteps')
            #     # print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            #     # New best model, you could save the agent here
            #     if mean_reward > best_mean_reward:
            #         best_mean_reward = mean_reward
            #         # Example for saving best model
            #         print("Saving new best model",best_mean_reward)
            #         if model is not None:
            #             model.save(log_dir + args.prefix + '/' + str(n_steps) +'_best_model.pkl')
            #         else:
            #             _locals['self'].save(log_dir + args.prefix + '/' + str(n_steps) +'_best_model.pkl')
            #     elif (n_steps + 1) % (divider*200) == 0:
            #         # if not os.path.exists(log_dir + args.prefix + '/checkpoints'):
            #         #     os.makedirs(log_dir + args.prefix + '/checkpoints')
            #         # print("Saving checkpoint",best_mean_reward)
            #         # if model is not None:
            #         #     model.save(log_dir + args.prefix + '/checkpoints/' + str(n_steps) +'_Check_model.pkl')
            #         # else:
            #         #     _locals['self'].save(log_dir + args.prefix + '/checkpoints/' + str(n_steps) +'_Check_model.pkl')
            #         pass
                    
            #     if model is not None:
            #         model.save(log_dir + args.prefix + '/model.pkl')
            #     else:
            #         _locals['self'].save(log_dir + args.prefix + '/model.pkl')
       
        n_steps += 1
        
        variables[("best_mean_reward{}".format(model_id))] = best_mean_reward
        variables[("n_steps{}".format(model_id))] = n_steps
        # if True:

        return True

    return bestRewardCallback

def rmsLogging(_locals, globals_):
    self_ = _locals['self']

    if self_.done:
        summary = tf.Summary(value=[tf.Summary.Value(tag='rms/com', simple_value=self_.info['rms_com'])])
        _locals['writer'].add_summary(summary, self_.num_timesteps)

        summary = tf.Summary(value=[tf.Summary.Value(tag='rms/phi', simple_value=self_.info['rms_phi'])])
        _locals['writer'].add_summary(summary, self_.num_timesteps)

        summary = tf.Summary(value=[tf.Summary.Value(tag='rms/theta', simple_value=self_.info['rms_theta'])])
        _locals['writer'].add_summary(summary, self_.num_timesteps)
    return True


def logDir():
    return log_dir

def resultsDir():
    return results_dir

def benchmarkDir():
    return benchmark_dir