
import math
import random

import matplotlib.pyplot as plt
import numpy as np

class Bernoulli:

    def __init__(self,p):
        # create a Bernoulli arm with mean p
        self.mean = p
        self.variance = p*(1-p)

    def sample(self):
        # generate a reward from a Bernoulli arm 
        return float(random.random() < self.mean)


class Exponential:
    def __init__(self,p):
        # create an Exponential arm with parameter p
        self.p = p
        self.mean = 1 / p
        self.variance = 1 / (p * p)

    def sample(self):
        # generate a reward from an Exponential arm 
        return random.expovariate(self.p)


class Bandit:
    def __init__(self, arms):
        """given a list of arms, create the MAB environnement"""
        self.arms = arms
        self.nbArms = len(arms)
        self.means = [arm.mean for arm in arms]
        self.bestarm = np.argmax(self.means)
    
    def generateReward(self,arm):
        return - self.arms[arm].sample()


class FTL:
    """
    Follow the leader (a.k.a. greedy strategy)
    """
    def __init__(self, nbArms):
        self.nbArms = nbArms
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)

    def randmax(self, A):
        maxValue=max(A)
        index = [i for i in range(len(A)) if A[i] == maxValue]
        return np.random.choice(index)

    def chooseArmToPlay(self):
        if (min(self.nbDraws) == 0):
            return self.randmax(-self.nbDraws)
        else:
            return self.randmax(self.cumRewards / self.nbDraws)

    def receiveReward(self, arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm] + reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1

    def name(self):
        return "FTL"


class UniformExploration:
    """
    A strategy that uniformly explores arms
    """
    def __init__(self, nbArms):
        self.nbArms = nbArms
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
    
    def chooseArmToPlay(self):
        return np.random.randint(0, self.nbArms)

    def receiveReward(self, arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm] + reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1

    def name(self):
        return "Uniform"



def OneBanditOneLearnerOneRun(bandit, strategy, timeHorizon):
    """
    Run a bandit strategy (strategy) on a MAB instance (bandit) for (timeHorizon) time steps
    output : sequence of arms chosen, sequence of rewards obtained
    """
    selections = []
    rewards = []
    strategy.clear() # reset previous history
    for t in range(timeHorizon):
        # choose the next arm to play with the bandit algorithm
        arm = strategy.chooseArmToPlay()

        # get the reward of the chosen arm
        reward = bandit.generateReward(arm)

        # update the algorithm with the observed reward
        strategy.receiveReward(arm, reward)

        # store what happened
        selections.append(arm)
        rewards.append(reward)
        
    return selections, rewards


class UCB1:
    """
    UCB1 with parameter alpha
    """
    def __init__(self, nbArms, alpha=1/2):
        self.nbArms = nbArms
        self.alpha = alpha
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.t = 0
    
    def chooseArmToPlay(self):
        """
        Choose the arm to play according to the UCB1 strategy
        """
        if any(self.nbDraws == 0):
            return np.argmin(self.nbDraws)
        UCB = (
            (self.cumRewards / self.nbDraws) + 
            np.sqrt(
                self.alpha * np.log(self.t) / 
                (self.nbDraws)
            )
        )
        return np.argmax(UCB)

    def receiveReward(self, arm, reward):
        self.t = self.t + 1
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.cumRewards[arm] = self.cumRewards[arm] + reward

    def name(self):
        return "UCB"


class ThompsonSampling:
    """
    Thompson Sampling with Beta(a,b) prior and Bernoulli likelihood
    """
    def __init__(self, nbArms, alpha=1, beta=1):
        self.nbArms = nbArms
        self.alpha = alpha
        self.beta = beta
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.t = 0

    def beta_function(self, arm_id):
        return np.random.beta(
            self.alpha + self.cumRewards[arm_id],
            self.beta + self.nbDraws[arm_id] - self.cumRewards[arm_id]
        )
    
    def chooseArmToPlay(self):
        """
        Choose the arm to play according to the UCB1 strategy
        """
        thompson_sample = np.array([
            self.beta_function(arm_id) for arm_id in range(self.nbArms)
        ])
        return np.argmax(thompson_sample)
        
    def receiveReward(self, arm, reward):
        self.t = self.t + 1
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.cumRewards[arm] = self.cumRewards[arm] + reward

    def apply_binarisation_trick(self):
        pass

    def name(self):
        return "Thomson Sampling"

