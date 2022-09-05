import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

VALUE_TO_CARD = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8',
                 7: '9', 8: '10', 9: 'Jack', 10: 'Queen', 11: 'King', 12: 'Ace'}


class Player:
    winnings = 0
    # Takes in state (containing whatever) and makes a decision based on it

    def act(self, state):
        pass


class LearningPlayer(Player):
    # def act(self, state):
    #     pass

    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.5, symbol=0):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.greedy = []
        self.symbol = symbol
        self.state = {}

        self.valueBet = np.zeros(13)
        self.valueCheck = np.zeros(13)
        self.valueCall = np.zeros(13)
        self.valueFold = np.zeros(13)

        self.hasBet = False
        self.hasChecked = False
        self.hasCalled = False
        self.hasFolded = False

        self.numBet = 0
        self.numCheck = 0
        self.numCall = 0
        self.numFold = 0

    def reset(self):
        self.greedy = []
        self.state = {}

    def set_state(self, state):
        self.greedy.append(True)
        self.state.append(state)

    # choose an action based on the state
    # This is where epsilon-greedy is implemented
    def act(self,state):

        self.hasBet = False
        self.hasChecked = False
        self.hasCalled = False
        self.hasFolded = False

        greedyChoice = 0

        if not (state['opponent_raised']):
            if self.valueBet[state['card']] > self.valueCheck[state['card']]:
                greedyChoice = 1
        elif self.valueCall[state['card']] > self.valueFold[state['card']]:
            greedyChoice = 1



        # With probability epsilon, we select a random action
        r = np.random.rand()
        #if r < self.epsilon:
        if r > self.epsilon:
           action = greedyChoice
        else:
           action = random.choice((0,1))

        if not (state['opponent_raised']) :
            if action == 1:
                self.hasBet = True
                self.numBet += 1
            else:
                self.hasChecked = True
                self.numCheck += 1
        else:
            if action == 1:
                self.hasCalled = True
                self.numCall += 1
            else:
                self.hasFolded = True
                self.numFold += 1
        # action.append(self.symbol)
        #self.greedy[-1] = False
        return action

        # values = []
        # # CALCULATE POSSIBLE VALUES HERE
        #
        # # to select one of the actions of maximum value
        # np.random.shuffle(values)
        # values.sort(key=lambda x: x[0], reverse=True)
        # action = values[0]
        # action.append(self.symbol)
        return action

    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.valueBet, f)
            pickle.dump(self.numBet, f)
            pickle.dump(self.valueCheck, f)
            pickle.dump(self.numCheck, f)
            pickle.dump(self.valueCall, f)
            pickle.dump(self.numCall, f)
            pickle.dump(self.valueFold, f)
            pickle.dump(self.numFold, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            #self.estimations = pickle.load(f)
            self.valueBet = pickle.load(f)
            self.numBet = pickle.load(f)
            self.valueCheck = pickle.load(f)
            self.numCheck = pickle.load(f)
            self.valueCall = pickle.load(f)
            self.numCall = pickle.load(f)
            self.valueFold = pickle.load(f)
            self.numFold = pickle.load(f)


class FixedStrategyPlayer(Player):
    def act(self, state):
        '''Very simple fixed strategy - bid 1 with probability equal to the likelihood that opponent's card is lower'''
        return 1 if state['card']/12 > random.random() else 0


class HumanPlayer(Player):
    def act(self, state):
        print(f"Your card is {VALUE_TO_CARD[state['card']]}")
        action = ''
        while action not in ['0', '1']:  # Make sure we get a valid input
            print('Input the amount to bid (either 0 or 1)')
            action = input()

        return int(action)

# Training phase
# epochs is the number of games to play during the training phase


def train(epochs, print_every_n=500):
    player1 = LearningPlayer(epsilon=0.01, symbol=0)
    player2 = LearningPlayer(epsilon=0.01, symbol=1)
    game = Game(player1, player2, epochs)
    player1ValueBet, player1ValueCheck, player1ValueCall,player1ValueFold,player2ValueBet, player2ValueCheck, player2ValueCall,player2ValueFold = game.play(train=True)
    player1.save_policy()
    player2.save_policy()
    for i in range(1):
        plt.plot(range(100000),np.transpose(player1ValueBet[i,:]),'-')
    plt.show()
    print(player1.valueBet, player1.valueCheck, player1.valueCall,player1.valueFold)


class Game():
    p1: Player
    p2: Player
    rounds: int

    def __init__(self, p1, p2, rounds):
        self.p1 = p1
        self.p2 = p2
        self.rounds = rounds

    def play(self, train=False):
        player1ValueBet = np.zeros((13,self.rounds))
        player2ValueBet = np.zeros((13, self.rounds))
        player1ValueCheck = np.zeros((13, self.rounds))
        player2ValueCheck = np.zeros((13, self.rounds))
        player1ValueCall = np.zeros((13, self.rounds))
        player2ValueCall = np.zeros((13, self.rounds))
        player1ValueFold = np.zeros((13, self.rounds))
        player2ValueFold = np.zeros((13, self.rounds))
        for i in range(self.rounds):
            if train == False:
                print(f'\nROUND {i+1}')
                print(
                    f'P1 currently has ${self.p1.winnings}; P2 has ${self.p2.winnings}')
            # Each round starts with each player ante $1
            p1_bid = 1
            p2_bid = 1

            p1_card, p2_card = self.deal_cards()
            p1_state = {'card': p1_card, 'opponent_raised': False}
            p2_state = {'card': p2_card, 'opponent_raised': False}

            # At this point, P1 bidding 0 is NOT folding, but P2 is
            p1_call = self.p1.act(p1_state)
            p1_bid += p1_call

            if(p1_call == 1):
                p2_state['opponent_raised'] = True
            p2_call = self.p2.act(p2_state)
            p2_bid += p2_call

            if p1_call == 1 and p2_call == 0:  # If P2 bids 0, that is considered folding
                self.end_round(1, p1_bid, p2_bid, p1_card, p2_card,train)
            else:
                if p1_call == 0 and p2_call == 1:  # Give P1 chance to call if P2 raised from 0 to 1
                    if train == False:
                        print('P2 raised to $1. P1 can call or fold')
                    p1_state['opponent_raised'] = True
                    p1_call2 = self.p1.act(p1_state)

                    if p1_call2 == 0:  # Bidding 0 here IS folding
                        self.end_round(2, p1_bid, p2_bid, p1_card, p2_card,train)
                        continue
                # To reach here, both bids must be either 0 or 1
                self.end_round(0, p1_bid, p2_bid, p1_card, p2_card,train)
            player1ValueBet[:,i] = self.p1.valueBet
            player2ValueBet[:,i] = self.p2.valueBet
            player1ValueCheck[:,i] = self.p1.valueCheck
            player2ValueCheck[:,i] = self.p2.valueCheck
            player1ValueCall[:,i] = self.p1.valueCall
            player2ValueCall[:,i] = self.p2.valueCall
            player1ValueFold[:,i] = self.p1.valueFold
            player2ValueFold[:,i] = self.p2.valueFold
            if i % print_every_n == 0 and train == True and i != 0:
                print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (
                    i, player1_win / (i + 1), player2_win / (i + 1)))
        return player1ValueBet, player1ValueCheck, player1ValueCall,player1ValueFold,player2ValueBet, player2ValueCheck, player2ValueCall,player2ValueFold
    def deal_cards(self):
        # Game is played with a single-suit (13-card) deck. Starting from 0 makes our
        # math easier i.e. probability for card X that opponent's card is lower is x/12
        p1_card = random.randint(0, 12)
        p2_card = p1_card
        while p2_card == p1_card:  # Make sure p1 and p2 get different cards
            p2_card = random.randint(0, 12)

        return p1_card, p2_card

    def end_round(self, winner, p1_bid, p2_bid, p1_card, p2_card,train):
        reward = 0
        global player1_win
        global player2_win
        self.p1.winnings -= p1_bid
        self.p2.winnings -= p2_bid
        if not train:
            print(
                f"P1's card was {VALUE_TO_CARD[p1_card]}; P2's card was {VALUE_TO_CARD[p2_card]}")
        if winner == 1:
            if not train:
                print(f'P2 folded! P1 wins by default, P1 gains ${p2_bid}')
            player1_win += 1
            self.p1.winnings += p2_bid + p1_bid
            reward = p2_bid + 2
        elif winner == 2:
            if not train:
                print(f'P1 folded! P2 wins by default, P2 gains ${p1_bid}')
            player2_win += 1
            self.p2.winnings += p1_bid + p2_bid
            reward = 2 - p1_bid
        elif winner == 0:
            if p1_card > p2_card:
                if not train:
                    print(f'P1 wins by having a bigger card, P1 gains ${p2_bid}')
                player1_win += 1
                self.p1.winnings += p2_bid + p1_bid
                reward = p2_bid
            else:
                if not train:
                    print(f'P2 wins by having a bigger card, P2 gains ${p1_bid}')
                player2_win += 1
                self.p2.winnings += p1_bid + p2_bid
                reward = 2 - p1_bid
        else:
            raise RuntimeError('Invalid winner provided to end_round()')

        if (isinstance(self.p1,LearningPlayer)):
            if self.p1.hasBet:
                self.p1.valueBet[p1_card] = (self.p1.valueBet[p1_card] * (self.p1.numBet - 1) + reward)/self.p1.numBet
            if self.p1.hasChecked:
                self.p1.valueCheck[p1_card] = (self.p1.valueCheck[p1_card] * (self.p1.numCheck - 1) + reward) / self.p1.numCheck
            if self.p1.hasCalled:
                self.p1.valueCall[p1_card] = (self.p1.valueCall[p1_card] * (self.p1.numCall - 1) + reward) / self.p1.numCall
            if self.p1.hasFolded:
                self.p1.valueFold[p1_card] = (self.p1.valueFold[p1_card] * (self.p1.numFold - 1) + reward) / self.p1.numFold
        if (isinstance(self.p2, LearningPlayer)):
            reward = -1 * (reward - 4)
            if self.p2.hasBet:
                self.p2.valueBet[p2_card] = (self.p2.valueBet[p2_card] * (self.p2.numBet - 1) + reward) / self.p2.numBet
            if self.p2.hasChecked:
                self.p2.valueCheck[p2_card] = (self.p2.valueCheck[p2_card] * (self.p2.numCheck - 1) + reward) / self.p2.numCheck
            if self.p2.hasCalled:
                self.p2.valueCall[p2_card] = (self.p2.valueCall[p2_card] * (self.p2.numCall - 1) + reward) / self.p2.numCall
            if self.p2.hasFolded:
                self.p2.valueFold[p2_card] = (self.p2.valueFold[p2_card] * (self.p2.numFold - 1) + reward) / self.p2.numFold


if __name__ == '__main__':
    global player1_win
    player1_win = 0
    global player2_win
    player2_win = 0
    global print_every_n
    print_every_n = 10000
    train(int(1e5),print_every_n=print_every_n)  # training phase first


    # compete(int(1e3)) # then two AI complete against each other
    # playerAI = LearningPlayer()
    # playerAI.load_policy()
    # game = Game(HumanPlayer(), playerAI, 10)
    # game.play()