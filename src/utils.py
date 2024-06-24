import matplotlib.pyplot as plt
from IPython import display

plt.ion()

class Plotter:

    def __init__(self):
        self.scores = []
        self.mean_scores = []
        self.n_updates = 0
        self.total = 0
    
    def add_score(self, score):

        self.n_updates += 1
        self.total += score
        mean_score = self.total / self.n_updates

        self.scores.append(score)
        self.mean_scores.append(mean_score)

    def plot(self):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(self.scores)
        plt.plot(self.mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(self.scores)-1, self.scores[-1], str(self.scores[-1]))
        plt.text(len(self.mean_scores)-1, self.mean_scores[-1], str(self.mean_scores[-1]))
        plt.show(block=False)
        plt.pause(.1)
